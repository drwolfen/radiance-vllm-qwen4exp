# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated CPU-offload process for PLE embedding layers.

This module implements a standalone process that:
1. Loads only the :class:`PleOffloadLayer` weights into CPU memory.
2. Accepts per-step computation requests from GPU worker processes.
3. Runs ``forward_impl()`` on CPU, copies results to every TP worker's GPU
   output buffer for the requesting DP rank, and signals the corresponding
   IPC semaphore.

The TP workers within one DP rank receive identical inputs, so the CPU result
is computed once per DP rank and fanned out to all of its TP ranks.

Class structure mirrors the GPU worker pattern in multiproc_executor.py:

  PleOffloadWorkerHandle -- handle held by the spawning GPU worker
  PleOffloadWorker       -- process lifecycle and READY handshake
  PleOffloadRunner       -- owns weights and serves inference requests
"""

import contextlib
import ctypes
import ctypes.util
import functools
import multiprocessing.process
import os
import pickle
import signal
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

import msgspec
import torch
import torch.distributed as dist
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
    mark_as_offload_worker,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.utils.system_utils import decorate_logs, get_mp_context
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.ple_offload.protocol import (
    _PLE_OFFLOAD_REQUEST_DECODER,
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)

_PLE_MMAP_HOST_REGISTER_ENV = "VLLM_PLE_MMAP_HOST_REGISTER"
_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES_ENV = (
    "VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES"
)
_PLE_PINNED_RESERVE_BYTES_ENV = "VLLM_PLE_PINNED_RESERVE_BYTES"
_PLE_RESIDENCY_MODE_ENV = "VLLM_PLE_RESIDENCY_MODE"
_PLE_WORKER_TIMING_ENV = "VLLM_PLE_WORKER_TIMING"
_QWEN38_PLE_QWEIGHT_BYTES = 28_800_138_240
_DEFAULT_PLE_PINNED_RESERVE_BYTES = 16 * 1024**3
_HIP_HOST_REGISTER_DEFAULT = 0


def _strict_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _ple_residency_mode() -> str:
    mode = os.environ.get(_PLE_RESIDENCY_MODE_ENV)
    if mode is None:
        mode = "pinned" if _strict_env_flag(_PLE_MMAP_HOST_REGISTER_ENV) else "ssd"
    mode = mode.lower()
    if mode not in {"ssd", "pinned", "bounded"}:
        raise ValueError(
            f"{_PLE_RESIDENCY_MODE_ENV} must be ssd, pinned, or bounded, got {mode!r}"
        )
    return mode


def _proc_mapping_rss_bytes(address: int, nbytes: int) -> int:
    mapping_end = address + nbytes
    rss_bytes = 0
    overlaps_mapping = False
    with open("/proc/self/smaps", encoding="utf-8") as smaps:
        for line in smaps:
            first_field = line.split(maxsplit=1)[0]
            if "-" in first_field:
                try:
                    start_text, end_text = first_field.split("-", maxsplit=1)
                    region_start = int(start_text, 16)
                    region_end = int(end_text, 16)
                except ValueError:
                    overlaps_mapping = False
                else:
                    overlaps_mapping = (
                        region_start < mapping_end and address < region_end
                    )
            elif overlaps_mapping and line.startswith("Rss:"):
                rss_bytes += int(line.split()[1]) * 1024
    return rss_bytes


def _proc_process_rss_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status did not contain VmRSS")


def _proc_mem_available_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo did not contain MemAvailable")


def _nonnegative_env_int(name: str, default: int) -> int:
    text = os.environ.get(name, str(default))
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {text!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


@functools.cache
def _hip_host_registration_api() -> ctypes.CDLL:
    library_name = ctypes.util.find_library("amdhip64")
    if library_name is None:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidate = Path(rocm_path) / "lib/libamdhip64.so"
        library_name = str(candidate) if candidate.is_file() else "libamdhip64.so"
    runtime = ctypes.CDLL(library_name)
    runtime.hipInit.argtypes = (ctypes.c_uint,)
    runtime.hipInit.restype = ctypes.c_int
    runtime.hipHostRegister.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    runtime.hipHostRegister.restype = ctypes.c_int
    runtime.hipHostUnregister.argtypes = (ctypes.c_void_p,)
    runtime.hipHostUnregister.restype = ctypes.c_int
    runtime.hipGetErrorString.argtypes = (ctypes.c_int,)
    runtime.hipGetErrorString.restype = ctypes.c_char_p
    return runtime


def _hip_error_string(runtime: Any, status: int) -> str:
    get_error_string = getattr(runtime, "hipGetErrorString", None)
    if get_error_string is None:
        return f"hipError_t={status}"
    message = get_error_string(status)
    return message.decode(errors="replace") if message else f"hipError_t={status}"


@dataclass
class HipHostRegistration:
    """Own one HIP page-lock registration until the mmap is released."""

    runtime: Any
    address: int
    nbytes: int
    label: str
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        status = self.runtime.hipHostUnregister(ctypes.c_void_p(self.address))
        if status:
            raise RuntimeError(
                f"hipHostUnregister failed for {self.label} at "
                f"0x{self.address:x} ({self.nbytes} bytes): "
                f"{_hip_error_string(self.runtime, status)}"
            )
        self.closed = True
        logger.info(
            "Unregistered PLE mmap from HIP: %s address=0x%x bytes=%d",
            self.label,
            self.address,
            self.nbytes,
        )


def _register_hip_host_range(
    runtime: Any,
    address: int,
    nbytes: int,
    label: str,
) -> HipHostRegistration:
    if address <= 0:
        raise ValueError(f"PLE mmap {label} has invalid address {address}")
    if nbytes <= 0:
        raise ValueError(f"PLE mmap {label} has invalid byte count {nbytes}")
    status = runtime.hipHostRegister(
        ctypes.c_void_p(address),
        ctypes.c_size_t(nbytes),
        ctypes.c_uint(_HIP_HOST_REGISTER_DEFAULT),
    )
    if status:
        raise RuntimeError(
            f"hipHostRegister failed for {label} at 0x{address:x} "
            f"({nbytes} bytes): {_hip_error_string(runtime, status)}"
        )
    return HipHostRegistration(runtime, address, nbytes, label)


def _ple_diagnostic(message: str, *args: object) -> None:
    if os.environ.get("VLLM_PLE_DIAGNOSTIC", "0") == "1":
        formatted_message = message % args if args else message
        logger.warning("PLE diagnostic: %s", formatted_message)


@dataclass
class PleOffloadOutputTarget:
    """GPU output destination and semaphore for one TP worker."""

    tp_rank: int
    gpu_output_buffer: torch.Tensor  # IPC-mapped GPU buffer for this TP worker
    sem: CpuGpuSemaphore  # semaphore paired with gpu_output_buffer
    copy_stream: torch.cuda.Stream


@dataclass
class PleOffloadInputBuffers:
    """Shared-memory input buffers registered for one DP rank."""

    input_ids_buf: torch.Tensor  # int32 (max_num_tokens,)
    query_start_loc_buf: torch.Tensor  # int32 (max_num_reqs + 1,)
    ngram_context_buf: torch.Tensor | None  # int32 (max_num_reqs, ngram_context_len)


@dataclass
class PleOffloadWorkerHandle:
    """Resources owned by the GPU worker that spawned the offload process."""

    proc: Any
    death_writer: Connection | None
    ready_pipe_reader: Connection | None

    def close(self) -> None:
        """Release all process resources. Safe to call more than once."""
        if self.ready_pipe_reader is not None:
            self.ready_pipe_reader.close()
            self.ready_pipe_reader = None
        if self.death_writer is not None:
            self.death_writer.close()
            self.death_writer = None
        # First allow the child to exit after observing the closed death pipe.
        if self.proc.is_alive():
            self.proc.join(timeout=5)
        # Fall back to SIGTERM if graceful shutdown times out.
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        # Use SIGKILL as the final fallback for a stuck child.
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)


def _init_offload_distributed() -> None:
    """Initialize the single-rank Gloo world required by TP-aware layers."""
    if dist.is_initialized():
        return

    # VocabParallelEmbedding reads the TP process group during construction.
    # The offload process owns the full embedding table, so it uses an isolated
    # TP1/PP1 Gloo world and never joins the GPU workers' NCCL groups.
    store_dir = tempfile.mkdtemp(prefix="vllm_ple_offload_")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{store_dir}/store",
        local_rank=0,
        backend="gloo",
    )
    # initialize_model_parallel reads the active VllmConfig in the current
    # vLLM version. Explicitly configure DP1/TP1/PP1 to match the isolated
    # world, regardless of any DP environment variables inherited from the GPU
    # worker. The real DP/TP configuration is used later for model construction,
    # registration, and request routing.
    offload_config = VllmConfig()
    offload_parallel_config = offload_config.parallel_config
    offload_parallel_config.data_parallel_size = 1
    offload_parallel_config.data_parallel_size_local = 1
    offload_parallel_config.data_parallel_rank = 0
    offload_parallel_config.data_parallel_rank_local = 0
    offload_parallel_config.data_parallel_index = 0
    offload_parallel_config.tensor_parallel_size = 1
    offload_parallel_config.pipeline_parallel_size = 1
    offload_parallel_config.prefill_context_parallel_size = 1
    offload_parallel_config.decode_context_parallel_size = 1
    offload_parallel_config.world_size = 1
    offload_parallel_config.nnodes = 1
    offload_parallel_config.node_rank = 0
    with set_current_vllm_config(offload_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
    logger.info(
        "Distributed environment initialized (backend=gloo, rank=0, world_size=1)."
    )


class PleOffloadWorker:
    """Manage process creation, READY handshake, and the child entry point."""

    READY_STR = "READY"

    @staticmethod
    def make_process(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
    ) -> PleOffloadWorkerHandle:
        """Spawn one CPU offload process for all local DP and TP workers."""
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        proc = context.Process(
            target=PleOffloadWorker.proc_main,
            kwargs={
                "vllm_config": vllm_config,
                "num_workers": num_workers,
                "ipc_addr": ipc_addr,
                "ready_pipe": (ready_reader, ready_writer),
                "death_pipe": death_reader,
            },
            name="PleOffloadWorker",
            daemon=True,
        )

        # Python normally forbids a daemon WorkerProc from spawning children.
        # vLLM owns this process through death_pipe and explicit shutdown, so
        # temporarily clear the daemon flag while the child is created.
        parent = multiprocessing.process._current_process  # type: ignore[attr-defined]
        saved_daemon = parent._config.get("daemon")
        parent._config["daemon"] = False
        try:
            proc.start()
        finally:
            parent._config["daemon"] = saved_daemon
        ready_writer.close()
        return PleOffloadWorkerHandle(
            proc=proc,
            death_writer=death_writer,
            ready_pipe_reader=ready_reader,
        )

    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
        reader = handle.ready_pipe_reader
        if reader is None:
            return
        if not reader.poll(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT):
            raise TimeoutError(
                "PLE offload worker did not become ready within "
                f"{envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT}s."
            )
        try:
            message = reader.recv()
        except EOFError as error:
            raise RuntimeError("PLE offload worker exited during startup") from error
        finally:
            reader.close()
            handle.ready_pipe_reader = None
        if message.get("status") != PleOffloadWorker.READY_STR:
            raise RuntimeError(
                "PLE offload worker failed during startup: "
                f"{message.get('error', 'unknown error')}"
            )
        layer_names = message["layer_names"]
        logger.info(
            "Worker ready - %d PleOffloadLayer(s): %s",
            len(layer_names),
            layer_names,
        )
        if os.environ.get("VLLM_PLE_DIAGNOSTIC", "0") == "1":
            logger.warning(
                "PLE diagnostic: worker ready pid=%s alive=%s exitcode=%s",
                handle.proc.pid,
                handle.proc.is_alive(),
                handle.proc.exitcode,
            )

            def monitor_exit() -> None:
                handle.proc.join()
                logger.error(
                    "PLE diagnostic: worker exited pid=%s exitcode=%s",
                    handle.proc.pid,
                    handle.proc.exitcode,
                )

            threading.Thread(
                target=monitor_exit,
                daemon=True,
                name="PleOffloadExitDiagnostic",
            ).start()

    @staticmethod
    def proc_main(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
        ready_pipe: tuple[Connection, Connection],
        death_pipe: Connection,
    ) -> None:
        """Load PLE weights, accept registrations, and run the request loop."""
        decorate_logs("PleOffloadWorker")
        # This subprocess does not pass through WorkerBase.init_worker(),
        # which is where normal GPU workers load out-of-tree model-loader
        # plugins.  Load them explicitly so the PLE-only weight stream can use
        # formats such as the registered GGUF loader.
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        ready_reader, ready_writer = ready_pipe
        ready_reader.close()
        shutdown_event = threading.Event()

        def monitor_parent() -> None:
            try:
                death_pipe.recv()
            except EOFError:
                logger.info("Parent exited, shutting down.")
                shutdown_event.set()

        def handle_signal(_signum: int, _frame: object) -> None:
            shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        threading.Thread(
            target=monitor_parent,
            daemon=True,
            name="PleOffloadDeathMonitor",
        ).start()

        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        runner: PleOffloadRunner | None = None
        try:
            # The flag lets PleOffloadLayer subclasses execute their complete
            # constructors instead of becoming empty GPU-worker placeholders.
            mark_as_offload_worker()

            # Initialize Gloo before installing the real VllmConfig. This keeps
            # the CPU process in an isolated rank-zero, world-size-one group.
            _init_offload_distributed()

            # Model components read the active VllmConfig while the meta model
            # is constructed, so keep the context around runner initialization.
            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)

            zmq_context = zmq.Context()
            pull_socket = zmq_context.socket(zmq.PULL)
            pull_socket.bind(ipc_addr)
            logger.info(
                "Bound IPC address %s; waiting for %d GPU worker registration(s).",
                ipc_addr,
                num_workers,
            )

            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers)
            ready_writer.send(
                {
                    "status": PleOffloadWorker.READY_STR,
                    "layer_names": sorted(runner.layer_names),
                }
            )
            ready_writer.close()
            ready_writer = None  # type: ignore[assignment]

            runner.busy_loop(pull_socket, shutdown_event)
        except Exception as error:
            logger.exception("Unexpected failure in PLE offload worker.")
            if ready_writer is not None:
                with contextlib.suppress(Exception):
                    ready_writer.send({"status": "FAILURE", "error": repr(error)})
            raise
        finally:
            if runner is not None:
                try:
                    runner.close()
                except Exception:
                    logger.exception("PLE offload runner cleanup failed")
            if pull_socket is not None:
                pull_socket.close(linger=0)
            if zmq_context is not None:
                zmq_context.term()
            if ready_writer is not None:
                ready_writer.close()
            death_pipe.close()


class PleOffloadRunner:
    """Own all discovered PLE tables and serve every local DP rank."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._clamp_input_ids = (
            getattr(vllm_config, "speculative_config", None) is not None
        )
        # name -> PleOffloadLayer (CPU)
        self._layers: dict[str, PleOffloadLayer] = {}
        # dp_rank -> layer_name -> one destination per TP rank
        self._worker_targets: dict[int, dict[str, list[PleOffloadOutputTarget]]] = {}
        # Each (dp_rank, layer_name) pair owns a separate pinned scratch buffer.
        # Sharing one buffer is unsafe because an asynchronous H2D copy may still
        # be reading it when another layer or DP rank starts writing.
        self._pinned_bufs: dict[int, dict[str, torch.Tensor]] = {}
        # Shared-memory inputs are registered once per DP rank by TP rank zero.
        self._input_bufs: dict[int, PleOffloadInputBuffers] = {}
        self._host_registrations: list[HipHostRegistration] = []
        self._residency_mode = _ple_residency_mode()
        self._detailed_timing = _strict_env_flag(_PLE_WORKER_TIMING_ENV)
        self._load_weights()
        logger.info(
            "PLE weights loaded with residency policy=%s; deferring any mmap "
            "host registration until all GPU workers finish model loading "
            "and register their IPC buffers",
            self._residency_mode,
        )

    @property
    def layer_names(self) -> list[str]:
        """Return PleOffloadLayer names in model traversal order."""
        return list(self._layers)

    def _file_backed_qweights(
        self,
    ) -> list[tuple[str, torch.nn.Module, torch.Tensor, Path, int]]:
        """Return validated GGUF PLE mmap tensors in traversal order."""
        mappings: list[tuple[str, torch.nn.Module, torch.Tensor, Path, int]] = []
        seen_ranges: dict[tuple[int, int], str] = {}
        for layer_name, layer in self._layers.items():
            for module_name, module in layer.named_modules():
                mmap_nbytes = getattr(module, "_vllm_gguf_mmap_nbytes", None)
                if mmap_nbytes is None:
                    continue
                label = ".".join(part for part in (layer_name, module_name) if part)
                qweight = getattr(module, "qweight", None)
                if not isinstance(qweight, torch.Tensor):
                    raise RuntimeError(f"PLE mmap owner {label} has no qweight tensor")
                if qweight.device.type != "cpu" or not qweight.is_contiguous():
                    raise RuntimeError(
                        f"PLE mmap qweight {label} must be contiguous CPU storage"
                    )
                tensor_nbytes = qweight.numel() * qweight.element_size()
                if int(mmap_nbytes) != tensor_nbytes:
                    raise RuntimeError(
                        f"PLE mmap qweight {label} reports {mmap_nbytes} bytes, "
                        f"but its tensor spans {tensor_nbytes} bytes"
                    )
                mmap_path_value = getattr(module, "_vllm_gguf_mmap_path", None)
                if not mmap_path_value:
                    raise RuntimeError(f"PLE mmap qweight {label} has no backing path")
                mmap_path = Path(mmap_path_value).resolve(strict=True)
                file_nbytes = mmap_path.stat().st_size
                if file_nbytes != tensor_nbytes:
                    raise RuntimeError(
                        f"PLE mmap file {mmap_path} has {file_nbytes} bytes, "
                        f"but {label} spans {tensor_nbytes} bytes"
                    )
                key = (qweight.data_ptr(), tensor_nbytes)
                previous = seen_ranges.get(key)
                if previous is not None:
                    logger.info(
                        "PLE mmap %s aliases already validated %s", label, previous
                    )
                    continue
                seen_ranges[key] = label
                mappings.append((label, module, qweight, mmap_path, tensor_nbytes))
        return mappings

    def _maybe_register_file_backed_qweights(self) -> None:
        """Opt-in page-lock of the complete file-backed PLE qweight mapping."""
        host_register = _strict_env_flag(_PLE_MMAP_HOST_REGISTER_ENV)
        residency_mode = getattr(self, "_residency_mode", None)
        if residency_mode is None:
            residency_mode = _ple_residency_mode()
        if residency_mode != "pinned":
            if host_register:
                raise RuntimeError(
                    f"{_PLE_MMAP_HOST_REGISTER_ENV}=1 conflicts with "
                    f"{_PLE_RESIDENCY_MODE_ENV}={residency_mode}"
                )
            return
        if not host_register:
            raise RuntimeError(
                f"{_PLE_RESIDENCY_MODE_ENV}=pinned requires "
                f"{_PLE_MMAP_HOST_REGISTER_ENV}=1"
            )
        mappings = self._file_backed_qweights()
        if not mappings:
            raise RuntimeError(
                f"{_PLE_MMAP_HOST_REGISTER_ENV}=1, but no file-backed PLE qweight "
                "mapping was discovered"
            )
        expected_text = os.environ.get(
            _PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES_ENV,
            str(_QWEN38_PLE_QWEIGHT_BYTES),
        )
        try:
            expected_bytes = int(expected_text)
        except ValueError as error:
            raise ValueError(
                f"{_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES_ENV} must be an integer"
            ) from error
        if expected_bytes <= 0:
            raise ValueError(
                f"{_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES_ENV} must be positive"
            )
        total_bytes = sum(item[4] for item in mappings)
        if total_bytes != expected_bytes:
            raise RuntimeError(
                f"PLE mmap registration expected {expected_bytes} bytes, but "
                f"validated {total_bytes} bytes across {len(mappings)} mapping(s)"
            )

        reserve_bytes = _nonnegative_env_int(
            _PLE_PINNED_RESERVE_BYTES_ENV,
            _DEFAULT_PLE_PINNED_RESERVE_BYTES,
        )
        available_bytes = _proc_mem_available_bytes()
        required_bytes = total_bytes + reserve_bytes
        logger.info(
            "PLE pinned residency preflight after GPU registration barrier: "
            "mapping_bytes=%d reserve_bytes=%d required_bytes=%d "
            "mem_available_bytes=%d",
            total_bytes,
            reserve_bytes,
            required_bytes,
            available_bytes,
        )
        if available_bytes < required_bytes:
            raise RuntimeError(
                "PLE pinned residency refused to register: MemAvailable "
                f"is {available_bytes} bytes, below mapping+reserve "
                f"requirement {required_bytes} bytes ({total_bytes}+{reserve_bytes})"
            )

        runtime = _hip_host_registration_api()
        init_status = runtime.hipInit(ctypes.c_uint(0))
        if init_status:
            raise RuntimeError(
                "hipInit failed before PLE mmap registration: "
                f"{_hip_error_string(runtime, init_status)}"
            )
        started_ns = time.perf_counter_ns()
        try:
            for label, module, qweight, mmap_path, nbytes in mappings:
                registration = _register_hip_host_range(
                    runtime,
                    qweight.data_ptr(),
                    nbytes,
                    label,
                )
                self._host_registrations.append(registration)
                # Registered pages must remain resident; file-backed trimming
                # would directly conflict with the page-lock contract.
                module._vllm_gguf_mmap_trim_rows = 0
                logger.info(
                    "Registered complete PLE mmap with HIP: %s path=%s "
                    "address=0x%x bytes=%d mapping_rss_bytes=%d "
                    "process_rss_bytes=%d",
                    label,
                    mmap_path,
                    registration.address,
                    registration.nbytes,
                    _proc_mapping_rss_bytes(
                        registration.address,
                        registration.nbytes,
                    ),
                    _proc_process_rss_bytes(),
                )
        except Exception:
            self._close_host_registrations()
            raise
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        logger.info(
            "PLE HIP mmap registration complete: mappings=%d bytes=%d elapsed_ms=%.3f",
            len(self._host_registrations),
            total_bytes,
            elapsed_ms,
        )

    def _close_host_registrations(self) -> None:
        errors = []
        while self._host_registrations:
            registration = self._host_registrations.pop()
            try:
                registration.close()
            except Exception as error:
                errors.append(error)
                logger.exception("Failed to unregister PLE mmap %s", registration.label)
        if errors:
            raise RuntimeError(
                f"Failed to unregister {len(errors)} PLE HIP mmap range(s)"
            ) from errors[0]

    def _close_file_backed_residency(self) -> None:
        closed: set[int] = set()
        for layer in self._layers.values():
            for module in layer.modules():
                residency = getattr(module, "_vllm_gguf_mmap_residency", None)
                if residency is None or id(residency) in closed:
                    continue
                residency.close()
                closed.add(id(residency))

    def close(self) -> None:
        """Release HIP registrations before the owning qweight mappings die."""
        if hasattr(self, "_layers"):
            self._close_file_backed_residency()
        if hasattr(self, "_host_registrations"):
            self._close_host_registrations()

    def _load_weights(self) -> None:
        """Load only :class:`PleOffloadLayer` subtrees into CPU memory.

        Strategy:
        1. Build the entire model on ``meta`` so non-offloaded parameters use no
           physical memory. PleOffloadLayer constructors explicitly target CPU.
        2. Discover all PleOffloadLayer modules from the complete model.
        3. Stream the checkpoint through a prefix filter so only matching PLE
           tensors are materialized and passed to ``model.load_weights``.
        4. Run post-load processing only on the CPU-owned PLE subtrees.
        """
        model_config = self.vllm_config.model_config
        load_config = self.vllm_config.load_config

        # Step 1: build complete structure, while only PLE subtrees allocate CPU
        # memory. All transformer, MoE, and vision parameters remain on meta.
        logger.info("Initializing model structure for PLE weight discovery ...")
        model_dtype = cast(torch.dtype, model_config.dtype)
        with set_default_torch_dtype(model_dtype), torch.device("meta"):
            model = initialize_model(
                vllm_config=self.vllm_config,
                model_config=model_config,
            )

        # Step 2: preserve named_modules DFS order so CPU execution follows the
        # same layer order as the GPU model forward.
        offload_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not offload_layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but no PleOffloadLayer "
                "was found in the initialized model"
            )
        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        offload_prefixes = tuple(f"{name}." for name in offload_layers)

        # Step 3: filter checkpoint tensors before model.load_weights(). The
        # conditional-generation checkpoint uses HF names such as
        # ``model.language_model.*`` while named_modules exposes mapped vLLM
        # names such as ``language_model.model.*``. Apply the model mapper only
        # for matching, then yield the original pair so load_weights performs
        # its normal single mapping pass.
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        matched_checkpoint_tensors = 0

        def offload_only_iter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal matched_checkpoint_tensors
            for weight_name, tensor in weights:
                mapped_name: str | None = weight_name
                if mapper is not None:
                    mapped_names = mapper.apply_list([weight_name])
                    mapped_name = mapped_names[0] if mapped_names else None
                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    yield weight_name, tensor

        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):
            logger.info(
                "Initializing dummy weights for %d PleOffloadLayer(s) ...",
                len(offload_layers),
            )
            for layer in offload_layers.values():
                initialize_dummy_weights(layer, model_config)
        elif callable(weight_stream := getattr(loader, "get_all_weights", None)):
            all_weights = weight_stream(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))
            if matched_checkpoint_tensors == 0:
                raise RuntimeError(
                    "PLE offload checkpoint filter matched no weights for "
                    f"layers: {sorted(offload_layers)}"
                )

            expected_offload_params = {
                f"{layer_name}.{param_name}"
                for layer_name, layer in offload_layers.items()
                for param_name, _ in layer.named_parameters()
            }
            loaded_offload_entries = {
                name for name in loaded_params if name.startswith(offload_prefixes)
            }
            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
            )
            if missing_offload_params:
                raise RuntimeError(
                    "PLE offload checkpoint did not load all materialized "
                    f"parameters: {missing_offload_params}"
                )
            logger.info(
                "PLE offload matched %d checkpoint tensor(s), loaded %d "
                "offload entries, and verified %d/%d materialized "
                "parameter(s) for layers: %s",
                matched_checkpoint_tensors,
                len(loaded_offload_entries),
                len(loaded_expected_params),
                len(expected_offload_params),
                sorted(offload_layers),
            )
        else:
            raise NotImplementedError(
                "PLE offload requires a streaming or dummy model loader, got "
                f"{type(loader).__name__}"
            )

        # Step 4: post-load processing is restricted to CPU-owned PLE modules;
        # the remainder of the model is still on meta and must not be visited.
        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        self._layers.update(offload_layers)
        del model
        logger.info("PLE weight loading complete.")

    def accept_registrations(
        self,
        pull_socket: zmq.Socket,
        num_workers: int,
    ) -> None:
        """Receive every local DP/TP worker's IPC and shared-memory buffers."""
        logger.info("Waiting for %d GPU worker registration(s) ...", num_workers)
        registrations: list[PleOffloadRegistration] = []
        for index in range(num_workers):
            item = pickle.loads(pull_socket.recv())
            if not isinstance(item, PleOffloadRegistration):
                raise RuntimeError(
                    "Expected PleOffloadRegistration during setup, got "
                    f"{type(item).__name__} ({index + 1}/{num_workers})"
                )
            registrations.append(item)
            logger.info(
                "GPU worker %d registered (dp_rank=%d, tp_rank=%d, layers=%s).",
                item.worker_id,
                item.dp_rank,
                item.tp_rank,
                sorted(item.gpu_output_buffers),
            )

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if num_workers != dp_size * tp_size:
            raise RuntimeError(
                f"Expected {dp_size * tp_size} registrations for DP={dp_size}, "
                f"TP={tp_size}, got {num_workers}"
            )

        registrations_by_dp: dict[int, list[PleOffloadRegistration]] = {}
        for registration in registrations:
            registrations_by_dp.setdefault(registration.dp_rank, []).append(
                registration
            )
        if set(registrations_by_dp) != set(range(dp_size)):
            raise RuntimeError(
                f"Expected DP ranks {set(range(dp_size))}, "
                f"got {set(registrations_by_dp)}"
            )
        for dp_rank, dp_registrations in registrations_by_dp.items():
            tp_ranks = {registration.tp_rank for registration in dp_registrations}
            if tp_ranks != set(range(tp_size)):
                raise RuntimeError(
                    f"DP rank {dp_rank} expected TP ranks {set(range(tp_size))}, "
                    f"got {tp_ranks}"
                )

        for registration in registrations:
            if set(registration.gpu_output_buffers) != set(self.layer_names):
                raise RuntimeError(
                    "Registered PLE layers do not match CPU layers: "
                    f"registered={sorted(registration.gpu_output_buffers)}, "
                    f"cpu={sorted(self.layer_names)}"
                )
            targets_for_dp = self._worker_targets.setdefault(registration.dp_rank, {})
            for layer_name, gpu_buffer in registration.gpu_output_buffers.items():
                target = PleOffloadOutputTarget(
                    tp_rank=registration.tp_rank,
                    gpu_output_buffer=gpu_buffer,
                    sem=CpuGpuSemaphore.from_ipc_tensor(
                        registration.sem_flag_tensors[layer_name]
                    ),
                    copy_stream=torch.cuda.Stream(device=gpu_buffer.device),
                )
                targets_for_dp.setdefault(layer_name, []).append(target)
            # All TP ranks in one DP group receive the same input, so buffers
            # registered by TP rank zero are sufficient for that DP rank.
            if registration.tp_rank == 0:
                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(
                    input_ids_buf=registration.input_ids_buf,
                    query_start_loc_buf=registration.query_start_loc_buf,
                    ngram_context_buf=registration.ngram_context_buf,
                )

        if set(self._input_bufs) != set(range(dp_size)):
            raise RuntimeError(
                "TP rank zero did not register PLE input buffers for every DP "
                f"rank: expected={set(range(dp_size))}, got={set(self._input_bufs)}"
            )

        config = self.vllm_config.model_config.hf_text_config
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        embedding_dim = int(config.ple_embed_dim)
        for dp_rank, layer_targets in self._worker_targets.items():
            self._pinned_bufs[dp_rank] = {}
            for layer_name, targets in layer_targets.items():
                if len(targets) != tp_size:
                    raise RuntimeError(
                        f"PLE layer {layer_name} for DP rank {dp_rank} received "
                        f"{len(targets)} targets, expected {tp_size}"
                    )
                targets.sort(key=lambda target: target.tp_rank)
                self._pinned_bufs[dp_rank][layer_name] = torch.empty(
                    max_tokens,
                    embedding_dim,
                    dtype=self._layers[layer_name].get_offload_output_dtype(
                        self.vllm_config.model_config.dtype
                    ),
                    pin_memory=True,
                )

        logger.info(
            "PLE GPU registration barrier complete: workers=%d; applying "
            "configured mmap residency policy before READY",
            num_workers,
        )
        self._maybe_register_file_backed_qweights()
        logger.info(
            "Registrations complete (dp_size=%d, tp_size=%d, layers=%s).",
            dp_size,
            tp_size,
            sorted(self.layer_names),
        )

    @torch.inference_mode()
    def busy_loop(
        self,
        pull_socket: zmq.Socket,
        shutdown_event: threading.Event,
    ) -> None:
        """Decode and batch available requests by DP rank until shutdown."""
        logger.info("Busy-loop started.")
        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        while not shutdown_event.is_set():
            if pull_socket not in dict(poller.poll(timeout=100)):
                continue

            requests = []
            try:
                requests.append(_PLE_OFFLOAD_REQUEST_DECODER.decode(pull_socket.recv()))
                while True:
                    requests.append(
                        _PLE_OFFLOAD_REQUEST_DECODER.decode(
                            pull_socket.recv(zmq.NOBLOCK)
                        )
                    )
            except zmq.Again:
                pass
            except msgspec.DecodeError as error:
                raise RuntimeError("Unexpected PLE offload request") from error

            self._handle_requests(requests)

    def _handle_requests(self, requests: list[PleOffloadRequest]) -> None:
        """Run requests layer-first so each DP rank can resume promptly."""
        _ple_diagnostic("worker received %d request(s)", len(requests))
        requests_by_dp: dict[int, PleOffloadRequest] = {}
        for request in requests:
            if request.dp_rank not in self._worker_targets:
                logger.warning(
                    "No PLE output targets for dp_rank=%d; skipping request.",
                    request.dp_rank,
                )
                continue
            if request.dp_rank in requests_by_dp:
                logger.warning(
                    "Duplicate PLE request for dp_rank=%d; skipping duplicate.",
                    request.dp_rank,
                )
                continue
            requests_by_dp[request.dp_rank] = request

        # Speculative placeholders are not vocabulary IDs. Normalize each DP
        # input once before all PLE layers consume the shared buffer.
        if self._clamp_input_ids:
            for dp_rank, request in requests_by_dp.items():
                self._input_bufs[dp_rank].input_ids_buf[
                    : request.num_tokens
                ].clamp_min_(0)

        for layer_name, layer in self._layers.items():
            for dp_rank, request in requests_by_dp.items():
                targets = self._worker_targets[dp_rank][layer_name]
                detailed_timing = getattr(self, "_detailed_timing", False)
                total_started_ns = time.perf_counter_ns() if detailed_timing else 0
                _ple_diagnostic(
                    "worker layer start layer=%s dp=%d tokens=%d reqs=%d",
                    layer_name,
                    dp_rank,
                    request.num_tokens,
                    request.num_reqs,
                )

                # The CPU must not overwrite a GPU output buffer until its
                # previous result has been consumed. The GPU runner resets the
                # flag after the complete model forward.
                input_wait_started_ns = time.perf_counter_ns() if detailed_timing else 0
                for target in targets:
                    target.copy_stream.synchronize()
                    target.sem.wait_reset(target.copy_stream)
                input_wait_us = (
                    (time.perf_counter_ns() - input_wait_started_ns) / 1_000
                    if detailed_timing
                    else 0.0
                )

                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (
                    input_bufs.ngram_context_buf[: request.num_reqs]
                    if input_bufs.ngram_context_buf is not None
                    else None
                )
                forward_started_ns = time.perf_counter_ns() if detailed_timing else 0
                result = layer.forward_impl(
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                    ngram_context,
                    output_buffer=self._pinned_bufs[dp_rank][layer_name],
                )
                forward_us = (
                    (time.perf_counter_ns() - forward_started_ns) / 1_000
                    if detailed_timing
                    else 0.0
                )
                _ple_diagnostic(
                    "worker lookup done layer=%s dp=%d shape=%s",
                    layer_name,
                    dp_rank,
                    tuple(result.shape),
                )

                # The result is identical on every TP rank in this DP group.
                # Each copy stream signals only after its DMA completes.
                slices = tuple(slice(0, size) for size in result.shape)
                h2d_started_ns = time.perf_counter_ns() if detailed_timing else 0
                for target in targets:
                    with torch.cuda.stream(target.copy_stream):
                        target.gpu_output_buffer[slices].copy_(
                            result[slices], non_blocking=True
                        )
                        target.sem.signal(target.copy_stream)
                h2d_enqueue_us = (
                    (time.perf_counter_ns() - h2d_started_ns) / 1_000
                    if detailed_timing
                    else 0.0
                )
                if detailed_timing:
                    layer_timing = getattr(layer, "_vllm_ple_last_timing", {})
                    total_us = (time.perf_counter_ns() - total_started_ns) / 1_000
                    logger.info(
                        "PLE worker timing layer=%s dp=%d tokens=%d reqs=%d "
                        "input_wait_us=%.3f ngram_id_us=%.3f "
                        "embedding_gather_dequant_us=%.3f "
                        "output_buffer_copy_us=%.3f forward_us=%.3f "
                        "h2d_enqueue_us=%.3f total_us=%.3f",
                        layer_name,
                        dp_rank,
                        request.num_tokens,
                        request.num_reqs,
                        input_wait_us,
                        float(layer_timing.get("ngram_id_us", 0.0)),
                        float(layer_timing.get("embedding_gather_dequant_us", 0.0)),
                        float(layer_timing.get("output_buffer_copy_us", 0.0)),
                        forward_us,
                        h2d_enqueue_us,
                        total_us,
                    )
                _ple_diagnostic(
                    "worker copies queued layer=%s dp=%d targets=%d",
                    layer_name,
                    dp_rank,
                    len(targets),
                )

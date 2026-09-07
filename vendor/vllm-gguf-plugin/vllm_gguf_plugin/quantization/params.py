# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import functools
import logging
import os
import threading
import weakref
from collections import OrderedDict
from functools import partial, wraps
from math import prod
from pathlib import Path

import gguf
import torch
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .tiered_compaction import (
    allocate_uva_host_owner,
    default_pinned_empty,
    is_tiered_expert_master,
    plan_uva_host_owner,
)

logger = logging.getLogger(__name__)

_PLE_MMAP_PATH_ENV = "GGUF_PLE_MMAP_PATH"
_PLE_MMAP_TRIM_ROWS_ENV = "GGUF_PLE_MMAP_TRIM_ROWS"
_DEFAULT_PLE_MMAP_TRIM_ROWS = 131_072
_PLE_RESIDENCY_MODE_ENV = "VLLM_PLE_RESIDENCY_MODE"
_PLE_BOUNDED_BYTES_ENV = "VLLM_PLE_BOUNDED_BYTES"
_PLE_BOUNDED_CHUNK_BYTES_ENV = "VLLM_PLE_BOUNDED_CHUNK_BYTES"
_PLE_MMAP_READAHEAD_ENV = "VLLM_PLE_MMAP_READAHEAD"
_PLE_RSS_LOG_ROWS_ENV = "VLLM_PLE_RSS_LOG_ROWS"
_DEFAULT_PLE_BOUNDED_BYTES = 4 * 1024**3
_DEFAULT_PLE_BOUNDED_CHUNK_BYTES = 4096
_DEFAULT_PLE_RSS_LOG_ROWS = 131_072
_UVA_HOST_NONCOHERENT_ENV = "RADIANCE_UVA_HOST_NONCOHERENT"
_UVA_HOST_COHERENCE_ENV = "RADIANCE_UVA_HOST_COHERENCE"
_HIP_HOST_MALLOC_COHERENT = 0x40000000
_HIP_HOST_MALLOC_NONCOHERENT = 0x80000000
_MADV_RANDOM = 1
_MADV_WILLNEED = 3
_MADV_DONTNEED = 4
_MADV_DONTDUMP = 16


@functools.cache
def _hip_host_api():
    rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    library = ctypes.CDLL(str(rocm_path / "lib/libamdhip64.so"))
    library.hipHostMalloc.argtypes = (
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    library.hipHostMalloc.restype = ctypes.c_int
    library.hipHostFree.argtypes = (ctypes.c_void_p,)
    library.hipHostFree.restype = ctypes.c_int
    library.hipGetErrorString.argtypes = (ctypes.c_int,)
    library.hipGetErrorString.restype = ctypes.c_char_p
    return library


def _hip_error_string(library, status: int) -> str:
    message = library.hipGetErrorString(status)
    return message.decode(errors="replace") if message else f"HIP error {status}"


def _hip_host_free_noexcept(library, address: int) -> None:
    try:
        status = library.hipHostFree(ctypes.c_void_p(address))
        if status:
            logger.warning(
                "hipHostFree failed for explicit-coherence UVA storage: %s",
                _hip_error_string(library, status),
            )
    except Exception:
        logger.warning(
            "hipHostFree raised while releasing explicit-coherence UVA storage",
            exc_info=True,
        )


@functools.cache
def _log_explicit_uva_coherence(mode: str) -> None:
    logger.warning("Using explicit %s HIP host memory for GGUF UVA", mode)


def _hip_host_empty(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    flag: int,
    mode: str,
) -> torch.Tensor:
    numel = prod(shape)
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device="cpu")

    # HIP pointer attributes are not available to PyTorch until its device
    # context exists. Model loading normally establishes it first, but keep the
    # allocator correct when it is called by an isolated probe as well.
    torch.cuda.current_device()
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    library = _hip_host_api()
    pointer = ctypes.c_void_p()
    status = library.hipHostMalloc(
        ctypes.byref(pointer),
        ctypes.c_size_t(nbytes),
        ctypes.c_uint(flag),
    )
    if status:
        raise RuntimeError(
            f"hipHostMalloc({mode}) failed: {_hip_error_string(library, status)}"
        )
    if pointer.value is None:
        raise RuntimeError(f"hipHostMalloc({mode}) returned a null pointer")

    owner = (ctypes.c_ubyte * nbytes).from_address(pointer.value)
    finalizer = weakref.finalize(owner, _hip_host_free_noexcept, library, pointer.value)
    finalizer.atexit = False
    tensor = torch.frombuffer(owner, dtype=dtype, count=numel).reshape(shape)
    if not tensor.is_pinned():
        raise RuntimeError(f"hipHostMalloc({mode}) storage is not recognized as pinned")
    _log_explicit_uva_coherence(mode)
    return tensor


def _hip_noncoherent_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return _hip_host_empty(
        shape,
        dtype,
        flag=_HIP_HOST_MALLOC_NONCOHERENT,
        mode="noncoherent",
    )


def _hip_coherent_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return _hip_host_empty(
        shape,
        dtype,
        flag=_HIP_HOST_MALLOC_COHERENT,
        mode="coherent",
    )


def _uva_host_coherence() -> str:
    """Resolve the explicit HIP host-allocation mode.

    ``default`` deliberately means the PyTorch pinned allocator. On ROCm its
    effective coherence can also depend on ``HIP_HOST_COHERENT``, so it must
    not be called a coherent baseline in an A/B. The legacy boolean remains a
    compatible alias for explicit ``noncoherent``.
    """
    mode = os.environ.get(_UVA_HOST_COHERENCE_ENV, "default").lower()
    if mode not in {"default", "coherent", "noncoherent"}:
        raise ValueError(
            f"{_UVA_HOST_COHERENCE_ENV} must be default, coherent, or noncoherent"
        )
    legacy = os.environ.get(_UVA_HOST_NONCOHERENT_ENV, "0")
    if legacy not in {"0", "1"}:
        raise ValueError(f"{_UVA_HOST_NONCOHERENT_ENV} must be 0 or 1")
    if legacy == "1":
        if mode not in {"default", "noncoherent"}:
            raise ValueError(
                f"{_UVA_HOST_NONCOHERENT_ENV}=1 conflicts with "
                f"{_UVA_HOST_COHERENCE_ENV}={mode}"
            )
        mode = "noncoherent"
    return mode


def _use_noncoherent_uva_host_memory(param: UninitializedParameter) -> bool:
    return (
        _uva_host_coherence() == "noncoherent"
        and bool(torch.version.hip)
        and getattr(param, "_vllm_uva_pin_memory", False)
    )


def _explicit_hip_uva_empty(
    shape: tuple[int, ...], dtype: torch.dtype, mode: str
) -> torch.Tensor:
    if mode == "coherent":
        return _hip_coherent_empty(shape, dtype)
    if mode == "noncoherent":
        return _hip_noncoherent_empty(shape, dtype)
    raise ValueError(f"explicit HIP UVA allocation does not support mode {mode!r}")


def allocate_uva_host_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate pinned host storage using the configured ROCm cache policy."""
    mode = _uva_host_coherence()
    if mode != "default" and bool(torch.version.hip):
        return _explicit_hip_uva_empty(shape, dtype, mode)
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


@functools.cache
def _libc_madvise():
    libc = ctypes.CDLL(None, use_errno=True)
    madvise = libc.madvise
    madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    madvise.restype = ctypes.c_int
    return madvise


def _madvise_tensor(tensor: torch.Tensor, advice: int) -> None:
    if tensor.device.type != "cpu" or tensor.numel() == 0:
        return
    result = _libc_madvise()(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(tensor.numel() * tensor.element_size()),
        advice,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _madvise_range(address: int, nbytes: int, advice: int) -> None:
    if nbytes <= 0:
        return
    result = _libc_madvise()(
        ctypes.c_void_p(address),
        ctypes.c_size_t(nbytes),
        advice,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


@functools.cache
def _native_mmap_readahead():
    try:
        from vllm_gguf_plugin import _C_gguf
    except ImportError:
        return None
    return getattr(_C_gguf, "mmap_readahead", None)


def _positive_env_int(name: str, default: int) -> int:
    text = os.environ.get(name, str(default))
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {text!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_flag(name: str, default: bool) -> bool:
    text = os.environ.get(name, "1" if default else "0")
    if text not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {text!r}")
    return text == "1"


def _ple_residency_mode() -> str:
    mode = os.environ.get(_PLE_RESIDENCY_MODE_ENV)
    if mode is None:
        mode = (
            "pinned"
            if os.environ.get("VLLM_PLE_MMAP_HOST_REGISTER", "0") == "1"
            else "ssd"
        )
    mode = mode.lower()
    if mode not in {"ssd", "pinned", "bounded"}:
        raise ValueError(
            f"{_PLE_RESIDENCY_MODE_ENV} must be ssd, pinned, or bounded, got {mode!r}"
        )
    return mode


def _mapping_rss_bytes(tensor: torch.Tensor) -> int:
    """Read this tensor mapping's resident bytes from ``/proc/self/smaps``."""
    mapping_start = tensor.data_ptr()
    mapping_end = mapping_start + tensor.numel() * tensor.element_size()
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
                        region_start < mapping_end and mapping_start < region_end
                    )
            elif overlaps_mapping and line.startswith("Rss:"):
                rss_bytes += int(line.split()[1]) * 1024
    return rss_bytes


def _process_rss_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status did not contain VmRSS")


class _BoundedMmapResidency:
    """Keep at most a fixed LRU set of mmap chunks resident in this process."""

    def __init__(
        self,
        tensor: torch.Tensor,
        path: str,
        budget_bytes: int,
        chunk_bytes: int,
    ) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("bounded PLE residency requires a contiguous CPU tensor")
        if tensor.ndim < 2 or tensor.shape[0] <= 0:
            raise ValueError("bounded PLE residency requires a nonempty row tensor")
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        if chunk_bytes % page_bytes:
            raise ValueError(
                f"{_PLE_BOUNDED_CHUNK_BYTES_ENV} must be a multiple of "
                f"the {page_bytes}-byte page size"
            )
        if budget_bytes < chunk_bytes or budget_bytes % chunk_bytes:
            raise ValueError(
                f"{_PLE_BOUNDED_BYTES_ENV} must be a positive multiple of "
                f"{_PLE_BOUNDED_CHUNK_BYTES_ENV}"
            )
        self.tensor = tensor
        self.address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        self.row_nbytes = tensor[0].numel() * tensor.element_size()
        if tensor.stride(0) * tensor.element_size() != self.row_nbytes:
            raise ValueError("bounded PLE residency requires packed contiguous rows")
        self.num_rows = tensor.shape[0]
        self.path = path
        self.budget_bytes = budget_bytes
        self.chunk_bytes = chunk_bytes
        self.max_chunks = budget_bytes // chunk_bytes
        self._chunks: OrderedDict[int, None] = OrderedDict()
        self._fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self._fd_finalizer = weakref.finalize(self, os.close, self._fd)
        self._owner_thread_id = threading.get_ident()
        self.eviction_count = 0
        self.evicted_bytes = 0

    @property
    def tracked_bytes(self) -> int:
        return sum(
            min(self.chunk_bytes, self.nbytes - chunk * self.chunk_bytes)
            for chunk in self._chunks
        )

    @property
    def tracked_chunks(self) -> int:
        return len(self._chunks)

    def _advise_chunks(self, chunks: list[int]) -> int:
        if not chunks:
            return 0
        evicted_bytes = 0
        sorted_chunks = sorted(chunks)
        range_start = sorted_chunks[0]
        previous = range_start
        ranges = []
        for chunk in sorted_chunks[1:]:
            if chunk != previous + 1:
                ranges.append((range_start, previous + 1))
                range_start = chunk
            previous = chunk
        ranges.append((range_start, previous + 1))
        for first_chunk, end_chunk in ranges:
            offset = first_chunk * self.chunk_bytes
            end = min(end_chunk * self.chunk_bytes, self.nbytes)
            length = end - offset
            _madvise_range(self.address + offset, length, _MADV_DONTNEED)
            os.posix_fadvise(
                self._fd,
                offset,
                length,
                os.POSIX_FADV_DONTNEED,
            )
            evicted_bytes += length
        self.eviction_count += len(chunks)
        self.evicted_bytes += evicted_bytes
        return evicted_bytes

    def prepare(self, row_ids: torch.Tensor) -> tuple[int, int, int]:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("bounded PLE residency is single-owner")
        touched_chunks: set[int] = set()
        for row in sorted(set(row_ids.detach().reshape(-1).tolist())):
            if not 0 <= row < self.num_rows:
                raise IndexError(f"PLE mmap row {row} is outside [0, {self.num_rows})")
            row_start = row * self.row_nbytes
            first_chunk = row_start // self.chunk_bytes
            final_chunk = (row_start + self.row_nbytes - 1) // self.chunk_bytes
            touched_chunks.update(range(first_chunk, final_chunk + 1))
        if len(touched_chunks) > self.max_chunks:
            raise RuntimeError(
                "one PLE lookup requires "
                f"{len(touched_chunks) * self.chunk_bytes} tracked bytes, "
                f"exceeding the {_PLE_BOUNDED_BYTES_ENV}={self.budget_bytes} budget"
            )

        new_chunks = 0
        for chunk in sorted(touched_chunks):
            if chunk in self._chunks:
                self._chunks.move_to_end(chunk)
            else:
                self._chunks[chunk] = None
                new_chunks += 1

        evicted_chunks = []
        while len(self._chunks) > self.max_chunks:
            chunk, _ = self._chunks.popitem(last=False)
            evicted_chunks.append(chunk)
        evicted_bytes = self._advise_chunks(evicted_chunks)
        return new_chunks, len(evicted_chunks), evicted_bytes

    def evict_all(self) -> int:
        tracked_chunks = len(self._chunks)
        self._chunks.clear()
        _madvise_range(self.address, self.nbytes, _MADV_DONTNEED)
        os.posix_fadvise(
            self._fd,
            0,
            self.nbytes,
            os.POSIX_FADV_DONTNEED,
        )
        self.eviction_count += tracked_chunks
        self.evicted_bytes += self.nbytes
        return self.nbytes

    def close(self) -> None:
        self._fd_finalizer()


class _MmapRowReadahead:
    """Batch page advice for sparse rows in a file-backed tensor.

    The GGUF PLE table packs each row into 90 bytes. A lookup may therefore
    touch one or two filesystem pages per row. Issuing one merged WILLNEED
    range per adjacent run lets the kernel submit the SSD reads before
    ``index_select`` synchronously faults the selected rows.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("PLE mmap readahead requires a contiguous CPU tensor")
        if tensor.ndim < 2 or tensor.shape[0] <= 0:
            raise ValueError("PLE mmap readahead requires a nonempty row tensor")
        self.address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        self.row_nbytes = tensor[0].numel() * tensor.element_size()
        if tensor.stride(0) * tensor.element_size() != self.row_nbytes:
            raise ValueError("PLE mmap readahead requires packed contiguous rows")
        self.num_rows = tensor.shape[0]
        self.page_bytes = os.sysconf("SC_PAGE_SIZE")
        if self.address % self.page_bytes:
            raise ValueError("PLE mmap readahead requires a page-aligned mapping")
        self._owner_thread_id = threading.get_ident()
        self.prepare_count = 0
        self.advised_pages = 0
        self.advised_ranges = 0
        self.advised_bytes = 0

    def prepare(self, row_ids: torch.Tensor) -> tuple[int, int, int]:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("PLE mmap readahead is single-owner")
        rows = row_ids.detach().reshape(-1).tolist()
        native_readahead = _native_mmap_readahead()
        if native_readahead is not None:
            pages, ranges, advised_bytes = native_readahead(
                self.address,
                self.nbytes,
                self.row_nbytes,
                self.num_rows,
                rows,
            )
            self.prepare_count += 1
            self.advised_pages += pages
            self.advised_ranges += ranges
            self.advised_bytes += advised_bytes
            return pages, ranges, advised_bytes

        pages: set[int] = set()
        for row in rows:
            if not 0 <= row < self.num_rows:
                raise IndexError(f"PLE mmap row {row} is outside [0, {self.num_rows})")
            row_start = row * self.row_nbytes
            first_page = row_start // self.page_bytes
            final_page = (row_start + self.row_nbytes - 1) // self.page_bytes
            pages.update(range(first_page, final_page + 1))

        if not pages:
            return 0, 0, 0
        sorted_pages = sorted(pages)
        range_start = sorted_pages[0]
        previous = range_start
        ranges: list[tuple[int, int]] = []
        for page in sorted_pages[1:]:
            if page != previous + 1:
                ranges.append((range_start, previous + 1))
                range_start = page
            previous = page
        ranges.append((range_start, previous + 1))

        advised_bytes = 0
        for first_page, end_page in ranges:
            offset = first_page * self.page_bytes
            end = min(end_page * self.page_bytes, self.nbytes)
            length = end - offset
            _madvise_range(self.address + offset, length, _MADV_WILLNEED)
            advised_bytes += length

        self.prepare_count += 1
        self.advised_pages += len(sorted_pages)
        self.advised_ranges += len(ranges)
        self.advised_bytes += advised_bytes
        return len(sorted_pages), len(ranges), advised_bytes


def _sample_matches_file_mapping(
    mapped: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> bool:
    """Check three small regions without making the source table resident."""
    mapped_bytes = mapped.view(torch.uint8).flatten()
    loaded_bytes = loaded_weight.view(torch.uint8).flatten()
    sample_bytes = min(4096, mapped_bytes.numel())
    if sample_bytes == 0:
        return True
    starts = {
        0,
        max(0, mapped_bytes.numel() // 2 - sample_bytes // 2),
        mapped_bytes.numel() - sample_bytes,
    }
    return all(
        torch.equal(
            mapped_bytes.narrow(0, start, sample_bytes),
            loaded_bytes.narrow(0, start, sample_bytes),
        )
        for start in starts
    )


def _materialize_file_backed_embedding(
    layer: VocabParallelEmbedding,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    path: str,
) -> bool:
    """Bind a CPU embedding parameter directly to an opt-in raw weight file."""
    if param.device.type != "cpu":
        return False

    mmap_path = Path(path).expanduser().resolve(strict=True)
    expected_bytes = loaded_weight.numel() * loaded_weight.element_size()
    actual_bytes = mmap_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{_PLE_MMAP_PATH_ENV} has {actual_bytes} bytes, but the checkpoint "
            f"embedding requires {expected_bytes} bytes"
        )

    mapped = torch.from_file(
        str(mmap_path),
        shared=False,
        size=loaded_weight.numel(),
        dtype=loaded_weight.dtype,
    ).view_as(loaded_weight)
    if not _sample_matches_file_mapping(mapped, loaded_weight):
        raise ValueError(
            f"{_PLE_MMAP_PATH_ENV} does not match the checkpoint embedding samples"
        )

    if isinstance(param, UninitializedParameter):
        param.materialize((0,), device=param.device, dtype=mapped.dtype)
    param.data = mapped
    residency_mode = _ple_residency_mode()
    rss_log_rows = _positive_env_int(
        _PLE_RSS_LOG_ROWS_ENV,
        _DEFAULT_PLE_RSS_LOG_ROWS,
    )
    layer._vllm_gguf_mmap_path = str(mmap_path)
    layer._vllm_gguf_mmap_nbytes = expected_bytes
    layer._vllm_gguf_mmap_residency_mode = residency_mode
    layer._vllm_gguf_mmap_rows_since_trim = 0
    layer._vllm_gguf_mmap_rows_since_rss_log = 0
    layer._vllm_gguf_mmap_rss_log_rows = rss_log_rows
    layer._vllm_gguf_mmap_trim_count = 0
    layer._vllm_gguf_mmap_readahead = _env_flag(
        _PLE_MMAP_READAHEAD_ENV,
        residency_mode in {"ssd", "bounded"},
    )
    layer._vllm_gguf_mmap_readahead_state = None
    if residency_mode == "ssd":
        layer._vllm_gguf_mmap_trim_rows = _positive_env_int(
            _PLE_MMAP_TRIM_ROWS_ENV,
            _DEFAULT_PLE_MMAP_TRIM_ROWS,
        )
    else:
        layer._vllm_gguf_mmap_trim_rows = 0
    if residency_mode == "bounded":
        layer._vllm_gguf_mmap_bounded_bytes = _positive_env_int(
            _PLE_BOUNDED_BYTES_ENV,
            _DEFAULT_PLE_BOUNDED_BYTES,
        )
        layer._vllm_gguf_mmap_bounded_chunk_bytes = _positive_env_int(
            _PLE_BOUNDED_CHUNK_BYTES_ENV,
            _DEFAULT_PLE_BOUNDED_CHUNK_BYTES,
        )
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        if layer._vllm_gguf_mmap_bounded_chunk_bytes % page_bytes:
            raise ValueError(
                f"{_PLE_BOUNDED_CHUNK_BYTES_ENV} must be a multiple of "
                f"the {page_bytes}-byte page size"
            )
        if (
            layer._vllm_gguf_mmap_bounded_bytes
            < layer._vllm_gguf_mmap_bounded_chunk_bytes
            or layer._vllm_gguf_mmap_bounded_bytes
            % layer._vllm_gguf_mmap_bounded_chunk_bytes
        ):
            raise ValueError(
                f"{_PLE_BOUNDED_BYTES_ENV} must be a positive multiple of "
                f"{_PLE_BOUNDED_CHUNK_BYTES_ENV}"
            )
    try:
        _madvise_tensor(mapped, _MADV_RANDOM)
        _madvise_tensor(mapped, _MADV_DONTDUMP)
    except OSError:
        logger.warning("Could not apply mmap advice to %s", mmap_path, exc_info=True)
    logger.info(
        "Using file-backed GGUF embedding: path=%s bytes=%d policy=%s "
        "trim_rows=%d readahead=%s rss_log_rows=%d mapping_rss_bytes=%d "
        "process_rss_bytes=%d",
        mmap_path,
        expected_bytes,
        residency_mode,
        layer._vllm_gguf_mmap_trim_rows,
        layer._vllm_gguf_mmap_readahead,
        rss_log_rows,
        _mapping_rss_bytes(mapped),
        _process_rss_bytes(),
    )
    return True


def prepare_file_backed_embedding_access(
    layer: torch.nn.Module,
    row_ids: torch.Tensor,
) -> None:
    """Prepare eviction and batched page readahead before selected-row gather."""
    mode = getattr(layer, "_vllm_gguf_mmap_residency_mode", None)
    if mode is None:
        return
    if mode == "bounded":
        residency = getattr(layer, "_vllm_gguf_mmap_residency", None)
        if residency is None:
            residency = _BoundedMmapResidency(
                layer.qweight,
                layer._vllm_gguf_mmap_path,
                layer._vllm_gguf_mmap_bounded_bytes,
                layer._vllm_gguf_mmap_bounded_chunk_bytes,
            )
            layer._vllm_gguf_mmap_residency = residency
            logger.info(
                "PLE mmap bounded residency initialized: path=%s bytes=%d "
                "budget_bytes=%d chunk_bytes=%d max_chunks=%d",
                residency.path,
                residency.nbytes,
                residency.budget_bytes,
                residency.chunk_bytes,
                residency.max_chunks,
            )
        new_chunks, evicted_chunks, evicted_bytes = residency.prepare(row_ids)
        layer._vllm_gguf_mmap_last_new_chunks = new_chunks
        layer._vllm_gguf_mmap_last_evicted_chunks = evicted_chunks
        layer._vllm_gguf_mmap_last_evicted_bytes = evicted_bytes

    if not getattr(layer, "_vllm_gguf_mmap_readahead", False):
        return
    readahead = getattr(layer, "_vllm_gguf_mmap_readahead_state", None)
    if readahead is None:
        readahead = _MmapRowReadahead(layer.qweight)
        layer._vllm_gguf_mmap_readahead_state = readahead
        logger.info(
            "PLE mmap row readahead initialized: path=%s bytes=%d "
            "row_bytes=%d page_bytes=%d",
            layer._vllm_gguf_mmap_path,
            readahead.nbytes,
            readahead.row_nbytes,
            readahead.page_bytes,
        )
    pages, ranges, advised_bytes = readahead.prepare(row_ids)
    layer._vllm_gguf_mmap_last_readahead_pages = pages
    layer._vllm_gguf_mmap_last_readahead_ranges = ranges
    layer._vllm_gguf_mmap_last_readahead_bytes = advised_bytes


def _log_file_backed_residency(
    layer: torch.nn.Module,
    *,
    event: str,
    rows: int,
    mapping_rss_before: int | None = None,
) -> int:
    mapping_rss = _mapping_rss_bytes(layer.qweight)
    residency = getattr(layer, "_vllm_gguf_mmap_residency", None)
    logger.info(
        "PLE mmap residency: policy=%s event=%s path=%s rows=%d "
        "mapping_rss_bytes=%d mapping_rss_before_bytes=%s "
        "process_rss_bytes=%d budget_bytes=%d tracked_bytes=%d "
        "tracked_chunks=%d evicted_chunks=%d evicted_bytes=%d trims=%d",
        layer._vllm_gguf_mmap_residency_mode,
        event,
        layer._vllm_gguf_mmap_path,
        rows,
        mapping_rss,
        "na" if mapping_rss_before is None else mapping_rss_before,
        _process_rss_bytes(),
        0 if residency is None else residency.budget_bytes,
        0 if residency is None else residency.tracked_bytes,
        0 if residency is None else residency.tracked_chunks,
        0 if residency is None else residency.eviction_count,
        0 if residency is None else residency.evicted_bytes,
        layer._vllm_gguf_mmap_trim_count,
    )
    return mapping_rss


def maybe_trim_file_backed_embedding(
    layer: torch.nn.Module,
    rows_read: int,
) -> None:
    """Apply the configured mmap policy and report its RSS through ``/proc``."""
    mode = getattr(layer, "_vllm_gguf_mmap_residency_mode", None)
    if mode is None:
        return
    rows_since_log = getattr(layer, "_vllm_gguf_mmap_rows_since_rss_log", 0)
    rows_since_log += rows_read
    layer._vllm_gguf_mmap_rows_since_rss_log = rows_since_log

    if mode == "ssd":
        trim_rows = layer._vllm_gguf_mmap_trim_rows
        rows_since_trim = layer._vllm_gguf_mmap_rows_since_trim + rows_read
        if rows_since_trim >= trim_rows:
            mapping_rss_before = _mapping_rss_bytes(layer.qweight)
            try:
                _madvise_tensor(layer.qweight, _MADV_DONTNEED)
            except OSError as error:
                raise RuntimeError(
                    "SSD PLE residency failed to trim its file-backed mapping"
                ) from error
            layer._vllm_gguf_mmap_rows_since_trim = 0
            layer._vllm_gguf_mmap_rows_since_rss_log = 0
            layer._vllm_gguf_mmap_trim_count += 1
            _log_file_backed_residency(
                layer,
                event="trim",
                rows=rows_since_trim,
                mapping_rss_before=mapping_rss_before,
            )
            return
        layer._vllm_gguf_mmap_rows_since_trim = rows_since_trim

    if rows_since_log < layer._vllm_gguf_mmap_rss_log_rows:
        return
    layer._vllm_gguf_mmap_rows_since_rss_log = 0
    mapping_rss = _log_file_backed_residency(
        layer,
        event="observe",
        rows=rows_since_log,
    )
    if mode != "bounded":
        return
    residency = layer._vllm_gguf_mmap_residency
    if mapping_rss <= residency.budget_bytes:
        return
    evicted_bytes = residency.evict_all()
    mapping_rss_after = _mapping_rss_bytes(layer.qweight)
    logger.warning(
        "PLE mmap bounded RSS exceeded its budget; reset complete mapping: "
        "path=%s budget_bytes=%d mapping_rss_before_bytes=%d "
        "mapping_rss_after_bytes=%d evicted_bytes=%d",
        residency.path,
        residency.budget_bytes,
        mapping_rss,
        mapping_rss_after,
        evicted_bytes,
    )
    if mapping_rss_after > residency.budget_bytes:
        raise RuntimeError(
            "bounded PLE residency could not return mapping RSS below its "
            f"{residency.budget_bytes}-byte budget; RSS is {mapping_rss_after}"
        )


def _clone_loaded_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    return loaded_weight.detach().clone()


def _resolve_gguf_weight_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    uses_weight_loader_v2 = hasattr(layer, "weight_loader_v2")
    base_loader = (
        layer.weight_loader_v2 if uses_weight_loader_v2 else fallback_weight_loader
    )
    if base_loader is None:
        return None

    @wraps(base_loader)
    def _gguf_weight_loader(param, loaded_weight, loaded_shard_id=None):
        # V2 loaders split packed logical shards and select the local TP
        # partition before storage. Legacy loaders expect an already
        # materialized parameter, so retain direct storage for their unsharded
        # tensors.
        if (
            not uses_weight_loader_v2
            and loaded_shard_id is None
            and hasattr(param, "_store")
        ):
            param._store(loaded_weight)
            return
        if loaded_shard_id is None:
            base_loader(param, loaded_weight)
        else:
            base_loader(param, loaded_weight, loaded_shard_id)

    return _gguf_weight_loader


def _resolve_gguf_weight_type_loader(
    layer: torch.nn.Module,
    fallback_weight_loader=None,
):
    """Weight loader for GGUF weight-type parameters."""
    base_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    if base_loader is None:
        return fallback_weight_loader

    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None and hasattr(param, "_store"):
            param._store(loaded_weight)
            return
        if isinstance(loaded_shard_id, tuple) and hasattr(param, "_store"):
            for shard_id in loaded_shard_id:
                param._store(loaded_weight, shard_id=shard_id)
            return
        base_loader(param, loaded_weight, loaded_shard_id)

    return _gguf_weight_type_loader_v2


def _materialize_parameter_data(
    param: Parameter | UninitializedParameter,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if isinstance(param, UninitializedParameter):
        if getattr(param, "_vllm_is_uva_offloaded", False):
            plan = plan_uva_host_owner(
                pin_memory=bool(getattr(param, "_vllm_uva_pin_memory", False)),
                tiered_master=is_tiered_expert_master(param),
            )
            mode = _uva_host_coherence()
            pinned_empty = (
                partial(_explicit_hip_uva_empty, mode=mode)
                if mode != "default" and bool(torch.version.hip)
                else default_pinned_empty
            )
            cpu_data = allocate_uva_host_owner(
                shape, dtype, plan, pinned_empty=pinned_empty
            )
            # A tiered expert master is only a compaction source, so it keeps
            # no view: get_accelerator_view_from_cpu_tensor pins unpinned input
            # internally, which is exactly the cost a pageable master avoids.
            accelerator_view = (
                get_accelerator_view_from_cpu_tensor(cpu_data)
                if plan.accelerator_view
                else cpu_data
            )
            # Materialize the Python parameter object without allocating the
            # final tensor on the accelerator, then attach the UVA view.
            param.materialize((0,), device=param.device, dtype=dtype)
            param.data = accelerator_view
            # get_accelerator_view_from_cpu_tensor aliases the allocation but
            # deliberately does not retain the CPU tensor.  Without this
            # strong reference, the pinned storage can be freed before the
            # first model forward and ROCm reports a host-UVA page fault.
            param._vllm_uva_cpu_data = cpu_data
            param._vllm_is_uva_offloaded = True
            return
        param.materialize(shape, device=param.device, dtype=dtype)


def _gguf_shard_id_as_int(shard_id: int | str) -> int:
    if isinstance(shard_id, int):
        return shard_id
    qkv_idxs = {"q": 0, "k": 1, "v": 2}
    return qkv_idxs[shard_id]


def _gguf_ordered_shard_ids(shard_ids: list[int | str]) -> list[int | str]:
    return sorted(shard_ids, key=_gguf_shard_id_as_int)


def _store_gguf_loaded_weight(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(device=param.device)
    if shard_id is None:
        _materialize_parameter_data(
            param, tuple(loaded_weight.shape), loaded_weight.dtype
        )
        param.data.copy_(loaded_weight)
        return

    if shard_id not in param.shard_id_map:
        param.shard_id_map[shard_id] = len(param.data_container)
        param.data_container.append(loaded_weight)
        param.shard_id.append(shard_id)
    else:
        param.data_container[param.shard_id_map[shard_id]] = loaded_weight
    if not isinstance(param, UninitializedParameter) and param.data.numel() == 0:
        param.data = loaded_weight


def _store_gguf_weight_type(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: int | str | None = None,
) -> None:
    loaded_weight = _clone_loaded_weight(loaded_weight).to(
        device=param.device, dtype=torch.uint8
    )
    weight_type = int(loaded_weight.item())
    num_elements = getattr(param, "num_elements", 1)
    if shard_id is None:
        _materialize_parameter_data(param, (num_elements,), torch.uint8)
        param.weight_type = weight_type
        if param.data.numel() == 1:
            param.data.fill_(weight_type)
        else:
            param.data.zero_()
            param.data[0] = weight_type
        return

    param.shard_weight_type[shard_id] = weight_type
    if len(param.shard_weight_type) == 1:
        param.weight_type = weight_type
    if not isinstance(param, UninitializedParameter):
        if param.data.numel() == 0:
            param.data = torch.empty(
                num_elements, dtype=torch.uint8, device=loaded_weight.device
            )
        param.data[_gguf_shard_id_as_int(shard_id)] = weight_type


def _gguf_embedding_weight_loader(
    layer: VocabParallelEmbedding,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    # Embeddings are copied directly into their final padded parameter below,
    # so cloning the staging tensor first is redundant.  For Qwen4Exp's PLE
    # embedding that clone alone is roughly 27 GiB.  Retain the GGUF mmap view
    # until the final copy completes instead.
    loaded_weight = loaded_weight.detach()
    mmap_path = os.environ.get(_PLE_MMAP_PATH_ENV)
    if mmap_path and _materialize_file_backed_embedding(
        layer,
        param,
        loaded_weight,
        mmap_path,
    ):
        return

    loaded_weight = loaded_weight.to(device=param.device)
    start_idx = layer.shard_indices.org_vocab_start_index
    shard_size = layer.shard_indices.org_vocab_end_index - start_idx
    loaded_weight = loaded_weight.narrow(param.output_dim, start_idx, shard_size)

    padded_shape = list(loaded_weight.shape)
    padded_shape[param.output_dim] = param.tensor_shape[param.output_dim]
    _materialize_parameter_data(param, tuple(padded_shape), loaded_weight.dtype)
    param.data.zero_()
    param.data.narrow(param.output_dim, 0, loaded_weight.shape[param.output_dim]).copy_(
        loaded_weight
    )


def _gguf_embedding_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
) -> None:
    _store_gguf_weight_type(param, loaded_weight)


def _materialize_gguf_moe_param(
    layer: RoutedExperts,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
) -> None:
    is_uninitialized = isinstance(param, UninitializedParameter)
    if not is_uninitialized and param.data.numel() != 0:
        return

    shard_dim = {"w1": 0, "w2": 1, "w3": 0}[shard_id]
    if getattr(param, "is_transposed", False):
        shard_dim = int(not shard_dim)

    if loaded_weight.ndim == 3:
        final_shape = list(loaded_weight.shape)
    elif loaded_weight.ndim == 2:
        final_shape = [param.tensor_shape[0], *loaded_weight.shape]
    else:
        return

    shard_dim += 1
    if shard_id in {"w1", "w3"}:
        final_shape[1] *= 2
    final_shape[shard_dim] = (
        final_shape[shard_dim] // layer.moe_config.moe_parallel_config.tp_size
    )
    if is_uninitialized:
        _materialize_parameter_data(
            param,
            tuple(final_shape),
            loaded_weight.dtype,
        )
    else:
        param.data = torch.empty(
            tuple(final_shape), dtype=loaded_weight.dtype, device=param.device
        )


def _gguf_moe_weight_loader(
    layer: RoutedExperts,
    base_weight_loader,
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    _materialize_gguf_moe_param(layer, param, loaded_weight, shard_id)
    return base_weight_loader(
        param,
        loaded_weight,
        weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=return_success,
    )


def _gguf_moe_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del weight_name, expert_id
    _store_gguf_weight_type(param, loaded_weight, shard_id)
    return True if return_success else None


class _GGUFParamLoadMixin:
    """Mixin providing GGUF parameter weight loading methods."""

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and loaded_weight.ndim >= 1:
            shard_size = loaded_weight.shape[0] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size
                )
        self._store(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        layout = getattr(self, "gguf_layout", None)
        if tp_size > 1 and layout is not None:
            weight_type_param = self.gguf_weight_type_parameter
            weight_type = weight_type_param.weight_type
            if weight_type not in gguf.GGML_QUANT_SIZES:
                raise ValueError(
                    f"Unknown GGUF weight type {weight_type} while sharding "
                    "a transformed row-parallel weight"
                )
            block_size, _ = gguf.GGML_QUANT_SIZES[weight_type]
            loaded_weight = layout.shard_weight(
                loaded_weight,
                dim=self.input_dim,
                logical_size=self.gguf_logical_input_size,
                block_size=block_size,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        elif tp_size > 1 and loaded_weight.ndim >= 2:
            shard_size = loaded_weight.shape[1] // tp_size
            if shard_size > 0:
                loaded_weight = loaded_weight.narrow(
                    1, tp_rank * shard_size, shard_size
                )
        self._store(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        shard_size = kwargs.get("shard_size")
        if (
            shard_size is not None
            and loaded_weight.ndim >= 1
            and shard_size > 0
            and shard_size < loaded_weight.shape[0]
        ):
            tp_rank = get_tensor_model_parallel_rank()
            loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        self._store(loaded_weight, shard_id=shard_id)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        shard_size = kwargs.get("shard_size")
        num_kv_head_replicas = kwargs.get("num_heads", 1)
        if (
            shard_size is not None
            and loaded_weight.ndim >= 1
            and shard_size > 0
            and shard_size < loaded_weight.shape[0]
        ):
            tp_rank = get_tensor_model_parallel_rank()
            effective_tp_rank = (
                tp_rank // num_kv_head_replicas if shard_id in ("k", "v") else tp_rank
            )
            loaded_weight = loaded_weight.narrow(
                0, effective_tp_rank * shard_size, shard_size
            )
        self._store(loaded_weight, shard_id=shard_id)


class GGUFWeightParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(
        self,
        *,
        data: torch.Tensor,
        weight_loader,
        input_dim: int,
        output_dim: int,
        tensor_shape: tuple[int, ...],
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.tensor_shape = tensor_shape
        self.data_container: list[torch.Tensor] = []
        self.shard_id: list[int | str] = []
        self.shard_id_map: dict[int | str, int] = {}
        super().__init__(data=data, weight_loader=weight_loader)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFWeightTypeParameter(_GGUFParamLoadMixin, BasevLLMParameter):
    def __init__(self, *, data: torch.Tensor, weight_loader):
        self.weight_type = 0
        self.shard_weight_type: dict[int | str, int] = {}
        self.num_elements = data.numel()
        super().__init__(data=data, weight_loader=weight_loader)

    def _store(
        self,
        loaded_weight: torch.Tensor,
        shard_id: int | str | None = None,
    ) -> None:
        _store_gguf_weight_type(self, loaded_weight, shard_id)


def _materialize_gguf_weight_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(0, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight = GGUFWeightParameter(
        data=data,
        weight_loader=weight_loader,
        input_dim=raw_param.input_dim,
        output_dim=raw_param.output_dim,
        tensor_shape=raw_param.tensor_shape,
    )
    qweight.data_container = list(raw_param.data_container)
    qweight.shard_id = list(raw_param.shard_id)
    qweight.shard_id_map = dict(raw_param.shard_id_map)
    if hasattr(raw_param, "ignore_warning"):
        qweight.ignore_warning = raw_param.ignore_warning
    for attr in (
        "_vllm_is_uva_offloaded",
        "_vllm_uva_pin_memory",
        "_vllm_uva_cpu_data",
    ):
        if hasattr(raw_param, attr):
            setattr(qweight, attr, getattr(raw_param, attr))
    # Hand the shard tensors over rather than sharing them: the source param
    # outlives this call, and a second reference to the shards would keep them
    # resident after _create_padded_weight_param builds the concatenated copy,
    # leaving every fused layer in VRAM twice.
    raw_param.data_container.clear()
    raw_param.shard_id.clear()
    raw_param.shard_id_map.clear()
    layer.register_parameter(param_name, qweight)


def _materialize_gguf_weight_type_parameter(
    layer: torch.nn.Module,
    param_name: str,
    fallback_weight_loader=None,
) -> None:
    raw_param = getattr(layer, param_name)
    if isinstance(raw_param, GGUFWeightTypeParameter):
        return

    if fallback_weight_loader is None:
        fallback_weight_loader = getattr(raw_param, "weight_loader", None)
    weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
    assert weight_loader is not None
    num_elements = getattr(raw_param, "num_elements", 1)
    if isinstance(raw_param, UninitializedParameter):
        data = torch.empty(num_elements, dtype=torch.uint8, device=raw_param.device)
    else:
        data = raw_param.data
    qweight_type = GGUFWeightTypeParameter(data=data, weight_loader=weight_loader)
    qweight_type.num_elements = num_elements
    qweight_type.weight_type = raw_param.weight_type
    qweight_type.shard_weight_type = dict(raw_param.shard_weight_type)
    if hasattr(raw_param, "ignore_warning"):
        qweight_type.ignore_warning = raw_param.ignore_warning
    for attr in (
        "_vllm_is_uva_offloaded",
        "_vllm_uva_pin_memory",
        "_vllm_uva_cpu_data",
    ):
        if hasattr(raw_param, attr):
            setattr(qweight_type, attr, getattr(raw_param, attr))
    layer.register_parameter(param_name, qweight_type)


class GGUFUninitializedParameter(_GGUFParamLoadMixin, UninitializedParameter):
    """Base class for uninitialized GGUF parameters."""

    cls_to_become = Parameter


class GGUFUninitializedWeightParameter(GGUFUninitializedParameter):
    data_container: list[torch.Tensor]

    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_loaded_weight(self, loaded_weight, shard_id)


class GGUFUninitializedWeightTypeParameter(GGUFUninitializedParameter):
    def _store(self, loaded_weight: torch.Tensor, shard_id=None):
        _store_gguf_weight_type(self, loaded_weight, shard_id)

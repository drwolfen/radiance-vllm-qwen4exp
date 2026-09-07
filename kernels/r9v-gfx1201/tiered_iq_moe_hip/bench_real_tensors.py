# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-GPU timing probe for the production tiered Q4 MoE kernel.

This benchmark uses packed expert rows from the released Qwen3.8-Flash-Next
IQ4_XS GGUF and reproduces one TP=2 rank's runtime shapes.  It deliberately
does not load the model, start vLLM, use collectives, or inspect GPU telemetry.

Each invocation selects exactly one accelerator with ``--device`` and one
logical TP shard with ``--tp-rank``.  The three synthetic modes answer separate
questions:

* ``hot`` keeps every sampled expert on the selected GPU.
* ``uva-cache-hot`` keeps ten UVA experts and repeatedly routes to
  them, measuring the best case after the GPU cache is warm.
* ``uva-rotating`` rotates routes over a UVA working set strictly
  larger than ``--rotating-min-mib`` (96 MiB by default).

The two replay modes consume route IDs from an existing route-profile NPZ,
while retaining the same hot-device or UVA placement.  Replay has a
separate storage cap and stops before admitting a route set that would exceed
it.  All modes are also protected by ``--max-storage-mib``.

``--uva-coherence coherent|noncoherent`` selects an explicit ``hipHostMalloc``
flag for a true cache-policy A/B. ``default`` uses PyTorch's pinned allocator
and is not a coherent control on ROCm.

Only the two quantized expert GEMVs are timed.  Routing, activation, weighted
reduction, GDN, PLE, and TP collectives are intentionally outside the result.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import gguf
import numpy as np
import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm_gguf_plugin.quantization.params import allocate_uva_host_empty

MIB = 1024 * 1024
NUM_EXPERTS = 512
TOP_K = 10
TP_SIZE = 2
DEVICE_PATTERN = re.compile(r"cuda:(\d+)")
KERNEL_VARIANTS = {
    "generic": 0,
    "auto": 1,
    "u2": 2,
    "u5": 5,
    "u10": 10,
    "reuse3": 30,
    "reuse3v2": 31,
}


@dataclass(frozen=True)
class LayerCase:
    name: str
    layer: int
    gate_qtype: int
    down_qtype: int
    model_layers: int

    @property
    def gate_name(self) -> str:
        return f"blk.{self.layer}.ffn_gate_exps.weight"

    @property
    def up_name(self) -> str:
        return f"blk.{self.layer}.ffn_up_exps.weight"

    @property
    def down_name(self) -> str:
        return f"blk.{self.layer}.ffn_down_exps.weight"


CASES = {
    "normal": LayerCase("normal", 0, 21, 20, 43),
    "q4xs-q8": LayerCase("q4xs-q8", 2, 23, 8, 1),
    "iq3s-q8": LayerCase("iq3s-q8", 4, 21, 8, 4),
}


@dataclass(frozen=True)
class PackedShape:
    gate_rows: int
    gate_row_bytes: int
    down_rows: int
    down_row_bytes: int

    @property
    def bytes_per_expert(self) -> int:
        return (
            self.gate_rows * self.gate_row_bytes
            + self.down_rows * self.down_row_bytes
        )


@dataclass
class WeightStorage:
    accelerator: torch.Tensor
    owner: torch.Tensor | None


@dataclass
class ExpertPair:
    gate: WeightStorage
    down: WeightStorage
    hot_map: torch.Tensor
    cold_map: torch.Tensor


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qwen38_tiered_iq_moe_hip", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tiered extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_device(device_text: str, memory_fraction: float) -> torch.device:
    match = DEVICE_PATTERN.fullmatch(device_text)
    if match is None:
        raise ValueError("--device must be explicit, for example cuda:0 or cuda:1")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("the benchmark requires a ROCm accelerator")
    index = int(match.group(1))
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"{device_text} does not exist; visible device count is "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(index)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, index)
    return torch.device(device_text)


def tensor_table(reader: gguf.GGUFReader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def require_tensor(tensors: dict[str, object], name: str):
    try:
        return tensors[name]
    except KeyError as error:
        raise KeyError(
            f"{name} is absent from this GGUF split; pass the split containing "
            "the requested representative layer"
        ) from error


def inspect_shape(tensors: dict[str, object], case: LayerCase) -> PackedShape:
    gate = require_tensor(tensors, case.gate_name)
    up = require_tensor(tensors, case.up_name)
    down = require_tensor(tensors, case.down_name)
    for tensor, expected_qtype in (
        (gate, case.gate_qtype),
        (up, case.gate_qtype),
        (down, case.down_qtype),
    ):
        if int(tensor.tensor_type) != expected_qtype:
            raise ValueError(
                f"{tensor.name} has qtype {int(tensor.tensor_type)}, expected "
                f"{expected_qtype}"
            )
        if tensor.data.ndim != 3 or tensor.data.shape[0] != NUM_EXPERTS:
            raise ValueError(
                f"{tensor.name} has packed shape {tensor.data.shape}, expected "
                f"({NUM_EXPERTS}, rows, row_bytes)"
            )

    if gate.data.shape != up.data.shape:
        raise ValueError("gate and up expert tensors do not have matching layouts")
    if gate.data.shape[1] % TP_SIZE:
        raise ValueError("gate/up output rows cannot be split evenly over TP=2")
    if down.data.shape[2] % TP_SIZE:
        raise ValueError("down packed input rows cannot be split evenly over TP=2")

    # Production fuses the local gate and up shards along output rows.  Each
    # projection contributes half of the final 640-row w13 tensor.
    gate_rows_per_projection = gate.data.shape[1] // TP_SIZE
    return PackedShape(
        gate_rows=gate_rows_per_projection * 2,
        gate_row_bytes=gate.data.shape[2],
        down_rows=down.data.shape[1],
        down_row_bytes=down.data.shape[2] // TP_SIZE,
    )


def make_storage(
    shape: tuple[int, int, int],
    placement: str,
    device: torch.device,
) -> WeightStorage:
    if placement == "hot":
        return WeightStorage(
            torch.empty(shape, dtype=torch.uint8, device=device),
            None,
        )
    owner = allocate_uva_host_empty(shape, torch.uint8)
    accelerator = get_accelerator_view_from_cpu_tensor(owner)
    if not accelerator.is_cuda:
        raise RuntimeError("pinned storage did not produce a UVA view")
    return WeightStorage(accelerator, owner)


def copy_numpy(destination: torch.Tensor, source: np.ndarray) -> None:
    staging = np.array(source, copy=True, order="C")
    destination.copy_(torch.from_numpy(staging))


def load_expert_pair(
    tensors: dict[str, object],
    case: LayerCase,
    packed: PackedShape,
    expert_ids: list[int],
    tp_rank: int,
    placement: str,
    device: torch.device,
) -> ExpertPair:
    gate_source = require_tensor(tensors, case.gate_name).data
    up_source = require_tensor(tensors, case.up_name).data
    down_source = require_tensor(tensors, case.down_name).data
    num_loaded = len(expert_ids)
    gate = make_storage(
        (num_loaded, packed.gate_rows, packed.gate_row_bytes),
        placement,
        device,
    )
    down = make_storage(
        (num_loaded, packed.down_rows, packed.down_row_bytes),
        placement,
        device,
    )
    gate_destination = gate.owner if gate.owner is not None else gate.accelerator
    down_destination = down.owner if down.owner is not None else down.accelerator

    projection_rows = packed.gate_rows // 2
    projection_start = tp_rank * projection_rows
    projection_stop = projection_start + projection_rows
    copy_numpy(
        gate_destination[:, :projection_rows],
        gate_source[expert_ids, projection_start:projection_stop, :],
    )
    copy_numpy(
        gate_destination[:, projection_rows:],
        up_source[expert_ids, projection_start:projection_stop, :],
    )

    down_byte_start = tp_rank * packed.down_row_bytes
    down_byte_stop = down_byte_start + packed.down_row_bytes
    copy_numpy(
        down_destination,
        down_source[expert_ids, :, down_byte_start:down_byte_stop],
    )

    compact_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
    compact_map[expert_ids] = torch.arange(num_loaded, dtype=torch.int32)
    absent_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    if placement == "hot":
        hot_map = compact_map.to(device)
        cold_map = absent_map
    else:
        hot_map = absent_map
        cold_map = compact_map.to(device)
    torch.cuda.synchronize(device)
    return ExpertPair(gate, down, hot_map, cold_map)


def synthetic_routes(
    mode: str,
    tokens: int,
    pool_experts: int,
) -> np.ndarray:
    if mode == "uva-cache-hot":
        row = np.arange(TOP_K, dtype=np.int32)
        return np.broadcast_to(row, (1, tokens, TOP_K)).copy()

    experts_per_set = tokens * TOP_K
    num_sets = math.ceil(pool_experts / experts_per_set)
    routes = np.empty((num_sets, tokens, TOP_K), dtype=np.int32)
    offsets = np.arange(experts_per_set, dtype=np.int32)
    for route_set in range(num_sets):
        routes[route_set] = (
            offsets + route_set * experts_per_set
        ).reshape(tokens, TOP_K) % pool_experts
    return routes


def replay_routes(
    profile_path: Path,
    layer: int,
    tokens: int,
    max_sets: int,
    max_pool_experts: int,
) -> tuple[np.ndarray, list[int]]:
    accepted: list[np.ndarray] = []
    pool: set[int] = set()
    with np.load(profile_path) as profile:
        routes = profile["routes"]
        rows = profile["rows"]
        if routes.ndim != 4 or routes.shape[1] != 48 or routes.shape[3] != TOP_K:
            raise ValueError(
                f"route profile has shape {routes.shape}, expected "
                f"(events, 48, positions, {TOP_K})"
            )
        for event, num_rows in enumerate(rows):
            usable = int(num_rows)
            if usable < tokens:
                continue
            for start in range(0, usable - tokens + 1, tokens):
                candidate = np.asarray(
                    routes[event, layer, start : start + tokens],
                    dtype=np.int32,
                )
                if np.any(candidate < 0) or np.any(candidate >= NUM_EXPERTS):
                    continue
                candidate_ids = set(int(value) for value in candidate.flat)
                if len(pool | candidate_ids) > max_pool_experts:
                    if accepted:
                        return np.stack(accepted), sorted(pool)
                    raise ValueError(
                        "the first replay route set exceeds --replay-max-mib"
                    )
                accepted.append(np.array(candidate, copy=True))
                pool.update(candidate_ids)
                if len(accepted) >= max_sets:
                    return np.stack(accepted), sorted(pool)
    if not accepted:
        raise ValueError("the route profile contained no usable route sets")
    return np.stack(accepted), sorted(pool)


def time_kernel(
    operation, route_sets: torch.Tensor, warmup: int, iterations: int
) -> float:
    num_sets = route_sets.shape[0]
    last = None
    for iteration in range(warmup):
        last = operation(route_sets[iteration % num_sets])
    if last is None or not bool(torch.isfinite(last).all().item()):
        raise RuntimeError("tiered kernel produced non-finite output during warmup")
    torch.cuda.synchronize(route_sets.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iterations):
        last = operation(route_sets[iteration % num_sets])
    end.record()
    end.synchronize()
    if last is None:
        raise RuntimeError("timed loop did not execute")
    return start.elapsed_time(end) / iterations


def time_pair(
    gate_operation,
    down_operation,
    route_sets: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    """Time interleaved w13/w2 reads so their combined cache footprint matters."""
    num_sets = route_sets.shape[0]
    last = None
    for iteration in range(warmup):
        ids = route_sets[iteration % num_sets]
        gate_operation(ids)
        last = down_operation(ids)
    if last is None or not bool(torch.isfinite(last).all().item()):
        raise RuntimeError(
            "tiered kernel pair produced non-finite output during warmup"
        )
    torch.cuda.synchronize(route_sets.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iterations):
        ids = route_sets[iteration % num_sets]
        gate_operation(ids)
        last = down_operation(ids)
    end.record()
    end.synchronize()
    if last is None:
        raise RuntimeError("timed pair loop did not execute")
    return start.elapsed_time(end) / iterations


def release_cached_storage() -> None:
    """Release allocator caches so sequential cases keep the advertised bound."""
    gc.collect()
    torch.cuda.empty_cache()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()


def benchmark_one(
    module: ModuleType,
    tensors: dict[str, object],
    case: LayerCase,
    args: argparse.Namespace,
    device: torch.device,
    tokens: int,
) -> tuple[float, int]:
    packed = inspect_shape(tensors, case)
    placement = "hot" if args.mode in {"hot", "replay-hot"} else "uva"
    if args.mode.startswith("replay-"):
        layer = case.layer if args.profile_layer is None else args.profile_layer
        max_pool_experts = min(
            NUM_EXPERTS,
            int(args.replay_max_mib * MIB) // packed.bytes_per_expert,
        )
        routes_np, expert_ids = replay_routes(
            args.route_profile,
            layer,
            tokens,
            args.replay_max_sets,
            max_pool_experts,
        )
    else:
        if args.mode == "uva-cache-hot":
            pool_experts = TOP_K
        elif args.mode == "uva-rotating":
            threshold = int(args.rotating_min_mib * MIB)
            pool_experts = max(tokens * TOP_K, threshold // packed.bytes_per_expert + 1)
            pool_experts = math.ceil(pool_experts / TOP_K) * TOP_K
        else:
            pool_experts = args.resident_experts or tokens * TOP_K
        if pool_experts > NUM_EXPERTS:
            raise ValueError(
                f"mode {args.mode} needs {pool_experts} experts, exceeding "
                f"the model's {NUM_EXPERTS}"
            )
        expert_ids = list(range(pool_experts))
        routes_np = synthetic_routes(args.mode, tokens, pool_experts)

    storage_bytes = len(expert_ids) * packed.bytes_per_expert
    if storage_bytes > args.max_storage_mib * MIB:
        raise ValueError(
            f"planned packed storage is {storage_bytes / MIB:.2f} MiB, above "
            f"--max-storage-mib={args.max_storage_mib}"
        )
    if args.mode == "uva-rotating" and storage_bytes <= args.rotating_min_mib * MIB:
        raise AssertionError("rotating UVA working set is not strictly above its floor")

    largest_staging = max(
        len(expert_ids)
        * (packed.gate_rows // 2)
        * packed.gate_row_bytes,
        len(expert_ids) * packed.down_rows * packed.down_row_bytes,
    )
    pinned_mib = storage_bytes / MIB if placement == "uva" else 0.0
    vram_mib = storage_bytes / MIB if placement == "hot" else 0.0
    print(
        f"PLAN mode={args.mode} variant={args.kernel_variant} "
        f"case={case.name} tokens={tokens} "
        f"device={device} tp_rank={args.tp_rank} experts={len(expert_ids)} "
        f"route_sets={routes_np.shape[0]} storage_mib={storage_bytes / MIB:.2f} "
        f"pinned_mib={pinned_mib:.2f} vram_mib={vram_mib:.2f} "
        f"max_pageable_staging_mib={largest_staging / MIB:.2f}",
        flush=True,
    )

    pair = load_expert_pair(
        tensors,
        case,
        packed,
        expert_ids,
        args.tp_rank,
        placement,
        device,
    )
    route_sets = torch.from_numpy(routes_np).to(device=device, dtype=torch.int32)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_gate = torch.randn(
        (tokens, 2560),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    x_down = torch.randn(
        (tokens * TOP_K, 320),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    empty_gate = torch.empty(
        (0, packed.gate_rows, packed.gate_row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    empty_down = torch.empty(
        (0, packed.down_rows, packed.down_row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    if placement == "hot":
        cold_gate, hot_gate = empty_gate, pair.gate.accelerator
        cold_down, hot_down = empty_down, pair.down.accelerator
    else:
        cold_gate, hot_gate = pair.gate.accelerator, empty_gate
        cold_down, hot_down = pair.down.accelerator, empty_down

    variant_code = KERNEL_VARIANTS[args.kernel_variant]

    def run_gemv(*gemv_args) -> torch.Tensor:
        if variant_code == 0:
            return module.tiered_iq_moe_gemv(*gemv_args)
        return module.tiered_iq_moe_gemv_variant(*gemv_args, variant_code)

    def gate_operation(ids: torch.Tensor) -> torch.Tensor:
        return run_gemv(
            x_gate,
            cold_gate,
            hot_gate,
            pair.hot_map,
            pair.cold_map,
            ids,
            TOP_K,
            case.gate_qtype,
            packed.gate_rows,
            tokens,
        )

    def down_operation(ids: torch.Tensor) -> torch.Tensor:
        return run_gemv(
            x_down,
            cold_down,
            hot_down,
            pair.hot_map,
            pair.cold_map,
            ids.reshape(-1),
            1,
            case.down_qtype,
            packed.down_rows,
            tokens * TOP_K,
        )

    gate_ms = time_kernel(
        gate_operation, route_sets, args.warmup, args.iterations
    )
    down_ms = time_kernel(
        down_operation, route_sets, args.warmup, args.iterations
    )
    pair_ms = time_pair(
        gate_operation,
        down_operation,
        route_sets,
        args.warmup,
        args.iterations,
    )
    logical_active_bytes = tokens * TOP_K * packed.bytes_per_expert
    effective_gbps = logical_active_bytes / pair_ms / 1_000_000
    print(
        f"RESULT mode={args.mode} variant={args.kernel_variant} "
        f"case={case.name} tokens={tokens} "
        f"gate_ms={gate_ms:.6f} down_ms={down_ms:.6f} pair_ms={pair_ms:.6f} "
        f"component_sum_ms={gate_ms + down_ms:.6f} "
        f"logical_active_mib={logical_active_bytes / MIB:.3f} "
        f"effective_gbps={effective_gbps:.2f}",
        flush=True,
    )
    return pair_ms, logical_active_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--device",
        required=True,
        help="explicit visible accelerator, for example cuda:0",
    )
    parser.add_argument("--tp-rank", type=int, required=True, choices=(0, 1))
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "hot",
            "uva-cache-hot",
            "uva-rotating",
            "replay-hot",
            "replay-uva",
        ),
    )
    parser.add_argument(
        "--case", choices=(*CASES, "all"), default="all"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--kernel-variant",
        choices=tuple(KERNEL_VARIANTS),
        default="generic",
        help="generic or an exact-shape 2D kernel A/B arm",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--resident-experts",
        type=int,
        help="hot-mode pool size; defaults to tokens * 10",
    )
    parser.add_argument("--rotating-min-mib", type=float, default=96.0)
    parser.add_argument("--max-storage-mib", type=float, default=160.0)
    parser.add_argument("--route-profile", type=Path)
    parser.add_argument("--profile-layer", type=int, choices=range(48))
    parser.add_argument("--replay-max-sets", type=int, default=64)
    parser.add_argument("--replay-max-mib", type=float, default=128.0)
    parser.add_argument("--vram-fraction", type=float, default=0.05)
    parser.add_argument(
        "--uva-coherence",
        choices=("default", "coherent", "noncoherent"),
        default="default",
        help=(
            "host allocation mode for UVA cases; coherent/noncoherent use "
            "explicit HIP flags"
        ),
    )
    args = parser.parse_args()

    if any(tokens < 1 or tokens > 3 for tokens in args.tokens):
        parser.error("--tokens values must be in 1..3")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if args.replay_max_sets < 1:
        parser.error("--replay-max-sets must be positive")
    if not 0 < args.vram_fraction <= 0.1:
        parser.error("--vram-fraction must be in (0, 0.1]")
    if args.max_storage_mib <= 0 or args.replay_max_mib <= 0:
        parser.error("storage caps must be positive")
    if args.rotating_min_mib < 96:
        parser.error("--rotating-min-mib cannot be lower than 96")
    if args.mode.startswith("replay-"):
        if args.route_profile is None:
            parser.error("replay modes require --route-profile")
    elif args.route_profile is not None or args.profile_layer is not None:
        parser.error("route-profile options are valid only in replay modes")
    if args.resident_experts is not None:
        if args.mode != "hot":
            parser.error("--resident-experts is valid only in hot mode")
        if not TOP_K <= args.resident_experts <= NUM_EXPERTS:
            parser.error(f"--resident-experts must be in {TOP_K}..{NUM_EXPERTS}")
    return args


def main() -> None:
    args = parse_args()
    os.environ["RADIANCE_UVA_HOST_NONCOHERENT"] = "0"
    os.environ["RADIANCE_UVA_HOST_COHERENCE"] = args.uva_coherence
    device = select_device(args.device, args.vram_fraction)
    extension = load_module(args.extension.resolve(strict=True))
    reader = gguf.GGUFReader(args.model.resolve(strict=True))
    tensors = tensor_table(reader)
    selected_cases = list(CASES.values()) if args.case == "all" else [CASES[args.case]]

    props = torch.cuda.get_device_properties(device)
    print(
        f"DEVICE selected={device} name={props.name!r} tp_rank={args.tp_rank} "
        f"torch_vram_fraction={args.vram_fraction:.3f} "
        f"uva_coherence={args.uva_coherence}",
        flush=True,
    )
    timings: dict[int, dict[str, float]] = {tokens: {} for tokens in args.tokens}
    logical_bytes: dict[int, dict[str, int]] = {
        tokens: {} for tokens in args.tokens
    }
    for tokens in args.tokens:
        for case in selected_cases:
            pair_ms, active_bytes = benchmark_one(
                extension,
                tensors,
                case,
                args,
                device,
                tokens,
            )
            timings[tokens][case.name] = pair_ms
            logical_bytes[tokens][case.name] = active_bytes
            release_cached_storage()

    if len(selected_cases) == len(CASES):
        for tokens in args.tokens:
            projected_ms = sum(
                timings[tokens][case.name] * case.model_layers
                for case in CASES.values()
            )
            projected_bytes = sum(
                logical_bytes[tokens][case.name] * case.model_layers
                for case in CASES.values()
            )
            print(
                f"MODEL_EXPERT_PROJECTION mode={args.mode} "
                f"variant={args.kernel_variant} tokens={tokens} "
                f"layers=48 projected_ms={projected_ms:.6f} "
                f"logical_active_mib={projected_bytes / MIB:.3f} "
                f"effective_gbps={projected_bytes / projected_ms / 1_000_000:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

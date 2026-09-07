# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight A/B for expert-grouped Qwen3.8 prefill.

The probe exposes one ROCm device, loads at most ``--max-storage-mib`` of
packed experts from a real release GGUF, and compares the production
route-at-a-time kernel with grouped-4/8/16 execution.  It checks bit-exact
BF16 W13 and W2 output before reporting timings.  It never starts vLLM or
loads the model.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import torch
from bench_real_tensors import (
    CASES,
    MIB,
    NUM_EXPERTS,
    TOP_K,
    inspect_shape,
    load_expert_pair,
    load_module,
    release_cached_storage,
    select_device,
    tensor_table,
)


def align_routes(
    routes: np.ndarray, group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = routes.reshape(-1)
    output_groups = flat.size
    sorted_routes: list[int] = []
    block_experts: list[int] = []
    for expert in range(NUM_EXPERTS):
        expert_routes = np.flatnonzero(flat == expert).astype(np.int32).tolist()
        if not expert_routes:
            continue
        padding = (-len(expert_routes)) % group_size
        expert_routes.extend([output_groups] * padding)
        sorted_routes.extend(expert_routes)
        block_experts.extend([expert] * (len(expert_routes) // group_size))
    return (
        np.asarray(sorted_routes, dtype=np.int32),
        np.asarray(block_experts, dtype=np.int32),
        np.asarray([len(sorted_routes)], dtype=np.int32),
    )


def time_operation(operation, warmup: int, iterations: int) -> float:
    result = None
    for _ in range(warmup):
        result = operation()
    if result is None or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("kernel produced non-finite output")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def require_exact(label: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    if torch.equal(expected, actual):
        return
    mismatch = int(torch.count_nonzero(expected != actual).item())
    max_abs = float((expected.float() - actual.float()).abs().max().item())
    raise AssertionError(
        f"{label} is not BF16-exact: mismatches={mismatch} max_abs={max_abs}"
    )


def benchmark_case(module, tensors, case, args, device) -> None:
    packed = inspect_shape(tensors, case)
    storage_bytes = args.experts * packed.bytes_per_expert
    if storage_bytes > args.max_storage_mib * MIB:
        raise ValueError(
            f"{case.name} needs {storage_bytes / MIB:.2f} MiB, above "
            f"--max-storage-mib={args.max_storage_mib}"
        )
    expert_ids = list(range(args.experts))
    pair = load_expert_pair(
        tensors,
        case,
        packed,
        expert_ids,
        args.tp_rank,
        args.placement,
        device,
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
    if args.placement == "hot":
        cold_gate, hot_gate = empty_gate, pair.gate.accelerator
        cold_down, hot_down = empty_down, pair.down.accelerator
    else:
        cold_gate, hot_gate = pair.gate.accelerator, empty_gate
        cold_down, hot_down = pair.down.accelerator, empty_down

    route_values = np.arange(args.tokens * TOP_K, dtype=np.int32)
    routes_np = (route_values % args.experts).reshape(args.tokens, TOP_K)
    routes = torch.from_numpy(routes_np).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + case.layer)
    x_gate = torch.randn(
        (args.tokens, 2560),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    x_down = torch.randn(
        (args.tokens * TOP_K, 320),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    def generic_gate():
        return module.tiered_iq_moe_gemv(
            x_gate,
            cold_gate,
            hot_gate,
            pair.hot_map,
            pair.cold_map,
            routes,
            TOP_K,
            case.gate_qtype,
            packed.gate_rows,
            args.tokens,
        )

    def generic_down():
        return module.tiered_iq_moe_gemv(
            x_down,
            cold_down,
            hot_down,
            pair.hot_map,
            pair.cold_map,
            routes.reshape(-1),
            1,
            case.down_qtype,
            packed.down_rows,
            args.tokens * TOP_K,
        )

    expected_gate = generic_gate()
    expected_down = generic_down()
    torch.cuda.synchronize(device)
    generic_gate_ms = time_operation(generic_gate, args.warmup, args.iterations)
    generic_down_ms = time_operation(generic_down, args.warmup, args.iterations)
    generic_ms = generic_gate_ms + generic_down_ms
    print(
        f"BASELINE case={case.name} tp_rank={args.tp_rank} "
        f"placement={args.placement} tokens={args.tokens} experts={args.experts} "
        f"storage_mib={storage_bytes / MIB:.3f} gate_ms={generic_gate_ms:.6f} "
        f"down_ms={generic_down_ms:.6f} pair_ms={generic_ms:.6f}",
        flush=True,
    )

    for group_size in args.groups:
        sorted_np, block_np, num_post_np = align_routes(routes_np, group_size)
        sorted_routes = torch.from_numpy(sorted_np).to(device)
        block_experts = torch.from_numpy(block_np).to(device)
        num_post = torch.from_numpy(num_post_np).to(device)

        grouped_gate = partial(
            module.tiered_iq_moe_prefill_grouped,
            x_gate,
            cold_gate,
            hot_gate,
            pair.hot_map,
            pair.cold_map,
            sorted_routes,
            block_experts,
            num_post,
            TOP_K,
            case.gate_qtype,
            packed.gate_rows,
            args.tokens,
            group_size,
        )
        grouped_down = partial(
            module.tiered_iq_moe_prefill_grouped,
            x_down,
            cold_down,
            hot_down,
            pair.hot_map,
            pair.cold_map,
            sorted_routes,
            block_experts,
            num_post,
            1,
            case.down_qtype,
            packed.down_rows,
            args.tokens * TOP_K,
            group_size,
        )

        actual_gate = grouped_gate()
        actual_down = grouped_down()
        require_exact(
            f"{case.name} grouped-{group_size} W13", expected_gate, actual_gate
        )
        require_exact(
            f"{case.name} grouped-{group_size} W2", expected_down, actual_down
        )
        group_gate_ms = time_operation(grouped_gate, args.warmup, args.iterations)
        group_down_ms = time_operation(grouped_down, args.warmup, args.iterations)
        group_ms = group_gate_ms + group_down_ms
        print(
            f"RESULT case={case.name} tp_rank={args.tp_rank} "
            f"placement={args.placement} tokens={args.tokens} group={group_size} "
            f"blocks={block_np.size} gate_ms={group_gate_ms:.6f} "
            f"down_ms={group_down_ms:.6f} pair_ms={group_ms:.6f} "
            f"speedup={generic_ms / group_ms:.4f} exact=1",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tp-rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--placement", choices=("hot", "uva"), default="uva")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--groups", nargs="+", type=int, default=(4, 8, 16))
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-storage-mib", type=float, default=160.0)
    parser.add_argument("--vram-fraction", type=float, default=0.05)
    parser.add_argument(
        "--uva-coherence", choices=("coherent", "noncoherent"), default="coherent"
    )
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if any(group not in {4, 8, 16} for group in args.groups):
        parser.error("--groups values must be 4, 8, or 16")
    if not 65 <= args.tokens <= 4096:
        parser.error("--tokens must be in 65..4096")
    if not TOP_K <= args.experts <= NUM_EXPERTS:
        parser.error(f"--experts must be in {TOP_K}..{NUM_EXPERTS}")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if not 0 < args.vram_fraction <= 0.1:
        parser.error("--vram-fraction must be in (0, 0.1]")
    return args


def main() -> None:
    args = parse_args()
    os.environ["RADIANCE_UVA_HOST_NONCOHERENT"] = "0"
    os.environ["RADIANCE_UVA_HOST_COHERENCE"] = args.uva_coherence
    device = select_device(args.device, args.vram_fraction)
    module = load_module(args.extension.resolve(strict=True))
    tensors = tensor_table(gguf.GGUFReader(args.model.resolve(strict=True)))
    cases = CASES.values() if args.case == "all" else (CASES[args.case],)
    print(
        f"DEVICE selected={device} name={torch.cuda.get_device_name(device)!r} "
        f"tp_rank={args.tp_rank} vram_fraction={args.vram_fraction:.3f} "
        f"uva_coherence={args.uva_coherence}",
        flush=True,
    )
    for case in cases:
        benchmark_case(module, tensors, case, args, device)
        release_cached_storage()


if __name__ == "__main__":
    main()

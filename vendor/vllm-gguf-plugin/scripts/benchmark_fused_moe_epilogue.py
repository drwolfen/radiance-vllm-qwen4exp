#!/usr/bin/env python3
# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0

"""Bounded parity/timing check for the opt-in fused MoE epilogue."""

from __future__ import annotations

import argparse

import torch

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.fused_moe.weighted_sum import (
    fused_weighted_moe_sum_shared,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 64:
        parser.error("--tokens must be in [1, 64]")
    if not 1 <= args.top_k <= 32:
        parser.error("--top-k must be in [1, 32]")
    if not 1 <= args.hidden <= 32768:
        parser.error("--hidden must be in [1, 32768]")
    if not 0 <= args.warmup <= 1000:
        parser.error("--warmup must be in [0, 1000]")
    if not 1 <= args.iterations <= 10000:
        parser.error("--iterations must be in [1, 10000]")
    return args


def legacy(
    routed: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
    weighted: torch.Tensor,
    output: torch.Tensor,
) -> None:
    torch.mul(routed, weights.unsqueeze(-1), out=weighted)
    ops.moe_sum(weighted, output)
    output.add_(shared)


def time_ms(function, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    routed = torch.randn(
        args.tokens, args.top_k, args.hidden, device="cuda", dtype=torch.bfloat16
    )
    weights = torch.rand(args.tokens, args.top_k, device="cuda")
    shared = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    legacy_output = torch.empty_like(shared)
    fused_output = torch.empty_like(shared)
    weighted = torch.empty_like(routed)

    legacy_call = lambda: legacy(routed, weights, shared, weighted, legacy_output)
    fused_call = lambda: fused_weighted_moe_sum_shared(
        routed, weights, shared, fused_output
    )
    for _ in range(args.warmup):
        legacy_call()
        fused_call()
    torch.cuda.synchronize()

    legacy_call()
    fused_call()
    torch.cuda.synchronize()
    if not torch.equal(legacy_output.view(torch.int16), fused_output.view(torch.int16)):
        raise AssertionError("fused epilogue is not bit-exact")

    legacy_ms = time_ms(legacy_call, args.iterations)
    fused_ms = time_ms(fused_call, args.iterations)
    routed_bytes = args.tokens * args.top_k * args.hidden * 2
    output_bytes = args.tokens * args.hidden * 2
    weight_bytes = args.tokens * args.top_k * 4
    legacy_bytes = 3 * routed_bytes + weight_bytes + 4 * output_bytes
    fused_bytes = routed_bytes + weight_bytes + 2 * output_bytes
    print(f"logical_bytes_saved={legacy_bytes - fused_bytes}")
    print("launches_saved=2")
    print(f"legacy_ms={legacy_ms:.6f}")
    print(f"fused_ms={fused_ms:.6f}")
    print(f"speedup={legacy_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()

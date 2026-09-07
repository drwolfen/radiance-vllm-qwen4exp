# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import gguf
import numpy as np
import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm_gguf_plugin.quantization.params import allocate_uva_host_empty


LAYERS = (0, 2, 4)
PCIE4_X4_PAYLOAD_GBPS = 16 * (128 / 130) / 8 * 4


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_tiered_iq_moe_hip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(reader: gguf.GGUFReader, name: str):
    match = next((item for item in reader.tensors if item.name == name), None)
    if match is None:
        raise KeyError(f"GGUF tensor is missing: {name}")
    return match


def real_tp_expert_pair(
    reader: gguf.GGUFReader, layer: int, num_loaded: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Load real packed experts into production-style pinned host UVA."""
    gate = tensor(reader, f"blk.{layer}.ffn_gate_exps.weight")
    up = tensor(reader, f"blk.{layer}.ffn_up_exps.weight")
    down = tensor(reader, f"blk.{layer}.ffn_down_exps.weight")
    if gate.data.shape[0] != 512 or up.data.shape[0] != 512:
        raise ValueError("unexpected routed-expert count")

    # Column-parallel W13 owns half the output rows on each TP rank.  Row-
    # parallel W2 owns half the quantized input blocks.  The cache copier sees
    # only contiguous bytes, exactly as it does after plugin materialization.
    gate_tp = np.ascontiguousarray(
        gate.data[:num_loaded, : gate.data.shape[1] // 2]
    )
    up_tp = np.ascontiguousarray(up.data[:num_loaded, : up.data.shape[1] // 2])
    down_tp = np.ascontiguousarray(
        down.data[:num_loaded, :, : down.data.shape[2] // 2]
    )
    w13 = np.concatenate(
        (gate_tp.reshape(num_loaded, -1), up_tp.reshape(num_loaded, -1)), axis=1
    )
    w2 = down_tp.reshape(num_loaded, -1)
    owner_w13 = allocate_uva_host_empty((num_loaded, 1, w13.shape[1]), torch.uint8)
    owner_w2 = allocate_uva_host_empty((num_loaded, 1, w2.shape[1]), torch.uint8)
    owner_w13.copy_(torch.from_numpy(w13).reshape_as(owner_w13))
    owner_w2.copy_(torch.from_numpy(w2).reshape_as(owner_w2))
    cold_w13 = get_accelerator_view_from_cpu_tensor(owner_w13)
    cold_w2 = get_accelerator_view_from_cpu_tensor(owner_w2)
    if not cold_w13.is_cuda or not cold_w2.is_cuda:
        raise RuntimeError("pinned host tensors did not produce UVA views")
    return cold_w13, cold_w2, (owner_w13, owner_w2)


def state(cold_w13: torch.Tensor, cold_w2: torch.Tensor, lru: bool):
    num_experts = cold_w13.shape[0]
    route = torch.zeros((1,), dtype=torch.int32, device="cuda")
    cache_w13 = torch.empty((1, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda")
    cache_w2 = torch.empty((1, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda")
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cold_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    cache_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache_tags = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    cache_clock = torch.zeros((2 if lru else 1,), dtype=torch.int32, device="cuda")
    # Pre-admit the legacy arm so both policies copy immediately on every
    # alternating miss.  LRU ignores this state for candidate selection.
    admission = torch.ones((num_experts,), dtype=torch.int32, device="cuda")
    stats = torch.zeros((9 if lru else 5,), dtype=torch.int32, device="cuda")
    args = (
        cold_w13,
        cold_w2,
        hot_map,
        cold_map,
        route,
        cache_w13,
        cache_w2,
        cache_map,
        cache_tags,
        cache_clock,
        admission,
        stats,
    )
    if lru:
        args += (torch.zeros((7,), dtype=torch.int32, device="cuda"),)
    return route, args


def capture_expert_scan(module, args: tuple[torch.Tensor, ...], lru: bool):
    route = args[4]
    prepare = (
        module.tiered_iq_moe_cache_lru_prepare
        if lru
        else module.tiered_iq_moe_cache_prepare
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for expert in range(args[0].shape[0]):
            route.fill_(expert)
            prepare(*args)
    torch.cuda.synchronize()
    return graph


def elapsed_per_fill_ms(
    graph: torch.cuda.CUDAGraph, iterations: int, fills_per_iteration: int
) -> float:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations / fills_per_iteration


def verify_final_copy(args: tuple[torch.Tensor, ...]) -> None:
    cold_w13, cold_w2 = args[0], args[1]
    cache_w13, cache_w2 = args[5], args[6]
    cache_map, cache_tags = args[7], args[8]
    final_expert = cold_w13.shape[0] - 1
    if (
        int(cache_tags[0]) != final_expert
        or int(cache_map[final_expert]) != 0
        or int(cache_map[0]) != -1
    ):
        raise AssertionError("scan benchmark published the wrong final expert")
    if not torch.equal(cache_w13[0], cold_w13[final_expert]) or not torch.equal(
        cache_w2[0], cold_w2[final_expert]
    ):
        raise AssertionError("scan benchmark copied incomplete expert bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.experts <= 512:
        raise ValueError("experts must be between 2 and 512")
    if args.iterations < 5 or args.rounds < 3:
        raise ValueError("timing requires at least 5 iterations and 3 rounds")

    module = load_module(args.extension)
    reader = gguf.GGUFReader(args.model)
    rows = []
    for layer in LAYERS:
        cold_w13, cold_w2, owners = real_tp_expert_pair(
            reader, layer, args.experts
        )
        _, single_args = state(cold_w13, cold_w2, False)
        _, multi_args = state(cold_w13, cold_w2, True)
        single_graph = capture_expert_scan(module, single_args, False)
        multi_graph = capture_expert_scan(module, multi_args, True)
        single_samples = []
        multi_samples = []
        for round_index in range(args.rounds):
            if round_index % 2 == 0:
                single_samples.append(
                    elapsed_per_fill_ms(single_graph, args.iterations, args.experts)
                )
                multi_samples.append(
                    elapsed_per_fill_ms(multi_graph, args.iterations, args.experts)
                )
            else:
                multi_samples.append(
                    elapsed_per_fill_ms(multi_graph, args.iterations, args.experts)
                )
                single_samples.append(
                    elapsed_per_fill_ms(single_graph, args.iterations, args.experts)
                )
        verify_final_copy(single_args)
        verify_final_copy(multi_args)
        single_median = statistics.median(single_samples)
        multi_median = statistics.median(multi_samples)
        row = {
            "layer": layer,
            "w13_bytes": cold_w13[0].numel(),
            "w2_bytes": cold_w2[0].numel(),
            "expert_bytes": cold_w13[0].numel() + cold_w2[0].numel(),
            "single_block_ms": single_median,
            "multiblock128_ms": multi_median,
            "speedup": single_median / multi_median,
            "single_block_gbps": (
                cold_w13[0].numel() + cold_w2[0].numel()
            )
            / (single_median / 1_000)
            / 1e9,
            "multiblock128_gbps": (
                cold_w13[0].numel() + cold_w2[0].numel()
            )
            / (multi_median / 1_000)
            / 1e9,
            "multiblock_payload_efficiency": (
                (cold_w13[0].numel() + cold_w2[0].numel())
                / (multi_median / 1_000)
                / 1e9
                / PCIE4_X4_PAYLOAD_GBPS
            ),
            "source": "pinned-host-UVA",
            "working_set_experts": args.experts,
            "working_set_bytes": args.experts
            * (cold_w13[0].numel() + cold_w2[0].numel()),
            "single_samples_ms": single_samples,
            "multiblock_samples_ms": multi_samples,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    result = {
        "schema": 1,
        "device": str(torch.cuda.get_device_name()),
        "iterations_per_round": args.iterations,
        "fills_per_iteration": args.experts,
        "rounds": args.rounds,
        "source": "pinned-host-UVA",
        "pcie4_x4_payload_gbps": PCIE4_X4_PAYLOAD_GBPS,
        "single_block_api": "tiered_iq_moe_cache_prepare",
        "multiblock_api": "tiered_iq_moe_cache_lru_prepare",
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

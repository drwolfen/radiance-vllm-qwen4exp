# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from statistics import median

import gguf
import numpy as np
import torch

HC_COUNT = 4
HC_RANK = 320
HIDDEN = 2560
HYPER_HIDDEN = HC_COUNT * HIDDEN
TOKENS = 3
CALLS_PER_CYCLE = 97
VARIANTS = {"w4-r1": 0, "w8-r1": 1, "w4-r2": 2, "w8-r2": 3}
TENSOR_NAMES = ("blk.0.hc_attn_up.weight", "blk.0.hc_ffn_up.weight")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_dense_mmvq_hip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def timed_samples(call, warmup: int, samples: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return median(values), percentile(values, 0.95)


def capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def find_tensors(model: Path):
    reader = gguf.GGUFReader(model)
    found = {
        tensor.name: tensor
        for tensor in reader.tensors
        if tensor.name in TENSOR_NAMES
    }
    missing = set(TENSOR_NAMES) - found.keys()
    if missing:
        raise RuntimeError(f"missing HC-up tensor(s): {', '.join(sorted(missing))}")
    for name, tensor in found.items():
        if int(tensor.tensor_type) != 8 or tuple(map(int, tensor.shape)) != (
            HC_RANK,
            HYPER_HIDDEN,
        ):
            raise RuntimeError(
                f"unexpected {name}: qtype={int(tensor.tensor_type)} "
                f"shape={tuple(map(int, tensor.shape))}"
            )
    return reader, found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded exact Qwen4Exp M=3 grouped HC-up sweep"
    )
    parser.add_argument("extension", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--parity-seeds", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.02)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    args = parser.parse_args()
    if not 0.0 < args.max_memory_fraction <= 0.05:
        parser.error("--max-memory-fraction must be in (0, 0.05]")
    if args.samples < 5 or args.parity_seeds < 2:
        parser.error("need at least five samples and two parity seeds")

    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    if free_bytes < args.min_free_gib * 1024**3:
        raise RuntimeError(
            f"device {args.device} has only {free_bytes / 1024**3:.2f} GiB "
            f"free of {total_bytes / 1024**3:.2f} GiB"
        )

    extension = load_module(args.extension)
    reader, tensors = find_tensors(args.model)
    legacy_ms = []
    legacy_graph_ms = []
    variant_ms = {name: [] for name in args.variants}
    variant_graph_ms = {name: [] for name in args.variants}
    weight_bytes = HYPER_HIDDEN * (HC_RANK // 32) * 34
    input_bytes = TOKENS * (512 // 32) * 36
    xn_bytes = TOKENS * HYPER_HIDDEN * 2
    output_bytes = TOKENS * HIDDEN * 2
    unique_bytes = weight_bytes + input_bytes + xn_bytes + output_bytes

    for tensor_name in TENSOR_NAMES:
        tensor = tensors[tensor_name]
        weight = torch.from_numpy(np.array(tensor.data, copy=True)).to(args.device)

        for seed in range(args.parity_seeds):
            torch.manual_seed(20260829 + seed)
            down = torch.randn(
                (TOKENS, 336), dtype=torch.bfloat16, device=args.device
            )
            raw_lora = down[:, :HC_RANK]
            xn = torch.randn(
                (TOKENS, HYPER_HIDDEN), dtype=torch.bfloat16, device=args.device
            )
            expected = extension.dense_gemv_q8_hc_mix(
                weight, raw_lora, xn, HC_COUNT
            )
            for name in args.variants:
                actual = extension.dense_gemv_q8_hc_mix_grouped(
                    weight, raw_lora, xn, HC_COUNT, VARIANTS[name]
                )
                torch.cuda.synchronize()
                if not torch.equal(actual, expected):
                    mismatch = actual != expected
                    difference = (actual.float() - expected.float()).abs()
                    raise AssertionError(
                        f"{tensor_name} {name} seed={seed} changed BF16 bits: "
                        f"mismatch={int(mismatch.sum())}/{mismatch.numel()} "
                        f"max_abs={float(difference.max()):.6f}"
                    )

        torch.manual_seed(20260939)
        down = torch.randn((TOKENS, 336), dtype=torch.bfloat16, device=args.device)
        raw_lora = down[:, :HC_RANK]
        xn = torch.randn(
            (TOKENS, HYPER_HIDDEN), dtype=torch.bfloat16, device=args.device
        )
        legacy = lambda: extension.dense_gemv_q8_hc_mix(
            weight, raw_lora, xn, HC_COUNT
        )
        expected = legacy()
        p50, p95 = timed_samples(legacy, args.warmup, args.samples)
        legacy_ms.append(p50)
        graph, graph_output = capture(legacy)
        graph_p50, graph_p95 = timed_samples(
            graph.replay, args.warmup, args.samples
        )
        legacy_graph_ms.append(graph_p50)
        if not torch.equal(graph_output, expected):
            raise AssertionError(f"{tensor_name} legacy graph changed BF16 bits")
        print(
            f"{tensor_name} legacy p50_ms={p50:.6f} p95_ms={p95:.6f} "
            f"graph_p50_ms={graph_p50:.6f} graph_p95_ms={graph_p95:.6f} "
            f"unique_gbps={unique_bytes / (p50 * 1e6):.1f}"
        )
        del graph

        for name in args.variants:
            variant = VARIANTS[name]
            call = lambda variant=variant: extension.dense_gemv_q8_hc_mix_grouped(
                weight, raw_lora, xn, HC_COUNT, variant
            )
            actual = call()
            if not torch.equal(actual, expected):
                raise AssertionError(f"{tensor_name} {name} timing parity failed")
            p50, p95 = timed_samples(call, args.warmup, args.samples)
            variant_ms[name].append(p50)
            graph, graph_output = capture(call)
            graph_p50, graph_p95 = timed_samples(
                graph.replay, args.warmup, args.samples
            )
            variant_graph_ms[name].append(graph_p50)
            if not torch.equal(graph_output, expected):
                raise AssertionError(f"{tensor_name} {name} graph changed BF16 bits")
            print(
                f"{tensor_name} {name} exact=1 p50_ms={p50:.6f} "
                f"p95_ms={p95:.6f} graph_p50_ms={graph_p50:.6f} "
                f"graph_p95_ms={graph_p95:.6f} "
                f"unique_gbps={unique_bytes / (p50 * 1e6):.1f}"
            )
            del graph

        del weight, down, raw_lora, xn, expected
        torch.cuda.empty_cache()

    legacy_mean = sum(legacy_ms) / len(legacy_ms)
    legacy_graph_mean = sum(legacy_graph_ms) / len(legacy_graph_ms)
    print("weighted-cycle summary")
    print(
        f"  legacy cycle_ms={CALLS_PER_CYCLE * legacy_mean:.6f} "
        f"graph_cycle_ms={CALLS_PER_CYCLE * legacy_graph_mean:.6f}"
    )
    for name in args.variants:
        candidate_mean = sum(variant_ms[name]) / len(variant_ms[name])
        candidate_graph_mean = sum(variant_graph_ms[name]) / len(
            variant_graph_ms[name]
        )
        print(
            f"  {name} cycle_ms={CALLS_PER_CYCLE * candidate_mean:.6f} "
            f"saving_ms={CALLS_PER_CYCLE * (legacy_mean - candidate_mean):.6f} "
            f"graph_cycle_ms={CALLS_PER_CYCLE * candidate_graph_mean:.6f} "
            "graph_saving_ms="
            f"{CALLS_PER_CYCLE * (legacy_graph_mean - candidate_graph_mean):.6f}"
        )

    del reader


if __name__ == "__main__":
    main()

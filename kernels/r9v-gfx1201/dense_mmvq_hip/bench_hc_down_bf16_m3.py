# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from statistics import median

import gguf
import numpy as np
import torch
from vllm import _custom_ops as ops

HC_RANK = 320
HC_COUNT = 4
PAD_ROWS = 12
ROWS = HC_RANK + HC_COUNT + PAD_ROWS
COLS = 10240
TOKENS = 3
CALLS_PER_CYCLE = 96
VARIANTS = {"cached": 0, "non-temporal": 1}
TENSOR_PAIRS = (
    ("blk.0.hc_attn_down.weight", "blk.0.hc_attn_inject.weight"),
    ("blk.0.hc_ffn_down.weight", "blk.0.hc_ffn_inject.weight"),
)


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
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return median(timings), percentile(timings, 0.95)


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
    wanted = {name for pair in TENSOR_PAIRS for name in pair}
    found = {tensor.name: tensor for tensor in reader.tensors if tensor.name in wanted}
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"missing HC tensor(s): {', '.join(sorted(missing))}")
    return reader, found


def runtime_weight(down_tensor, inject_tensor, device: int) -> torch.Tensor:
    if int(down_tensor.tensor_type) != 8:
        raise RuntimeError(
            f"expected Q8_0 HC-down tensor, got qtype {int(down_tensor.tensor_type)}"
        )
    if int(inject_tensor.tensor_type) != 0:
        raise RuntimeError(
            "expected F32 HC-injection tensor, got qtype "
            f"{int(inject_tensor.tensor_type)}"
        )
    down = gguf.dequantize(
        np.asarray(down_tensor.data),
        gguf.GGMLQuantizationType(int(down_tensor.tensor_type)),
    )
    inject = np.asarray(inject_tensor.data)
    down = np.asarray(down).reshape(HC_RANK, COLS)
    inject = np.asarray(inject).reshape(HC_COUNT, COLS)
    weight = torch.cat(
        (
            torch.from_numpy(np.array(down, copy=True)),
            torch.from_numpy(np.array(inject, copy=True)),
            torch.zeros((PAD_ROWS, COLS), dtype=torch.float32),
        ),
        dim=0,
    ).to(device=device, dtype=torch.bfloat16)
    if tuple(weight.shape) != (ROWS, COLS) or not weight.is_contiguous():
        raise AssertionError(f"unexpected runtime HC weight {tuple(weight.shape)}")
    return weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded exact BF16 Qwen4Exp HC-down M=3 kernel sweep"
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
    native_ms = []
    native_graph_ms = []
    variant_ms = {name: [] for name in args.variants}
    variant_graph_ms = {name: [] for name in args.variants}
    unique_bytes = ROWS * COLS * 2 + TOKENS * COLS * 2 + TOKENS * ROWS * 2

    for down_name, inject_name in TENSOR_PAIRS:
        label = down_name.removesuffix(".weight")
        weight = runtime_weight(tensors[down_name], tensors[inject_name], args.device)
        outputs = {name: torch.empty((TOKENS, ROWS), **{
            "dtype": torch.bfloat16, "device": args.device
        }) for name in args.variants}

        for seed in range(args.parity_seeds):
            torch.manual_seed(20260829 + seed)
            x = torch.randn((TOKENS, COLS), dtype=torch.bfloat16, device=args.device)
            expected = ops.wvSplitK(weight, x, 32, None)
            for name, variant in VARIANTS.items():
                if name not in outputs:
                    continue
                extension.hc_down_bf16_m3_out(weight, x, outputs[name], variant)
                torch.cuda.synchronize()
                if not torch.equal(outputs[name], expected):
                    mismatch = outputs[name] != expected
                    difference = (outputs[name].float() - expected.float()).abs()
                    raise AssertionError(
                        f"{label} {name} seed={seed} changed BF16 bits: "
                        f"mismatch={int(mismatch.sum())}/{mismatch.numel()} "
                        f"max_abs={float(difference.max()):.6f}"
                    )

        torch.manual_seed(20260938)
        x = torch.randn((TOKENS, COLS), dtype=torch.bfloat16, device=args.device)
        native = lambda: ops.wvSplitK(weight, x, 32, None)
        native_p50, native_p95 = timed_samples(native, args.warmup, args.samples)
        native_ms.append(native_p50)
        native_graph, native_graph_output = capture(native)
        native_graph_p50, native_graph_p95 = timed_samples(
            native_graph.replay, args.warmup, args.samples
        )
        native_graph_ms.append(native_graph_p50)
        expected = native()
        if not torch.equal(native_graph_output, expected):
            raise AssertionError(f"{label} native graph replay changed BF16 bits")
        print(
            f"{label} native p50_ms={native_p50:.6f} p95_ms={native_p95:.6f} "
            f"graph_p50_ms={native_graph_p50:.6f} "
            f"graph_p95_ms={native_graph_p95:.6f} "
            f"unique_gbps={unique_bytes / (native_p50 * 1e6):.1f}"
        )

        for name in args.variants:
            variant = VARIANTS[name]
            output = outputs[name]

            def call(
                weight=weight, x=x, output=output, variant=variant
            ) -> torch.Tensor:
                extension.hc_down_bf16_m3_out(weight, x, output, variant)
                return output

            actual = call()
            if not torch.equal(actual, expected):
                raise AssertionError(f"{label} {name} timing input parity failed")
            p50, p95 = timed_samples(call, args.warmup, args.samples)
            variant_ms[name].append(p50)
            graph, graph_output = capture(call)
            graph_p50, graph_p95 = timed_samples(
                graph.replay, args.warmup, args.samples
            )
            variant_graph_ms[name].append(graph_p50)
            if not torch.equal(graph_output, expected):
                raise AssertionError(f"{label} {name} graph replay changed BF16 bits")
            print(
                f"{label} {name} exact=1 p50_ms={p50:.6f} p95_ms={p95:.6f} "
                f"graph_p50_ms={graph_p50:.6f} graph_p95_ms={graph_p95:.6f} "
                f"unique_gbps={unique_bytes / (p50 * 1e6):.1f}"
            )
            del graph

        del native_graph, weight, x, expected, outputs
        torch.cuda.empty_cache()

    print("weighted-cycle summary")
    native_mean = sum(native_ms) / len(native_ms)
    native_graph_mean = sum(native_graph_ms) / len(native_graph_ms)
    print(
        f"  native cycle_ms={CALLS_PER_CYCLE * native_mean:.6f} "
        f"graph_cycle_ms={CALLS_PER_CYCLE * native_graph_mean:.6f}"
    )
    for name in args.variants:
        candidate_mean = sum(variant_ms[name]) / len(variant_ms[name])
        candidate_graph_mean = sum(variant_graph_ms[name]) / len(
            variant_graph_ms[name]
        )
        print(
            f"  {name} cycle_ms={CALLS_PER_CYCLE * candidate_mean:.6f} "
            f"saving_ms={CALLS_PER_CYCLE * (native_mean - candidate_mean):.6f} "
            f"graph_cycle_ms={CALLS_PER_CYCLE * candidate_graph_mean:.6f} "
            "graph_saving_ms="
            f"{CALLS_PER_CYCLE * (native_graph_mean - candidate_graph_mean):.6f}"
        )

    del reader


if __name__ == "__main__":
    main()

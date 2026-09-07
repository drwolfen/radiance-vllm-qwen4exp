# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from statistics import median

import gguf
import numpy as np
import torch
import vllm_gguf_plugin  # noqa: F401
from vllm_gguf_plugin import ops as gguf_ops

SHAPE_COUNTS = {(8192, 2560): 36, (6656, 2560): 12}
COMPONENTS = {
    (8192, 2560): (
        ("blk.0.attn_qkv.weight", 10240),
        ("blk.0.attn_gate.weight", 6144),
    ),
    (6656, 2560): (
        ("blk.3.attn_q.weight", 12288),
        ("blk.3.attn_k.weight", 512),
        ("blk.3.attn_v.weight", 512),
    ),
}
VARIANTS = {
    "reuse3": 0,
    "exact4": 1,
    "exact4-w8": 2,
    "group4": 3,
    "group4-w8": 4,
    "group4-w10": 5,
}
GENERIC_CYCLE_MS = 3.062245


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_dense_mmvq_hip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


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


def graph_call(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph.replay, output


def find_tensors(paths: list[Path]):
    found = {}
    readers = []
    wanted = {name for components in COMPONENTS.values() for name, _ in components}
    for path in paths:
        reader = gguf.GGUFReader(path)
        readers.append(reader)
        for tensor in reader.tensors:
            if tensor.name in wanted and tensor.name not in found:
                found[tensor.name] = (path, tensor)
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(
            f"Q8 component tensor(s) not found: {', '.join(sorted(missing))}"
        )
    for components in COMPONENTS.values():
        for name, rows in components:
            _, tensor = found[name]
            shape = tuple(map(int, tensor.shape))
            if int(tensor.tensor_type) != 8 or shape != (2560, rows):
                raise RuntimeError(
                    f"unexpected {name}: qtype={int(tensor.tensor_type)} shape={shape}"
                )
    return readers, found


def tp_weight(tensors, shape: tuple[int, int], rank: int) -> np.ndarray:
    parts = []
    for name, full_rows in COMPONENTS[shape]:
        _, tensor = tensors[name]
        local_rows = full_rows // 2
        start = rank * local_rows
        parts.append(np.asarray(tensor.data)[start : start + local_rows])
    packed = np.concatenate(parts, axis=0)
    expected_rows, cols = shape
    expected_shape = (expected_rows, (cols // 32) * 34)
    if packed.shape != expected_shape:
        raise RuntimeError(
            f"packed TP{rank} weight is {packed.shape}, expected {expected_shape}"
        )
    return packed


def parity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, ...]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = (actual_f32 - expected_f32).abs()
    relative_l2 = float(difference.norm() / expected_f32.norm().clamp_min(1e-12))
    return (
        relative_l2,
        float(difference.max()),
        float(difference.mean()),
        float((actual == expected).float().mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded real-Q8 Qwen3.8 M=3 attention kernel sweep"
    )
    parser.add_argument("extension", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-memory-fraction", type=float, default=0.05)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--tp-ranks", type=int, nargs="+", choices=(0, 1), default=(0, 1)
    )
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.max_memory_fraction <= 0.05:
        parser.error("--max-memory-fraction must be in (0, 0.05]")
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    if os.environ.get("VLLM_GGUF_USE_CUDA", "1") != "1":
        parser.error("native generic control requires VLLM_GGUF_USE_CUDA=1")

    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    if free_bytes < args.min_free_gib * 1024**3:
        raise RuntimeError(
            f"device {args.device} has only {free_bytes / 1024**3:.2f} GiB "
            f"free of {total_bytes / 1024**3:.2f} GiB"
        )

    module = load_module(args.extension)
    readers, tensors = find_tensors(args.models)
    torch.manual_seed(23)
    generic_cycle_p50 = 0.0
    generic_cycle_graph_p50 = 0.0
    cycle_p50 = {name: 0.0 for name in args.variants}
    cycle_graph_p50 = {name: 0.0 for name in args.variants}

    rank_weight = 1.0 / len(args.tp_ranks)
    for (rows, cols), count in SHAPE_COUNTS.items():
        component_names = ",".join(name for name, _ in COMPONENTS[(rows, cols)])
        for rank in args.tp_ranks:
            print(f"components={component_names} tp_rank={rank} shape={rows}x{cols}")
            packed = tp_weight(tensors, (rows, cols), rank)
            weight = torch.from_numpy(packed).to("cuda")
            x = torch.randn((3, cols), dtype=torch.bfloat16, device="cuda")
            native = lambda weight=weight, x=x, rows=rows: gguf_ops.ggml_mul_mat_vec_a8(
                weight, x, 8, rows
            )
            native_output = native()
            if not bool(torch.isfinite(native_output).all()):
                raise AssertionError("generic MMVQ emitted a non-finite value")
            native_p50, native_p95 = timed_samples(native, args.warmup, args.samples)
            generic_cycle_p50 += rank_weight * count * native_p50
            print(f"  generic p50_ms={native_p50:.6f} p95_ms={native_p95:.6f}")
            if not args.skip_graphs:
                native_replay, native_graph_output = graph_call(native)
                native_graph_p50, native_graph_p95 = timed_samples(
                    native_replay, args.warmup, args.samples
                )
                if not torch.equal(native_graph_output, native_output):
                    raise AssertionError("generic graph replay parity failed")
                generic_cycle_graph_p50 += rank_weight * count * native_graph_p50
                print(
                    f"    graph_p50_ms={native_graph_p50:.6f} "
                    f"graph_p95_ms={native_graph_p95:.6f}"
                )

            control = lambda weight=weight, x=x, rows=rows: (
                module.dense_gemv_q8_attention_m3(weight, x, rows, VARIANTS["reuse3"])
            )
            control_output = control()
            control_metrics = parity(control_output, native_output)
            print(
                "  control-vs-generic "
                f"rel_l2={control_metrics[0]:.8f} "
                f"max_abs={control_metrics[1]:.6f} "
                f"mean_abs={control_metrics[2]:.8f} "
                f"bf16_exact={control_metrics[3]:.6f}"
            )
            if not torch.equal(control_output, native_output):
                raise AssertionError("reuse3 control parity failed")

            unique_bytes = rows * (cols // 32) * 34
            for name in args.variants:
                variant = VARIANTS[name]
                call = lambda weight=weight, x=x, rows=rows, variant=variant: (
                    module.dense_gemv_q8_attention_m3(weight, x, rows, variant)
                )
                output = call()
                if not bool(torch.isfinite(output).all()):
                    raise AssertionError(f"{name} emitted a non-finite value")
                metrics = parity(output, control_output)
                if name.startswith("exact4") and not torch.equal(
                    output, control_output
                ):
                    raise AssertionError(f"{name} changed BF16 output bits")
                if metrics[0] > 2e-3 or metrics[1] > 0.25:
                    raise AssertionError(f"{name} parity failed")
                p50, p95 = timed_samples(call, args.warmup, args.samples)
                bandwidth = unique_bytes / (p50 * 1e6)
                cycle_p50[name] += rank_weight * count * p50
                print(
                    f"  {name} rel_l2={metrics[0]:.8f} "
                    f"max_abs={metrics[1]:.6f} mean_abs={metrics[2]:.8f} "
                    f"bf16_exact={metrics[3]:.6f} p50_ms={p50:.6f} "
                    f"p95_ms={p95:.6f} unique_gbps={bandwidth:.1f}"
                )
                if not args.skip_graphs:
                    replay, graph_output = graph_call(call)
                    graph_p50, graph_p95 = timed_samples(
                        replay, args.warmup, args.samples
                    )
                    if not torch.equal(graph_output, output):
                        raise AssertionError(f"{name} graph replay parity failed")
                    cycle_graph_p50[name] += rank_weight * count * graph_p50
                    print(
                        f"    graph_p50_ms={graph_p50:.6f} graph_p95_ms={graph_p95:.6f}"
                    )
            del packed, weight, x, native_output, control_output
            torch.cuda.empty_cache()

    del readers

    total_bytes = sum(
        count * rows * (cols // 32) * 34 for (rows, cols), count in SHAPE_COUNTS.items()
    )
    print("weighted-cycle summary")
    generic_bandwidth = total_bytes / (generic_cycle_p50 * 1e6)
    generic_line = (
        f"  generic cycle_p50_ms={generic_cycle_p50:.6f} "
        f"unique_gbps={generic_bandwidth:.1f} "
        f"historical_cycle_ms={GENERIC_CYCLE_MS:.6f}"
    )
    if not args.skip_graphs:
        generic_line += f" graph_cycle_p50_ms={generic_cycle_graph_p50:.6f}"
    print(generic_line)
    for name in args.variants:
        cycle_ms = cycle_p50[name]
        bandwidth = total_bytes / (cycle_ms * 1e6)
        saving = generic_cycle_p50 - cycle_ms
        line = (
            f"  {name} cycle_p50_ms={cycle_ms:.6f} "
            f"unique_gbps={bandwidth:.1f} measured_saving_ms={saving:.6f}"
        )
        if not args.skip_graphs:
            line += f" graph_cycle_p50_ms={cycle_graph_p50[name]:.6f}"
        print(line)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import gguf
import numpy as np
import torch

import vllm_gguf_plugin  # noqa: F401
from vllm_gguf_plugin import ops as gguf_ops


REUSE_SHAPES = {
    8: {
        (10240, 2560),
        (5120, 2560),
        (2560, 6144),
        (2560, 3072),
        (6144, 2560),
        (3072, 2560),
        (12288, 2560),
    },
    12: {(248320, 2560), (124160, 2560)},
    14: {(248320, 2560), (124160, 2560)},
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_dense_mmvq_hip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def elapsed(call, warmup: int, iterations: int) -> float:
    for _ in range(warmup): call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations): call()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension")
    parser.add_argument("model")
    parser.add_argument("--vectors", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-memory-fraction", type=float, default=0.05)
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="run only exact-shape reuse3/reuse4 parity and timing cases",
    )
    args = parser.parse_args()
    if not 0.0 < args.max_memory_fraction <= 0.05:
        parser.error("--max-memory-fraction must be in (0, 0.05]")
    if args.reuse_only and args.vectors not in (3, 4):
        parser.error("--reuse-only requires --vectors 3 or 4")
    torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, 0)
    module = load_module(Path(args.extension))
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    cases = (
        "blk.0.hc_attn_down.weight",
        "blk.0.hc_attn_up.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_output.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.3.attn_k.weight",
        "blk.3.attn_v.weight",
        "blk.0.ssm_out.weight",
        "output.weight",
    )
    torch.manual_seed(23)
    for name in cases:
        if name not in tensors: continue
        tensor = tensors[name]
        qtype = int(tensor.tensor_type)
        cols = int(tensor.shape[0]); rows = int(tensor.shape[1])
        if args.reuse_only and (rows, cols) not in REUSE_SHAPES.get(qtype, ()):
            continue
        weight = torch.from_numpy(np.array(tensor.data, copy=True)).to("cuda")
        x = torch.randn((args.vectors, cols), dtype=torch.bfloat16, device="cuda")
        # Go through the same public dispatch used by the live server.  Setting
        # VLLM_GGUF_USE_CUDA=0 makes this the production Triton fallback,
        # avoiding a misleading comparison against a separately built HIP
        # extension that the current image does not use for GEMV.
        native = lambda: gguf_ops.ggml_mul_mat_vec_a8(weight, x, qtype, rows)
        expected = native()
        native_ms = elapsed(native, args.warmup, args.iterations)
        print(f"{name} qtype={qtype} shape={rows}x{cols} native_ms={native_ms:.6f}")
        if qtype in (12, 13, 14) and not args.reuse_only:
            for waves in (1, 2, 4, 5, 6, 8):
                custom = lambda waves=waves: module.dense_gemv(
                    weight, x, qtype, rows, waves
                )
                actual = custom()
                rel = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-12)
                )
                ms = elapsed(custom, args.warmup, args.iterations)
                print(
                    f"  waves={waves} rel_l2={rel:.8f} ms={ms:.6f} "
                    f"speedup={native_ms/ms:.3f}x"
                )
        if qtype == 12 and args.vectors == 2:
            for waves in (1, 2, 4):
                reuse = lambda waves=waves: module.dense_gemv_q4_reuse2(
                    weight, x, rows, waves
                )
                actual = reuse()
                rel = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-12)
                )
                ms = elapsed(reuse, args.warmup, args.iterations)
                print(
                    f"  reuse2_waves={waves} rel_l2={rel:.8f} "
                    f"ms={ms:.6f} speedup={native_ms/ms:.3f}x"
                )
                if waves == 1:
                    graph = torch.cuda.CUDAGraph()
                    # Materialize every lazy path before capture, then time
                    # the exact replay mode used by vLLM decode graphs.
                    reuse()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        graph_output = reuse()
                    for _ in range(args.warmup):
                        graph.replay()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    graph_ms = start.elapsed_time(end) / args.iterations
                    graph_rel = float(
                        (graph_output.float() - expected.float()).norm()
                        / expected.float().norm().clamp_min(1e-12)
                    )
                    print(
                        f"  reuse2_graph rel_l2={graph_rel:.8f} "
                        f"ms={graph_ms:.6f} speedup={native_ms/graph_ms:.3f}x"
                    )
        if qtype == 8 and args.vectors == 2:
            for waves in (1, 2, 4):
                reuse = lambda waves=waves: module.dense_gemv_q8_reuse2(
                    weight, x, rows, waves
                )
                actual = reuse()
                rel = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-12)
                )
                ms = elapsed(reuse, args.warmup, args.iterations)
                print(
                    f"  q8_reuse2_waves={waves} rel_l2={rel:.8f} "
                    f"ms={ms:.6f} speedup={native_ms/ms:.3f}x"
                )
        if qtype in (8, 12, 14) and args.vectors in (3, 4):
            exact_shape = (rows, cols)
            if exact_shape in REUSE_SHAPES[qtype]:
                reuse = (
                    (lambda: module.dense_gemv_reuse3(weight, x, qtype, rows))
                    if args.vectors == 3
                    else (
                        lambda: module.dense_gemv_reuse4(
                            weight, x, qtype, rows
                        )
                    )
                )
                actual = reuse()
                rel = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-12)
                )
                max_abs = float(
                    (actual.float() - expected.float()).abs().max()
                )
                ms = elapsed(reuse, args.warmup, args.iterations)
                print(
                    f"  reuse{args.vectors} rel_l2={rel:.8f} "
                    f"max_abs={max_abs:.6f} ms={ms:.6f} "
                    f"speedup={native_ms/ms:.3f}x"
                )
                if rel > 2e-3 or max_abs > 0.25:
                    raise AssertionError(
                        f"reuse{args.vectors} parity failed for {name}"
                    )
        del weight
        torch.cuda.empty_cache()


if __name__ == "__main__": main()

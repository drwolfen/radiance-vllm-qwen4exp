# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import gguf
import numpy as np
import torch
import vllm_gguf_plugin  # noqa: F401

CASES = (
    ("blk.0.ffn_gate_exps.weight", 21, 2560, 640, "IQ3_S"),
    ("blk.2.ffn_gate_exps.weight", 23, 2560, 640, "IQ4_XS"),
    ("blk.0.ffn_down_exps.weight", 20, 320, 2560, "IQ4_NL_TP"),
    ("blk.0.ffn_down_exps.weight", 20, 640, 2560, "IQ4_NL"),
    ("blk.2.ffn_down_exps.weight", 8, 320, 2560, "Q8_0_TP"),
    ("blk.2.ffn_down_exps.weight", 8, 640, 2560, "Q8_0"),
)

VARIANTS = {
    "generic": 0,
    "auto": 1,
    "u2": 2,
    "u5": 5,
    "u10": 10,
    "reuse3": 30,
    "reuse3v2": 31,
}
VARIANT_BITS = {
    "auto": 1,
    "u2": 2,
    "u5": 4,
    "u10": 8,
    "reuse3": 16,
    "reuse3v2": 32,
}
BLOCK_BYTES = {8: (32, 34), 20: (32, 18), 21: (256, 110), 23: (256, 136)}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_tiered_iq_moe_hip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    relative_l2 = float(difference.norm() / expected.float().norm().clamp_min(1e-12))
    return relative_l2, float(difference.abs().max())


def selected_variants(module, requested: str) -> list[str]:
    if requested != "all":
        return [requested]
    compiled = (
        int(module.tiered_iq_moe_compiled_variants())
        if hasattr(module, "tiered_iq_moe_compiled_variants")
        else 0
    )
    return ["generic"] + [name for name, bit in VARIANT_BITS.items() if compiled & bit]


def run_gemv(module, variant: str, *args):
    code = VARIANTS[variant]
    if code == 0:
        return module.tiered_iq_moe_gemv(*args)
    return module.tiered_iq_moe_gemv_variant(*args, code)


def run_cached_gemv(module, variant: str, *args):
    code = VARIANTS[variant]
    if code == 0:
        return module.tiered_iq_moe_cached_gemv(*args)
    return module.tiered_iq_moe_cached_gemv_variant(*args, code)


def verify_cache8_graph_safety(module, generator: torch.Generator) -> None:
    """Exercise all eight slots, eviction, and fixed-address graph replay."""
    num_experts = 10
    cold_w13 = torch.randint(
        0,
        256,
        (num_experts, 4, 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cold_w2 = torch.randint(
        0,
        256,
        (num_experts, 3, 48),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cache_w13 = torch.empty((8, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda")
    cache_w2 = torch.empty((8, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda")
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cold_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    cache_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache_tags = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    cache_clock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    admission = torch.zeros((num_experts,), dtype=torch.int32, device="cuda")
    stats = torch.zeros((5,), dtype=torch.int32, device="cuda")
    route = torch.zeros((1,), dtype=torch.int32, device="cuda")
    prepare_args = (
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

    # Each single-use expert bypasses once and is admitted on its second call.
    # Eight slots must fill without eviction or reallocating any graph operand.
    pointers = tuple(tensor.data_ptr() for tensor in prepare_args)
    for expert in range(8):
        route.fill_(expert)
        module.tiered_iq_moe_cache_prepare(*prepare_args)
        module.tiered_iq_moe_cache_prepare(*prepare_args)
    torch.cuda.synchronize()
    if int(stats[1]) != 8 or int(stats[3]) != 8 or int(stats[4]) != 0:
        raise AssertionError("cache8 fill/admission counters differ")
    if not torch.equal(cache_tags, torch.arange(8, dtype=torch.int32, device="cuda")):
        raise AssertionError("cache8 tags do not cover the initial eight experts")
    if not torch.equal(cache_map[:8], torch.arange(8, dtype=torch.int32, device="cuda")):
        raise AssertionError("cache8 map does not cover the initial eight experts")
    if not torch.equal(cache_w13, cold_w13[:8]) or not torch.equal(
        cache_w2, cold_w2[:8]
    ):
        raise AssertionError("cache8 packed bytes differ after initial fills")

    # Capture a cache hit, then replay the same fixed-address graph with two
    # new route IDs. Pre-admission makes each replay take the fill path. The
    # device round-robin policy must evict slots 0 then 1 without host route
    # readback or a change to any captured buffer address.
    route.zero_()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.tiered_iq_moe_cache_prepare(*prepare_args)
    fills_before = int(stats[1])
    evictions_before = int(stats[4])
    for expert in (8, 9):
        admission[expert] = 1
        route.fill_(expert)
        graph.replay()
    torch.cuda.synchronize()
    if int(stats[1]) - fills_before != 2 or int(stats[4]) - evictions_before != 2:
        raise AssertionError("cache8 graph replay did not fill and evict twice")
    if int(cache_map[0]) != -1 or int(cache_map[1]) != -1:
        raise AssertionError("cache8 eviction left stale reverse-map entries")
    if int(cache_map[8]) != 0 or int(cache_map[9]) != 1:
        raise AssertionError("cache8 graph replay published the wrong slots")
    if int(cache_tags[0]) != 8 or int(cache_tags[1]) != 9:
        raise AssertionError("cache8 graph replay published the wrong tags")
    if not torch.equal(cache_w13[0], cold_w13[8]) or not torch.equal(
        cache_w2[1], cold_w2[9]
    ):
        raise AssertionError("cache8 graph replay copied the wrong packed bytes")
    if pointers != tuple(tensor.data_ptr() for tensor in prepare_args):
        raise AssertionError("cache8 graph operands changed addresses")

    # Seventeen physical slots are accepted for async logical-cache16's
    # breakable-graph synchronous fallback.  Launcher/plugin logical capacity
    # remains capped at sixteen.
    fallback17_args = list(prepare_args)
    fallback17_args[5] = torch.empty(
        (17, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    fallback17_args[6] = torch.empty(
        (17, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    fallback17_args[7] = torch.full(
        (num_experts,), -1, dtype=torch.int32, device="cuda"
    )
    fallback17_args[8] = torch.full((17,), -1, dtype=torch.int32, device="cuda")
    fallback17_args[9] = torch.zeros((1,), dtype=torch.int32, device="cuda")
    fallback17_args[10] = torch.ones(
        (num_experts,), dtype=torch.int32, device="cuda"
    )
    fallback17_args[11] = torch.zeros((5,), dtype=torch.int32, device="cuda")
    route.fill_(2)
    module.tiered_iq_moe_cache_prepare(*fallback17_args)
    torch.cuda.synchronize()
    if int(fallback17_args[11][1]) != 1:
        raise AssertionError("seventeen-slot synchronous fallback did not fill")

    # The public boundary stays deliberately bounded even if a caller bypasses
    # launcher/plugin validation.
    oversized_args = list(prepare_args)
    oversized_args[5] = torch.empty(
        (18, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[6] = torch.empty(
        (18, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[8] = torch.full((18,), -1, dtype=torch.int32, device="cuda")
    try:
        module.tiered_iq_moe_cache_prepare(*oversized_args)
    except RuntimeError as error:
        if "one through seventeen physical slots" not in str(error):
            raise
    else:
        raise AssertionError("cache boundary accepted eighteen physical slots")


def verify_lru16_graph_safety(module, generator: torch.Generator) -> None:
    """Verify immediate admission, exact LRU order, and fixed graph state."""
    num_experts = 18
    cache_slots = 16
    cold_w13 = torch.randint(
        0,
        256,
        (num_experts, 4, 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cold_w2 = torch.randint(
        0,
        256,
        (num_experts, 3, 48),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cache_w13 = torch.empty(
        (cache_slots, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    cache_w2 = torch.empty(
        (cache_slots, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cold_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    cache_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache_tags = torch.full((cache_slots,), -1, dtype=torch.int32, device="cuda")
    cache_clock = torch.zeros((cache_slots + 1,), dtype=torch.int32, device="cuda")
    admission = torch.zeros((num_experts,), dtype=torch.int32, device="cuda")
    stats = torch.zeros((9,), dtype=torch.int32, device="cuda")
    pending = torch.zeros((7,), dtype=torch.int32, device="cuda")
    route = torch.zeros((1,), dtype=torch.int32, device="cuda")
    prepare_args = (
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
        pending,
    )
    pointers = tuple(tensor.data_ptr() for tensor in prepare_args)

    # LRU admits a singleton immediately.  Filling in ascending order creates
    # dense ranks [0..15], with expert 0 oldest and expert 15 newest.
    for expert in range(cache_slots):
        route.fill_(expert)
        module.tiered_iq_moe_cache_lru_prepare(*prepare_args)
    torch.cuda.synchronize()
    expected = torch.arange(cache_slots, dtype=torch.int32, device="cuda")
    if int(stats[1]) != cache_slots or int(stats[3]) != 0 or int(stats[4]) != 0:
        raise AssertionError("LRU16 immediate-fill counters differ")
    if not torch.equal(cache_tags, expected) or not torch.equal(
        cache_map[:16], expected
    ):
        raise AssertionError("LRU16 initial tags/maps differ")
    if not torch.equal(cache_clock[1:], expected):
        raise AssertionError("LRU16 initial recency ranks are not dense")

    # Exact route-order touches make expert 0 then expert 1 the two newest;
    # expert 2 becomes the oldest and must be the next victim.
    touch_route = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    touch_args = list(prepare_args)
    touch_args[4] = touch_route
    module.tiered_iq_moe_cache_lru_prepare(*touch_args)
    route.fill_(16)
    module.tiered_iq_moe_cache_lru_prepare(*prepare_args)
    torch.cuda.synchronize()
    if int(cache_map[2]) != -1 or int(cache_map[16]) != 2 or int(cache_tags[2]) != 16:
        raise AssertionError("LRU16 did not evict the exact oldest slot")
    if not torch.equal(cache_w13[2], cold_w13[16]) or not torch.equal(
        cache_w2[2], cold_w2[16]
    ):
        raise AssertionError("LRU16 published incomplete packed bytes")

    # Capture on a hit, then replay with a new singleton.  Expert 3 is now the
    # oldest entry, and all graph operands must retain their fixed addresses.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.tiered_iq_moe_cache_lru_prepare(*prepare_args)
    fills_before = int(stats[1])
    evictions_before = int(stats[4])
    route.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    if int(stats[1]) - fills_before != 1 or int(stats[4]) - evictions_before != 1:
        raise AssertionError("LRU16 graph replay did not fill and evict once")
    if int(cache_map[3]) != -1 or int(cache_map[17]) != 3 or int(cache_tags[3]) != 17:
        raise AssertionError("LRU16 graph replay chose the wrong victim")
    live_ranks = sorted(
        int(cache_clock[int(cache_map[e]) + 1])
        for e in range(18)
        if int(cache_map[e]) >= 0
    )
    if live_ranks != list(range(cache_slots)):
        raise AssertionError("LRU16 recency ranks lost dense ordering")
    if pointers != tuple(tensor.data_ptr() for tensor in prepare_args):
        raise AssertionError("LRU16 graph operands changed addresses")

    oversized_args = list(prepare_args)
    oversized_args[5] = torch.empty(
        (17, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[6] = torch.empty(
        (17, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[8] = torch.full((17,), -1, dtype=torch.int32, device="cuda")
    oversized_args[9] = torch.zeros((18,), dtype=torch.int32, device="cuda")
    try:
        module.tiered_iq_moe_cache_lru_prepare(*oversized_args)
    except RuntimeError as error:
        if "one through sixteen slots" not in str(error):
            raise
    else:
        raise AssertionError("LRU cache boundary accepted seventeen slots")


def verify_async_cache8_graph_safety(module, generator: torch.Generator) -> None:
    """Verify the 8-logical/9-physical staging ring and graph replay."""
    num_experts = 10
    cold_w13 = torch.randint(
        0,
        256,
        (num_experts, 4, 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cold_w2 = torch.randint(
        0,
        256,
        (num_experts, 3, 48),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cache_w13 = torch.empty((9, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda")
    cache_w2 = torch.empty((9, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda")
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cold_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    cache_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache_tags = torch.full((9,), -1, dtype=torch.int32, device="cuda")
    cache_clock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    admission = torch.zeros((num_experts,), dtype=torch.int32, device="cuda")
    stats = torch.zeros((9,), dtype=torch.int32, device="cuda")
    pending = torch.zeros((7,), dtype=torch.int32, device="cuda")
    route = torch.zeros((1,), dtype=torch.int32, device="cuda")
    prepare_args = (
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
        pending,
    )
    commit_args = (cache_map, cache_tags, admission, stats, pending)
    pointers = tuple(tensor.data_ptr() for tensor in prepare_args)
    module.tiered_iq_moe_cache_async_init()

    for expert in range(8):
        route.fill_(expert)
        module.tiered_iq_moe_cache_async_prepare(*prepare_args)
        module.tiered_iq_moe_cache_async_commit(*commit_args, True)
        module.tiered_iq_moe_cache_async_prepare(*prepare_args)
        # Finishing the copy stream without scheduling commit must not publish
        # the staging slot.  The current pass therefore still falls back cold.
        torch.cuda.synchronize()
        if int(cache_map[expert]) != -1:
            raise AssertionError("async fill published before safe-to-publish")
        module.tiered_iq_moe_cache_async_commit(*commit_args, True)
    torch.cuda.synchronize()
    if int(stats[1]) != 8 or int(stats[5]) != 8 or int(stats[4]) != 0:
        raise AssertionError("async cache8 fill/schedule counters differ")
    if int((cache_tags >= 0).sum()) != 8 or int((cache_tags < 0).sum()) != 1:
        raise AssertionError("async cache8 lost its unique staging slot")
    for expert in range(8):
        slot = int(cache_map[expert])
        if slot < 0 or int(cache_tags[slot]) != expert:
            raise AssertionError("async cache8 map/tag publication differs")
        if not torch.equal(cache_w13[slot], cold_w13[expert]) or not torch.equal(
            cache_w2[slot], cold_w2[expert]
        ):
            raise AssertionError("async cache8 published incomplete packed bytes")

    # Capture a fixed-address fill/cold-fallback/publish sequence.  Admission
    # is preset so each replay fills.  The final join represents layer 47 and
    # makes the cross-stream graph self-contained.
    route.zero_()
    admission[8:] = 1
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.tiered_iq_moe_cache_async_prepare(*prepare_args)
        # Any current-stream op here stands in for W13/activation/W2.  Commit's
        # event cannot pass it and publication therefore cannot race a reader.
        cache_clock.add_(0)
        module.tiered_iq_moe_cache_async_commit(*commit_args, True)
    fills_before = int(stats[1])
    evictions_before = int(stats[4])
    route.fill_(8)
    graph.replay()
    route.fill_(9)
    graph.replay()
    torch.cuda.synchronize()
    if int(stats[1]) - fills_before != 2 or int(stats[4]) - evictions_before != 2:
        raise AssertionError("async cache8 graph replay did not rotate twice")
    if int(cache_map[0]) != -1 or int(cache_map[1]) != -1:
        raise AssertionError("async cache8 eviction left stale reverse maps")
    if int(cache_map[8]) != 8 or int(cache_map[9]) != 0:
        raise AssertionError("async cache8 staging rotation published wrong slots")
    if int((cache_tags < 0).sum()) != 1 or int(cache_tags[1]) != -1:
        raise AssertionError("async cache8 staging invariant failed after replay")
    if pointers != tuple(tensor.data_ptr() for tensor in prepare_args):
        raise AssertionError("async cache graph operands changed addresses")

    oversized_args = list(prepare_args)
    oversized_args[5] = torch.empty(
        (18, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[6] = torch.empty(
        (18, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda"
    )
    oversized_args[8] = torch.full((18,), -1, dtype=torch.int32, device="cuda")
    try:
        module.tiered_iq_moe_cache_async_prepare(*oversized_args)
    except RuntimeError as error:
        if "two through seventeen physical slots" not in str(error):
            raise
    else:
        raise AssertionError("async cache boundary accepted eighteen physical slots")


def verify_hybrid_duplicate_immediate_graph(module, generator: torch.Generator) -> None:
    """Prove duplicate fills are immediate while singleton fills publish late."""
    num_experts = 4
    cold_w13 = torch.randint(
        0,
        256,
        (num_experts, 4, 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cold_w2 = torch.randint(
        0,
        256,
        (num_experts, 3, 48),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    cache_w13 = torch.empty((3, *cold_w13.shape[1:]), dtype=torch.uint8, device="cuda")
    cache_w2 = torch.empty((3, *cold_w2.shape[1:]), dtype=torch.uint8, device="cuda")
    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cold_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    cache_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache_tags = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    cache_clock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    admission = torch.zeros((num_experts,), dtype=torch.int32, device="cuda")
    stats = torch.zeros((9,), dtype=torch.int32, device="cuda")
    pending = torch.zeros((7,), dtype=torch.int32, device="cuda")
    route = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    observed = torch.full((1,), -2, dtype=torch.int32, device="cuda")
    prepare_args = (
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
        pending,
    )
    commit_args = (cache_map, cache_tags, admission, stats, pending)

    # Duplicate expert 2 is admitted on first touch and must be visible before
    # commit.  The copy/publish mode is mutually exclusive with async stats.
    module.tiered_iq_moe_cache_async_prepare(*prepare_args)
    torch.cuda.synchronize()
    slot2 = int(cache_map[2])
    if slot2 < 0 or int(cache_tags[slot2]) != 2:
        raise AssertionError("duplicate route did not publish synchronously")
    if not torch.equal(cache_w13[slot2], cold_w13[2]) or not torch.equal(
        cache_w2[slot2], cold_w2[2]
    ):
        raise AssertionError("duplicate route published incomplete bytes")
    if int(stats[7]) != 1 or int(stats[8]) != 2 or int(stats[5]) != 0:
        raise AssertionError("hybrid duplicate admission modes overlapped")
    module.tiered_iq_moe_cache_async_commit(*commit_args, True)

    # Capture fixed pointers while hitting expert 2.  Replay first with a new
    # duplicate and observe its map on the current stream before commit.
    selected_map = cache_map.index_select(0, route[:1])
    del selected_map
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.tiered_iq_moe_cache_async_prepare(*prepare_args)
        torch.index_select(cache_map, 0, route[:1], out=observed)
        module.tiered_iq_moe_cache_async_commit(*commit_args, True)
    route.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    if int(observed[0]) < 0 or int(cache_map[0]) != int(observed[0]):
        raise AssertionError("captured duplicate was not visible before commit")

    # First singleton touch bypasses.  Its second graph replay schedules async
    # copy: the captured pre-commit observation stays cold, then final join
    # proves publication completed after the observation.
    route[0] = 1
    route[1] = -1
    graph.replay()
    torch.cuda.synchronize()
    if int(cache_map[1]) != -1:
        raise AssertionError("singleton first touch unexpectedly filled")
    graph.replay()
    torch.cuda.synchronize()
    if int(observed[0]) != -1 or int(cache_map[1]) < 0:
        raise AssertionError("singleton did not preserve delayed publication")
    if int(stats[5]) != 1 or int(stats[6]) != 1 or int(stats[7]) != 2:
        raise AssertionError("hybrid sync/async admission counters differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--variant",
        choices=("all", *VARIANTS),
        default="all",
        help="exact-shape variant to verify; all tests every compiled variant",
    )
    args = parser.parse_args()

    module = load_module(args.extension)
    variants = selected_variants(module, args.variant)
    print("variants", ",".join(variants))
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(29)
    verify_cache8_graph_safety(module, generator)
    print("cache8 graph-safe fill/eviction boundary passed")
    verify_lru16_graph_safety(module, generator)
    print("LRU16 immediate-fill/recency graph boundary passed")
    verify_async_cache8_graph_safety(module, generator)
    print("async cache8 staging/publication graph boundary passed")
    verify_hybrid_duplicate_immediate_graph(module, generator)
    print("hybrid duplicate-immediate/singleton-delayed graph boundary passed")

    for tensor_name, qtype, cols, rows, label in CASES:
        packed = np.array(tensors[tensor_name].data[:4], copy=True)
        block_values, block_bytes = BLOCK_BYTES[qtype]
        packed = np.ascontiguousarray(
            packed[..., : (cols // block_values) * block_bytes]
        )
        full = torch.from_numpy(packed).to("cuda")
        x = torch.randn(
            (2, cols), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        expected = torch.ops._C_gguf.ggml_moe_a8_vec(x, full, ids, 2, qtype, rows, 2)

        cold_map = torch.arange(4, dtype=torch.int32, device="cuda")
        no_hot_map = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        all_hot_map = torch.arange(4, dtype=torch.int32, device="cuda")
        no_cold_map = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        hot_ids = torch.tensor([1, 3], dtype=torch.long, device="cuda")
        hot = full.index_select(0, hot_ids).contiguous()
        cold_ids = torch.tensor([0, 2], dtype=torch.long, device="cuda")
        cold = full.index_select(0, cold_ids).contiguous()
        mixed_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device="cuda")
        mixed_cold_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device="cuda")
        invalid_ids = torch.tensor([[-1, 4]], dtype=torch.int32, device="cuda")
        cache_w1 = torch.empty_like(cold[:1])
        cache_w2 = torch.empty_like(cold[:1])
        cache_map = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        cache_tags = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        cache_clock = torch.zeros((1,), dtype=torch.int32, device="cuda")
        admission = torch.zeros((4,), dtype=torch.int32, device="cuda")
        cache_stats = torch.zeros((5,), dtype=torch.int32, device="cuda")

        # The explicit LRU policy is the default-off multiblock arm.  Its
        # current-stream prepare must make the copied expert visible to the
        # immediately following GEMV for every packed qtype.  Shadowing the
        # cold source after prepare makes a missing/delayed publication fail
        # parity instead of silently falling back to the original bytes.
        lru_cache_w1 = torch.empty_like(cold[:1])
        lru_cache_w2 = torch.empty_like(cold[:1])
        lru_cache_map = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        lru_cache_tags = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        lru_cache_clock = torch.zeros((2,), dtype=torch.int32, device="cuda")
        lru_admission = torch.zeros((4,), dtype=torch.int32, device="cuda")
        lru_stats = torch.zeros((9,), dtype=torch.int32, device="cuda")
        lru_pending = torch.zeros((7,), dtype=torch.int32, device="cuda")
        lru_cold_shadowed = cold.clone()
        lru_cold_shadowed[0].zero_()
        lru_prepare_args = (
            cold,
            cold,
            mixed_map,
            mixed_cold_map,
            ids,
            lru_cache_w1,
            lru_cache_w2,
            lru_cache_map,
            lru_cache_tags,
            lru_cache_clock,
            lru_admission,
            lru_stats,
            lru_pending,
        )
        module.tiered_iq_moe_cache_lru_prepare(*lru_prepare_args)
        lru_cached_by_variant = {
            variant: run_cached_gemv(
                module,
                variant,
                x,
                lru_cold_shadowed,
                hot,
                lru_cache_w1,
                mixed_map,
                mixed_cold_map,
                lru_cache_map,
                ids,
                2,
                qtype,
                rows,
                2,
            )
            for variant in variants
        }
        if int(lru_stats[1]) != 1 or int(lru_cache_map[0]) != 0:
            raise AssertionError(f"{label} LRU multiblock fill was not immediate")
        if not torch.equal(lru_cache_w1[0], cold[0]) or not torch.equal(
            lru_cache_w2[0], cold[0]
        ):
            raise AssertionError(f"{label} LRU multiblock packed bytes differ")

        # Single-occurrence cold experts bypass on first touch.  A second
        # identical route admits expert 0 and copies both projections into the
        # fixed cache without a host-side route decision.
        prepare_args = (
            cold,
            cold,
            mixed_map,
            mixed_cold_map,
            ids,
            cache_w1,
            cache_w2,
            cache_map,
            cache_tags,
            cache_clock,
            admission,
            cache_stats,
        )
        module.tiered_iq_moe_cache_prepare(*prepare_args)
        if int(cache_stats[1]) != 0:
            raise AssertionError(f"{label} cache ignored first-touch admission")
        module.tiered_iq_moe_cache_prepare(*prepare_args)
        if int(cache_stats[1]) != 1 or int(cache_map[0]) != 0:
            raise AssertionError(f"{label} cache did not admit expert 0")
        if not torch.equal(cache_w1[0], cold[0]):
            raise AssertionError(f"{label} cached expert bytes differ")
        for variant in variants:
            all_cold = run_gemv(
                module,
                variant,
                x,
                full,
                full[:1],
                no_hot_map,
                cold_map,
                ids,
                2,
                qtype,
                rows,
                2,
            )
            all_hot = run_gemv(
                module,
                variant,
                x,
                full[:1],
                full,
                all_hot_map,
                no_cold_map,
                ids,
                2,
                qtype,
                rows,
                2,
            )
            mixed = run_gemv(
                module,
                variant,
                x,
                cold,
                hot,
                mixed_map,
                mixed_cold_map,
                ids,
                2,
                qtype,
                rows,
                2,
            )
            cached = run_cached_gemv(
                module,
                variant,
                x,
                cold,
                hot,
                cache_w1,
                mixed_map,
                mixed_cold_map,
                cache_map,
                ids,
                2,
                qtype,
                rows,
                2,
            )

            reports = {
                "cold": error(all_cold, expected),
                "hot": error(all_hot, expected),
                "mixed": error(mixed, expected),
                "cached": error(cached, expected),
                "lru-multiblock": error(lru_cached_by_variant[variant], expected),
            }
            print(
                label,
                variant,
                " ".join(
                    f"{name}=({relative_l2:.8f},{max_abs:.6f})"
                    for name, (relative_l2, max_abs) in reports.items()
                ),
            )
            for relative_l2, max_abs in reports.values():
                if relative_l2 > 2e-3 or max_abs > 0.25:
                    raise AssertionError(
                        f"{label} {variant} tiered kernel parity failed"
                    )

            invalid = run_gemv(
                module,
                variant,
                x[:1],
                full,
                full[:1],
                no_hot_map,
                cold_map,
                invalid_ids,
                2,
                qtype,
                rows,
                1,
            )
            if torch.count_nonzero(invalid):
                raise AssertionError(f"{label} {variant} invalid expert guard failed")

        reuse3_supported = (rows == 640 and qtype in (21, 23)) or (
            cols == 320 and rows == 2560 and qtype in (8, 20)
        )
        reuse3v2_supported = reuse3_supported or (
            cols == 640 and rows == 2560 and qtype in (8, 20)
        )
        reuse_variants = []
        if "reuse3" in variants and reuse3_supported:
            reuse_variants.append("reuse3")
        if "reuse3v2" in variants and reuse3v2_supported:
            reuse_variants.append("reuse3v2")
        if not reuse_variants:
            continue

        reuse_packed = np.array(tensors[tensor_name].data[:12], copy=True)
        reuse_packed = np.ascontiguousarray(
            reuse_packed[..., : (cols // block_values) * block_bytes]
        )
        reuse_full = torch.from_numpy(reuse_packed).to("cuda")
        reuse_ids = torch.tensor(
            [
                list(range(10)),
                [0, 1, 2, 3, 4, 10, 11, 5, 6, 7],
                [0, 1, 2, 3, 8, 9, 10, 11, 4, 5],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        route_k = 10 if rows == 640 else 1
        token_count = 3 if route_k == 10 else 30
        kernel_ids = reuse_ids if route_k == 10 else reuse_ids.reshape(30, 1)
        reuse_x = torch.randn(
            (token_count, cols),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        reuse_expected = torch.ops._C_gguf.ggml_moe_a8_vec(
            reuse_x,
            reuse_full,
            kernel_ids,
            route_k,
            qtype,
            rows,
            token_count,
        )
        hot_ids = torch.arange(1, 12, 2, dtype=torch.long, device="cuda")
        cold_ids = torch.arange(0, 12, 2, dtype=torch.long, device="cuda")
        reuse_hot = reuse_full.index_select(0, hot_ids).contiguous()
        reuse_cold = reuse_full.index_select(0, cold_ids).contiguous()
        reuse_cold_shadowed = reuse_cold.clone()
        reuse_cold_shadowed[0].zero_()
        reuse_hot_map = torch.tensor(
            [-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5],
            dtype=torch.int32,
            device="cuda",
        )
        reuse_cold_map = torch.tensor(
            [0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1],
            dtype=torch.int32,
            device="cuda",
        )
        reuse_cache_map = torch.full((12,), -1, dtype=torch.int32, device="cuda")
        reuse_cache_map[0] = 0
        for reuse_variant in reuse_variants:
            reuse_actual = run_cached_gemv(
                module,
                reuse_variant,
                reuse_x,
                reuse_cold_shadowed,
                reuse_hot,
                reuse_full[:1],
                reuse_hot_map,
                reuse_cold_map,
                reuse_cache_map,
                kernel_ids,
                route_k,
                qtype,
                rows,
                token_count,
            )
            relative_l2, max_abs = error(reuse_actual, reuse_expected)
            print(
                label,
                f"{reuse_variant}-overlap",
                f"mixed-cache=({relative_l2:.8f},{max_abs:.6f})",
            )
            if relative_l2 > 2e-3 or max_abs > 0.25:
                raise AssertionError(
                    f"{label} {reuse_variant} repeated-route parity failed"
                )

        invalid_reuse_ids = kernel_ids.clone()
        invalid_reuse_ids.reshape(-1)[0] = -1
        invalid_reuse_ids.reshape(-1)[-1] = 12
        safe_reuse_ids = invalid_reuse_ids.clone()
        safe_reuse_ids.reshape(-1)[[0, -1]] = 0
        invalid_expected = torch.ops._C_gguf.ggml_moe_a8_vec(
            reuse_x,
            reuse_full,
            safe_reuse_ids,
            route_k,
            qtype,
            rows,
            token_count,
        )
        invalid_expected_routes = invalid_expected.reshape(-1, rows)
        invalid_expected_routes[0].zero_()
        invalid_expected_routes[-1].zero_()
        if torch.count_nonzero(invalid_expected_routes[[0, -1]]):
            raise AssertionError(f"{label} invalid-route oracle was not zeroed")
        for reuse_variant in reuse_variants:
            invalid_actual = run_gemv(
                module,
                reuse_variant,
                reuse_x,
                reuse_cold,
                reuse_hot,
                reuse_hot_map,
                reuse_cold_map,
                invalid_reuse_ids,
                route_k,
                qtype,
                rows,
                token_count,
            )
            relative_l2, max_abs = error(invalid_actual, invalid_expected)
            if relative_l2 > 2e-3 or max_abs > 0.25:
                raise AssertionError(
                    f"{label} {reuse_variant} invalid-route parity failed"
                )


if __name__ == "__main__":
    main()

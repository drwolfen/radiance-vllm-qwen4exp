# Bounded cold-expert cache

The tiered IQ MoE extension has an opt-in, decode-only VRAM cache for cold
experts. It is disabled unless both the tiered kernel and a nonzero cache size
are selected:

```bash
export QWEN38_USE_TIERED_IQ_MOE_HIP=1
export QWEN38_TIERED_EXPERT_CACHE_SLOTS=1
export QWEN38_TIERED_EXPERT_CACHE_RANKS=1
```

Rank 1 is the default cache rank, but the cache itself defaults to zero slots.
The hard limit is sixteen logical slots per layer. On the Qwen3.8 Q4 TP=2 shapes, one
slot holds one W13 and W2 expert pair. Because the packed expert size varies by
layer, one uniform slot uses 58,124,800 bytes (55.43 MiB) over all 48 layers.
Two slots use 110.86 MiB, four use 221.73 MiB, eight use 443.46 MiB, and
sixteen use 886.92 MiB.

`QWEN38_TIERED_EXPERT_CACHE_RANKS=1` keeps the cache on rank 1 only. Set it to
`0,1` only with a manifest that removes the same number of static experts from
both ranks; for example, 325 static + 4 cache slots on rank 0 and 377 static +
8 cache slots on rank 1 preserve the former 329/385 packed-expert byte budgets
exactly. Buffers and policy state remain fixed-address allocations made before
graph capture, independent of the selected rank list.

All weights and state are allocated during tiered-weight materialization. A
captured forward only launches kernels against those stable addresses. Route
selection, admission, replacement, copy, and publication happen on device;
there is no route readback or host synchronization.

The first single use of a cold expert is recorded but remains on the direct UVA
path. A second use, or multiple occurrences in one routed group, admits it. One
block copies both packed projections to the same round-robin slot and publishes
the device map only after both copies complete. W13 and W2 GEMVs resolve sources
in this order: static hot VRAM, dynamic cache VRAM, cold UVA. Shapes with more
than 64 routed occurrences bypass cache preparation.

The default path remains deliberately same-stream: a miss fills before the
current GEMV, while later uses are hits. A default-off asynchronous A/B is
available with:

```bash
export QWEN38_TIERED_EXPERT_CACHE_ASYNC=1
```

Async mode is hybrid and treats `CACHE_SLOTS` as logical capacity, with one
extra physical staging slot per layer. The small planner runs on the current
stream. If the chosen expert occurs more than once in the current routed group,
its packed W13+W2 bytes are synchronously copied and published before W13; this
preserves the large same-cycle MTP reuse that cache4 measured. A singleton that
passes second-touch admission instead copies on one pre-created nonblocking
stream per device. That current pass keeps the old map and falls back cold.
After W2, an event releases singleton publication. The two modes are selected
by one device planner and guarded by disjoint mode values, so they cannot both
fill the same pending record.

At full capacity publication rotates staging into the map and makes the old
victim the next unpublished staging slot. Thus an asynchronous fill never
overwrites bytes that an already-enqueued GEMV could still read.

All copy streams and events are created during materialization. A single wait
at layer 47 drains the serialized fill stream, rather than adding 48 waits.
Breakable graph capture falls back to the same-stream path because its graph
can end before layer 47. The full graph retains fixed pointers and event state
across replay.

The staging slot is real headroom: logical cache8 consumes nine physical slots,
exactly 523,123,200 bytes (498.89 MiB) over 48 layers. The headroom-neutral
rank-1 placement is therefore 376 static + 8 logical cache + 1 staging = the
former 385-static packed-expert budget. Cache16 uses seventeen physical slots
(988,121,600 bytes / 942.35 MiB) and requires 368 static experts for the same
rank-1 budget.

`_gguf_cache_stats` is an int32 device buffer per layer containing:

1. prepare calls
2. fills
3. routed cache hits
4. first-touch bypasses
5. evictions
6. async fills scheduled
7. routed cold fallbacks while an async fill is scheduled
8. synchronous duplicate fills
9. same-pass duplicate routes served by synchronous fills

`test_async_cache_protocol.py` is the CPU-only staging/publication proof.
`test_bounded.py` additionally verifies first-touch admission, pre-commit cold
fallback, unique staging ownership, packed-byte copies, two graph-replay
rotations, cached GEMV parity, every compiled exact-shape variant, and invalid
expert guards. It is a GPU test and must only be run in a deliberately bounded
test window.

## MTP2 multi-token weight reuse

`reuse3` is a separate, default-off exact-shape variant for the target
verification pass:

```bash
export QWEN38_TIERED_IQ_MOE_VARIANT=reuse3
```

It activates only when the GEMV has exactly 30 routed occurrences, which is
both target W13 (`tokens=3`, `top_k=10`) and target W2 (`tokens=30`,
`top_k=1`) under MTP2. Other shapes fall through to the generic kernel.

Each of the 30 graph-stable route blocks scans the same bounded device route
array. A duplicate block returns before reading weights. The first occurrence
of an expert computes all of its matching route outputs with independent
accumulators. Packed blocks are copied from hot VRAM, dynamic cache VRAM, or
cold UVA into a small per-workgroup LDS tile once; IQ3_S, IQ4_XS, IQ4_NL, and
Q8_0 lane decodes are then held in registers across matching activations.
Invalid IDs still write zero. More than three copies of one expert take an
in-kernel per-occurrence fallback, which keeps malformed routing exact without
host readback or a different graph launch.

The recorded MTP2 profile contains 383 complete three-token events. Across
18,384 layer calls, 30 routes reduce to 21.8347 unique experts on average.
Using the actual packed projection formats per layer, the target expert stream
drops from 1.62399 GiB to 1.18423 GiB per 48-layer pass: 0.43976 GiB (27.08%)
fewer weight bytes and a 1.3713x weight-only ceiling. W13 saves 0.25786 GiB and
W2 saves 0.18189 GiB per pass.

`test_reuse_plan.py` is a CPU-only exact ordering/parity test for W13, W2,
invalid IDs, and the greater-than-three fallback. `test_bounded.py` adds real
packed-tensor parity for repeated routes and mixed hot/cache/cold sources. The
latter remains deliberately GPU-only and is not part of a static build step.

`reuse3v2` is the default-off, direct-register A/B arm:

```bash
export QWEN38_TIERED_IQ_MOE_VARIANT=reuse3v2
```

It keeps the same fixed ownership plan and hot, then cache, then cold source
priority as `reuse3`, but does not copy packed weights into LDS. Each lane
loads and decodes its own packed values directly into registers, then reuses
those decoded values across the one to three matching activations. There is
one workgroup barrier to publish route ownership and the selected source
pointer, versus that barrier plus two barriers for every quant block in the
first implementation. The TP W13 path has 10 blocks per output row and TP W2
also has 10, so this removes 20 K-loop barriers from each kernel workgroup.

The recorded routing byte reduction is unchanged: 1.62399 GiB becomes
1.18423 GiB per 48-layer target pass, saving 0.43976 GiB (27.08%). Against the
recorded `u10` replay times, a purely weight-bandwidth-limited ceiling is
3.6198 ms hot instead of 4.964 ms and 170.232 ms UVA instead of 233.447 ms.
Those are byte-ceiling projections, not measured kernel results; promote the
arm only after bounded headless parity and replay timing.

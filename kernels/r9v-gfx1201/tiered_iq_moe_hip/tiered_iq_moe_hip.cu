// SPDX-License-Identifier: Apache-2.0
// R9V gfx1201 specializations. GGUF quant primitives are supplied by the
// vLLM GGUF plugin; see the repository's THIRD_PARTY_NOTICES.md.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <type_traits>

#include "hip_compat.h"
#include "gguf/ggml-common.h"
#include "gguf/vecdotq.cuh"

namespace {

using bf16 = __hip_bfloat16;
constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;

// The generic flattened kernel remains the default.  These bits only control
// which opt-in exact-shape A/B variants are emitted into the extension.
#ifndef QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS
#define QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS 15
#endif

constexpr int kCompiledSpecializedVariants =
    QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS;
static_assert((kCompiledSpecializedVariants & ~63) == 0,
              "unknown exact-shape variant bits");

constexpr int kReuseRoutes = 30;
constexpr int kReuseMatches = 3;
constexpr int kMaxPrefillGroup = 16;
constexpr int kMaxLogicalCacheSlots = 16;
constexpr int kMaxHIPDevices = 16;
constexpr int kCacheCopyBlocks = 128;

// Async cache state uses one process-local copy stream per device.  vLLM calls
// tiered_iq_moe_cache_async_init during model materialization, before CUDA graph
// capture, so stream/event objects never change during replay.  Resources are
// intentionally process-lifetime objects: destroying HIP events after PyTorch
// has torn down its runtime is less safe than letting the OS reclaim them.
struct AsyncCacheContext {
  hipStream_t stream = nullptr;
  hipEvent_t fork = nullptr;
  hipEvent_t safe_to_publish = nullptr;
  hipEvent_t done = nullptr;
};

std::array<AsyncCacheContext, kMaxHIPDevices> g_async_cache_contexts;
std::array<std::once_flag, kMaxHIPDevices> g_async_cache_once;

AsyncCacheContext& async_cache_context() {
  int device = -1;
  AT_CUDA_CHECK(hipGetDevice(&device));
  TORCH_CHECK(device >= 0 && device < kMaxHIPDevices,
              "async expert cache device index is out of range");
  std::call_once(g_async_cache_once[device], [device]() {
    auto& context = g_async_cache_contexts[device];
    AT_CUDA_CHECK(hipStreamCreateWithFlags(&context.stream,
                                            hipStreamNonBlocking));
    AT_CUDA_CHECK(hipEventCreateWithFlags(&context.fork,
                                           hipEventDisableTiming));
    AT_CUDA_CHECK(hipEventCreateWithFlags(&context.safe_to_publish,
                                           hipEventDisableTiming));
    AT_CUDA_CHECK(hipEventCreateWithFlags(&context.done,
                                           hipEventDisableTiming));
  });
  return g_async_cache_contexts[device];
}

template <typename scalar_t>
__global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                              block_q8_1* __restrict__ y, int kx,
                              int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) return;
  const int iy = blockIdx.y;
  const int i_padded = iy * kx_padded + ix;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;
  const float value = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    sum += __shfl_xor(sum, mask, 32);
  }
  const float scale = amax / 127.0f;
  y[ib].qs[iqs] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (iqs == 0) y[ib].ds = __floats2half2_rn(scale, sum);
}

// A deliberately small decode cache.  One block owns the entire admission,
// copy, and publish sequence, so publication cannot race the copy and no
// grid-wide synchronization or host decision is required.  A first-touch
// filter avoids paying copy + VRAM read for one-use tail experts; an expert is
// admitted on its second call or immediately when it occurs more than once in
// the current routed group (notably MTP decode).
__global__ __launch_bounds__(256) void prepare_expert_cache(
    const uint8_t* __restrict__ cold_w13,
    const uint8_t* __restrict__ cold_w2,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ expert_ids,
    uint8_t* __restrict__ cache_w13,
    uint8_t* __restrict__ cache_w2,
    int* __restrict__ cache_map,
    int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ admission,
    int* __restrict__ stats,
    int num_experts, int cold_count, int cache_slots, int routes,
    int64_t w13_expert_bytes, int64_t w2_expert_bytes) {
  __shared__ int selected_expert;
  __shared__ int selected_cold_slot;
  __shared__ int selected_cache_slot;
  __shared__ int should_fill;

  if (threadIdx.x == 0) {
    selected_expert = -1;
    selected_cold_slot = -1;
    selected_cache_slot = -1;
    should_fill = 0;
    stats[0] += 1;  // prepare calls

    int best_score = -1;
    for (int route = 0; route < routes; ++route) {
      const int expert = expert_ids[route];
      if (expert < 0 || expert >= num_experts || hot_map[expert] >= 0 ||
          cold_map[expert] < 0 || cold_map[expert] >= cold_count) {
        continue;
      }

      const int cache_slot = cache_map[expert];
      if (cache_slot >= 0 && cache_slot < cache_slots &&
          cache_tags[cache_slot] == expert) {
        stats[2] += 1;  // routed cache hits
        continue;
      }

      bool first_occurrence = true;
      int occurrences = 0;
      for (int other = 0; other < routes; ++other) {
        occurrences += expert_ids[other] == expert;
        if (other < route && expert_ids[other] == expert) {
          first_occurrence = false;
        }
      }
      if (!first_occurrence) continue;

      const int prior = admission[expert];
      if (prior == 0) admission[expert] = 1;
      if (occurrences == 1 && prior == 0) {
        stats[3] += 1;  // first-touch bypasses
        continue;
      }
      const int score = occurrences * 8 + (prior > 7 ? 7 : prior);
      if (score > best_score) {
        best_score = score;
        selected_expert = expert;
        selected_cold_slot = cold_map[expert];
      }
    }

    if (selected_expert >= 0) {
      int victim = -1;
      for (int slot = 0; slot < cache_slots; ++slot) {
        if (cache_tags[slot] < 0) {
          victim = slot;
          break;
        }
      }
      if (victim < 0) {
        victim = cache_clock[0] % cache_slots;
      }
      cache_clock[0] = (victim + 1) % cache_slots;
      const int old_expert = cache_tags[victim];
      if (old_expert >= 0 && old_expert < num_experts) {
        if (cache_map[old_expert] == victim) cache_map[old_expert] = -1;
        stats[4] += 1;  // evictions
      }
      // The tag/map stay invalid until every byte has arrived.
      cache_tags[victim] = -1;
      selected_cache_slot = victim;
      should_fill = 1;
    }
  }
  __syncthreads();

  if (!should_fill) return;

  const uint8_t* source_w13 =
      cold_w13 + static_cast<int64_t>(selected_cold_slot) * w13_expert_bytes;
  const uint8_t* source_w2 =
      cold_w2 + static_cast<int64_t>(selected_cold_slot) * w2_expert_bytes;
  uint8_t* target_w13 =
      cache_w13 + static_cast<int64_t>(selected_cache_slot) * w13_expert_bytes;
  uint8_t* target_w2 =
      cache_w2 + static_cast<int64_t>(selected_cache_slot) * w2_expert_bytes;

  // Torch allocations and Qwen's packed expert strides are 16-byte aligned.
  // Keep a scalar tail so the helper remains correct for bounded fixtures.
  const int64_t w13_vectors = w13_expert_bytes / sizeof(uint4);
  for (int64_t index = threadIdx.x; index < w13_vectors;
       index += blockDim.x) {
    reinterpret_cast<uint4*>(target_w13)[index] =
        reinterpret_cast<const uint4*>(source_w13)[index];
  }
  for (int64_t index = w13_vectors * sizeof(uint4) + threadIdx.x;
       index < w13_expert_bytes; index += blockDim.x) {
    target_w13[index] = source_w13[index];
  }
  const int64_t w2_vectors = w2_expert_bytes / sizeof(uint4);
  for (int64_t index = threadIdx.x; index < w2_vectors;
       index += blockDim.x) {
    reinterpret_cast<uint4*>(target_w2)[index] =
        reinterpret_cast<const uint4*>(source_w2)[index];
  }
  for (int64_t index = w2_vectors * sizeof(uint4) + threadIdx.x;
       index < w2_expert_bytes; index += blockDim.x) {
    target_w2[index] = source_w2[index];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    cache_tags[selected_cache_slot] = selected_expert;
    cache_map[selected_expert] = selected_cache_slot;
    admission[selected_expert] = 2;
    stats[1] += 1;  // fills
  }
}

// Exact synchronous LRU policy used by the headroom-neutral cache16 arm.
// cache_clock[0] remains reserved for the legacy RR cursor; entries 1..N are
// dense LRU ranks for the corresponding cache slots (0=LRU, published-1=MRU).
// Renormalizing ranks on every touch avoids a graph-replay clock overflow.
__device__ __forceinline__ void touch_lru_slot(
    const int* __restrict__ cache_tags, int* __restrict__ cache_clock,
    int cache_slots, int slot) {
  const int old_rank = cache_clock[slot + 1];
  int published = 0;
  for (int other = 0; other < cache_slots; ++other) {
    if (cache_tags[other] < 0) continue;
    ++published;
    if (other != slot && cache_clock[other + 1] > old_rank) {
      cache_clock[other + 1] -= 1;
    }
  }
  cache_clock[slot + 1] = published > 0 ? published - 1 : 0;
}

// pending fields match the hybrid planner: active, expert, cold slot, target
// slot, victim slot, victim expert, mode.  Mode 3 is synchronous LRU and is
// disjoint from hybrid modes 1/2.  Planning, the 128-block copy, publication,
// and both GEMVs all execute on the current stream.
__global__ __launch_bounds__(256) void plan_expert_cache_lru_fill(
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ expert_ids,
    const int* __restrict__ cache_map,
    const int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int cold_count, int cache_slots, int routes) {
  if (threadIdx.x != 0) return;
  pending[0] = 0;
  if (routes <= 0 || routes > 64) return;
  stats[0] += 1;  // prepare calls

  // Touch every resident route in exact order, including duplicates.  This
  // matches the authoritative offline LRU simulation for interleaved routes.
  for (int route = 0; route < routes; ++route) {
    const int expert = expert_ids[route];
    if (expert < 0 || expert >= num_experts) continue;
    const int slot = cache_map[expert];
    if (slot < 0 || slot >= cache_slots || cache_tags[slot] != expert) {
      continue;
    }
    stats[2] += 1;
    touch_lru_slot(cache_tags, cache_clock, cache_slots, slot);
  }

  int selected_expert = -1;
  int selected_cold_slot = -1;
  int selected_occurrences = 0;
  int selected_first_route = routes;
  for (int route = 0; route < routes; ++route) {
    const int expert = expert_ids[route];
    if (expert < 0 || expert >= num_experts || hot_map[expert] >= 0 ||
        cold_map[expert] < 0 || cold_map[expert] >= cold_count) {
      continue;
    }
    const int slot = cache_map[expert];
    if (slot >= 0 && slot < cache_slots && cache_tags[slot] == expert) {
      continue;
    }
    bool first_occurrence = true;
    int occurrences = 0;
    for (int other = 0; other < routes; ++other) {
      occurrences += expert_ids[other] == expert;
      if (other < route && expert_ids[other] == expert) {
        first_occurrence = false;
      }
    }
    if (!first_occurrence) continue;
    if (occurrences > selected_occurrences ||
        (occurrences == selected_occurrences && route < selected_first_route)) {
      selected_expert = expert;
      selected_cold_slot = cold_map[expert];
      selected_occurrences = occurrences;
      selected_first_route = route;
    }
  }
  if (selected_expert < 0) return;

  int target = -1;
  for (int slot = 0; slot < cache_slots; ++slot) {
    if (cache_tags[slot] < 0) {
      target = slot;
      break;
    }
  }
  if (target < 0) {
    int oldest_rank = cache_slots + 1;
    for (int slot = 0; slot < cache_slots; ++slot) {
      const int rank = cache_clock[slot + 1];
      if (rank < oldest_rank) {
        oldest_rank = rank;
        target = slot;
      }
    }
  }
  if (target < 0) return;

  pending[1] = selected_expert;
  pending[2] = selected_cold_slot;
  pending[3] = target;
  pending[4] = target;
  pending[5] = cache_tags[target];
  pending[6] = 3;
  __threadfence();
  pending[0] = 1;
  stats[7] += 1;                     // synchronous LRU fills selected
  stats[8] += selected_occurrences;  // same-pass routes served from cache
}

__global__ void publish_expert_cache_lru(
    int* __restrict__ cache_map,
    int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ admission,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int cache_slots) {
  if (threadIdx.x != 0 || pending[0] == 0 || pending[6] != 3) return;
  const int expert = pending[1];
  const int target = pending[3];
  const int old_expert = pending[5];
  if (expert < 0 || expert >= num_experts || target < 0 ||
      target >= cache_slots) {
    pending[0] = 0;
    return;
  }

  int old_rank = -1;
  if (old_expert >= 0 && old_expert < num_experts &&
      cache_tags[target] == old_expert) {
    old_rank = cache_clock[target + 1];
    if (cache_map[old_expert] == target) cache_map[old_expert] = -1;
    cache_tags[target] = -1;
    stats[4] += 1;
  }

  int published = 0;
  for (int slot = 0; slot < cache_slots; ++slot) {
    if (cache_tags[slot] < 0) continue;
    ++published;
    if (old_rank >= 0 && cache_clock[slot + 1] > old_rank) {
      cache_clock[slot + 1] -= 1;
    }
  }
  cache_clock[target + 1] = published;
  __threadfence();
  cache_tags[target] = expert;
  __threadfence();
  cache_map[expert] = target;
  admission[expert] = 2;
  stats[1] += 1;
  __threadfence();
  pending[0] = 0;
}

// The asynchronous cache has N logical entries backed by N+1 physical slots.
// Exactly one physical slot remains unpublished while a fill is in flight.
// The old victim stays mapped until the current layer has consumed W13 and W2,
// which prevents the copy stream from racing a GEMV that already read its map.
// pending fields: active, expert, cold slot, staging slot, victim slot,
// victim expert, mode (1 synchronous duplicate, 2 async singleton).  All state
// and addresses are fixed across graph replay.
__global__ __launch_bounds__(256) void plan_expert_cache_fill(
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ expert_ids,
    const int* __restrict__ cache_map,
    const int* __restrict__ cache_tags,
    int* __restrict__ cache_clock,
    int* __restrict__ admission,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int cold_count, int physical_slots, int routes) {
  if (threadIdx.x != 0) return;
  pending[0] = 0;
  if (routes <= 0 || routes > 64) return;

  stats[0] += 1;  // prepare calls
  const int logical_slots = physical_slots - 1;
  int selected_expert = -1;
  int selected_cold_slot = -1;
  int selected_occurrences = 0;
  int best_score = -1;
  for (int route = 0; route < routes; ++route) {
    const int expert = expert_ids[route];
    if (expert < 0 || expert >= num_experts || hot_map[expert] >= 0 ||
        cold_map[expert] < 0 || cold_map[expert] >= cold_count) {
      continue;
    }

    const int cache_slot = cache_map[expert];
    if (cache_slot >= 0 && cache_slot < physical_slots &&
        cache_tags[cache_slot] == expert) {
      stats[2] += 1;  // routed cache hits
      continue;
    }

    bool first_occurrence = true;
    int occurrences = 0;
    for (int other = 0; other < routes; ++other) {
      occurrences += expert_ids[other] == expert;
      if (other < route && expert_ids[other] == expert) {
        first_occurrence = false;
      }
    }
    if (!first_occurrence) continue;

    const int prior = admission[expert];
    if (prior == 0) admission[expert] = 1;
    if (occurrences == 1 && prior == 0) {
      stats[3] += 1;  // first-touch bypasses
      continue;
    }
    const int score = occurrences * 8 + (prior > 7 ? 7 : prior);
    if (score > best_score) {
      best_score = score;
      selected_expert = expert;
      selected_cold_slot = cold_map[expert];
      selected_occurrences = occurrences;
    }
  }
  if (selected_expert < 0) return;

  int published = 0;
  int staging = -1;
  for (int slot = 0; slot < physical_slots; ++slot) {
    if (cache_tags[slot] >= 0) {
      ++published;
    } else if (staging < 0) {
      staging = slot;
    }
  }
  // A pending fill is only legal while an unpublished staging slot exists.
  // If state is corrupt, cold fallback is safer than overwriting live data.
  if (staging < 0 || published > logical_slots) return;

  int victim = -1;
  int old_expert = -1;
  if (published == logical_slots) {
    const int start = cache_clock[0] % physical_slots;
    for (int offset = 0; offset < physical_slots; ++offset) {
      const int candidate = (start + offset) % physical_slots;
      if (cache_tags[candidate] >= 0) {
        victim = candidate;
        old_expert = cache_tags[candidate];
        cache_clock[0] = (candidate + 1) % physical_slots;
        break;
      }
    }
    if (victim < 0) return;
  }

  const int mode = selected_occurrences > 1 ? 1 : 2;
  pending[1] = selected_expert;
  pending[2] = selected_cold_slot;
  pending[3] = staging;
  pending[4] = victim;
  pending[5] = old_expert;
  pending[6] = mode;
  __threadfence();
  pending[0] = 1;
  if (mode == 1) {
    stats[7] += 1;                    // synchronous duplicate fills
    stats[8] += selected_occurrences; // same-pass routes served from cache
  } else {
    stats[5] += 1;                    // asynchronous fills scheduled
    stats[6] += selected_occurrences; // cold routes during async fill
  }
}

__global__ __launch_bounds__(256) void copy_planned_expert_cache(
    const uint8_t* __restrict__ cold_w13,
    const uint8_t* __restrict__ cold_w2,
    uint8_t* __restrict__ cache_w13,
    uint8_t* __restrict__ cache_w2,
    const int* __restrict__ pending,
    int64_t w13_expert_bytes, int64_t w2_expert_bytes,
    int expected_mode) {
  if (pending[0] == 0 || pending[6] != expected_mode) return;
  const int cold_slot = pending[2];
  const int cache_slot = pending[3];
  const uint8_t* source_w13 =
      cold_w13 + static_cast<int64_t>(cold_slot) * w13_expert_bytes;
  const uint8_t* source_w2 =
      cold_w2 + static_cast<int64_t>(cold_slot) * w2_expert_bytes;
  uint8_t* target_w13 =
      cache_w13 + static_cast<int64_t>(cache_slot) * w13_expert_bytes;
  uint8_t* target_w2 =
      cache_w2 + static_cast<int64_t>(cache_slot) * w2_expert_bytes;

  const int64_t w13_vectors = w13_expert_bytes / sizeof(uint4);
  const int64_t w2_vectors = w2_expert_bytes / sizeof(uint4);
  const int64_t vector_count = w13_vectors + w2_vectors;
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread; index < vector_count; index += stride) {
    if (index < w13_vectors) {
      reinterpret_cast<uint4*>(target_w13)[index] =
          reinterpret_cast<const uint4*>(source_w13)[index];
    } else {
      const int64_t w2_index = index - w13_vectors;
      reinterpret_cast<uint4*>(target_w2)[w2_index] =
          reinterpret_cast<const uint4*>(source_w2)[w2_index];
    }
  }
  for (int64_t index = w13_vectors * sizeof(uint4) + thread;
       index < w13_expert_bytes; index += stride) {
    target_w13[index] = source_w13[index];
  }
  for (int64_t index = w2_vectors * sizeof(uint4) + thread;
       index < w2_expert_bytes; index += stride) {
    target_w2[index] = source_w2[index];
  }
}

__global__ void publish_planned_expert_cache(
    int* __restrict__ cache_map,
    int* __restrict__ cache_tags,
    int* __restrict__ admission,
    int* __restrict__ stats,
    int* __restrict__ pending,
    int num_experts, int physical_slots, int expected_mode) {
  if (threadIdx.x != 0 || pending[0] == 0 ||
      pending[6] != expected_mode) return;
  const int expert = pending[1];
  const int staging = pending[3];
  const int victim = pending[4];
  const int old_expert = pending[5];
  if (expert < 0 || expert >= num_experts || staging < 0 ||
      staging >= physical_slots) {
    pending[0] = 0;
    return;
  }

  if (victim >= 0 && victim < physical_slots) {
    if (old_expert >= 0 && old_expert < num_experts &&
        cache_map[old_expert] == victim) {
      cache_map[old_expert] = -1;
    }
    cache_tags[victim] = -1;
    stats[4] += 1;  // evictions
  }
  __threadfence();
  cache_tags[staging] = expert;
  __threadfence();
  cache_map[expert] = staging;
  admission[expert] = 2;
  stats[1] += 1;  // completed fills
  __threadfence();
  pending[0] = 0;
}

template <int lanes_per_row, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot>
__global__ __launch_bounds__(256) void tiered_moe_vec_subgroup(
    const void* __restrict__ cold_weight,
    const void* __restrict__ hot_weight,
    const void* __restrict__ cache_weight,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ cache_map,
    const block_q8_1* __restrict__ input, bf16* __restrict__ output,
    const int* __restrict__ expert_ids, int num_experts, int hot_count,
    int cold_count, int cache_count, int top_k, int ncols, int nrows,
    int token_stride, int output_groups) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int row_in_wave = lane / lanes_per_row;
  const int lane_in_row = lane % lanes_per_row;
  const int blocks_per_row = ncols / qk;
  const int64_t expert_stride = static_cast<int64_t>(nrows) * blocks_per_row;
  const int total_rows = output_groups * nrows;

  for (int linear_row = blockIdx.x * rows_per_block +
                        wave * rows_per_wave + row_in_wave;
       linear_row < total_rows;
       linear_row += gridDim.x * rows_per_block) {
    const int group = linear_row / nrows;
    const int row = linear_row - group * nrows;
    const int token = group / top_k;
    const int expert = expert_ids[group];
    float sum = 0.0f;

    if (expert >= 0 && expert < num_experts) {
      const int hot_slot = hot_map[expert];
      const bool use_hot = hot_slot >= 0 && hot_slot < hot_count;
      const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
      const bool use_cache = cache_slot >= 0 && cache_slot < cache_count;
      const int cold_slot = cold_map[expert];
      const bool use_cold = cold_slot >= 0 && cold_slot < cold_count;
      if (use_hot || use_cache || use_cold) {
        const block_q_t* base =
            use_hot
                ? static_cast<const block_q_t*>(hot_weight) +
                      hot_slot * expert_stride
                : use_cache
                      ? static_cast<const block_q_t*>(cache_weight) +
                            cache_slot * expert_stride
                      : static_cast<const block_q_t*>(cold_weight) +
                            cold_slot * expert_stride;
        const block_q_t* weight =
            base + static_cast<int64_t>(row) * blocks_per_row;
        const block_q8_1* activation = reinterpret_cast<const block_q8_1*>(
            reinterpret_cast<const int*>(input) + token * token_stride);

        for (int block = 0; block < blocks_per_row; ++block) {
          const int activation_block = block * (qk / QK8_1);
          const int quant_index = vdr * lane_in_row;
          sum +=
              vec_dot(&weight[block], &activation[activation_block], quant_index);
        }
      }
    }

#pragma unroll
    for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down(sum, offset, lanes_per_row);
    }
    if (lane_in_row == 0) output[linear_row] = __float2bfloat16(sum);
  }
}

template <int lanes_per_row, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot>
void launch_tiered(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, int ncols, int nrows,
    int tokens, hipStream_t stream) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const int output_groups = tokens * top_k;
  const int64_t total_rows = static_cast<int64_t>(output_groups) * nrows;
  const int64_t grid = (total_rows + rows_per_block - 1) / rows_per_block;
  tiered_moe_vec_subgroup<lanes_per_row, qk, qi, block_q_t, vdr, vec_dot>
      <<<grid, 256, 0, stream>>>(
          cold_weight.data_ptr(), hot_weight.data_ptr(),
          cache_weight == nullptr ? nullptr : cache_weight->data_ptr(),
          hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
          cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
          reinterpret_cast<const block_q8_1*>(quantized.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()), expert_ids.data_ptr<int>(),
          hot_map.numel(), hot_weight.size(0), cold_weight.size(0),
          cache_weight == nullptr ? 0 : cache_weight->size(0), top_k, ncols,
          nrows, quantized.stride(0), output_groups);
}

template <int unroll, int blocks_per_row, int qk, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot>
__device__ __forceinline__ float exact_shape_dot(
    const block_q_t* __restrict__ weight,
    const block_q8_1* __restrict__ activation, int lane_in_row) {
  float sum = 0.0f;
  const int quant_index = vdr * lane_in_row;
  if constexpr (unroll == 2) {
#pragma unroll 2
    for (int block = 0; block < blocks_per_row; ++block) {
      sum += vec_dot(&weight[block],
                     &activation[block * (qk / QK8_1)], quant_index);
    }
  } else if constexpr (unroll == 5) {
#pragma unroll 5
    for (int block = 0; block < blocks_per_row; ++block) {
      sum += vec_dot(&weight[block],
                     &activation[block * (qk / QK8_1)], quant_index);
    }
  } else if constexpr (unroll == 10) {
#pragma unroll 10
    for (int block = 0; block < blocks_per_row; ++block) {
      sum += vec_dot(&weight[block],
                     &activation[block * (qk / QK8_1)], quant_index);
    }
  } else {
    static_assert(unroll == 1, "unsupported exact-shape unroll variant");
    for (int block = 0; block < blocks_per_row; ++block) {
      sum += vec_dot(&weight[block],
                     &activation[block * (qk / QK8_1)], quant_index);
    }
  }
  return sum;
}

// Qwen3.8 decode uses fixed expert matrix shapes.  The y dimension owns one
// routed occurrence, so every block resolves one expert/source/activation and
// broadcasts those uniform pointers to all row workers.  This avoids the
// flattened kernel's per-row division and repeated hot/cold map resolution.
template <int unroll, int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
__global__ __launch_bounds__(256) void tiered_moe_vec_exact_2d(
    const void* __restrict__ cold_weight,
    const void* __restrict__ hot_weight,
    const void* __restrict__ cache_weight,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ cache_map,
    const block_q8_1* __restrict__ input, bf16* __restrict__ output,
    const int* __restrict__ expert_ids, int num_experts, int hot_count,
    int cold_count, int cache_count, int top_k, int token_stride) {
  static_assert(ncols % qk == 0, "exact shape must contain whole quant blocks");
  static_assert(kWave % lanes_per_row == 0,
                "row subgroups must divide Wave32");
  constexpr int blocks_per_row = ncols / qk;
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  constexpr int64_t expert_stride =
      static_cast<int64_t>(nrows) * blocks_per_row;

  __shared__ const block_q_t* selected_weight;
  __shared__ const block_q8_1* selected_activation;
  __shared__ int selected_valid;

  const int group = blockIdx.y;
  if (threadIdx.x == 0) {
    selected_weight = nullptr;
    selected_activation = nullptr;
    selected_valid = 0;
    const int expert = expert_ids[group];
    if (expert >= 0 && expert < num_experts) {
      const int hot_slot = hot_map[expert];
      const bool use_hot = hot_slot >= 0 && hot_slot < hot_count;
      const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
      const bool use_cache = cache_slot >= 0 && cache_slot < cache_count;
      const int cold_slot = cold_map[expert];
      const bool use_cold = cold_slot >= 0 && cold_slot < cold_count;
      if (use_hot || use_cache || use_cold) {
        selected_weight =
            (use_hot ? static_cast<const block_q_t*>(hot_weight) +
                           hot_slot * expert_stride
                     : use_cache
                           ? static_cast<const block_q_t*>(cache_weight) +
                                 cache_slot * expert_stride
                           : static_cast<const block_q_t*>(cold_weight) +
                                 cold_slot * expert_stride);
        const int token = group / top_k;
        selected_activation = reinterpret_cast<const block_q8_1*>(
            reinterpret_cast<const int*>(input) + token * token_stride);
        selected_valid = 1;
      }
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int row_in_wave = lane / lanes_per_row;
  const int lane_in_row = lane % lanes_per_row;
  const int row = blockIdx.x * rows_per_block + wave * rows_per_wave +
                  row_in_wave;
  if (row >= nrows) return;

  float sum = 0.0f;
  if (selected_valid) {
    const block_q_t* weight =
        selected_weight + static_cast<int64_t>(row) * blocks_per_row;
    sum = exact_shape_dot<unroll, blocks_per_row, qk, block_q_t, vdr,
                          vec_dot>(weight, selected_activation, lane_in_row);
  }

#pragma unroll
  for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset, lanes_per_row);
  }
  if (lane_in_row == 0) {
    output[static_cast<int64_t>(group) * nrows + row] =
        __float2bfloat16(sum);
  }
}

template <int unroll, int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
void launch_tiered_exact_2d(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, int tokens,
    hipStream_t stream) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const int output_groups = tokens * top_k;
  const dim3 grid((nrows + rows_per_block - 1) / rows_per_block,
                  output_groups, 1);
  tiered_moe_vec_exact_2d<unroll, lanes_per_row, qk, ncols, nrows,
                           block_q_t, vdr, vec_dot>
      <<<grid, 256, 0, stream>>>(
          cold_weight.data_ptr(), hot_weight.data_ptr(),
          cache_weight == nullptr ? nullptr : cache_weight->data_ptr(),
          hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
          cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
          reinterpret_cast<const block_q8_1*>(quantized.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()), expert_ids.data_ptr<int>(),
          hot_map.numel(), hot_weight.size(0), cold_weight.size(0),
          cache_weight == nullptr ? 0 : cache_weight->size(0), top_k,
          quantized.stride(0));
}

template <int max_matches, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot>
__device__ __forceinline__ void accumulate_reused_block(
    const block_q_t* __restrict__ weight,
    const block_q8_1* const* __restrict__ activations, int match_count,
    int lane_in_row, float* __restrict__ sums) {
  if constexpr (std::is_same_v<block_q_t, block_iq4_nl>) {
    static_assert(vdr == VDR_Q4_0_Q8_1_MMVQ);
    const int quant_index = vdr * lane_in_row;
    const uint16_t* q4 = reinterpret_cast<const uint16_t*>(weight->qs) +
                         2 * quant_index;
    const uint8_t* values = reinterpret_cast<const uint8_t*>(kvalues_iq4nl);
    int weight_low[VDR_Q4_0_Q8_1_MMVQ];
    int weight_high[VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
    for (int item = 0; item < VDR_Q4_0_Q8_1_MMVQ; ++item) {
      const uint32_t packed = q4[2 * item] | (q4[2 * item + 1] << 16);
      get_int_from_table_16(packed, values, weight_low[item],
                            weight_high[item]);
    }
    const float weight_scale = __half2float(weight->d);
#pragma unroll
    for (int match = 0; match < max_matches; ++match) {
      if (match < match_count) {
        const block_q8_1* activation = activations[match];
        const int32_t* q8 =
            reinterpret_cast<const int32_t*>(activation->qs) + quant_index;
        int sum_low = 0;
        int sum_high = 0;
#pragma unroll
        for (int item = 0; item < VDR_Q4_0_Q8_1_MMVQ; ++item) {
          sum_low = __dp4a(weight_low[item], q8[item], sum_low);
          sum_high = __dp4a(weight_high[item], q8[item + 4], sum_high);
        }
        const float scale =
            weight_scale * __low2float(activation->ds);
        sums[match] += scale * (sum_low + sum_high);
      }
    }
  } else if constexpr (std::is_same_v<block_q_t, block_iq4_xs>) {
    const int ib32 = lane_in_row;
    const uint8_t* values = reinterpret_cast<const uint8_t*>(kvalues_iq4nl);
    const uint32_t* q4 = reinterpret_cast<const uint32_t*>(weight->qs) +
                         4 * ib32;
    int weight_low[4];
    int weight_high[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      get_int_from_table_16(q4[item], values, weight_low[item],
                            weight_high[item]);
    }
    const int8_t local_scale =
        ((weight->scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf) |
        (((weight->scales_h >> (2 * ib32)) & 3) << 4);
    const float weight_scale =
        __half2float(weight->d) * (local_scale - 32);
#pragma unroll
    for (int match = 0; match < max_matches; ++match) {
      if (match < match_count) {
        const block_q8_1* activation = activations[match] + ib32;
        const int32_t* q8 = reinterpret_cast<const int32_t*>(activation->qs);
        int sum_low = 0;
        int sum_high = 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          sum_low = __dp4a(weight_low[item], q8[item], sum_low);
          sum_high = __dp4a(weight_high[item], q8[item + 4], sum_high);
        }
        const float scale =
            weight_scale * __low2float(activation->ds);
        sums[match] += scale * (sum_low + sum_high);
      }
    }
  } else if constexpr (std::is_same_v<block_q_t, block_iq3_s>) {
    const int ib32 = lane_in_row;
    const uint8_t* qs = weight->qs + 8 * ib32;
    int weight_low[4];
    int weight_high[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const uint32_t* grid_low =
          iq3xs_grid +
          (qs[2 * item] |
           ((weight->qh[ib32] << (8 - 2 * item)) & 256));
      const uint32_t* grid_high =
          iq3xs_grid +
          (qs[2 * item + 1] |
           ((weight->qh[ib32] << (7 - 2 * item)) & 256));
      const uint32_t signs_low = __vcmpeq4(
          ((weight->signs[4 * ib32 + item] & 0xf) * 0x01010101) &
              0x08040201,
          0x08040201);
      const uint32_t signs_high = __vcmpeq4(
          ((weight->signs[4 * ib32 + item] >> 4) * 0x01010101) &
              0x08040201,
          0x08040201);
      weight_low[item] = __vsub4(grid_low[0] ^ signs_low, signs_low);
      weight_high[item] = __vsub4(grid_high[0] ^ signs_high, signs_high);
    }
    const float weight_scale =
        __half2float(weight->d) *
        (0.5f +
         ((weight->scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf));
#pragma unroll
    for (int match = 0; match < max_matches; ++match) {
      if (match < match_count) {
        const block_q8_1* activation = activations[match] + ib32;
        const int32_t* q8 = reinterpret_cast<const int32_t*>(activation->qs);
        int integer_sum = 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          integer_sum = __dp4a(weight_low[item], q8[2 * item], integer_sum);
          integer_sum =
              __dp4a(weight_high[item], q8[2 * item + 1], integer_sum);
        }
        const float scale =
            weight_scale * __low2float(activation->ds) * 0.5f;
        sums[match] += scale * integer_sum;
      }
    }
  } else if constexpr (std::is_same_v<block_q_t, block_q8_0>) {
    const int quant_index = vdr * lane_in_row;
    int decoded_weight[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int item = 0; item < VDR_Q8_0_Q8_1_MMVQ; ++item) {
      decoded_weight[item] = get_int_from_int8(weight->qs, quant_index + item);
    }
    const float weight_scale = __half2float(weight->d);
#pragma unroll
    for (int match = 0; match < max_matches; ++match) {
      if (match < match_count) {
        int activation_values[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
        for (int item = 0; item < VDR_Q8_0_Q8_1_MMVQ; ++item) {
          activation_values[item] = get_int_from_int8_aligned(
              activations[match]->qs, quant_index + item);
        }
        sums[match] +=
            vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
                decoded_weight, activation_values, weight_scale,
                __low2float(activations[match]->ds));
      }
    }
  } else {
#pragma unroll
    for (int match = 0; match < max_matches; ++match) {
      if (match < match_count) {
        sums[match] +=
            vec_dot(weight, activations[match], vdr * lane_in_row);
      }
    }
  }
}

// Prompt processing has thousands of routed occurrences. vLLM's alignment
// kernel supplies stable expert-grouped blocks of four route indices. One
// workgroup owns one aligned expert block and one output-row tile, decodes
// each packed weight value once, and applies it to four Q8_1 activations.
// This preserves every route's original accumulation order while reducing
// cold UVA weight reads by up to 4x. Padding route ids are skipped and every
// valid result is stored at its original flattened route position.
template <int group_size, int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
__global__ __launch_bounds__(256) void tiered_moe_prefill_grouped4(
    const void* __restrict__ cold_weight,
    const void* __restrict__ hot_weight,
    const void* __restrict__ cache_weight,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ cache_map,
    const block_q8_1* __restrict__ input, bf16* __restrict__ output,
    const int* __restrict__ sorted_route_ids,
    const int* __restrict__ block_expert_ids,
    const int* __restrict__ num_routes_post_padded, int num_experts,
    int hot_count, int cold_count, int cache_count, int top_k,
    int token_stride, int output_groups, int sorted_route_capacity) {
  static_assert(ncols % qk == 0);
  constexpr int blocks_per_row = ncols / qk;
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  constexpr int64_t expert_stride =
      static_cast<int64_t>(nrows) * blocks_per_row;

  __shared__ const block_q_t* selected_weight;
  __shared__ int selected_routes[group_size];
  __shared__ int selected_count;
  __shared__ int selected_valid;

  const int grouped_block = blockIdx.y;
  const int route_base = grouped_block * group_size;
  if (route_base >= num_routes_post_padded[0] ||
      route_base >= sorted_route_capacity) {
    return;
  }
  if (threadIdx.x == 0) {
    selected_weight = nullptr;
    selected_count = 0;
    selected_valid = 0;
    const int expert = block_expert_ids[grouped_block];
#pragma unroll
    for (int match = 0; match < group_size; ++match) {
      const int route_slot = route_base + match;
      const int route = route_slot < sorted_route_capacity
                            ? sorted_route_ids[route_slot]
                            : -1;
      if (route >= 0 && route < output_groups) {
        selected_routes[selected_count++] = route;
      }
    }
#pragma unroll
    for (int match = 0; match < group_size; ++match) {
      if (match >= selected_count) selected_routes[match] = -1;
    }
    if (expert >= 0 && expert < num_experts && selected_count > 0) {
      const int hot_slot = hot_map[expert];
      const bool use_hot = hot_slot >= 0 && hot_slot < hot_count;
      const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
      const bool use_cache = cache_slot >= 0 && cache_slot < cache_count;
      const int cold_slot = cold_map[expert];
      const bool use_cold = cold_slot >= 0 && cold_slot < cold_count;
      if (use_hot || use_cache || use_cold) {
        selected_weight =
            use_hot
                ? static_cast<const block_q_t*>(hot_weight) +
                      hot_slot * expert_stride
                : use_cache
                      ? static_cast<const block_q_t*>(cache_weight) +
                            cache_slot * expert_stride
                      : static_cast<const block_q_t*>(cold_weight) +
                            cold_slot * expert_stride;
        selected_valid = 1;
      }
    }
  }
  __syncthreads();
  if (selected_count == 0) return;

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int row_in_wave = lane / lanes_per_row;
  const int lane_in_row = lane % lanes_per_row;
  const int row = blockIdx.x * rows_per_block + wave * rows_per_wave +
                  row_in_wave;
  if (row >= nrows) return;

  float sums[group_size] = {};
  const block_q8_1* activations[group_size] = {};
#pragma unroll
  for (int match = 0; match < group_size; ++match) {
    const int route = selected_routes[match];
    if (route >= 0 && route < output_groups) {
      const int token = route / top_k;
      activations[match] = reinterpret_cast<const block_q8_1*>(
          reinterpret_cast<const int*>(input) + token * token_stride);
    }
  }

  if (selected_valid) {
    const block_q_t* weight =
        selected_weight + static_cast<int64_t>(row) * blocks_per_row;
#pragma unroll
    for (int block = 0; block < blocks_per_row; ++block) {
      const block_q8_1* block_activations[group_size];
#pragma unroll
      for (int match = 0; match < group_size; ++match) {
        block_activations[match] =
            activations[match] == nullptr
                ? nullptr
                : activations[match] + block * (qk / QK8_1);
      }
      accumulate_reused_block<group_size, block_q_t, vdr, vec_dot>(
          &weight[block], block_activations, selected_count, lane_in_row, sums);
    }
  }

#pragma unroll
  for (int match = 0; match < group_size; ++match) {
    const int route = selected_routes[match];
    if (route >= 0 && route < output_groups) {
      float sum = sums[match];
#pragma unroll
      for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, lanes_per_row);
      }
      if (lane_in_row == 0) {
        output[static_cast<int64_t>(route) * nrows + row] =
            __float2bfloat16(sum);
      }
    }
  }
}

template <int group_size, int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
void launch_tiered_prefill_grouped(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& sorted_route_ids,
    const torch::Tensor& block_expert_ids,
    const torch::Tensor& num_routes_post_padded, int top_k, int tokens,
    hipStream_t stream) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const int output_groups = tokens * top_k;
  const dim3 grid((nrows + rows_per_block - 1) / rows_per_block,
                  block_expert_ids.numel(), 1);
  tiered_moe_prefill_grouped4<group_size, lanes_per_row, qk, ncols, nrows,
                              block_q_t, vdr, vec_dot>
      <<<grid, 256, 0, stream>>>(
          cold_weight.data_ptr(), hot_weight.data_ptr(),
          cache_weight == nullptr ? nullptr : cache_weight->data_ptr(),
          hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
          cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
          reinterpret_cast<const block_q8_1*>(quantized.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()),
          sorted_route_ids.data_ptr<int>(), block_expert_ids.data_ptr<int>(),
          num_routes_post_padded.data_ptr<int>(), hot_map.numel(),
          hot_weight.size(0), cold_weight.size(0),
          cache_weight == nullptr ? 0 : cache_weight->size(0), top_k,
          quantized.stride(0), output_groups, sorted_route_ids.numel());
}

template <int group_size>
bool try_launch_tiered_prefill_grouped(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& sorted_route_ids,
    const torch::Tensor& block_expert_ids,
    const torch::Tensor& num_routes_post_padded, int top_k, int qtype,
    int ncols, int nrows, int tokens, hipStream_t stream) {
  if (ncols == 2560 && nrows == 640) {
    if (qtype == 21) {
      launch_tiered_prefill_grouped<group_size, 8, QK_K, 2560, 640,
                                    block_iq3_s, 1,
                                    vec_dot_iq3_s_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
    if (qtype == 23) {
      launch_tiered_prefill_grouped<group_size, 8, QK_K, 2560, 640,
                                    block_iq4_xs, 1,
                                    vec_dot_iq4_xs_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
  }
  if (nrows == 2560 && (ncols == 320 || ncols == 640)) {
    if (qtype == 20 && ncols == 320) {
      launch_tiered_prefill_grouped<group_size, 2, QK4_NL, 320, 2560,
                                    block_iq4_nl,
                                    VDR_Q4_0_Q8_1_MMVQ,
                                    vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
    if (qtype == 20 && ncols == 640) {
      launch_tiered_prefill_grouped<group_size, 2, QK4_NL, 640, 2560,
                                    block_iq4_nl,
                                    VDR_Q4_0_Q8_1_MMVQ,
                                    vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
    if (qtype == 8 && ncols == 320) {
      launch_tiered_prefill_grouped<group_size, 4, QK8_0, 320, 2560,
                                    block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                    vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
    if (qtype == 8 && ncols == 640) {
      launch_tiered_prefill_grouped<group_size, 4, QK8_0, 640, 2560,
                                    block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                    vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, sorted_route_ids, block_expert_ids,
          num_routes_post_padded, top_k, tokens, stream);
      return true;
    }
  }
  return false;
}

// MTP2 verification routes exactly 30 expert occurrences (three target
// tokens by top-10).  One block still maps to each route so the launch stays
// graph-stable, but duplicate blocks return before touching weights.  The
// first occurrence computes all matching outputs while a packed weight block
// is resident in LDS and its lane-owned decode is held in registers.  Normal
// top-k routing can match an expert at most once per token, hence three live
// accumulators.  Malformed routing with more matches falls back in-kernel to
// one occurrence per block instead of changing correctness or launch shape.
template <int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
__global__ __launch_bounds__(256) void tiered_moe_vec_exact_reuse3(
    const void* __restrict__ cold_weight,
    const void* __restrict__ hot_weight,
    const void* __restrict__ cache_weight,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ cache_map,
    const block_q8_1* __restrict__ input, bf16* __restrict__ output,
    const int* __restrict__ expert_ids, int num_experts, int hot_count,
    int cold_count, int cache_count, int top_k, int token_stride) {
  static_assert(ncols % qk == 0);
  constexpr int blocks_per_row = ncols / qk;
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  constexpr int64_t expert_stride =
      static_cast<int64_t>(nrows) * blocks_per_row;

  __shared__ const block_q_t* selected_weight;
  __shared__ int selected_groups[kReuseMatches];
  __shared__ int selected_count;
  __shared__ int selected_valid;
  __shared__ block_q_t weight_tiles[rows_per_block];

  const int group = blockIdx.y;
  if (threadIdx.x == 0) {
    selected_weight = nullptr;
    selected_count = 0;
    selected_valid = 0;
    const int expert = expert_ids[group];
    if (expert < 0 || expert >= num_experts) {
      selected_groups[0] = group;
      selected_count = 1;
    } else {
      int first_group = kReuseRoutes;
      int occurrences = 0;
      int matches[kReuseMatches] = {-1, -1, -1};
#pragma unroll
      for (int route = 0; route < kReuseRoutes; ++route) {
        if (expert_ids[route] == expert) {
          if (occurrences < kReuseMatches) matches[occurrences] = route;
          if (route < first_group) first_group = route;
          ++occurrences;
        }
      }
      if (occurrences > kReuseMatches) {
        selected_groups[0] = group;
        selected_count = 1;
      } else if (group == first_group) {
        selected_count = occurrences;
#pragma unroll
        for (int match = 0; match < kReuseMatches; ++match) {
          selected_groups[match] = matches[match];
        }
      }

      if (selected_count > 0) {
        const int hot_slot = hot_map[expert];
        const bool use_hot = hot_slot >= 0 && hot_slot < hot_count;
        const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
        const bool use_cache = cache_slot >= 0 && cache_slot < cache_count;
        const int cold_slot = cold_map[expert];
        const bool use_cold = cold_slot >= 0 && cold_slot < cold_count;
        if (use_hot || use_cache || use_cold) {
          selected_weight =
              use_hot
                  ? static_cast<const block_q_t*>(hot_weight) +
                        hot_slot * expert_stride
                  : use_cache
                        ? static_cast<const block_q_t*>(cache_weight) +
                              cache_slot * expert_stride
                        : static_cast<const block_q_t*>(cold_weight) +
                              cold_slot * expert_stride;
          selected_valid = 1;
        }
      }
    }
  }
  __syncthreads();
  if (selected_count == 0) return;

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int row_in_wave = lane / lanes_per_row;
  const int lane_in_row = lane % lanes_per_row;
  const int row_in_block = wave * rows_per_wave + row_in_wave;
  const int row = blockIdx.x * rows_per_block + row_in_block;
  float sums[kReuseMatches] = {0.0f, 0.0f, 0.0f};

  const block_q8_1* activations[kReuseMatches] = {nullptr, nullptr, nullptr};
#pragma unroll
  for (int match = 0; match < kReuseMatches; ++match) {
    if (match < selected_count) {
      const int token = selected_groups[match] / top_k;
      activations[match] = reinterpret_cast<const block_q8_1*>(
          reinterpret_cast<const int*>(input) + token * token_stride);
    }
  }

  if (selected_valid) {
#pragma unroll
    for (int block = 0; block < blocks_per_row; ++block) {
      if (row < nrows) {
        const uint16_t* source = reinterpret_cast<const uint16_t*>(
            selected_weight + static_cast<int64_t>(row) * blocks_per_row +
            block);
        uint16_t* target =
            reinterpret_cast<uint16_t*>(&weight_tiles[row_in_block]);
#pragma unroll
        for (int item = lane_in_row; item < sizeof(block_q_t) / 2;
             item += lanes_per_row) {
          target[item] = source[item];
        }
      }
      __syncthreads();
      if (row < nrows) {
        const block_q8_1* block_activations[kReuseMatches];
#pragma unroll
        for (int match = 0; match < kReuseMatches; ++match) {
          block_activations[match] =
              match < selected_count
                  ? activations[match] + block * (qk / QK8_1)
                  : nullptr;
        }
        accumulate_reused_block<kReuseMatches, block_q_t, vdr, vec_dot>(
            &weight_tiles[row_in_block], block_activations, selected_count,
            lane_in_row, sums);
      }
      __syncthreads();
    }
  }

  if (row >= nrows) return;
#pragma unroll
  for (int match = 0; match < kReuseMatches; ++match) {
    if (match < selected_count) {
      float sum = sums[match];
#pragma unroll
      for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, lanes_per_row);
      }
      if (lane_in_row == 0) {
        output[static_cast<int64_t>(selected_groups[match]) * nrows + row] =
            __float2bfloat16(sum);
      }
    }
  }
}

template <int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
void launch_tiered_reuse3(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, hipStream_t stream) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const dim3 grid((nrows + rows_per_block - 1) / rows_per_block,
                  kReuseRoutes, 1);
  tiered_moe_vec_exact_reuse3<lanes_per_row, qk, ncols, nrows, block_q_t,
                               vdr, vec_dot>
      <<<grid, 256, 0, stream>>>(
          cold_weight.data_ptr(), hot_weight.data_ptr(),
          cache_weight == nullptr ? nullptr : cache_weight->data_ptr(),
          hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
          cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
          reinterpret_cast<const block_q8_1*>(quantized.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()), expert_ids.data_ptr<int>(),
          hot_map.numel(), hot_weight.size(0), cold_weight.size(0),
          cache_weight == nullptr ? 0 : cache_weight->size(0), top_k,
          quantized.stride(0));
}

bool try_launch_tiered_reuse3(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, int qtype, int ncols,
    int nrows, int tokens, hipStream_t stream) {
  if (tokens * top_k != kReuseRoutes) return false;
  if (ncols == 2560 && nrows == 640) {
    if (qtype == 21) {
      launch_tiered_reuse3<8, QK_K, 2560, 640, block_iq3_s, 1,
                           vec_dot_iq3_s_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 23) {
      launch_tiered_reuse3<8, QK_K, 2560, 640, block_iq4_xs, 1,
                           vec_dot_iq4_xs_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
  }
  if (ncols == 320 && nrows == 2560) {
    if (qtype == 20) {
      launch_tiered_reuse3<2, QK4_NL, 320, 2560, block_iq4_nl,
                           VDR_Q4_0_Q8_1_MMVQ,
                           vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 8) {
      launch_tiered_reuse3<4, QK8_0, 320, 2560, block_q8_0,
                           VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
  }
  return false;
}

// The first reuse kernel stages every packed quant block through LDS and
// brackets every K iteration with two workgroup barriers.  That makes the
// weight bytes reusable, but the synchronization cost is larger than the
// saved VRAM traffic for resident experts.  reuse3v2 retains the identical
// fixed 30-route ownership plan while letting each lane decode its portion of
// the packed block directly from hot/cache/cold storage.  The decoded values
// in accumulate_reused_block stay live in registers across all matching
// activations.  Only the one control barrier which publishes route ownership
// and the selected source pointer remains; there is no packed-weight LDS and
// no barrier in the K loop.
template <int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
__global__ __launch_bounds__(256) void tiered_moe_vec_exact_reuse3v2(
    const void* __restrict__ cold_weight,
    const void* __restrict__ hot_weight,
    const void* __restrict__ cache_weight,
    const int* __restrict__ hot_map,
    const int* __restrict__ cold_map,
    const int* __restrict__ cache_map,
    const block_q8_1* __restrict__ input, bf16* __restrict__ output,
    const int* __restrict__ expert_ids, int num_experts, int hot_count,
    int cold_count, int cache_count, int top_k, int token_stride) {
  static_assert(ncols % qk == 0);
  constexpr int blocks_per_row = ncols / qk;
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  constexpr int64_t expert_stride =
      static_cast<int64_t>(nrows) * blocks_per_row;

  __shared__ const block_q_t* selected_weight;
  __shared__ int selected_groups[kReuseMatches];
  __shared__ int selected_count;
  __shared__ int selected_valid;

  const int group = blockIdx.y;
  if (threadIdx.x == 0) {
    selected_weight = nullptr;
    selected_count = 0;
    selected_valid = 0;
    const int expert = expert_ids[group];
    if (expert < 0 || expert >= num_experts) {
      selected_groups[0] = group;
      selected_count = 1;
    } else {
      int first_group = kReuseRoutes;
      int occurrences = 0;
      int matches[kReuseMatches] = {-1, -1, -1};
#pragma unroll
      for (int route = 0; route < kReuseRoutes; ++route) {
        if (expert_ids[route] == expert) {
          if (occurrences < kReuseMatches) matches[occurrences] = route;
          if (route < first_group) first_group = route;
          ++occurrences;
        }
      }
      if (occurrences > kReuseMatches) {
        selected_groups[0] = group;
        selected_count = 1;
      } else if (group == first_group) {
        selected_count = occurrences;
#pragma unroll
        for (int match = 0; match < kReuseMatches; ++match) {
          selected_groups[match] = matches[match];
        }
      }

      if (selected_count > 0) {
        const int hot_slot = hot_map[expert];
        const bool use_hot = hot_slot >= 0 && hot_slot < hot_count;
        const int cache_slot = cache_count > 0 ? cache_map[expert] : -1;
        const bool use_cache = cache_slot >= 0 && cache_slot < cache_count;
        const int cold_slot = cold_map[expert];
        const bool use_cold = cold_slot >= 0 && cold_slot < cold_count;
        if (use_hot || use_cache || use_cold) {
          selected_weight =
              use_hot
                  ? static_cast<const block_q_t*>(hot_weight) +
                        hot_slot * expert_stride
                  : use_cache
                        ? static_cast<const block_q_t*>(cache_weight) +
                              cache_slot * expert_stride
                        : static_cast<const block_q_t*>(cold_weight) +
                              cold_slot * expert_stride;
          selected_valid = 1;
        }
      }
    }
  }
  __syncthreads();
  if (selected_count == 0) return;

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int row_in_wave = lane / lanes_per_row;
  const int lane_in_row = lane % lanes_per_row;
  const int row = blockIdx.x * rows_per_block + wave * rows_per_wave +
                  row_in_wave;
  float sums[kReuseMatches] = {0.0f, 0.0f, 0.0f};

  const block_q8_1* activations[kReuseMatches] = {nullptr, nullptr, nullptr};
#pragma unroll
  for (int match = 0; match < kReuseMatches; ++match) {
    if (match < selected_count) {
      const int token = selected_groups[match] / top_k;
      activations[match] = reinterpret_cast<const block_q8_1*>(
          reinterpret_cast<const int*>(input) + token * token_stride);
    }
  }

  if (selected_valid && row < nrows) {
    const block_q_t* row_weight =
        selected_weight + static_cast<int64_t>(row) * blocks_per_row;
#pragma unroll
    for (int block = 0; block < blocks_per_row; ++block) {
      const block_q8_1* block_activations[kReuseMatches];
#pragma unroll
      for (int match = 0; match < kReuseMatches; ++match) {
        block_activations[match] =
            match < selected_count
                ? activations[match] + block * (qk / QK8_1)
                : nullptr;
      }
      accumulate_reused_block<kReuseMatches, block_q_t, vdr, vec_dot>(
          &row_weight[block], block_activations, selected_count, lane_in_row,
          sums);
    }
  }

  if (row >= nrows) return;
#pragma unroll
  for (int match = 0; match < kReuseMatches; ++match) {
    if (match < selected_count) {
      float sum = sums[match];
#pragma unroll
      for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, lanes_per_row);
      }
      if (lane_in_row == 0) {
        output[static_cast<int64_t>(selected_groups[match]) * nrows + row] =
            __float2bfloat16(sum);
      }
    }
  }
}

template <int lanes_per_row, int qk, int ncols, int nrows,
          typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
void launch_tiered_reuse3v2(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, hipStream_t stream) {
  constexpr int rows_per_wave = kWave / lanes_per_row;
  constexpr int rows_per_block = kWavesPerBlock * rows_per_wave;
  const dim3 grid((nrows + rows_per_block - 1) / rows_per_block,
                  kReuseRoutes, 1);
  tiered_moe_vec_exact_reuse3v2<lanes_per_row, qk, ncols, nrows, block_q_t,
                                 vdr, vec_dot>
      <<<grid, 256, 0, stream>>>(
          cold_weight.data_ptr(), hot_weight.data_ptr(),
          cache_weight == nullptr ? nullptr : cache_weight->data_ptr(),
          hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
          cache_map == nullptr ? nullptr : cache_map->data_ptr<int>(),
          reinterpret_cast<const block_q8_1*>(quantized.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()), expert_ids.data_ptr<int>(),
          hot_map.numel(), hot_weight.size(0), cold_weight.size(0),
          cache_weight == nullptr ? 0 : cache_weight->size(0), top_k,
          quantized.stride(0));
}

bool try_launch_tiered_reuse3v2(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, int qtype, int ncols,
    int nrows, int tokens, hipStream_t stream) {
  if (tokens * top_k != kReuseRoutes) return false;
  if (ncols == 2560 && nrows == 640) {
    if (qtype == 21) {
      launch_tiered_reuse3v2<8, QK_K, 2560, 640, block_iq3_s, 1,
                             vec_dot_iq3_s_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 23) {
      launch_tiered_reuse3v2<8, QK_K, 2560, 640, block_iq4_xs, 1,
                             vec_dot_iq4_xs_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
  }
  if (nrows == 2560 && (ncols == 320 || ncols == 640)) {
    if (qtype == 20 && ncols == 320) {
      launch_tiered_reuse3v2<2, QK4_NL, 320, 2560, block_iq4_nl,
                             VDR_Q4_0_Q8_1_MMVQ,
                             vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 20 && ncols == 640) {
      launch_tiered_reuse3v2<2, QK4_NL, 640, 2560, block_iq4_nl,
                             VDR_Q4_0_Q8_1_MMVQ,
                             vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 8 && ncols == 320) {
      launch_tiered_reuse3v2<4, QK8_0, 320, 2560, block_q8_0,
                             VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
    if (qtype == 8 && ncols == 640) {
      launch_tiered_reuse3v2<4, QK8_0, 640, 2560, block_q8_0,
                             VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, stream);
      return true;
    }
  }
  return false;
}

template <int unroll>
bool try_launch_tiered_exact_q4(
    const torch::Tensor& cold_weight, const torch::Tensor& hot_weight,
    const torch::Tensor& hot_map, const torch::Tensor& cold_map,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map,
    const torch::Tensor& quantized, torch::Tensor& output,
    const torch::Tensor& expert_ids, int top_k, int qtype, int ncols,
    int nrows, int tokens, hipStream_t stream) {
  const int64_t output_groups = static_cast<int64_t>(tokens) * top_k;
  // HIP's portable grid-y limit is 65535.  Larger/pre-fill shapes retain the
  // generic flattened path rather than changing its launch semantics.
  if (output_groups > 65535) return false;

  if (ncols == 2560 && nrows == 640) {
    if (qtype == 21) {
      launch_tiered_exact_2d<unroll, 8, QK_K, 2560, 640, block_iq3_s, 1,
                             vec_dot_iq3_s_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
    if (qtype == 23) {
      launch_tiered_exact_2d<unroll, 8, QK_K, 2560, 640, block_iq4_xs, 1,
                             vec_dot_iq4_xs_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
  }

  // TP decode uses K=320 for W2.  The bounded GGUF parity fixture uses the
  // corresponding unsharded K=640 tensor, so both exact shapes are emitted.
  if (nrows == 2560 && (ncols == 320 || ncols == 640)) {
    if (qtype == 20 && ncols == 320) {
      launch_tiered_exact_2d<unroll, 2, QK4_NL, 320, 2560, block_iq4_nl,
                             VDR_Q4_0_Q8_1_MMVQ,
                             vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
    if (qtype == 20 && ncols == 640) {
      launch_tiered_exact_2d<unroll, 2, QK4_NL, 640, 2560, block_iq4_nl,
                             VDR_Q4_0_Q8_1_MMVQ,
                             vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
    if (qtype == 8 && ncols == 320) {
      launch_tiered_exact_2d<unroll, 4, QK8_0, 320, 2560, block_q8_0,
                             VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
    if (qtype == 8 && ncols == 640) {
      launch_tiered_exact_2d<unroll, 4, QK8_0, 640, 2560, block_q8_0,
                             VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, tokens, stream);
      return true;
    }
  }
  return false;
}

torch::Tensor tiered_iq_moe_prefill_grouped_impl(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor sorted_route_ids, torch::Tensor block_expert_ids,
    torch::Tensor num_routes_post_padded, int64_t top_k, int64_t qtype,
    int64_t rows, int64_t tokens, int64_t group_size,
    const torch::Tensor* cache_weight, const torch::Tensor* cache_map) {
  TORCH_CHECK(x.is_cuda() && cold_weight.is_cuda() && hot_weight.is_cuda() &&
                  hot_map.is_cuda() && cold_map.is_cuda() &&
                  sorted_route_ids.is_cuda() && block_expert_ids.is_cuda() &&
                  num_routes_post_padded.is_cuda(),
              "all grouped-prefill tensors must be GPU or UVA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(cold_weight.scalar_type() == torch::kUInt8 &&
                  hot_weight.scalar_type() == torch::kUInt8,
              "weights must contain packed GGUF bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  sorted_route_ids.scalar_type() == torch::kInt32 &&
                  block_expert_ids.scalar_type() == torch::kInt32 &&
                  num_routes_post_padded.scalar_type() == torch::kInt32,
              "grouped-prefill routing tensors must be int32");
  TORCH_CHECK(x.is_contiguous() && cold_weight.is_contiguous() &&
                  hot_weight.is_contiguous() && hot_map.is_contiguous() &&
                  cold_map.is_contiguous() && sorted_route_ids.is_contiguous() &&
                  block_expert_ids.is_contiguous() &&
                  num_routes_post_padded.is_contiguous(),
              "all grouped-prefill tensors must be contiguous");
  TORCH_CHECK(qtype == 8 || qtype == 20 || qtype == 21 || qtype == 23,
              "only Q8_0, IQ4_NL, IQ3_S, and IQ4_XS are supported");
  TORCH_CHECK(group_size == 4 || group_size == 8 || group_size == 16,
              "grouped prefill size must be 4, 8, or 16");
  TORCH_CHECK(top_k > 0 && tokens > 0 && rows > 0,
              "grouped-prefill dimensions must be positive");
  TORCH_CHECK(cold_weight.dim() == 3 && hot_weight.dim() == 3,
              "expert weights must be three-dimensional");
  TORCH_CHECK(cold_weight.size(1) == rows && hot_weight.size(1) == rows,
              "rows does not match packed expert weights");
  TORCH_CHECK(cold_weight.size(2) == hot_weight.size(2),
              "cold and hot expert byte strides differ");
  TORCH_CHECK(hot_map.numel() == cold_map.numel(),
              "hot_map and cold_map must cover the same logical experts");
  TORCH_CHECK(x.size(0) >= tokens, "input does not cover every token");
  TORCH_CHECK(num_routes_post_padded.numel() >= 1,
              "num_routes_post_padded must contain one device scalar");
  TORCH_CHECK(block_expert_ids.numel() > 0 &&
                  block_expert_ids.numel() <= 65535,
              "grouped-prefill block count exceeds HIP grid-y capacity");
  TORCH_CHECK(sorted_route_ids.numel() > 0,
              "sorted route storage must not be empty");
  const auto device = x.device();
  TORCH_CHECK(cold_weight.device() == device && hot_weight.device() == device &&
                  hot_map.device() == device && cold_map.device() == device &&
                  sorted_route_ids.device() == device &&
                  block_expert_ids.device() == device &&
                  num_routes_post_padded.device() == device,
              "grouped-prefill tensors must share one device");
  if (cache_weight != nullptr || cache_map != nullptr) {
    TORCH_CHECK(cache_weight != nullptr && cache_map != nullptr,
                "cache weight and map must be supplied together");
    TORCH_CHECK(cache_weight->is_cuda() && cache_map->is_cuda(),
                "cache tensors must be GPU tensors");
    TORCH_CHECK(cache_weight->scalar_type() == torch::kUInt8 &&
                    cache_map->scalar_type() == torch::kInt32,
                "cache weight must be uint8 and cache map must be int32");
    TORCH_CHECK(cache_weight->is_contiguous() && cache_map->is_contiguous(),
                "cache tensors must be contiguous");
    TORCH_CHECK(cache_weight->dim() == 3 &&
                    cache_weight->size(1) == rows &&
                    cache_weight->size(2) == cold_weight.size(2),
                "cache expert layout does not match packed weights");
    TORCH_CHECK(cache_map->numel() == hot_map.numel(),
                "cache map must cover every logical expert");
    TORCH_CHECK(cache_weight->device() == device &&
                    cache_map->device() == device,
                "cache tensors must be on the input device");
  }

  const int64_t cols = x.size(1);
  const int64_t qk = qtype == 8 || qtype == 20 ? 32 : 256;
  TORCH_CHECK(cols % qk == 0, "input width is not aligned to the GGUF block");
  const int64_t expected_bytes =
      qtype == 8 ? (cols / 32) * sizeof(block_q8_0)
                 : qtype == 20 ? (cols / 32) * sizeof(block_iq4_nl)
                               : qtype == 21
                                     ? (cols / 256) * sizeof(block_iq3_s)
                                     : (cols / 256) * sizeof(block_iq4_xs);
  TORCH_CHECK(cold_weight.size(2) == expected_bytes,
              "packed row byte width does not match qtype and input width");

  c10::cuda::CUDAGuard guard(device);
  const int64_t padded = (cols + 511) / 512 * 512;
  auto quantized = torch::empty(
      {tokens, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto output = torch::zeros({tokens * top_k, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const dim3 quant_grid((padded + 255) / 256, tokens, 1);
  quantize_q8_1<<<quant_grid, 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);

  bool launched = false;
  if (group_size == 4) {
    launched = try_launch_tiered_prefill_grouped<4>(
        cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
        quantized, output, sorted_route_ids, block_expert_ids,
        num_routes_post_padded, top_k, qtype, cols, rows, tokens, stream);
  } else if (group_size == 8) {
    launched = try_launch_tiered_prefill_grouped<8>(
        cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
        quantized, output, sorted_route_ids, block_expert_ids,
        num_routes_post_padded, top_k, qtype, cols, rows, tokens, stream);
  } else {
    launched = try_launch_tiered_prefill_grouped<16>(
        cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
        quantized, output, sorted_route_ids, block_expert_ids,
        num_routes_post_padded, top_k, qtype, cols, rows, tokens, stream);
  }
  TORCH_CHECK(launched,
              "grouped prefill does not support this qtype/shape combination");
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor tiered_iq_moe_gemv_impl(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map, torch::Tensor expert_ids,
    int64_t top_k, int64_t qtype, int64_t rows, int64_t tokens,
    int64_t specialized_variant, const torch::Tensor* cache_weight,
    const torch::Tensor* cache_map) {
  TORCH_CHECK(x.is_cuda() && cold_weight.is_cuda() && hot_weight.is_cuda() &&
                  hot_map.is_cuda() && cold_map.is_cuda() &&
                  expert_ids.is_cuda(),
              "all tensors must be GPU or UVA accelerator tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(cold_weight.scalar_type() == torch::kUInt8 &&
                  hot_weight.scalar_type() == torch::kUInt8,
              "weights must contain packed GGUF bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  expert_ids.scalar_type() == torch::kInt32,
              "expert maps and expert_ids must be int32");
  TORCH_CHECK(x.is_contiguous() && cold_weight.is_contiguous() &&
                  hot_weight.is_contiguous() && hot_map.is_contiguous() &&
                  cold_map.is_contiguous() && expert_ids.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(qtype == 8 || qtype == 20 || qtype == 21 || qtype == 23,
              "only Q8_0, IQ4_NL, IQ3_S, and IQ4_XS are supported");
  TORCH_CHECK(specialized_variant == 0 || specialized_variant == 1 ||
                  specialized_variant == 2 || specialized_variant == 5 ||
                  specialized_variant == 10 || specialized_variant == 30 ||
                  specialized_variant == 31,
              "specialized variant must be 0 (generic), 1 (auto), 2, 5, 10, "
              "30 (reuse3), or 31 (reuse3v2)");
  TORCH_CHECK(cold_weight.dim() == 3 && hot_weight.dim() == 3,
              "expert weights must be three-dimensional");
  TORCH_CHECK(cold_weight.size(1) == rows && hot_weight.size(1) == rows,
              "rows does not match packed expert weights");
  TORCH_CHECK(cold_weight.size(2) == hot_weight.size(2),
              "cold and hot expert byte strides differ");
  TORCH_CHECK(hot_map.numel() == cold_map.numel(),
              "hot_map and cold_map must cover the same logical experts");
  if (cache_weight != nullptr || cache_map != nullptr) {
    TORCH_CHECK(cache_weight != nullptr && cache_map != nullptr,
                "cache weight and map must be supplied together");
    TORCH_CHECK(cache_weight->is_cuda() && cache_map->is_cuda(),
                "cache tensors must be GPU tensors");
    TORCH_CHECK(cache_weight->scalar_type() == torch::kUInt8 &&
                    cache_map->scalar_type() == torch::kInt32,
                "cache weight must be uint8 and cache map must be int32");
    TORCH_CHECK(cache_weight->is_contiguous() && cache_map->is_contiguous(),
                "cache tensors must be contiguous");
    TORCH_CHECK(cache_weight->dim() == 3 &&
                    cache_weight->size(1) == rows &&
                    cache_weight->size(2) == cold_weight.size(2),
                "cache expert layout does not match packed weights");
    TORCH_CHECK(cache_map->numel() == hot_map.numel(),
                "cache map must cover every logical expert");
    TORCH_CHECK(cache_weight->device() == x.device() &&
                    cache_map->device() == x.device(),
                "cache tensors must be on the input device");
  }
  TORCH_CHECK(x.size(0) >= tokens, "input does not cover every token");
  TORCH_CHECK(expert_ids.numel() >= tokens * top_k,
              "expert_ids does not cover every output group");

  const int64_t cols = x.size(1);
  const int64_t qk = qtype == 8 || qtype == 20 ? 32 : 256;
  TORCH_CHECK(cols % qk == 0, "input width is not aligned to the GGUF block");
  const int64_t expected_bytes =
      qtype == 8 ? (cols / 32) * sizeof(block_q8_0)
                 : qtype == 20 ? (cols / 32) * sizeof(block_iq4_nl)
                               : qtype == 21
                                     ? (cols / 256) * sizeof(block_iq3_s)
                                     : (cols / 256) * sizeof(block_iq4_xs);
  TORCH_CHECK(cold_weight.size(2) == expected_bytes,
              "packed row byte width does not match qtype and input width");

  c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded = (cols + 511) / 512 * 512;
  auto quantized = torch::empty(
      {tokens, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({tokens * top_k, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const dim3 quant_grid((padded + 255) / 256, tokens, 1);
  quantize_q8_1<<<quant_grid, 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);

  bool specialized_launched = false;
  switch (specialized_variant) {
    case 0:
      break;
    case 1:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 1
      specialized_launched = try_launch_tiered_exact_q4<1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, qtype, cols, rows, tokens, stream);
#else
      TORCH_CHECK(false, "auto exact-shape variant was not compiled");
#endif
      break;
    case 2:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 2
      specialized_launched = try_launch_tiered_exact_q4<2>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, qtype, cols, rows, tokens, stream);
#else
      TORCH_CHECK(false, "unroll-2 exact-shape variant was not compiled");
#endif
      break;
    case 5:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 4
      specialized_launched = try_launch_tiered_exact_q4<5>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, qtype, cols, rows, tokens, stream);
#else
      TORCH_CHECK(false, "unroll-5 exact-shape variant was not compiled");
#endif
      break;
    case 10:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 8
      specialized_launched = try_launch_tiered_exact_q4<10>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids, top_k, qtype, cols, rows, tokens, stream);
#else
      TORCH_CHECK(false, "unroll-10 exact-shape variant was not compiled");
#endif
      break;
    case 30:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 16
      specialized_launched = try_launch_tiered_reuse3(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, qtype, cols, rows, tokens,
          stream);
#else
      TORCH_CHECK(false, "reuse3 exact-shape variant was not compiled");
#endif
      break;
    case 31:
#if QWEN38_TIERED_IQ_MOE_SPECIALIZED_VARIANTS & 32
      specialized_launched = try_launch_tiered_reuse3v2(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output, expert_ids, top_k, qtype, cols, rows, tokens,
          stream);
#else
      TORCH_CHECK(false, "reuse3v2 exact-shape variant was not compiled");
#endif
      break;
  }

  if (specialized_launched) {
    AT_CUDA_CHECK(hipGetLastError());
    return output;
  }

  switch (qtype) {
    case 8:
      launch_tiered<4, QK8_0, QI8_0, block_q8_0,
                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids,
          top_k, cols, rows, tokens, stream);
      break;
    case 20:
      launch_tiered<2, QK4_NL, QI4_NL, block_iq4_nl,
                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids,
          top_k, cols, rows, tokens, stream);
      break;
    case 21:
      launch_tiered<8, QK_K, QI3_XS, block_iq3_s, 1,
                    vec_dot_iq3_s_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids,
          top_k, cols, rows, tokens, stream);
      break;
    case 23:
      launch_tiered<8, QK_K, QI4_XS, block_iq4_xs, 1,
                    vec_dot_iq4_xs_q8_1>(
          cold_weight, hot_weight, hot_map, cold_map, cache_weight, cache_map,
          quantized, output,
          expert_ids,
          top_k, cols, rows, tokens, stream);
      break;
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

void tiered_iq_moe_cache_prepare(
    torch::Tensor cold_w13, torch::Tensor cold_w2, torch::Tensor hot_map,
    torch::Tensor cold_map, torch::Tensor expert_ids,
    torch::Tensor cache_w13, torch::Tensor cache_w2,
    torch::Tensor cache_map, torch::Tensor cache_tags,
    torch::Tensor cache_clock, torch::Tensor admission, torch::Tensor stats) {
  TORCH_CHECK(cold_w13.is_cuda() && cold_w2.is_cuda() && hot_map.is_cuda() &&
                  cold_map.is_cuda() && expert_ids.is_cuda() &&
                  cache_w13.is_cuda() && cache_w2.is_cuda() &&
                  cache_map.is_cuda() && cache_tags.is_cuda() &&
                  cache_clock.is_cuda() && admission.is_cuda() &&
                  stats.is_cuda(),
              "cache preparation tensors must be GPU or UVA tensors");
  const auto device = cold_w13.device();
  TORCH_CHECK(cold_w2.device() == device && hot_map.device() == device &&
                  cold_map.device() == device && expert_ids.device() == device &&
                  cache_w13.device() == device && cache_w2.device() == device &&
                  cache_map.device() == device && cache_tags.device() == device &&
                  cache_clock.device() == device && admission.device() == device &&
                  stats.device() == device,
              "cache preparation tensors must share one device");
  TORCH_CHECK(cold_w13.scalar_type() == torch::kUInt8 &&
                  cold_w2.scalar_type() == torch::kUInt8 &&
                  cache_w13.scalar_type() == torch::kUInt8 &&
                  cache_w2.scalar_type() == torch::kUInt8,
              "cache weights must contain packed uint8 bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  cache_map.scalar_type() == torch::kInt32 &&
                  cache_tags.scalar_type() == torch::kInt32 &&
                  cache_clock.scalar_type() == torch::kInt32 &&
                  admission.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32,
              "cache maps and state must be int32");
  TORCH_CHECK(cold_w13.is_contiguous() && cold_w2.is_contiguous() &&
                  hot_map.is_contiguous() && cold_map.is_contiguous() &&
                  expert_ids.is_contiguous() && cache_w13.is_contiguous() &&
                  cache_w2.is_contiguous() && cache_map.is_contiguous() &&
                  cache_tags.is_contiguous() && cache_clock.is_contiguous() &&
                  admission.is_contiguous() && stats.is_contiguous(),
              "cache preparation tensors must be contiguous");
  TORCH_CHECK(cold_w13.dim() == 3 && cold_w2.dim() == 3 &&
                  cache_w13.dim() == 3 && cache_w2.dim() == 3,
              "expert weights must be three-dimensional");
  TORCH_CHECK(cold_w13.size(0) == cold_w2.size(0),
              "cold projections must contain the same experts");
  TORCH_CHECK(cache_w13.size(0) == cache_w2.size(0) &&
                  cache_w13.size(0) == cache_tags.numel(),
              "cache projections and tags must contain the same slots");
  TORCH_CHECK(cache_w13.size(1) == cold_w13.size(1) &&
                  cache_w13.size(2) == cold_w13.size(2) &&
                  cache_w2.size(1) == cold_w2.size(1) &&
                  cache_w2.size(2) == cold_w2.size(2),
              "cache projection layouts must match cold expert layouts");
  // Seventeen is accepted only so async logical-cache16 can use its physical
  // staging allocation when breakable graph capture falls back synchronously.
  // Launcher/plugin logical capacity remains capped at sixteen.
  TORCH_CHECK(cache_w13.size(0) > 0 &&
                  cache_w13.size(0) <= kMaxLogicalCacheSlots + 1,
              "bounded cache supports one through seventeen physical slots");
  TORCH_CHECK(hot_map.numel() == cold_map.numel() &&
                  hot_map.numel() == cache_map.numel() &&
                  hot_map.numel() == admission.numel(),
              "all expert maps/state must cover the same logical experts");
  TORCH_CHECK(cache_clock.numel() >= 1 && stats.numel() >= 5,
              "cache clock/stats tensors are too small");

  // This cache is intentionally decode-only.  Larger graph shapes retain the
  // normal mixed VRAM/UVA path and do not incur an O(routes^2) policy scan.
  const int64_t routes = expert_ids.numel();
  if (routes == 0 || routes > 64) return;

  c10::cuda::CUDAGuard guard(cold_w13.device());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const int64_t w13_expert_bytes = cold_w13.size(1) * cold_w13.size(2);
  const int64_t w2_expert_bytes = cold_w2.size(1) * cold_w2.size(2);
  prepare_expert_cache<<<1, 256, 0, stream>>>(
      cold_w13.data_ptr<uint8_t>(), cold_w2.data_ptr<uint8_t>(),
      hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
      expert_ids.data_ptr<int>(), cache_w13.data_ptr<uint8_t>(),
      cache_w2.data_ptr<uint8_t>(), cache_map.data_ptr<int>(),
      cache_tags.data_ptr<int>(), cache_clock.data_ptr<int>(),
      admission.data_ptr<int>(), stats.data_ptr<int>(), hot_map.numel(),
      cold_w13.size(0), cache_w13.size(0), routes, w13_expert_bytes,
      w2_expert_bytes);
  AT_CUDA_CHECK(hipGetLastError());
}

void tiered_iq_moe_cache_lru_prepare(
    torch::Tensor cold_w13, torch::Tensor cold_w2, torch::Tensor hot_map,
    torch::Tensor cold_map, torch::Tensor expert_ids,
    torch::Tensor cache_w13, torch::Tensor cache_w2,
    torch::Tensor cache_map, torch::Tensor cache_tags,
    torch::Tensor cache_clock, torch::Tensor admission, torch::Tensor stats,
    torch::Tensor pending) {
  TORCH_CHECK(cold_w13.is_cuda() && cold_w2.is_cuda() && hot_map.is_cuda() &&
                  cold_map.is_cuda() && expert_ids.is_cuda() &&
                  cache_w13.is_cuda() && cache_w2.is_cuda() &&
                  cache_map.is_cuda() && cache_tags.is_cuda() &&
                  cache_clock.is_cuda() && admission.is_cuda() &&
                  stats.is_cuda() && pending.is_cuda(),
              "LRU cache tensors must be GPU or UVA tensors");
  const auto device = cold_w13.device();
  TORCH_CHECK(cold_w2.device() == device && hot_map.device() == device &&
                  cold_map.device() == device && expert_ids.device() == device &&
                  cache_w13.device() == device && cache_w2.device() == device &&
                  cache_map.device() == device && cache_tags.device() == device &&
                  cache_clock.device() == device && admission.device() == device &&
                  stats.device() == device && pending.device() == device,
              "LRU cache tensors must share one device");
  TORCH_CHECK(cold_w13.scalar_type() == torch::kUInt8 &&
                  cold_w2.scalar_type() == torch::kUInt8 &&
                  cache_w13.scalar_type() == torch::kUInt8 &&
                  cache_w2.scalar_type() == torch::kUInt8,
              "LRU cache weights must contain packed uint8 bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  cache_map.scalar_type() == torch::kInt32 &&
                  cache_tags.scalar_type() == torch::kInt32 &&
                  cache_clock.scalar_type() == torch::kInt32 &&
                  admission.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32 &&
                  pending.scalar_type() == torch::kInt32,
              "LRU cache maps and state must be int32");
  TORCH_CHECK(cold_w13.is_contiguous() && cold_w2.is_contiguous() &&
                  hot_map.is_contiguous() && cold_map.is_contiguous() &&
                  expert_ids.is_contiguous() && cache_w13.is_contiguous() &&
                  cache_w2.is_contiguous() && cache_map.is_contiguous() &&
                  cache_tags.is_contiguous() && cache_clock.is_contiguous() &&
                  admission.is_contiguous() && stats.is_contiguous() &&
                  pending.is_contiguous(),
              "LRU cache tensors must be contiguous");
  TORCH_CHECK(cold_w13.dim() == 3 && cold_w2.dim() == 3 &&
                  cache_w13.dim() == 3 && cache_w2.dim() == 3,
              "LRU expert weights must be three-dimensional");
  TORCH_CHECK(cold_w13.size(0) == cold_w2.size(0),
              "LRU cold projections must contain the same experts");
  TORCH_CHECK(cache_w13.size(0) == cache_w2.size(0) &&
                  cache_w13.size(0) == cache_tags.numel(),
              "LRU cache projections and tags must contain the same slots");
  TORCH_CHECK(cache_w13.size(1) == cold_w13.size(1) &&
                  cache_w13.size(2) == cold_w13.size(2) &&
                  cache_w2.size(1) == cold_w2.size(1) &&
                  cache_w2.size(2) == cold_w2.size(2),
              "LRU cache projection layouts must match cold expert layouts");
  TORCH_CHECK(cache_w13.size(0) > 0 &&
                  cache_w13.size(0) <= kMaxLogicalCacheSlots,
              "synchronous LRU cache supports one through sixteen slots");
  TORCH_CHECK(hot_map.numel() == cold_map.numel() &&
                  hot_map.numel() == cache_map.numel() &&
                  hot_map.numel() == admission.numel(),
              "all LRU expert maps/state must cover the same logical experts");
  TORCH_CHECK(cache_clock.numel() >= cache_tags.numel() + 1 &&
                  stats.numel() >= 9 && pending.numel() >= 7,
              "LRU cache clock/stats/pending tensors are too small");

  const int routes = expert_ids.numel();
  if (routes == 0 || routes > 64) return;
  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const int64_t w13_expert_bytes = cold_w13.size(1) * cold_w13.size(2);
  const int64_t w2_expert_bytes = cold_w2.size(1) * cold_w2.size(2);
  plan_expert_cache_lru_fill<<<1, 256, 0, stream>>>(
      hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
      expert_ids.data_ptr<int>(), cache_map.data_ptr<int>(),
      cache_tags.data_ptr<int>(), cache_clock.data_ptr<int>(),
      stats.data_ptr<int>(), pending.data_ptr<int>(), hot_map.numel(),
      cold_w13.size(0), cache_w13.size(0), routes);
  copy_planned_expert_cache<<<kCacheCopyBlocks, 256, 0, stream>>>(
      cold_w13.data_ptr<uint8_t>(), cold_w2.data_ptr<uint8_t>(),
      cache_w13.data_ptr<uint8_t>(), cache_w2.data_ptr<uint8_t>(),
      pending.data_ptr<int>(), w13_expert_bytes, w2_expert_bytes, 3);
  publish_expert_cache_lru<<<1, 1, 0, stream>>>(
      cache_map.data_ptr<int>(), cache_tags.data_ptr<int>(),
      cache_clock.data_ptr<int>(), admission.data_ptr<int>(),
      stats.data_ptr<int>(), pending.data_ptr<int>(), cache_map.numel(),
      cache_tags.numel());
  AT_CUDA_CHECK(hipGetLastError());
}

void tiered_iq_moe_cache_async_init() {
  // Force stream/event creation during model materialization.  The schedule
  // functions intentionally reject no work here; calling this before graph
  // capture is the plugin's responsibility.
  (void)async_cache_context();
}

void tiered_iq_moe_cache_async_prepare(
    torch::Tensor cold_w13, torch::Tensor cold_w2, torch::Tensor hot_map,
    torch::Tensor cold_map, torch::Tensor expert_ids,
    torch::Tensor cache_w13, torch::Tensor cache_w2,
    torch::Tensor cache_map, torch::Tensor cache_tags,
    torch::Tensor cache_clock, torch::Tensor admission, torch::Tensor stats,
    torch::Tensor pending) {
  TORCH_CHECK(cold_w13.is_cuda() && cold_w2.is_cuda() && hot_map.is_cuda() &&
                  cold_map.is_cuda() && expert_ids.is_cuda() &&
                  cache_w13.is_cuda() && cache_w2.is_cuda() &&
                  cache_map.is_cuda() && cache_tags.is_cuda() &&
                  cache_clock.is_cuda() && admission.is_cuda() &&
                  stats.is_cuda() && pending.is_cuda(),
              "async cache tensors must be GPU or UVA tensors");
  const auto device = cold_w13.device();
  TORCH_CHECK(cold_w2.device() == device && hot_map.device() == device &&
                  cold_map.device() == device && expert_ids.device() == device &&
                  cache_w13.device() == device && cache_w2.device() == device &&
                  cache_map.device() == device && cache_tags.device() == device &&
                  cache_clock.device() == device && admission.device() == device &&
                  stats.device() == device && pending.device() == device,
              "async cache tensors must share one device");
  TORCH_CHECK(cold_w13.scalar_type() == torch::kUInt8 &&
                  cold_w2.scalar_type() == torch::kUInt8 &&
                  cache_w13.scalar_type() == torch::kUInt8 &&
                  cache_w2.scalar_type() == torch::kUInt8,
              "async cache weights must contain packed uint8 bytes");
  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 &&
                  cold_map.scalar_type() == torch::kInt32 &&
                  expert_ids.scalar_type() == torch::kInt32 &&
                  cache_map.scalar_type() == torch::kInt32 &&
                  cache_tags.scalar_type() == torch::kInt32 &&
                  cache_clock.scalar_type() == torch::kInt32 &&
                  admission.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32 &&
                  pending.scalar_type() == torch::kInt32,
              "async cache maps and state must be int32");
  TORCH_CHECK(cold_w13.is_contiguous() && cold_w2.is_contiguous() &&
                  hot_map.is_contiguous() && cold_map.is_contiguous() &&
                  expert_ids.is_contiguous() && cache_w13.is_contiguous() &&
                  cache_w2.is_contiguous() && cache_map.is_contiguous() &&
                  cache_tags.is_contiguous() && cache_clock.is_contiguous() &&
                  admission.is_contiguous() && stats.is_contiguous() &&
                  pending.is_contiguous(),
              "async cache tensors must be contiguous");
  TORCH_CHECK(cold_w13.dim() == 3 && cold_w2.dim() == 3 &&
                  cache_w13.dim() == 3 && cache_w2.dim() == 3,
              "async expert weights must be three-dimensional");
  TORCH_CHECK(cold_w13.size(0) == cold_w2.size(0),
              "cold projections must contain the same experts");
  TORCH_CHECK(cache_w13.size(0) == cache_w2.size(0) &&
                  cache_w13.size(0) == cache_tags.numel(),
              "async cache projections and tags must contain the same slots");
  TORCH_CHECK(cache_w13.size(1) == cold_w13.size(1) &&
                  cache_w13.size(2) == cold_w13.size(2) &&
                  cache_w2.size(1) == cold_w2.size(1) &&
                  cache_w2.size(2) == cold_w2.size(2),
              "async cache projection layouts must match cold expert layouts");
  TORCH_CHECK(cache_w13.size(0) >= 2 &&
                  cache_w13.size(0) <= kMaxLogicalCacheSlots + 1,
              "async cache requires two through seventeen physical slots");
  TORCH_CHECK(hot_map.numel() == cold_map.numel() &&
                  hot_map.numel() == cache_map.numel() &&
                  hot_map.numel() == admission.numel(),
              "all async expert maps/state must cover the same experts");
  TORCH_CHECK(cache_clock.numel() >= 1 && stats.numel() >= 9 &&
                  pending.numel() >= 7,
              "async cache clock/stats/pending tensors are too small");

  c10::cuda::CUDAGuard guard(device);
  const auto current = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  auto& context = async_cache_context();

  const int64_t w13_expert_bytes = cold_w13.size(1) * cold_w13.size(2);
  const int64_t w2_expert_bytes = cold_w2.size(1) * cold_w2.size(2);
  const int routes = expert_ids.numel();
  // Planning is deliberately tiny and stays on the current stream.  A route
  // duplicated within this MTP group uses mode 1: copy + publish now so every
  // occurrence in W13/W2 gets the same-cycle VRAM benefit.  A singleton that
  // passed admission uses mode 2 and remains unpublished while the copy stream
  // fills staging.  The mode guards make the two admissions mutually exclusive.
  plan_expert_cache_fill<<<1, 256, 0, current>>>(
      hot_map.data_ptr<int>(), cold_map.data_ptr<int>(),
      expert_ids.data_ptr<int>(), cache_map.data_ptr<int>(),
      cache_tags.data_ptr<int>(), cache_clock.data_ptr<int>(),
      admission.data_ptr<int>(), stats.data_ptr<int>(), pending.data_ptr<int>(),
      hot_map.numel(), cold_w13.size(0), cache_w13.size(0), routes);
  copy_planned_expert_cache<<<kCacheCopyBlocks, 256, 0, current>>>(
      cold_w13.data_ptr<uint8_t>(), cold_w2.data_ptr<uint8_t>(),
      cache_w13.data_ptr<uint8_t>(), cache_w2.data_ptr<uint8_t>(),
      pending.data_ptr<int>(), w13_expert_bytes, w2_expert_bytes, 1);
  publish_planned_expert_cache<<<1, 1, 0, current>>>(
      cache_map.data_ptr<int>(), cache_tags.data_ptr<int>(),
      admission.data_ptr<int>(), stats.data_ptr<int>(), pending.data_ptr<int>(),
      cache_map.numel(), cache_tags.numel(), 1);

  AT_CUDA_CHECK(hipEventRecord(context.fork, current));
  AT_CUDA_CHECK(hipStreamWaitEvent(context.stream, context.fork, 0));
  copy_planned_expert_cache<<<kCacheCopyBlocks, 256, 0, context.stream>>>(
      cold_w13.data_ptr<uint8_t>(), cold_w2.data_ptr<uint8_t>(),
      cache_w13.data_ptr<uint8_t>(), cache_w2.data_ptr<uint8_t>(),
      pending.data_ptr<int>(), w13_expert_bytes, w2_expert_bytes, 2);
  AT_CUDA_CHECK(hipGetLastError());
}

void tiered_iq_moe_cache_async_commit(
    torch::Tensor cache_map, torch::Tensor cache_tags,
    torch::Tensor admission, torch::Tensor stats, torch::Tensor pending,
    bool join) {
  TORCH_CHECK(cache_map.is_cuda() && cache_tags.is_cuda() &&
                  admission.is_cuda() && stats.is_cuda() && pending.is_cuda(),
              "async cache publication state must be on GPU");
  const auto device = cache_map.device();
  TORCH_CHECK(cache_tags.device() == device && admission.device() == device &&
                  stats.device() == device && pending.device() == device,
              "async cache publication state must share one device");
  TORCH_CHECK(cache_map.scalar_type() == torch::kInt32 &&
                  cache_tags.scalar_type() == torch::kInt32 &&
                  admission.scalar_type() == torch::kInt32 &&
                  stats.scalar_type() == torch::kInt32 &&
                  pending.scalar_type() == torch::kInt32,
              "async cache publication state must be int32");
  TORCH_CHECK(cache_map.is_contiguous() && cache_tags.is_contiguous() &&
                  admission.is_contiguous() && stats.is_contiguous() &&
                  pending.is_contiguous(),
              "async cache publication state must be contiguous");
  TORCH_CHECK(stats.numel() >= 9 && pending.numel() >= 7,
              "async cache publication state is too small");

  c10::cuda::CUDAGuard guard(device);
  const auto current = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  auto& context = async_cache_context();
  AT_CUDA_CHECK(hipEventRecord(context.safe_to_publish, current));
  AT_CUDA_CHECK(
      hipStreamWaitEvent(context.stream, context.safe_to_publish, 0));
  publish_planned_expert_cache<<<1, 1, 0, context.stream>>>(
      cache_map.data_ptr<int>(), cache_tags.data_ptr<int>(),
      admission.data_ptr<int>(), stats.data_ptr<int>(), pending.data_ptr<int>(),
      cache_map.numel(), cache_tags.numel(), 2);
  AT_CUDA_CHECK(hipGetLastError());
  AT_CUDA_CHECK(hipEventRecord(context.done, context.stream));
  if (join) {
    // Layer 47 is the only required join for the shared serialized fill stream.
    // It drains every earlier layer's work while avoiding 48 current-stream
    // waits.  Breakable graph capture uses the synchronous fallback in Python.
    AT_CUDA_CHECK(hipStreamWaitEvent(current, context.done, 0));
  }
}

torch::Tensor tiered_iq_moe_gemv(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map, torch::Tensor expert_ids,
    int64_t top_k, int64_t qtype, int64_t rows, int64_t tokens) {
  return tiered_iq_moe_gemv_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, expert_ids, top_k, qtype,
      rows, tokens, 0, nullptr, nullptr);
}

torch::Tensor tiered_iq_moe_gemv_variant(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map, torch::Tensor expert_ids,
    int64_t top_k, int64_t qtype, int64_t rows, int64_t tokens,
    int64_t specialized_variant) {
  return tiered_iq_moe_gemv_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, expert_ids, top_k, qtype,
      rows, tokens, specialized_variant, nullptr, nullptr);
}

torch::Tensor tiered_iq_moe_cached_gemv(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor cache_weight, torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor cache_map, torch::Tensor expert_ids, int64_t top_k,
    int64_t qtype, int64_t rows, int64_t tokens) {
  return tiered_iq_moe_gemv_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, expert_ids, top_k, qtype,
      rows, tokens, 0, &cache_weight, &cache_map);
}

torch::Tensor tiered_iq_moe_cached_gemv_variant(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor cache_weight, torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor cache_map, torch::Tensor expert_ids, int64_t top_k,
    int64_t qtype, int64_t rows, int64_t tokens,
    int64_t specialized_variant) {
  return tiered_iq_moe_gemv_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, expert_ids, top_k, qtype,
      rows, tokens, specialized_variant, &cache_weight, &cache_map);
}

torch::Tensor tiered_iq_moe_prefill_grouped(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor sorted_route_ids, torch::Tensor block_expert_ids,
    torch::Tensor num_routes_post_padded, int64_t top_k, int64_t qtype,
    int64_t rows, int64_t tokens, int64_t group_size) {
  return tiered_iq_moe_prefill_grouped_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, sorted_route_ids,
      block_expert_ids, num_routes_post_padded, top_k, qtype, rows, tokens,
      group_size, nullptr, nullptr);
}

torch::Tensor tiered_iq_moe_cached_prefill_grouped(
    torch::Tensor x, torch::Tensor cold_weight, torch::Tensor hot_weight,
    torch::Tensor cache_weight, torch::Tensor hot_map, torch::Tensor cold_map,
    torch::Tensor cache_map, torch::Tensor sorted_route_ids,
    torch::Tensor block_expert_ids, torch::Tensor num_routes_post_padded,
    int64_t top_k, int64_t qtype, int64_t rows, int64_t tokens,
    int64_t group_size) {
  return tiered_iq_moe_prefill_grouped_impl(
      x, cold_weight, hot_weight, hot_map, cold_map, sorted_route_ids,
      block_expert_ids, num_routes_post_padded, top_k, qtype, rows, tokens,
      group_size, &cache_weight, &cache_map);
}

int64_t tiered_iq_moe_compiled_variants() {
  return kCompiledSpecializedVariants;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tiered_iq_moe_gemv", &tiered_iq_moe_gemv,
             "Qwen3.8 Q4 mixed VRAM/UVA expert GEMV (HIP)");
  module.def("tiered_iq_moe_gemv_variant", &tiered_iq_moe_gemv_variant,
             "Opt-in exact-shape Qwen3.8 Q4 mixed expert GEMV (HIP)");
  module.def("tiered_iq_moe_cache_prepare", &tiered_iq_moe_cache_prepare,
             "Graph-safe bounded cold-expert VRAM cache fill (HIP)");
  module.def("tiered_iq_moe_cache_lru_prepare",
             &tiered_iq_moe_cache_lru_prepare,
             "Graph-safe synchronous LRU cold-expert cache fill (HIP)");
  module.def("tiered_iq_moe_cache_async_init",
             &tiered_iq_moe_cache_async_init,
             "Initialize fixed async cache stream/event state (HIP)");
  module.def("tiered_iq_moe_cache_async_prepare",
             &tiered_iq_moe_cache_async_prepare,
             "Schedule a graph-safe cold-expert fill on the copy stream (HIP)");
  module.def("tiered_iq_moe_cache_async_commit",
             &tiered_iq_moe_cache_async_commit,
             "Publish a completed async cache fill after W2 (HIP)");
  module.def("tiered_iq_moe_cached_gemv", &tiered_iq_moe_cached_gemv,
             "Qwen3.8 Q4 mixed expert GEMV with dynamic cache (HIP)");
  module.def("tiered_iq_moe_cached_gemv_variant",
             &tiered_iq_moe_cached_gemv_variant,
             "Exact-shape Q4 mixed expert GEMV with dynamic cache (HIP)");
  module.def("tiered_iq_moe_prefill_grouped",
             &tiered_iq_moe_prefill_grouped,
             "Expert-grouped Q4 mixed prefill with shared packed reads (HIP)");
  module.def("tiered_iq_moe_cached_prefill_grouped",
             &tiered_iq_moe_cached_prefill_grouped,
             "Expert-grouped Q4 mixed prefill with dynamic cache (HIP)");
  module.def("tiered_iq_moe_compiled_variants",
             &tiered_iq_moe_compiled_variants,
             "Bit mask of compiled exact-shape unroll variants");
}

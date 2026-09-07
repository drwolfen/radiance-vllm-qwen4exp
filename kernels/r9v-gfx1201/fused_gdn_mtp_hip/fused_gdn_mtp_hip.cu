// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

using bf16 = __hip_bfloat16;

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
constexpr int kChunkV = 32;
constexpr int kRowsPerWave = kChunkV / kWaves;
constexpr int kMaxTokens = 8;

struct Strides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, kWave);
  }
  return x;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 wave_sum2(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, kWave);
    y += __shfl_xor(y, offset, kWave);
  }
  return {x, y};
}

__device__ __forceinline__ float load_dt_bias(const void* ptr, int head,
                                               int dtype) {
  if (dtype == 1) {
    return __bfloat162float(static_cast<const bf16*>(ptr)[head]);
  }
  if (dtype == 2) {
    return __half2float(static_cast<const half*>(ptr)[head]);
  }
  return static_cast<const float*>(ptr)[head];
}

// Qwen3.8 Flash Next TP2 specialization: H=8, HV=24, HV/H=3, FP32 state.
// ApplyNorm=false is the graph-friendly core-only entry point: it preserves
// vLLM's external RMSNormGated/output-projection seam.
template <bool ApplyNorm, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_mtp_tp2_fp32_kernel(
    const bf16* __restrict__ mixed_qkv, const bf16* __restrict__ a,
    const bf16* __restrict__ b, const float* __restrict__ a_log,
    const void* __restrict__ dt_bias, const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, float* __restrict__ state,
    const bf16* __restrict__ output_gate, const void* __restrict__ norm_weight,
    bf16* __restrict__ out, int state_indices_width, int dt_bias_type,
    bool norm_weight_bf16, int num_state_slots, int total_tokens, float scale,
    float norm_eps, Strides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  if (bos < 0 || eos < bos || eos > total_tokens) return;
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || source_slot >= num_state_slots ||
      num_tokens > kMaxTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      out[(static_cast<int64_t>(token) * 24 + value_head) * kDimV + value] =
          __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / 3;
  __shared__ float shared_state[kChunkV][kDimK];
  __shared__ float shared_q[kMaxTokens][kDimK];
  __shared__ float shared_k[kMaxTokens][kDimK];
  __shared__ bf16 shared_v[kMaxTokens][kDimV];
  __shared__ bf16 shared_out[kMaxTokens][kDimV];
  __shared__ float shared_decay[kMaxTokens];
  __shared__ float shared_beta[kMaxTokens];

  if (wave < num_tokens) {
    const int t = wave;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float qv[4];
    float kv[4];
    float q2 = 0.0f;
    float k2 = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * kWave;
      qv[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      kv[i] = __bfloat162float(
          mixed_qkv[mixed_base + 8 * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 16 * kDimK + value_head * kDimV + dim];
      q2 += qv[i] * qv[i];
      k2 += kv[i] * kv[i];
    }
    const Sum2 sums = wave_sum2(q2, k2);
    const float qscale = __shfl(lane == 0 ? rsqrtf(sums.x + 1.0e-6f) * scale
                                          : 0.0f,
                                  0, kWave);
    const float kscale = __shfl(
        lane == 0 ? rsqrtf(sums.y + 1.0e-6f) : 0.0f, 0, kWave);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * kWave;
      shared_q[t][dim] = qv[i] * qscale;
      shared_k[t][dim] = kv[i] * kscale;
    }
    if (lane == 0) {
      const float av = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float bv = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(av + load_dt_bias(dt_bias, value_head,
                                                      dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(bv);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  const int rows[kRowsPerWave] = {wave, wave + kWaves, wave + 2 * kWaves,
                                  wave + 3 * kWaves};
  const float* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;

#pragma unroll
  for (int chunk = 0; chunk < kDimV / kChunkV; ++chunk) {
    for (int linear = tid; linear < kChunkV * kDimK; linear += kThreads) {
      shared_state[0][linear] = source_state[chunk * kChunkV * kDimK + linear];
    }
    __syncthreads();

    float h[kRowsPerWave][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWave; ++row) {
      const float4 sv = *reinterpret_cast<const float4*>(
          &shared_state[rows[row]][k_base]);
      h[row][0] = sv.x;
      h[row][1] = sv.y;
      h[row][2] = sv.z;
      h[row][3] = sv.w;
    }
    __syncthreads();

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 =
          *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 =
          *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float qvals[4] = {q4.x, q4.y, q4.z, q4.w};
      const float kvals[4] = {k4.x, k4.y, k4.z, k4.w};
      float hk[kRowsPerWave] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWave; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          hk[row] += h[row][i] * kvals[i];
        }
      }
      const Sum2 hk01 = wave_sum2(hk[0], hk[1]);
      const Sum2 hk23 = wave_sum2(hk[2], hk[3]);
      const float reduced_hk[kRowsPerWave] = {hk01.x, hk01.y, hk23.x, hk23.y};

      float hq[kRowsPerWave] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWave; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += kvals[i] * delta;
          hq[row] += h[row][i] * qvals[i];
        }
      }
      const Sum2 hq01 = wave_sum2(hq[0], hq[1]);
      const Sum2 hq23 = wave_sum2(hq[2], hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] = __float2bfloat16(hq01.x);
        shared_out[t][chunk * kChunkV + rows[1]] = __float2bfloat16(hq01.y);
        shared_out[t][chunk * kChunkV + rows[2]] = __float2bfloat16(hq23.x);
        shared_out[t][chunk * kChunkV + rows[3]] = __float2bfloat16(hq23.y);
      }

      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0 && destination_slot < num_state_slots) {
        float* destination_state =
            state + static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWave; ++row) {
          const int value = chunk * kChunkV + rows[row];
          *reinterpret_cast<float4*>(destination_state + value * kDimK +
                                     k_base) =
              make_float4(h[row][0], h[row][1], h[row][2], h[row][3]);
        }
      }
    }
  }
  __syncthreads();

  if (wave < num_tokens) {
    const int t = wave;
    const int token = bos + t;
    if constexpr (!ApplyNorm) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * kWave;
        out[(static_cast<int64_t>(token) * 24 + value_head) * kDimV + value] =
            shared_out[t][value];
      }
    } else {
      float ov[4];
      float sum2 = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * kWave;
        ov[i] = __bfloat162float(shared_out[t][value]);
        sum2 += ov[i] * ov[i];
      }
      sum2 = wave_sum(sum2);
      const float rstd = rsqrtf(sum2 / static_cast<float>(kDimV) + norm_eps);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * kWave;
        const float gate_input = __bfloat162float(
            output_gate[static_cast<int64_t>(token) * strides.gate_row +
                        value_head * kDimV + value]);
        const float gate =
            SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
        const float weight =
            norm_weight_bf16
                ? __bfloat162float(static_cast<const bf16*>(norm_weight)[value])
                : static_cast<const float*>(norm_weight)[value];
        out[(static_cast<int64_t>(token) * 24 + value_head) * kDimV + value] =
            __float2bfloat16(ov[i] * rstd * weight * gate);
      }
    }
  }
}

torch::Tensor fused_gdn_mtp_tp2_fp32(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens,
    torch::Tensor state, torch::Tensor output_gate, torch::Tensor norm_weight,
    torch::Tensor out, double scale, double norm_eps, bool sigmoid_gate) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kBFloat16,
              "mixed_qkv must be a HIP bfloat16 tensor");
  const auto device = mixed_qkv.device();
  const auto on_device = [&](const torch::Tensor& tensor) {
    return tensor.is_cuda() && tensor.device() == device;
  };
  TORCH_CHECK(on_device(a) && a.scalar_type() == at::kBFloat16,
              "a must be a HIP bfloat16 tensor on the same device");
  TORCH_CHECK(on_device(b) && b.scalar_type() == at::kBFloat16,
              "b must be a HIP bfloat16 tensor on the same device");
  TORCH_CHECK(on_device(a_log) && a_log.scalar_type() == at::kFloat,
              "A_log must be a HIP float32 tensor on the same device");
  TORCH_CHECK(on_device(dt_bias) &&
                  (dt_bias.scalar_type() == at::kFloat ||
                   dt_bias.scalar_type() == at::kBFloat16 ||
                   dt_bias.scalar_type() == at::kHalf),
              "dt_bias must be HIP float32, bfloat16, or float16");
  TORCH_CHECK(on_device(state_indices) &&
                  state_indices.scalar_type() == at::kInt,
              "state_indices must be a HIP int32 tensor on the same device");
  TORCH_CHECK(on_device(cu_seqlens) && cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be a HIP int32 tensor on the same device");
  TORCH_CHECK(on_device(num_accepted_tokens) &&
                  num_accepted_tokens.scalar_type() == at::kInt,
              "num_accepted_tokens must be HIP int32 on the same device");
  TORCH_CHECK(on_device(state) && state.scalar_type() == at::kFloat,
              "state must be a HIP float32 tensor on the same device");
  TORCH_CHECK(on_device(output_gate) &&
                  output_gate.scalar_type() == at::kBFloat16,
              "output_gate must be HIP bfloat16 on the same device");
  TORCH_CHECK(on_device(norm_weight) &&
                  (norm_weight.scalar_type() == at::kFloat ||
                   norm_weight.scalar_type() == at::kBFloat16),
              "norm_weight must be HIP float32 or bfloat16 on the same device");
  TORCH_CHECK(on_device(out) && out.scalar_type() == at::kBFloat16,
              "out must be HIP bfloat16 on the same device");

  TORCH_CHECK(mixed_qkv.dim() == 2 &&
                  mixed_qkv.size(1) == 16 * kDimK + 24 * kDimV,
              "mixed_qkv must have shape [L, 5120]");
  const int total_tokens = static_cast<int>(mixed_qkv.size(0));
  TORCH_CHECK(total_tokens > 0, "at least one token is required");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == total_tokens && a.size(1) == 24,
              "a must have shape [L, 24]");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == total_tokens && b.size(1) == 24,
              "b must have shape [L, 24]");
  TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == 24,
              "A_log must be contiguous with 24 elements");
  TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == 24,
              "dt_bias must be contiguous with 24 elements");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 1 && state.size(1) == 24 &&
                  state.size(2) == kDimV && state.size(3) == kDimK,
              "state must have shape [slots, 24, 128, 128]");
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) > 0 &&
                  state_indices.size(1) > 0 &&
                  state_indices.size(1) <= kMaxTokens,
              "state_indices must have shape [N, S] with 1 <= S <= 8");
  const int requests = static_cast<int>(state_indices.size(0));
  TORCH_CHECK(cu_seqlens.dim() == 1 &&
                  cu_seqlens.numel() == requests + 1,
              "cu_seqlens must have N + 1 elements");
  TORCH_CHECK(num_accepted_tokens.dim() == 1 &&
                  num_accepted_tokens.numel() == requests,
              "num_accepted_tokens must have N elements");
  TORCH_CHECK(output_gate.dim() == 3 &&
                  output_gate.size(0) == total_tokens &&
                  output_gate.size(1) == 24 &&
                  output_gate.size(2) == kDimV,
              "output_gate must have shape [L, 24, 128]");
  TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV,
              "norm_weight must be contiguous with 128 elements");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == total_tokens &&
                  out.size(1) == 24 && out.size(2) == kDimV,
              "out must have shape [L, 24, 128]");
  TORCH_CHECK(mixed_qkv.stride(1) == 1,
              "mixed_qkv channels must be contiguous");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
              "a and b heads must be contiguous");
  TORCH_CHECK(state.stride(0) >= 24 * kDimV * kDimK &&
                  state.stride(1) == kDimV * kDimK &&
                  state.stride(2) == kDimK && state.stride(3) == 1,
              "state slots must have contiguous [24, 128, 128] contents");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                  state.stride(0) % 4 == 0,
              "state slots must preserve 16-byte alignment");
  TORCH_CHECK(state_indices.is_contiguous() && cu_seqlens.is_contiguous() &&
                  num_accepted_tokens.is_contiguous(),
              "GDN metadata tensors must be contiguous");
  TORCH_CHECK(output_gate.stride(2) == 1 &&
                  output_gate.stride(1) == kDimV,
              "output_gate head rows must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");

  const int dt_type = dt_bias.scalar_type() == at::kFloat
                          ? 0
                          : (dt_bias.scalar_type() == at::kBFloat16 ? 1 : 2);
  const c10::cuda::OptionalCUDAGuard guard(device_of(mixed_qkv));
  hipStream_t stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const dim3 grid(requests, 24);
  Strides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                  output_gate.stride(0), state.stride(0)};
  auto launch = [&](auto sigmoid_tag) {
    constexpr bool Sigmoid = decltype(sigmoid_tag)::value;
    hipLaunchKernelGGL((gdn_mtp_tp2_fp32_kernel<true, Sigmoid>), grid,
                       dim3(kThreads), 0, stream,
                       reinterpret_cast<const bf16*>(mixed_qkv.data_ptr()),
                       reinterpret_cast<const bf16*>(a.data_ptr()),
                       reinterpret_cast<const bf16*>(b.data_ptr()),
                       a_log.data_ptr<float>(), dt_bias.data_ptr(),
                       state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
                       num_accepted_tokens.data_ptr<int>(), state.data_ptr<float>(),
                       reinterpret_cast<const bf16*>(output_gate.data_ptr()),
                       norm_weight.data_ptr(),
                       reinterpret_cast<bf16*>(out.data_ptr()),
                       state_indices.size(1), dt_type,
                       norm_weight.scalar_type() == at::kBFloat16,
                       static_cast<int>(state.size(0)), total_tokens,
                       static_cast<float>(scale), static_cast<float>(norm_eps),
                       strides);
  };
  if (sigmoid_gate) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  AT_CUDA_CHECK(hipGetLastError());
  return out;
}

torch::Tensor fused_gdn_mtp_tp2_fp32_core(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens,
    torch::Tensor state, torch::Tensor out, double scale) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kBFloat16,
              "mixed_qkv must be a HIP bfloat16 tensor");
  const auto device = mixed_qkv.device();
  const auto on_device = [&](const torch::Tensor& tensor) {
    return tensor.is_cuda() && tensor.device() == device;
  };
  TORCH_CHECK(on_device(a) && a.scalar_type() == at::kBFloat16,
              "a must be a HIP bfloat16 tensor on the same device");
  TORCH_CHECK(on_device(b) && b.scalar_type() == at::kBFloat16,
              "b must be a HIP bfloat16 tensor on the same device");
  TORCH_CHECK(on_device(a_log) && a_log.scalar_type() == at::kFloat,
              "A_log must be a HIP float32 tensor on the same device");
  TORCH_CHECK(on_device(dt_bias) &&
                  (dt_bias.scalar_type() == at::kFloat ||
                   dt_bias.scalar_type() == at::kBFloat16 ||
                   dt_bias.scalar_type() == at::kHalf),
              "dt_bias must be HIP float32, bfloat16, or float16");
  TORCH_CHECK(on_device(state_indices) &&
                  state_indices.scalar_type() == at::kInt,
              "state_indices must be a HIP int32 tensor on the same device");
  TORCH_CHECK(on_device(cu_seqlens) && cu_seqlens.scalar_type() == at::kInt,
              "cu_seqlens must be a HIP int32 tensor on the same device");
  TORCH_CHECK(on_device(num_accepted_tokens) &&
                  num_accepted_tokens.scalar_type() == at::kInt,
              "num_accepted_tokens must be HIP int32 on the same device");
  TORCH_CHECK(on_device(state) && state.scalar_type() == at::kFloat,
              "state must be a HIP float32 tensor on the same device");
  TORCH_CHECK(on_device(out) && out.scalar_type() == at::kBFloat16,
              "out must be HIP bfloat16 on the same device");

  TORCH_CHECK(mixed_qkv.dim() == 2 &&
                  mixed_qkv.size(1) == 16 * kDimK + 24 * kDimV,
              "mixed_qkv must have shape [L, 5120]");
  const int total_tokens = static_cast<int>(mixed_qkv.size(0));
  TORCH_CHECK(total_tokens > 0, "at least one token is required");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == total_tokens && a.size(1) == 24,
              "a must have shape [L, 24]");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == total_tokens && b.size(1) == 24,
              "b must have shape [L, 24]");
  TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == 24,
              "A_log must be contiguous with 24 elements");
  TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == 24,
              "dt_bias must be contiguous with 24 elements");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 1 && state.size(1) == 24 &&
                  state.size(2) == kDimV && state.size(3) == kDimK,
              "state must have shape [slots, 24, 128, 128]");
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) > 0 &&
                  state_indices.size(1) > 0 &&
                  state_indices.size(1) <= kMaxTokens,
              "state_indices must have shape [N, S] with 1 <= S <= 8");
  const int requests = static_cast<int>(state_indices.size(0));
  TORCH_CHECK(cu_seqlens.dim() == 1 &&
                  cu_seqlens.numel() == requests + 1,
              "cu_seqlens must have N + 1 elements");
  TORCH_CHECK(num_accepted_tokens.dim() == 1 &&
                  num_accepted_tokens.numel() == requests,
              "num_accepted_tokens must have N elements");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == total_tokens &&
                  out.size(1) == 24 && out.size(2) == kDimV,
              "out must have shape [L, 24, 128]");
  TORCH_CHECK(mixed_qkv.stride(1) == 1,
              "mixed_qkv channels must be contiguous");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
              "a and b heads must be contiguous");
  TORCH_CHECK(state.stride(0) >= 24 * kDimV * kDimK &&
                  state.stride(1) == kDimV * kDimK &&
                  state.stride(2) == kDimK && state.stride(3) == 1,
              "state slots must have contiguous [24, 128, 128] contents");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                  state.stride(0) % 4 == 0,
              "state slots must preserve 16-byte alignment");
  TORCH_CHECK(state_indices.is_contiguous() && cu_seqlens.is_contiguous() &&
                  num_accepted_tokens.is_contiguous(),
              "GDN metadata tensors must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  const int dt_type = dt_bias.scalar_type() == at::kFloat
                          ? 0
                          : (dt_bias.scalar_type() == at::kBFloat16 ? 1 : 2);
  const c10::cuda::OptionalCUDAGuard guard(device_of(mixed_qkv));
  hipStream_t stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const dim3 grid(requests, 24);
  Strides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0), 0,
                  state.stride(0)};
  hipLaunchKernelGGL((gdn_mtp_tp2_fp32_kernel<false, false>), grid,
                     dim3(kThreads), 0, stream,
                     reinterpret_cast<const bf16*>(mixed_qkv.data_ptr()),
                     reinterpret_cast<const bf16*>(a.data_ptr()),
                     reinterpret_cast<const bf16*>(b.data_ptr()),
                     a_log.data_ptr<float>(), dt_bias.data_ptr(),
                     state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
                     num_accepted_tokens.data_ptr<int>(), state.data_ptr<float>(),
                     nullptr, nullptr, reinterpret_cast<bf16*>(out.data_ptr()),
                     state_indices.size(1), dt_type, false,
                     static_cast<int>(state.size(0)), total_tokens,
                     static_cast<float>(scale), 0.0f, strides);
  AT_CUDA_CHECK(hipGetLastError());
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gdn_mtp_tp2_fp32", &fused_gdn_mtp_tp2_fp32,
        "Qwen3.8 Flash Next fused speculative GDN (HIP, TP2, FP32 state)");
  m.def("fused_gdn_mtp_tp2_fp32_core", &fused_gdn_mtp_tp2_fp32_core,
        "Qwen3.8 Flash Next speculative GDN core (HIP, TP2, FP32 state)");
}

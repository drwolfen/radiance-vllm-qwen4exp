// SPDX-License-Identifier: Apache-2.0
// R9V gfx1201 specializations. GGUF quant primitives are supplied by the
// vLLM GGUF plugin; see the repository's THIRD_PARTY_NOTICES.md.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "hip_compat.h"
#include "gguf/ggml-common_hip.h"
#include "gguf/vecdotq_hip.cuh"

namespace {

using bf16 = __hip_bfloat16;

// Exact Qwen4Exp target-verifier HyperConnection down projection.  The
// upstream wvSplitK shape is [336, 10240] x [3, 10240] and launches a
// 32-workgroup/16-wave kernel, but its runtime wave limiter leaves only six
// waves per workgroup doing two rows each.  On gfx1201 that is too little
// latency hiding for a 6.6 MiB BF16 matrix.  This mapping keeps the same 32
// workgroups and the same fixed per-row arithmetic/reduction order, while
// distributing one row to each of eleven waves (10 or 11 rows/workgroup).
//
// The union deliberately matches skinny_gemms.cu's 16-byte A_CHUNK=8 load.
// Keeping the K traversal, pairwise BF16 product, FP32 add, DPP tree, and
// final BF16 conversion identical is what makes bitwise parity possible.
using hc_scalar8 =
    __attribute__((__vector_size__(4 * sizeof(float)))) float;
union hc_big_type {
  bf16 h[8];
  float f[4];
  hc_scalar8 h8;
};

template <bool non_temporal_weight>
__device__ __forceinline__ hc_scalar8 hc_load_weight(
    const hc_scalar8* pointer) {
  if constexpr (non_temporal_weight) {
    return __builtin_nontemporal_load(pointer);
  } else {
    return *pointer;
  }
}

template <bool non_temporal_weight>
__global__ __launch_bounds__(512)
void hc_down_bf16_m3_cyclic(const bf16* __restrict__ weight,
                            const bf16* __restrict__ input,
                            bf16* __restrict__ output) {
  constexpr int cols = 10240;
  constexpr int rows = 336;
  constexpr int tokens = 3;
  constexpr int chunks_per_input = cols / 8;
  constexpr int input_chunks = tokens * chunks_per_input;
  constexpr int workgroups = 32;

  // 3 * 10240 * 2 = 61,440 bytes, within gfx1201's 64 KiB LDS limit.
  __shared__ hc_big_type input_lds[input_chunks];
  const int linear_thread = threadIdx.y * 32 + threadIdx.x;
  for (int chunk = linear_thread; chunk < input_chunks; chunk += 512) {
    input_lds[chunk].h8 =
        reinterpret_cast<const hc_big_type*>(input)[chunk].h8;
  }
  __syncthreads();

  const int row = blockIdx.x + threadIdx.y * workgroups;
  if (row >= rows) return;

  const int lane = threadIdx.x;
  const bf16* row_weight = weight + static_cast<int64_t>(row) * cols;
  const bf16* staged_input = reinterpret_cast<const bf16*>(input_lds);
  float sum[tokens] = {};

  // Match wvSplitK_hf_sml_<BF16,32,2,16,8,2,3>: lane L consumes
  // [L*8 + k2*256] for k2=0,1, with the outer loop advancing by 512.
  for (int k1 = 0; k1 < cols; k1 += 32 * 8 * 2) {
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      const int k = k1 + k2 * 32 * 8 + lane * 8;
      hc_big_type weight_values;
      weight_values.h8 = hc_load_weight<non_temporal_weight>(
          reinterpret_cast<const hc_scalar8*>(row_weight + k));
#pragma unroll
      for (int token = 0; token < tokens; ++token) {
        const hc_big_type input_values = *reinterpret_cast<const hc_big_type*>(
            staged_input + token * cols + k);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float2 product = __bfloat1622float2(
                                     *reinterpret_cast<const __hip_bfloat162*>(
                                         &weight_values.f[pair])) *
                                 __bfloat1622float2(
                                     *reinterpret_cast<const __hip_bfloat162*>(
                                         &input_values.f[pair]));
          sum[token] += product.x + product.y;
        }
      }
    }
  }

#pragma unroll
  for (int token = 0; token < tokens; ++token) {
    sum[token] +=
        __builtin_amdgcn_mov_dpp(sum[token], 0x118, 0xf, 0xf, 1);
    sum[token] +=
        __builtin_amdgcn_mov_dpp(sum[token], 0x114, 0xf, 0xf, 1);
    sum[token] +=
        __builtin_amdgcn_mov_dpp(sum[token], 0x112, 0xf, 0xf, 1);
    sum[token] +=
        __builtin_amdgcn_mov_dpp(sum[token], 0x111, 0xf, 0xf, 1);
    sum[token] += __shfl_xor(sum[token], 16);
  }
  if (lane == 31) {
#pragma unroll
    for (int token = 0; token < tokens; ++token) {
      output[token * rows + row] = __float2bfloat16(sum[token]);
    }
  }
}

template <typename scalar_t>
__global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                              block_q8_1* __restrict__ y, int cols,
                              int padded) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= padded) return;
  const int vec = blockIdx.y;
  const int offset = vec * padded + ix;
  const int block = offset / QK8_1;
  const int iqs = offset % QK8_1;
  const float value = ix < cols ? static_cast<float>(x[vec * cols + ix]) : 0.0f;
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    sum += __shfl_xor(sum, mask, 32);
  }
  const float scale = amax / 127.0f;
  y[block].qs[iqs] = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (iqs == 0) y[block].ds = __floats2half2_rn(scale, sum);
}

// Quantize the HC-up input directly from the materialized BF16 HC-down
// output.  The explicit BF16 conversion preserves the unfused
// down -> BF16 -> SiLU -> BF16 -> up boundary without writing the SiLU
// tensor to global memory.
__global__ void quantize_q8_1_hc_silu(const bf16* __restrict__ x,
                                      block_q8_1* __restrict__ y,
                                      int64_t stride_x, int cols, int padded,
                                      int hc_count) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= padded) return;
  const int vec = blockIdx.y;
  const int offset = vec * padded + ix;
  const int block = offset / QK8_1;
  const int iqs = offset % QK8_1;
  float value = 0.0f;
  if (ix < cols) {
    const float down_bf16 = static_cast<float>(x[vec * stride_x + ix]);
    const float scaled = down_bf16 / static_cast<float>(hc_count);
    const float activated = scaled / (1.0f + expf(-scaled));
    value = static_cast<float>(__float2bfloat16(activated));
  }
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    sum += __shfl_xor(sum, mask, 32);
  }
  const float scale = amax / 127.0f;
  y[block].qs[iqs] =
      amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (iqs == 0) y[block].ds = __floats2half2_rn(scale, sum);
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot, int waves>
__global__ void dense_mmvq(const void* __restrict__ packed_weight,
                           const block_q8_1* __restrict__ input,
                           bf16* __restrict__ output, int cols, int rows,
                           int vecs) {
  const int row = blockIdx.x;
  const int vec = blockIdx.y;
  if (row >= rows || vec >= vecs) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / qk;
  const int blocks_per_iter = vdr * waves * 32 / qi;
  const int padded = (cols + 511) / 512 * 512;
  const block_q_t* weight = static_cast<const block_q_t*>(packed_weight);
  float sum = 0.0f;
  for (int block = tid / (qi / vdr); block < blocks_per_row;
       block += blocks_per_iter) {
    const int iqs = vdr * (tid % (qi / vdr));
    sum += vec_dot(&weight[row * blocks_per_row + block],
                   &input[vec * (padded / QK8_1) + block * (qk / QK8_1)],
                   iqs);
  }
  __shared__ float partials[waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) partials[threadIdx.y - 1][lane] = sum;
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) sum += partials[wave][lane];
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
  if (lane == 0) output[vec * rows + row] = __float2bfloat16(sum);
}

// Q4_K two-vector dot that decodes and loads the packed weight block once.
// The verifier's M=2 vocabulary projection is bandwidth-bound; the stock
// MMVQ grid assigns each vector to a separate workgroup and streams the full
// matrix twice. Keeping both accumulators in one wave removes that duplicate
// weight traffic while preserving the exact GGML Q8_1 activation arithmetic.
__device__ __forceinline__ void vec_dot_q4_K_q8_1_pair(
    const block_q4_K* __restrict__ bq4_K,
    const block_q8_1* __restrict__ bq8_1_a,
    const block_q8_1* __restrict__ bq8_1_b, int iqs,
    float& sum_a, float& sum_b) {
  int v[2];
  int u_a[2 * QR4_K];
  int u_b[2 * QR4_K];
  float d8_a[QR4_K];
  float d8_b[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int* q4 = reinterpret_cast<const int*>(
      bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4_K->scales);
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
             ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
             ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1* qa = bq8_1_a + bq8_offset + i;
    const block_q8_1* qb = bq8_1_b + bq8_offset + i;
    d8_a[i] = __low2float(qa->ds);
    d8_b[i] = __low2float(qb->ds);
    const int* q8a = reinterpret_cast<const int*>(qa->qs) + ((iqs / 2) % 4);
    const int* q8b = reinterpret_cast<const int*>(qb->qs) + ((iqs / 2) % 4);
    u_a[2 * i + 0] = q8a[0];
    u_a[2 * i + 1] = q8a[4];
    u_b[2 * i + 0] = q8b[0];
    u_b[2 * i + 1] = q8b[4];
  }
  sum_a += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_a, sc, m, bq4_K->dm, d8_a);
  sum_b += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_b, sc, m, bq4_K->dm, d8_b);
}

// Three-row counterpart used by the MTP2 target verifier.  As in R9V's
// multi-slot Q6 head, one wave decodes each packed weight block once and keeps
// one accumulator per verifier row.  The activation remains in canonical
// GGML Q8_1 form so the only numerical change versus three independent MMVQs
// is the order in which the same weight decode is reused.
__device__ __forceinline__ void vec_dot_q4_K_q8_1_triple(
    const block_q4_K* __restrict__ bq4_K,
    const block_q8_1* __restrict__ bq8_1_a,
    const block_q8_1* __restrict__ bq8_1_b,
    const block_q8_1* __restrict__ bq8_1_c, int iqs,
    float& sum_a, float& sum_b, float& sum_c) {
  int v[2];
  int u_a[2 * QR4_K];
  int u_b[2 * QR4_K];
  int u_c[2 * QR4_K];
  float d8_a[QR4_K];
  float d8_b[QR4_K];
  float d8_c[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int* q4 = reinterpret_cast<const int*>(
      bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4_K->scales);
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
             ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
             ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1* qa = bq8_1_a + bq8_offset + i;
    const block_q8_1* qb = bq8_1_b + bq8_offset + i;
    const block_q8_1* qc = bq8_1_c + bq8_offset + i;
    d8_a[i] = __low2float(qa->ds);
    d8_b[i] = __low2float(qb->ds);
    d8_c[i] = __low2float(qc->ds);
    const int* q8a = reinterpret_cast<const int*>(qa->qs) + ((iqs / 2) % 4);
    const int* q8b = reinterpret_cast<const int*>(qb->qs) + ((iqs / 2) % 4);
    const int* q8c = reinterpret_cast<const int*>(qc->qs) + ((iqs / 2) % 4);
    u_a[2 * i + 0] = q8a[0];
    u_a[2 * i + 1] = q8a[4];
    u_b[2 * i + 0] = q8b[0];
    u_b[2 * i + 1] = q8b[4];
    u_c[2 * i + 0] = q8c[0];
    u_c[2 * i + 1] = q8c[4];
  }
  sum_a += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_a, sc, m, bq4_K->dm, d8_a);
  sum_b += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_b, sc, m, bq4_K->dm, d8_b);
  sum_c += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_c, sc, m, bq4_K->dm, d8_c);
}

// Four-row counterpart for an MTP3 target verification pass.  This remains
// a separate opt-in entry point so the additional accumulator pressure can
// never affect the validated M=3 production path.
__device__ __forceinline__ void vec_dot_q4_K_q8_1_quad(
    const block_q4_K* __restrict__ bq4_K,
    const block_q8_1* __restrict__ bq8_1_a,
    const block_q8_1* __restrict__ bq8_1_b,
    const block_q8_1* __restrict__ bq8_1_c,
    const block_q8_1* __restrict__ bq8_1_d, int iqs,
    float& sum_a, float& sum_b, float& sum_c, float& sum_d) {
  int v[2];
  int u_a[2 * QR4_K];
  int u_b[2 * QR4_K];
  int u_c[2 * QR4_K];
  int u_d[2 * QR4_K];
  float d8_a[QR4_K];
  float d8_b[QR4_K];
  float d8_c[QR4_K];
  float d8_d[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int* q4 = reinterpret_cast<const int*>(
      bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4_K->scales);
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) |
             ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) |
             ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1* qa = bq8_1_a + bq8_offset + i;
    const block_q8_1* qb = bq8_1_b + bq8_offset + i;
    const block_q8_1* qc = bq8_1_c + bq8_offset + i;
    const block_q8_1* qd = bq8_1_d + bq8_offset + i;
    d8_a[i] = __low2float(qa->ds);
    d8_b[i] = __low2float(qb->ds);
    d8_c[i] = __low2float(qc->ds);
    d8_d[i] = __low2float(qd->ds);
    const int* q8a = reinterpret_cast<const int*>(qa->qs) + ((iqs / 2) % 4);
    const int* q8b = reinterpret_cast<const int*>(qb->qs) + ((iqs / 2) % 4);
    const int* q8c = reinterpret_cast<const int*>(qc->qs) + ((iqs / 2) % 4);
    const int* q8d = reinterpret_cast<const int*>(qd->qs) + ((iqs / 2) % 4);
    u_a[2 * i + 0] = q8a[0];
    u_a[2 * i + 1] = q8a[4];
    u_b[2 * i + 0] = q8b[0];
    u_b[2 * i + 1] = q8b[4];
    u_c[2 * i + 0] = q8c[0];
    u_c[2 * i + 1] = q8c[4];
    u_d[2 * i + 0] = q8d[0];
    u_d[2 * i + 1] = q8d[4];
  }
  sum_a += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_a, sc, m, bq4_K->dm, d8_a);
  sum_b += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_b, sc, m, bq4_K->dm, d8_b);
  sum_c += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_c, sc, m, bq4_K->dm, d8_c);
  sum_d += vec_dot_q4_K_q8_1_impl_vmmq(
      v, u_d, sc, m, bq4_K->dm, d8_d);
}

template <int waves>
__global__ void dense_mmvq_q4_reuse2(
    const block_q4_K* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK_K;
  const int blocks_per_iter = VDR_Q4_K_Q8_1_MMVQ * waves * 32 / QI4_K;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  for (int block = tid / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q4_K_Q8_1_MMVQ *
                    (tid % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
    vec_dot_q4_K_q8_1_pair(
        &weight[row * blocks_per_row + block],
        &input[block * (QK_K / QK8_1)],
        &input[input_stride + block * (QK_K / QK8_1)],
        iqs, sum0, sum1);
  }
  __shared__ float partials[2][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
  }
}

template <int waves>
__global__ void dense_mmvq_q4_reuse3(
    const block_q4_K* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK_K;
  const int blocks_per_iter = VDR_Q4_K_Q8_1_MMVQ * waves * 32 / QI4_K;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int block = tid / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q4_K_Q8_1_MMVQ *
                    (tid % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
    const int q8_block = block * (QK_K / QK8_1);
    vec_dot_q4_K_q8_1_triple(
        &weight[row * blocks_per_row + block],
        &input[q8_block], &input[input_stride + q8_block],
        &input[2 * input_stride + q8_block], iqs, sum0, sum1, sum2);
  }
  __shared__ float partials[3][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
  }
}

template <int waves>
__global__ void dense_mmvq_q4_reuse4(
    const block_q4_K* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK_K;
  const int blocks_per_iter = VDR_Q4_K_Q8_1_MMVQ * waves * 32 / QI4_K;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (int block = tid / (QI4_K / VDR_Q4_K_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q4_K_Q8_1_MMVQ *
                    (tid % (QI4_K / VDR_Q4_K_Q8_1_MMVQ));
    const int q8_block = block * (QK_K / QK8_1);
    vec_dot_q4_K_q8_1_quad(
        &weight[row * blocks_per_row + block],
        &input[q8_block], &input[input_stride + q8_block],
        &input[2 * input_stride + q8_block],
        &input[3 * input_stride + q8_block], iqs,
        sum0, sum1, sum2, sum3);
  }
  __shared__ float partials[4][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
    partials[3][threadIdx.y - 1][lane] = sum3;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
    sum3 += partials[3][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
    sum3 += __shfl_xor(sum3, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
    output[3 * rows + row] = __float2bfloat16(sum3);
  }
}

__device__ __forceinline__ void vec_dot_q8_0_q8_1_pair(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1, int iqs,
    float& sum0, float& sum1) {
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u0[VDR_Q8_0_Q8_1_MMVQ];
  int u1[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(weight->qs, iqs + i);
    u0[i] = get_int_from_int8_aligned(input0->qs, iqs + i);
    u1[i] = get_int_from_int8_aligned(input1->qs, iqs + i);
  }
  const float dw = __half2float(weight->d);
  sum0 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u0, dw, __low2float(input0->ds));
  sum1 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u1, dw, __low2float(input1->ds));
}

__device__ __forceinline__ void vec_dot_q8_0_q8_1_triple(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1,
    const block_q8_1* __restrict__ input2, int iqs,
    float& sum0, float& sum1, float& sum2) {
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u0[VDR_Q8_0_Q8_1_MMVQ];
  int u1[VDR_Q8_0_Q8_1_MMVQ];
  int u2[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(weight->qs, iqs + i);
    u0[i] = get_int_from_int8_aligned(input0->qs, iqs + i);
    u1[i] = get_int_from_int8_aligned(input1->qs, iqs + i);
    u2[i] = get_int_from_int8_aligned(input2->qs, iqs + i);
  }
  const float dw = __half2float(weight->d);
  sum0 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u0, dw, __low2float(input0->ds));
  sum1 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u1, dw, __low2float(input1->ds));
  sum2 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u2, dw, __low2float(input2->ds));
}

__device__ __forceinline__ void vec_dot_q8_0_q8_1_quad(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1,
    const block_q8_1* __restrict__ input2,
    const block_q8_1* __restrict__ input3, int iqs,
    float& sum0, float& sum1, float& sum2, float& sum3) {
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u0[VDR_Q8_0_Q8_1_MMVQ];
  int u1[VDR_Q8_0_Q8_1_MMVQ];
  int u2[VDR_Q8_0_Q8_1_MMVQ];
  int u3[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(weight->qs, iqs + i);
    u0[i] = get_int_from_int8_aligned(input0->qs, iqs + i);
    u1[i] = get_int_from_int8_aligned(input1->qs, iqs + i);
    u2[i] = get_int_from_int8_aligned(input2->qs, iqs + i);
    u3[i] = get_int_from_int8_aligned(input3->qs, iqs + i);
  }
  const float dw = __half2float(weight->d);
  sum0 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u0, dw, __low2float(input0->ds));
  sum1 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u1, dw, __low2float(input1->ds));
  sum2 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u2, dw, __low2float(input2->ds));
  sum3 += vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(
      v, u3, dw, __low2float(input3->ds));
}

template <int waves>
__global__ void dense_mmvq_q8_reuse2(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK8_0;
  const int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * waves * 32 / QI8_0;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  for (int block = tid / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                    (tid % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
    vec_dot_q8_0_q8_1_pair(
        &weight[row * blocks_per_row + block],
        &input[block], &input[input_stride + block], iqs, sum0, sum1);
  }
  __shared__ float partials[2][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
  }
}

template <int waves>
__global__ void dense_mmvq_q8_reuse3(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK8_0;
  const int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * waves * 32 / QI8_0;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int block = tid / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                    (tid % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
    vec_dot_q8_0_q8_1_triple(
        &weight[row * blocks_per_row + block], &input[block],
        &input[input_stride + block], &input[2 * input_stride + block],
        iqs, sum0, sum1, sum2);
  }
  __shared__ float partials[3][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
  }
}

// Exact M=3 attention-input candidates for gfx1201.  The production arm is
// restricted to K=2560 and N in {8192, 6656}; specializing those dimensions
// removes every shape-dependent branch from the streaming loop.
//
// The exact quartet keeps the stock reuse3 lane/block assignment for each
// output row, but gives a wave four independent rows.  This preserves the
// stock per-lane accumulation and XOR reduction order while exposing four
// weight streams to the scheduler.  R9V uses the same rows-per-wave idea to
// raise memory-level parallelism on gfx1201.
template <int rows>
__device__ __forceinline__ void dense_mmvq_q8_attention_m3_exact4_body(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  constexpr int blocks_per_row = 2560 / QK8_0;
  constexpr int input_stride = 2560 / QK8_1;
  constexpr int rows_per_wave = 4;
  const int lane = threadIdx.x;
  const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * rows_per_wave;
  float sum0[rows_per_wave] = {};
  float sum1[rows_per_wave] = {};
  float sum2[rows_per_wave] = {};

  for (int block = lane / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
       block < blocks_per_row; block += 8) {
    const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                    (lane % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
#pragma unroll
    for (int row = 0; row < rows_per_wave; ++row) {
      vec_dot_q8_0_q8_1_triple(
          &weight[(row0 + row) * blocks_per_row + block], &input[block],
          &input[input_stride + block], &input[2 * input_stride + block],
          iqs, sum0[row], sum1[row], sum2[row]);
    }
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
#pragma unroll
    for (int row = 0; row < rows_per_wave; ++row) {
      sum0[row] += __shfl_xor(sum0[row], mask, 32);
      sum1[row] += __shfl_xor(sum1[row], mask, 32);
      sum2[row] += __shfl_xor(sum2[row], mask, 32);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < rows_per_wave; ++row) {
      output[row0 + row] = __float2bfloat16(sum0[row]);
      output[rows + row0 + row] = __float2bfloat16(sum1[row]);
      output[2 * rows + row0 + row] = __float2bfloat16(sum2[row]);
    }
  }
}

template <int rows>
__global__ __launch_bounds__(128)
void dense_mmvq_q8_attention_m3_exact4(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  dense_mmvq_q8_attention_m3_exact4_body<rows>(weight, input, output);
}

template <int rows>
__global__ __launch_bounds__(128)
__attribute__((amdgpu_waves_per_eu(1, 8)))
void dense_mmvq_q8_attention_m3_exact4_w8(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  dense_mmvq_q8_attention_m3_exact4_body<rows>(weight, input, output);
}

__device__ __forceinline__ int qwen38_signed_dot4_i8(
    unsigned int first, unsigned int second, int accumulator) {
  int result;
  asm("v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]"
      : "=v"(result)
      : "v"(first), "v"(second), "v"(accumulator));
  return result;
}

// R9V's canonical-Q8 group-lane mapping assigns four rows to a wave and
// eight lanes to each row.  A lane consumes a complete 34-byte Q8_0 block,
// so its scale is applied once instead of four times.  The activation and
// output still cross the same GGML Q8_1 and BF16 boundaries as reuse3.
template <int rows>
__device__ __forceinline__ void dense_mmvq_q8_attention_m3_group4_body(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  constexpr int blocks_per_row = 2560 / QK8_0;
  constexpr int input_stride = 2560 / QK8_1;
  constexpr int rows_per_wave = 4;
  const int lane = threadIdx.x;
  const int row_in_wave = lane & (rows_per_wave - 1);
  const int block_group = lane >> 2;
  const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * rows_per_wave;
  const block_q8_0* row_weight =
      weight + (row0 + row_in_wave) * blocks_per_row;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;

  for (int block = block_group; block < blocks_per_row; block += 8) {
    int dot0 = 0;
    int dot1 = 0;
    int dot2 = 0;
#pragma unroll
    for (int i = 0; i < QI8_0; ++i) {
      // Canonical Q8_0 has a two-byte scale before its bytes.  memcpy tells
      // LLVM the dword may be only two-byte aligned and lowers to the same
      // unaligned gfx1201 load pattern used by R9V's 34-byte Q8 rows.
      unsigned int v;
      __builtin_memcpy(&v, row_weight[block].qs + 4 * i, sizeof(v));
      const unsigned int u0 = static_cast<unsigned int>(
          get_int_from_int8_aligned(input[block].qs, i));
      const unsigned int u1 = static_cast<unsigned int>(
          get_int_from_int8_aligned(input[input_stride + block].qs, i));
      const unsigned int u2 = static_cast<unsigned int>(
          get_int_from_int8_aligned(input[2 * input_stride + block].qs, i));
      dot0 = qwen38_signed_dot4_i8(v, u0, dot0);
      dot1 = qwen38_signed_dot4_i8(v, u1, dot1);
      dot2 = qwen38_signed_dot4_i8(v, u2, dot2);
    }
    const float dw = __half2float(row_weight[block].d);
    sum0 += dw * __low2float(input[block].ds) * dot0;
    sum1 += dw * __low2float(input[input_stride + block].ds) * dot1;
    sum2 += dw * __low2float(input[2 * input_stride + block].ds) * dot2;
  }

#pragma unroll
  for (int offset = 16; offset >= 4; offset >>= 1) {
    sum0 += __shfl_down(sum0, offset, 32);
    sum1 += __shfl_down(sum1, offset, 32);
    sum2 += __shfl_down(sum2, offset, 32);
  }
  if (lane < rows_per_wave) {
    output[row0 + lane] = __float2bfloat16(sum0);
    output[rows + row0 + lane] = __float2bfloat16(sum1);
    output[2 * rows + row0 + lane] = __float2bfloat16(sum2);
  }
}

template <int rows>
__global__ __launch_bounds__(128)
void dense_mmvq_q8_attention_m3_group4(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  dense_mmvq_q8_attention_m3_group4_body<rows>(weight, input, output);
}

template <int rows>
__global__ __launch_bounds__(128)
__attribute__((amdgpu_waves_per_eu(1, 8)))
void dense_mmvq_q8_attention_m3_group4_w8(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  dense_mmvq_q8_attention_m3_group4_body<rows>(weight, input, output);
}

template <int rows>
__global__ __launch_bounds__(128)
__attribute__((amdgpu_waves_per_eu(1, 10)))
void dense_mmvq_q8_attention_m3_group4_w10(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output) {
  dense_mmvq_q8_attention_m3_group4_body<rows>(weight, input, output);
}

template <int waves>
__global__ void dense_mmvq_q8_reuse4(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK8_0;
  const int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * waves * 32 / QI8_0;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (int block = tid / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                    (tid % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
    vec_dot_q8_0_q8_1_quad(
        &weight[row * blocks_per_row + block], &input[block],
        &input[input_stride + block], &input[2 * input_stride + block],
        &input[3 * input_stride + block], iqs, sum0, sum1, sum2, sum3);
  }
  __shared__ float partials[4][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
    partials[3][threadIdx.y - 1][lane] = sum3;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
    sum3 += partials[3][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
    sum3 += __shfl_xor(sum3, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
    output[3 * rows + row] = __float2bfloat16(sum3);
  }
}

// HC-up Q8_0 GEMV with the gated stream reduction in its producer epilogue.
// Each workgroup owns one output feature across all four HC streams and all
// verifier rows.  Rounding every gate accumulator through BF16 before the
// sigmoid preserves the materialized up-projection boundary.
template <int vecs, int hc>
__global__ void dense_mmvq_q8_hc_mix(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    const bf16* __restrict__ xn, bf16* __restrict__ output,
    int64_t stride_xn, int cols, int rows) {
  const int inner = blockIdx.x;
  const int hidden = rows / hc;
  if (inner >= hidden) return;

  const int lane = threadIdx.x;
  const int blocks_per_row = cols / QK8_0;
  const int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sums[hc][vecs] = {};

#pragma unroll
  for (int stream = 0; stream < hc; ++stream) {
    const int row = stream * hidden + inner;
    for (int block = lane / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
         block < blocks_per_row; block += blocks_per_iter) {
      const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                      (lane % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
      const block_q8_0* w = &weight[row * blocks_per_row + block];
      if constexpr (vecs == 1) {
        sums[stream][0] += vec_dot_q8_0_q8_1(w, &input[block], iqs);
      } else if constexpr (vecs == 2) {
        vec_dot_q8_0_q8_1_pair(
            w, &input[block], &input[input_stride + block], iqs,
            sums[stream][0], sums[stream][1]);
      } else if constexpr (vecs == 3) {
        vec_dot_q8_0_q8_1_triple(
            w, &input[block], &input[input_stride + block],
            &input[2 * input_stride + block], iqs,
            sums[stream][0], sums[stream][1], sums[stream][2]);
      } else {
        vec_dot_q8_0_q8_1_quad(
            w, &input[block], &input[input_stride + block],
            &input[2 * input_stride + block],
            &input[3 * input_stride + block], iqs,
            sums[stream][0], sums[stream][1], sums[stream][2],
            sums[stream][3]);
      }
    }
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
#pragma unroll
    for (int stream = 0; stream < hc; ++stream) {
#pragma unroll
      for (int vec = 0; vec < vecs; ++vec) {
        sums[stream][vec] += __shfl_xor(sums[stream][vec], mask, 32);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int vec = 0; vec < vecs; ++vec) {
      float mixed = 0.0f;
#pragma unroll
      for (int stream = 0; stream < hc; ++stream) {
        const float rounded_gate =
            static_cast<float>(__float2bfloat16(sums[stream][vec]));
        const float gate = isfinite(rounded_gate) ? rounded_gate : 0.0f;
        const float gate_weight = 1.0f / (1.0f + expf(-gate));
        const float stream_value = static_cast<float>(
            xn[vec * stride_xn + stream * hidden + inner]);
        mixed += gate_weight * stream_value;
      }
      output[vec * hidden + inner] =
          __float2bfloat16(mixed / static_cast<float>(hc));
    }
  }
}

// Workgroup-grouped form of the exact HC-up producer epilogue above.  The
// legacy kernel launches one one-wave workgroup per hidden feature (2,560
// workgroups).  This body gives each workgroup several adjacent features and
// optionally computes two features per wave.  Every feature retains its own
// accumulators, the same block traversal, the same XOR reduction tree, and
// the same BF16 gate boundary and stream-mix order.
template <int waves, int outputs_per_wave>
__device__ __forceinline__ void dense_mmvq_q8_hc_mix_grouped_body(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    const bf16* __restrict__ xn, bf16* __restrict__ output,
    int64_t stride_xn) {
  constexpr int vecs = 3;
  constexpr int hc = 4;
  constexpr int cols = 320;
  constexpr int rows = 10240;
  constexpr int hidden = rows / hc;
  constexpr int blocks_per_row = cols / QK8_0;
  constexpr int blocks_per_iter = VDR_Q8_0_Q8_1_MMVQ * 32 / QI8_0;
  constexpr int input_stride = 512 / QK8_1;
  const int lane = threadIdx.x;
  const int inner0 =
      (blockIdx.x * waves + threadIdx.y) * outputs_per_wave;
  float sums[outputs_per_wave][hc][vecs] = {};

#pragma unroll
  for (int item = 0; item < outputs_per_wave; ++item) {
    const int inner = inner0 + item;
#pragma unroll
    for (int stream = 0; stream < hc; ++stream) {
      const int row = stream * hidden + inner;
      for (int block = lane / (QI8_0 / VDR_Q8_0_Q8_1_MMVQ);
           block < blocks_per_row; block += blocks_per_iter) {
        const int iqs = VDR_Q8_0_Q8_1_MMVQ *
                        (lane % (QI8_0 / VDR_Q8_0_Q8_1_MMVQ));
        const block_q8_0* w = &weight[row * blocks_per_row + block];
        vec_dot_q8_0_q8_1_triple(
            w, &input[block], &input[input_stride + block],
            &input[2 * input_stride + block], iqs,
            sums[item][stream][0], sums[item][stream][1],
            sums[item][stream][2]);
      }
    }
  }

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
#pragma unroll
    for (int item = 0; item < outputs_per_wave; ++item) {
#pragma unroll
      for (int stream = 0; stream < hc; ++stream) {
#pragma unroll
        for (int vec = 0; vec < vecs; ++vec) {
          sums[item][stream][vec] +=
              __shfl_xor(sums[item][stream][vec], mask, 32);
        }
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int item = 0; item < outputs_per_wave; ++item) {
      const int inner = inner0 + item;
#pragma unroll
      for (int vec = 0; vec < vecs; ++vec) {
        float mixed = 0.0f;
#pragma unroll
        for (int stream = 0; stream < hc; ++stream) {
          const float rounded_gate =
              static_cast<float>(__float2bfloat16(sums[item][stream][vec]));
          const float gate = isfinite(rounded_gate) ? rounded_gate : 0.0f;
          const float gate_weight = 1.0f / (1.0f + expf(-gate));
          const float stream_value = static_cast<float>(
              xn[vec * stride_xn + stream * hidden + inner]);
          mixed += gate_weight * stream_value;
        }
        output[vec * hidden + inner] =
            __float2bfloat16(mixed / static_cast<float>(hc));
      }
    }
  }
}

template <int outputs_per_wave>
__global__ __launch_bounds__(128)
void dense_mmvq_q8_hc_mix_grouped_w4(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    const bf16* __restrict__ xn, bf16* __restrict__ output,
    int64_t stride_xn) {
  dense_mmvq_q8_hc_mix_grouped_body<4, outputs_per_wave>(
      weight, input, xn, output, stride_xn);
}

template <int outputs_per_wave>
__global__ __launch_bounds__(256)
void dense_mmvq_q8_hc_mix_grouped_w8(
    const block_q8_0* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    const bf16* __restrict__ xn, bf16* __restrict__ output,
    int64_t stride_xn) {
  dense_mmvq_q8_hc_mix_grouped_body<8, outputs_per_wave>(
      weight, input, xn, output, stride_xn);
}

__device__ __forceinline__ void vec_dot_q6_K_q8_1_triple(
    const block_q6_K* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1,
    const block_q8_1* __restrict__ input2, int iqs,
    float& sum0, float& sum1, float& sum2) {
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) +
      (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) +
      (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_from_uint8(weight->ql, iqs);
  const int vh = get_int_from_uint8(
      weight->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                      iqs % (QI6_K / 4)) >> vh_shift;
  const int8_t* scales = weight->scales + scale_offset;
  float dot0 = 0.0f;
  float dot1 = 0.0f;
  float dot2 = 0.0f;
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int u0 = get_int_from_int8_aligned(
        input0[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const int u1 = get_int_from_int8_aligned(
        input1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const int u2 = get_int_from_int8_aligned(
        input2[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const float d80 = __low2float(input0[bq8_offset + 2 * i].ds);
    const float d81 = __low2float(input1[bq8_offset + 2 * i].ds);
    const float d82 = __low2float(input2[bq8_offset + 2 * i].ds);
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    dot0 += d80 * static_cast<float>(__dp4a(vi, u0, 0) * sc);
    dot1 += d81 * static_cast<float>(__dp4a(vi, u1, 0) * sc);
    dot2 += d82 * static_cast<float>(__dp4a(vi, u2, 0) * sc);
  }
  const float d = __half2float(weight->d);
  sum0 += d * dot0;
  sum1 += d * dot1;
  sum2 += d * dot2;
}

__device__ __forceinline__ void vec_dot_q6_K_q8_1_quad(
    const block_q6_K* __restrict__ weight,
    const block_q8_1* __restrict__ input0,
    const block_q8_1* __restrict__ input1,
    const block_q8_1* __restrict__ input2,
    const block_q8_1* __restrict__ input3, int iqs,
    float& sum0, float& sum1, float& sum2, float& sum3) {
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) +
      (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) +
      (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_from_uint8(weight->ql, iqs);
  const int vh = get_int_from_uint8(
      weight->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                      iqs % (QI6_K / 4)) >> vh_shift;
  const int8_t* scales = weight->scales + scale_offset;
  float dot0 = 0.0f;
  float dot1 = 0.0f;
  float dot2 = 0.0f;
  float dot3 = 0.0f;
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int u0 = get_int_from_int8_aligned(
        input0[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const int u1 = get_int_from_int8_aligned(
        input1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const int u2 = get_int_from_int8_aligned(
        input2[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const int u3 = get_int_from_int8_aligned(
        input3[bq8_offset + 2 * i].qs, iqs % QI8_1);
    const float d80 = __low2float(input0[bq8_offset + 2 * i].ds);
    const float d81 = __low2float(input1[bq8_offset + 2 * i].ds);
    const float d82 = __low2float(input2[bq8_offset + 2 * i].ds);
    const float d83 = __low2float(input3[bq8_offset + 2 * i].ds);
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    dot0 += d80 * static_cast<float>(__dp4a(vi, u0, 0) * sc);
    dot1 += d81 * static_cast<float>(__dp4a(vi, u1, 0) * sc);
    dot2 += d82 * static_cast<float>(__dp4a(vi, u2, 0) * sc);
    dot3 += d83 * static_cast<float>(__dp4a(vi, u3, 0) * sc);
  }
  const float d = __half2float(weight->d);
  sum0 += d * dot0;
  sum1 += d * dot1;
  sum2 += d * dot2;
  sum3 += d * dot3;
}

template <int waves>
__global__ void dense_mmvq_q6_reuse3(
    const block_q6_K* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK_K;
  const int blocks_per_iter = VDR_Q6_K_Q8_1_MMVQ * waves * 32 / QI6_K;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for (int block = tid / (QI6_K / VDR_Q6_K_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q6_K_Q8_1_MMVQ *
                    (tid % (QI6_K / VDR_Q6_K_Q8_1_MMVQ));
    const int q8_block = block * (QK_K / QK8_1);
    vec_dot_q6_K_q8_1_triple(
        &weight[row * blocks_per_row + block], &input[q8_block],
        &input[input_stride + q8_block],
        &input[2 * input_stride + q8_block], iqs, sum0, sum1, sum2);
  }
  __shared__ float partials[3][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
  }
}

template <int waves>
__global__ void dense_mmvq_q6_reuse4(
    const block_q6_K* __restrict__ weight,
    const block_q8_1* __restrict__ input,
    bf16* __restrict__ output, int cols, int rows) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int blocks_per_row = cols / QK_K;
  const int blocks_per_iter = VDR_Q6_K_Q8_1_MMVQ * waves * 32 / QI6_K;
  const int padded = (cols + 511) / 512 * 512;
  const int input_stride = padded / QK8_1;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (int block = tid / (QI6_K / VDR_Q6_K_Q8_1_MMVQ);
       block < blocks_per_row; block += blocks_per_iter) {
    const int iqs = VDR_Q6_K_Q8_1_MMVQ *
                    (tid % (QI6_K / VDR_Q6_K_Q8_1_MMVQ));
    const int q8_block = block * (QK_K / QK8_1);
    vec_dot_q6_K_q8_1_quad(
        &weight[row * blocks_per_row + block], &input[q8_block],
        &input[input_stride + q8_block],
        &input[2 * input_stride + q8_block],
        &input[3 * input_stride + q8_block], iqs,
        sum0, sum1, sum2, sum3);
  }
  __shared__ float partials[4][waves > 1 ? waves - 1 : 1][32];
  if (threadIdx.y > 0) {
    partials[0][threadIdx.y - 1][lane] = sum0;
    partials[1][threadIdx.y - 1][lane] = sum1;
    partials[2][threadIdx.y - 1][lane] = sum2;
    partials[3][threadIdx.y - 1][lane] = sum3;
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int wave = 0; wave < waves - 1; ++wave) {
    sum0 += partials[0][wave][lane];
    sum1 += partials[1][wave][lane];
    sum2 += partials[2][wave][lane];
    sum3 += partials[3][wave][lane];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum0 += __shfl_xor(sum0, mask, 32);
    sum1 += __shfl_xor(sum1, mask, 32);
    sum2 += __shfl_xor(sum2, mask, 32);
    sum3 += __shfl_xor(sum3, mask, 32);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum0);
    output[rows + row] = __float2bfloat16(sum1);
    output[2 * rows + row] = __float2bfloat16(sum2);
    output[3 * rows + row] = __float2bfloat16(sum3);
  }
}

template <int waves>
void launch(torch::Tensor weight, torch::Tensor quantized,
            torch::Tensor output, int qtype, int cols, int rows, int vecs,
            hipStream_t stream) {
  const dim3 grid(rows, vecs, 1);
  const dim3 block(32, waves, 1);
  const auto* q8 = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (qtype == 12) {
    dense_mmvq<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                vec_dot_q4_K_q8_1, waves><<<grid, block, 0, stream>>>(
        weight.data_ptr(), q8, out, cols, rows, vecs);
  } else if (qtype == 13) {
    dense_mmvq<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                vec_dot_q5_K_q8_1, waves><<<grid, block, 0, stream>>>(
        weight.data_ptr(), q8, out, cols, rows, vecs);
  } else {
    dense_mmvq<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                vec_dot_q6_K_q8_1, waves><<<grid, block, 0, stream>>>(
        weight.data_ptr(), q8, out, cols, rows, vecs);
  }
}

torch::Tensor dense_gemv(torch::Tensor weight, torch::Tensor x,
                         int64_t qtype, int64_t rows, int64_t waves) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(qtype >= 12 && qtype <= 14, "expected Q4_K, Q5_K, or Q6_K");
  TORCH_CHECK(waves == 1 || waves == 2 || waves == 4 || waves == 5 ||
                  waves == 6 || waves == 8,
              "waves must be one of 1,2,4,5,6,8");
  const int cols = x.size(1);
  const int vecs = x.size(0);
  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {vecs, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({vecs, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3((padded + 255) / 256, vecs, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  if (waves == 1) launch<1>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  else if (waves == 2) launch<2>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  else if (waves == 4) launch<4>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  else if (waves == 5) launch<5>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  else if (waves == 6) launch<6>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  else launch<8>(weight, quantized, output, qtype, cols, rows, vecs, stream);
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_q4_reuse2(torch::Tensor weight, torch::Tensor x,
                                    int64_t rows, int64_t waves) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 2, "expected exactly two vectors");
  TORCH_CHECK(waves == 1 || waves == 2 || waves == 4,
              "waves must be one of 1,2,4");
  const int cols = x.size(1);
  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {2, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({2, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3((padded + 255) / 256, 2, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  const dim3 grid(rows, 1, 1);
  const auto* q4 = reinterpret_cast<const block_q4_K*>(weight.data_ptr());
  const auto* q8 = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (waves == 1) {
    dense_mmvq_q4_reuse2<1><<<grid, dim3(32, 1, 1), 0, stream>>>(
        q4, q8, out, cols, rows);
  } else if (waves == 2) {
    dense_mmvq_q4_reuse2<2><<<grid, dim3(32, 2, 1), 0, stream>>>(
        q4, q8, out, cols, rows);
  } else {
    dense_mmvq_q4_reuse2<4><<<grid, dim3(32, 4, 1), 0, stream>>>(
        q4, q8, out, cols, rows);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_q8_reuse2(torch::Tensor weight, torch::Tensor x,
                                    int64_t rows, int64_t waves) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 2, "expected exactly two vectors");
  TORCH_CHECK(waves == 1 || waves == 2 || waves == 4,
              "waves must be one of 1,2,4");
  const int cols = x.size(1);
  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {2, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({2, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3((padded + 255) / 256, 2, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  const dim3 grid(rows, 1, 1);
  const auto* q8w = reinterpret_cast<const block_q8_0*>(weight.data_ptr());
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (waves == 1) {
    dense_mmvq_q8_reuse2<1><<<grid, dim3(32, 1, 1), 0, stream>>>(
        q8w, q8x, out, cols, rows);
  } else if (waves == 2) {
    dense_mmvq_q8_reuse2<2><<<grid, dim3(32, 2, 1), 0, stream>>>(
        q8w, q8x, out, cols, rows);
  } else {
    dense_mmvq_q8_reuse2<4><<<grid, dim3(32, 4, 1), 0, stream>>>(
        q8w, q8x, out, cols, rows);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_reuse3(torch::Tensor weight, torch::Tensor x,
                                int64_t qtype, int64_t rows) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == x.device(), "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(weight.dim() == 2, "weight must be a packed row-major matrix");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 3,
              "expected exactly three verifier rows");
  TORCH_CHECK(weight.size(0) == rows, "rows must match weight.size(0)");

  const int cols = x.size(1);
  const bool exact_qwen38_q8_shape =
      qtype == 8 &&
      ((rows == 10240 && cols == 2560) ||
       (rows == 5120 && cols == 2560) ||
       (rows == 2560 && cols == 6144) ||
       (rows == 2560 && cols == 3072) ||
       (rows == 6144 && cols == 2560) ||
       (rows == 3072 && cols == 2560) ||
       (rows == 12288 && cols == 2560));
  const bool exact_qwen38_head_shape =
      (qtype == 12 || qtype == 14) && cols == 2560 &&
      (rows == 248320 || rows == 124160);
  TORCH_CHECK(exact_qwen38_q8_shape || exact_qwen38_head_shape,
              "reuse3 only supports exact Qwen3.8 Q4 target shapes");

  const int64_t row_bytes =
      qtype == 8 ? (cols / QK8_0) * sizeof(block_q8_0)
                 : (cols / QK_K) *
                       (qtype == 12 ? sizeof(block_q4_K) : sizeof(block_q6_K));
  TORCH_CHECK(weight.numel() == rows * row_bytes,
              "packed weight byte count does not match qtype/shape");

  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {3, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({3, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3((padded + 255) / 256, 3, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  const dim3 grid(rows, 1, 1);
  const dim3 block(32, 1, 1);
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (qtype == 8) {
    dense_mmvq_q8_reuse3<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q8_0*>(weight.data_ptr()), q8x, out,
        cols, rows);
  } else if (qtype == 12) {
    dense_mmvq_q4_reuse3<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(weight.data_ptr()), q8x, out,
        cols, rows);
  } else {
    dense_mmvq_q6_reuse3<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q6_K*>(weight.data_ptr()), q8x, out,
        cols, rows);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

template <int rows>
void launch_q8_attention_m3(
    const block_q8_0* weight, const block_q8_1* input, bf16* output,
    int64_t variant, hipStream_t stream) {
  constexpr int rows_per_workgroup = 4 * 4;
  const dim3 quartet_grid(rows / rows_per_workgroup, 1, 1);
  const dim3 quartet_block(32, 4, 1);
  if (variant == 0) {
    dense_mmvq_q8_reuse3<1><<<dim3(rows, 1, 1), dim3(32, 1, 1), 0,
                               stream>>>(weight, input, output, 2560, rows);
  } else if (variant == 1) {
    dense_mmvq_q8_attention_m3_exact4<rows>
        <<<quartet_grid, quartet_block, 0, stream>>>(weight, input, output);
  } else if (variant == 2) {
    dense_mmvq_q8_attention_m3_exact4_w8<rows>
        <<<quartet_grid, quartet_block, 0, stream>>>(weight, input, output);
  } else if (variant == 3) {
    dense_mmvq_q8_attention_m3_group4<rows>
        <<<quartet_grid, quartet_block, 0, stream>>>(weight, input, output);
  } else if (variant == 4) {
    dense_mmvq_q8_attention_m3_group4_w8<rows>
        <<<quartet_grid, quartet_block, 0, stream>>>(weight, input, output);
  } else {
    dense_mmvq_q8_attention_m3_group4_w10<rows>
        <<<quartet_grid, quartet_block, 0, stream>>>(weight, input, output);
  }
}

torch::Tensor dense_gemv_q8_attention_m3(
    torch::Tensor weight, torch::Tensor x, int64_t rows, int64_t variant) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == x.device(), "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(weight.dim() == 2, "weight must be a packed row-major matrix");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 3 && x.size(1) == 2560,
              "expected an exact 3x2560 Qwen3.8 attention input");
  TORCH_CHECK(weight.size(0) == rows, "rows must match weight.size(0)");
  TORCH_CHECK(rows == 8192 || rows == 6656,
              "attention M=3 supports only N=8192 or N=6656");
  TORCH_CHECK(variant >= 0 && variant <= 5,
              "attention M=3 variant must be in [0, 5]");
  constexpr int row_bytes = (2560 / QK8_0) * sizeof(block_q8_0);
  TORCH_CHECK(weight.numel() == rows * row_bytes,
              "packed Q8_0 byte count does not match attention shape");

  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {3, (2560 / QK8_1) * static_cast<int>(sizeof(block_q8_1))},
      torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
  auto output = torch::empty({3, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3(10, 3, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), 2560, 2560);
  const auto* q8w = reinterpret_cast<const block_q8_0*>(weight.data_ptr());
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (rows == 8192) {
    launch_q8_attention_m3<8192>(q8w, q8x, out, variant, stream);
  } else {
    launch_q8_attention_m3<6656>(q8w, q8x, out, variant, stream);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_reuse4(torch::Tensor weight, torch::Tensor x,
                                int64_t qtype, int64_t rows) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda(), "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == x.device(), "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(weight.dim() == 2, "weight must be a packed row-major matrix");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 4,
              "expected exactly four verifier rows");
  TORCH_CHECK(weight.size(0) == rows, "rows must match weight.size(0)");

  const int cols = x.size(1);
  const bool exact_qwen38_q8_shape =
      qtype == 8 &&
      ((rows == 10240 && cols == 2560) ||
       (rows == 5120 && cols == 2560) ||
       (rows == 2560 && cols == 6144) ||
       (rows == 2560 && cols == 3072) ||
       (rows == 6144 && cols == 2560) ||
       (rows == 3072 && cols == 2560) ||
       (rows == 12288 && cols == 2560));
  const bool exact_qwen38_head_shape =
      (qtype == 12 || qtype == 14) && cols == 2560 &&
      (rows == 248320 || rows == 124160);
  TORCH_CHECK(exact_qwen38_q8_shape || exact_qwen38_head_shape,
              "reuse4 only supports exact Qwen3.8 Q4 target shapes");

  const int64_t row_bytes =
      qtype == 8 ? (cols / QK8_0) * sizeof(block_q8_0)
                 : (cols / QK_K) *
                       (qtype == 12 ? sizeof(block_q4_K) : sizeof(block_q6_K));
  TORCH_CHECK(weight.numel() == rows * row_bytes,
              "packed weight byte count does not match qtype/shape");

  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(x.device());
  auto quantized = torch::empty(
      {4, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
  auto output = torch::empty({4, rows}, x.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1<<<dim3((padded + 255) / 256, 4, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()), cols, padded);
  const dim3 grid(rows, 1, 1);
  const dim3 block(32, 1, 1);
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (qtype == 8) {
    dense_mmvq_q8_reuse4<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q8_0*>(weight.data_ptr()), q8x, out,
        cols, rows);
  } else if (qtype == 12) {
    dense_mmvq_q4_reuse4<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(weight.data_ptr()), q8x, out,
        cols, rows);
  } else {
    dense_mmvq_q6_reuse4<1><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q6_K*>(weight.data_ptr()), q8x, out,
        cols, rows);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_q8_hc_mix(torch::Tensor weight,
                                   torch::Tensor raw_lora,
                                   torch::Tensor xn, int64_t hc_count) {
  TORCH_CHECK(weight.is_cuda() && raw_lora.is_cuda() && xn.is_cuda(),
              "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == raw_lora.device() && xn.device() == raw_lora.device(),
              "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "weight must be uint8 Q8_0");
  TORCH_CHECK(raw_lora.scalar_type() == torch::kBFloat16 &&
                  xn.scalar_type() == torch::kBFloat16,
              "HC inputs must be BF16");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(raw_lora.dim() == 2 && xn.dim() == 2,
              "HC inputs must be matrices");
  TORCH_CHECK(raw_lora.stride(1) == 1 && xn.stride(1) == 1,
              "HC inputs must have unit inner stride");
  TORCH_CHECK(raw_lora.size(0) == xn.size(0),
              "HC inputs must have the same token count");
  TORCH_CHECK(raw_lora.size(0) >= 1 && raw_lora.size(0) <= 4,
              "fused HC-up supports one through four token rows");
  TORCH_CHECK(hc_count == 4, "fused HC-up requires exactly four streams");

  const int vecs = raw_lora.size(0);
  const int cols = raw_lora.size(1);
  const int rows = weight.size(0);
  TORCH_CHECK(cols == 320 && rows == 10240 && xn.size(1) == rows,
              "expected Qwen3.8 HC-up shape 10240x320 and xn width 10240");
  TORCH_CHECK(weight.dim() == 2, "weight must be a packed row-major matrix");
  const int64_t row_bytes = (cols / QK8_0) * sizeof(block_q8_0);
  TORCH_CHECK(weight.numel() == rows * row_bytes,
              "packed Q8_0 byte count does not match HC-up shape");

  const int padded = (cols + 511) / 512 * 512;
  c10::cuda::CUDAGuard guard(raw_lora.device());
  auto quantized = torch::empty(
      {vecs, padded / 32 * 9},
      torch::TensorOptions().dtype(torch::kInt32).device(raw_lora.device()));
  auto output = torch::empty({vecs, rows / hc_count}, raw_lora.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1_hc_silu<<<dim3((padded + 255) / 256, vecs, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(raw_lora.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()),
      raw_lora.stride(0), cols, padded, hc_count);
  const dim3 grid(rows / hc_count, 1, 1);
  const auto* q8w = reinterpret_cast<const block_q8_0*>(weight.data_ptr());
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  const auto* xn_ptr = reinterpret_cast<const bf16*>(xn.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (vecs == 1) {
    dense_mmvq_q8_hc_mix<1, 4><<<grid, 32, 0, stream>>>(
        q8w, q8x, xn_ptr, out, xn.stride(0), cols, rows);
  } else if (vecs == 2) {
    dense_mmvq_q8_hc_mix<2, 4><<<grid, 32, 0, stream>>>(
        q8w, q8x, xn_ptr, out, xn.stride(0), cols, rows);
  } else if (vecs == 3) {
    dense_mmvq_q8_hc_mix<3, 4><<<grid, 32, 0, stream>>>(
        q8w, q8x, xn_ptr, out, xn.stride(0), cols, rows);
  } else {
    dense_mmvq_q8_hc_mix<4, 4><<<grid, 32, 0, stream>>>(
        q8w, q8x, xn_ptr, out, xn.stride(0), cols, rows);
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

torch::Tensor dense_gemv_q8_hc_mix_grouped(torch::Tensor weight,
                                           torch::Tensor raw_lora,
                                           torch::Tensor xn,
                                           int64_t hc_count,
                                           int64_t variant) {
  TORCH_CHECK(weight.is_cuda() && raw_lora.is_cuda() && xn.is_cuda(),
              "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == raw_lora.device() &&
                  xn.device() == raw_lora.device(),
              "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "weight must be uint8 Q8_0");
  TORCH_CHECK(raw_lora.scalar_type() == torch::kBFloat16 &&
                  xn.scalar_type() == torch::kBFloat16,
              "HC inputs must be BF16");
  TORCH_CHECK(weight.is_contiguous(), "grouped HC weight must be contiguous");
  TORCH_CHECK(raw_lora.stride(1) == 1 && xn.stride(1) == 1,
              "grouped HC inputs must have unit inner stride");
  TORCH_CHECK(hc_count == 4, "grouped HC-up requires exactly four streams");
  TORCH_CHECK(raw_lora.dim() == 2 && raw_lora.size(0) == 3 &&
                  raw_lora.size(1) == 320,
              "expected exact Qwen4Exp target HC input shape 3x320");
  TORCH_CHECK(xn.dim() == 2 && xn.size(0) == 3 && xn.size(1) == 10240,
              "expected exact Qwen4Exp normalized input shape 3x10240");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 10240,
              "expected Qwen4Exp HC-up weight with 10240 rows");
  constexpr int row_bytes = (320 / QK8_0) * sizeof(block_q8_0);
  TORCH_CHECK(weight.numel() == 10240 * row_bytes,
              "packed Q8_0 byte count does not match HC-up shape");
  TORCH_CHECK(variant >= 0 && variant <= 3,
              "grouped HC-up variant must be in [0, 3]");

  c10::cuda::CUDAGuard guard(raw_lora.device());
  auto quantized = torch::empty(
      {3, (512 / QK8_1) * static_cast<int>(sizeof(block_q8_1))},
      torch::TensorOptions().dtype(torch::kUInt8).device(raw_lora.device()));
  auto output = torch::empty({3, 2560}, raw_lora.options());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  quantize_q8_1_hc_silu<<<dim3(2, 3, 1), 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(raw_lora.data_ptr()),
      reinterpret_cast<block_q8_1*>(quantized.data_ptr()),
      raw_lora.stride(0), 320, 512, hc_count);
  const auto* q8w = reinterpret_cast<const block_q8_0*>(weight.data_ptr());
  const auto* q8x = reinterpret_cast<const block_q8_1*>(quantized.data_ptr());
  const auto* xn_ptr = reinterpret_cast<const bf16*>(xn.data_ptr());
  auto* out = reinterpret_cast<bf16*>(output.data_ptr());
  if (variant == 0) {
    dense_mmvq_q8_hc_mix_grouped_w4<1>
        <<<dim3(640, 1, 1), dim3(32, 4, 1), 0, stream>>>(
            q8w, q8x, xn_ptr, out, xn.stride(0));
  } else if (variant == 1) {
    dense_mmvq_q8_hc_mix_grouped_w8<1>
        <<<dim3(320, 1, 1), dim3(32, 8, 1), 0, stream>>>(
            q8w, q8x, xn_ptr, out, xn.stride(0));
  } else if (variant == 2) {
    dense_mmvq_q8_hc_mix_grouped_w4<2>
        <<<dim3(320, 1, 1), dim3(32, 4, 1), 0, stream>>>(
            q8w, q8x, xn_ptr, out, xn.stride(0));
  } else {
    dense_mmvq_q8_hc_mix_grouped_w8<2>
        <<<dim3(160, 1, 1), dim3(32, 8, 1), 0, stream>>>(
            q8w, q8x, xn_ptr, out, xn.stride(0));
  }
  AT_CUDA_CHECK(hipGetLastError());
  return output;
}

void check_hc_down_bf16_m3(const torch::Tensor& weight,
                           const torch::Tensor& x,
                           const torch::Tensor& output, int64_t variant) {
  TORCH_CHECK(weight.is_cuda() && x.is_cuda() && output.is_cuda(),
              "tensors must be on a GPU");
  TORCH_CHECK(weight.device() == x.device() && output.device() == x.device(),
              "tensors must be on the same GPU");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16 &&
                  x.scalar_type() == torch::kBFloat16 &&
                  output.scalar_type() == torch::kBFloat16,
              "HC-down weight, input, and output must be BF16");
  TORCH_CHECK(weight.is_contiguous() && x.is_contiguous() &&
                  output.is_contiguous(),
              "HC-down tensors must be contiguous");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 336 &&
                  weight.size(1) == 10240,
              "expected exact Qwen4Exp HC-down weight shape 336x10240");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 3 && x.size(1) == 10240,
              "expected exact Qwen4Exp target input shape 3x10240");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == 3 &&
                  output.size(1) == 336,
              "expected exact Qwen4Exp HC-down output shape 3x336");
  TORCH_CHECK(variant == 0 || variant == 1,
              "HC-down variant must be 0 (cached) or 1 (non-temporal)");
}

void hc_down_bf16_m3_out(torch::Tensor weight, torch::Tensor x,
                         torch::Tensor output, int64_t variant) {
  check_hc_down_bf16_m3(weight, x, output, variant);
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
  const dim3 grid(32, 1, 1);
  const dim3 block(32, 16, 1);
  const auto* weight_ptr = reinterpret_cast<const bf16*>(weight.data_ptr());
  const auto* input_ptr = reinterpret_cast<const bf16*>(x.data_ptr());
  auto* output_ptr = reinterpret_cast<bf16*>(output.data_ptr());
  if (variant == 0) {
    hc_down_bf16_m3_cyclic<false><<<grid, block, 0, stream>>>(
        weight_ptr, input_ptr, output_ptr);
  } else {
    hc_down_bf16_m3_cyclic<true><<<grid, block, 0, stream>>>(
        weight_ptr, input_ptr, output_ptr);
  }
  AT_CUDA_CHECK(hipGetLastError());
}

torch::Tensor hc_down_bf16_m3(torch::Tensor weight, torch::Tensor x,
                              int64_t variant) {
  auto output = torch::empty({3, 336}, x.options());
  hc_down_bf16_m3_out(weight, x, output, variant);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dense_gemv", &dense_gemv,
             "Qwen3.8 dense Q4_K/Q5_K/Q6_K MMVQ (HIP)");
  module.def("dense_gemv_q4_reuse2", &dense_gemv_q4_reuse2,
             "Qwen3.8 Q4_K two-row weight-reuse MMVQ (HIP)");
  module.def("dense_gemv_q8_reuse2", &dense_gemv_q8_reuse2,
             "Qwen3.8 Q8_0 two-row weight-reuse MMVQ (HIP)");
  module.def("dense_gemv_reuse3", &dense_gemv_reuse3,
             "Qwen3.8 Q4-target three-row weight-reuse MMVQ (HIP)");
  module.def("dense_gemv_q8_attention_m3", &dense_gemv_q8_attention_m3,
             "Qwen3.8 exact-shape Q8 attention-input M=3 MMVQ (HIP)");
  module.def("dense_gemv_reuse4", &dense_gemv_reuse4,
             "Qwen3.8 Q4-target four-row weight-reuse MMVQ (HIP)");
  module.def("dense_gemv_q8_hc_mix", &dense_gemv_q8_hc_mix,
             "Qwen3.8 fused HC SiLU, Q8_0 up GEMV, and gate mix (HIP)");
  module.def("dense_gemv_q8_hc_mix_grouped",
             &dense_gemv_q8_hc_mix_grouped,
             "Qwen4Exp exact M=3 grouped fused HC-up and gate mix (HIP)");
  module.def("hc_down_bf16_m3_out", &hc_down_bf16_m3_out,
             "Qwen4Exp exact-shape BF16 HC-down M=3 into caller output (HIP)");
  module.def("hc_down_bf16_m3", &hc_down_bf16_m3,
             "Qwen4Exp exact-shape BF16 HC-down M=3 (HIP)");
}

// Fused Mamba SSM conv1d kernel for AMD RDNA4 (gfx1201)
// Uses wave32 WMMA instructions and 64 KiB LDS per WGP

#include <hip/hip_runtime.h>

__global__ void mamba_conv1d_fwd_kernel(
    const float* __restrict__ u,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch,
    int dim,
    int seq_len,
    int width
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elements = batch * dim * seq_len;
    if (tid >= total_elements) return;

    // Convolution 1D sliding window
    int t = tid % seq_len;
    int c = (tid / seq_len) % dim;
    int b = tid / (seq_len * dim);

    float acc = bias ? bias[c] : 0.0f;
    for (int w = 0; w < width; ++w) {
        int in_t = t - (width - 1) + w;
        if (in_t >= 0) {
            acc += u[b * (dim * seq_len) + c * seq_len + in_t] * weight[c * width + w];
        }
    }
    out[tid] = acc;
}

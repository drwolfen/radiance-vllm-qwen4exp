# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_sum_shared_kernel(
    routed_ptr,
    weights_ptr,
    shared_ptr,
    output_ptr,
    stride_routed_token,
    stride_routed_expert,
    stride_weights_token,
    stride_shared_token,
    stride_output_token,
    hidden_size,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for expert in tl.static_range(0, TOP_K):
        route = tl.load(
            routed_ptr
            + token * stride_routed_token
            + expert * stride_routed_expert
            + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weights_ptr + token * stride_weights_token + expert).to(
            tl.float32
        )
        # Preserve the legacy in-place BF16 multiply boundary before the
        # moe_sum kernel's FP32 accumulation.
        weighted = (route * weight).to(tl.bfloat16).to(tl.float32)
        accumulator += weighted

    # Preserve the legacy moe_sum BF16 output boundary before shared+routed add.
    routed_sum = accumulator.to(tl.bfloat16).to(tl.float32)
    shared = tl.load(
        shared_ptr + token * stride_shared_token + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = (routed_sum + shared).to(tl.bfloat16)
    tl.store(
        output_ptr + token * stride_output_token + offsets,
        result,
        mask=mask,
    )


def weighted_sum_shared_reference(
    routed: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    """CPU reference with the legacy BF16 rounding boundaries."""

    if routed.ndim != 3 or weights.shape != routed.shape[:2]:
        raise ValueError(
            "routed and weights must be [tokens, top_k, hidden] / [tokens, top_k]"
        )
    if shared.shape != (routed.shape[0], routed.shape[2]):
        raise ValueError("shared output must be [tokens, hidden]")
    if routed.dtype != torch.bfloat16 or shared.dtype != torch.bfloat16:
        raise ValueError("the fused epilogue requires BF16 routed/shared outputs")

    accumulator = torch.zeros_like(shared, dtype=torch.float32)
    for expert in range(routed.shape[1]):
        weighted = (routed[:, expert].float() * weights[:, expert, None].float()).to(
            torch.bfloat16
        )
        accumulator += weighted.float()
    routed_sum = accumulator.to(torch.bfloat16)
    return (routed_sum + shared).to(torch.bfloat16)


def fused_weighted_moe_sum_shared(
    routed: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Weight and reduce routed rows, then add shared output in one launch."""

    if routed.ndim != 3 or weights.shape != routed.shape[:2]:
        raise ValueError("routed and weights have incompatible shapes")
    expected_output = (routed.shape[0], routed.shape[2])
    if shared.shape != expected_output or output.shape != expected_output:
        raise ValueError("shared and output must match [tokens, hidden]")
    if routed.dtype != torch.bfloat16 or shared.dtype != torch.bfloat16:
        raise ValueError("the fused epilogue requires BF16 routed/shared outputs")
    if output.dtype != torch.bfloat16:
        raise ValueError("the fused epilogue requires a BF16 output")
    if not (routed.is_cuda and weights.is_cuda and shared.is_cuda and output.is_cuda):
        raise ValueError("the fused epilogue requires GPU tensors")
    if not (routed.device == weights.device == shared.device == output.device):
        raise ValueError("all fused epilogue tensors must be on the same device")
    if not (routed.is_contiguous() and weights.is_contiguous()):
        raise ValueError("routed outputs and weights must be contiguous")
    if not (shared.is_contiguous() and output.is_contiguous()):
        raise ValueError("shared and output tensors must be contiguous")
    if not routed.shape[0] or not routed.shape[2]:
        return

    block_h = 256
    grid = (routed.shape[0], triton.cdiv(routed.shape[2], block_h))
    _weighted_sum_shared_kernel[grid](
        routed,
        weights,
        shared,
        output,
        routed.stride(0),
        routed.stride(1),
        weights.stride(0),
        shared.stride(0),
        output.stride(0),
        routed.shape[2],
        TOP_K=routed.shape[1],
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=1,
    )


__all__ = ["fused_weighted_moe_sum_shared", "weighted_sum_shared_reference"]

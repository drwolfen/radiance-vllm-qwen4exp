# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_gguf_plugin.triton.fused_moe.weighted_sum import (
    weighted_sum_shared_reference,
)


def _legacy_epilogue(
    routed: torch.Tensor,
    weights: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    weighted = routed.clone()
    weighted.mul_(weights.unsqueeze(-1))
    accumulator = torch.zeros_like(shared, dtype=torch.float32)
    for expert in range(weighted.shape[1]):
        accumulator += weighted[:, expert].float()
    routed_sum = accumulator.to(torch.bfloat16)
    return (routed_sum + shared).to(torch.bfloat16)


def test_weighted_sum_shared_reference_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(1234)
    routed = torch.randn(3, 10, 2560, generator=generator).to(torch.bfloat16)
    weights = torch.rand(3, 10, generator=generator)
    shared = torch.randn(3, 2560, generator=generator).to(torch.bfloat16)

    expected = _legacy_epilogue(routed, weights, shared)
    actual = weighted_sum_shared_reference(routed, weights, shared)

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_weighted_sum_shared_reference_rejects_wrong_shape() -> None:
    routed = torch.zeros(3, 10, 8, dtype=torch.bfloat16)
    weights = torch.zeros(3, 9)
    shared = torch.zeros(3, 8, dtype=torch.bfloat16)

    try:
        weighted_sum_shared_reference(routed, weights, shared)
    except ValueError as error:
        assert "routed and weights" in str(error)
    else:
        raise AssertionError("shape mismatch was not rejected")

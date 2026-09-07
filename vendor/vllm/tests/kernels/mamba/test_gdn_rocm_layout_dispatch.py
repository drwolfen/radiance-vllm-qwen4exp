# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side coverage for ROCm GDN layout dispatch."""

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    _resolve_aiter_conv_layout_kwargs,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

PREFIX = "model.layers.0.linear_attn"
H = 2
HV = 4
K = 8
V = 8


def _make_metadata(
    *,
    num_prefills: int = 0,
    num_decodes: int = 2,
    spec_sequence_masks: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_prefills + num_decodes,
        spec_sequence_masks=spec_sequence_masks,
    )


def _make_layer(gqa_interleaved_layout: bool, supports_layout: bool):
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.gqa_interleaved_layout = gqa_interleaved_layout
    layer.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        supports_layout,
    ):
        layer._aiter_conv_layout_kwargs = _resolve_aiter_conv_layout_kwargs(
            layer.qkvz_layout
        )

    layer.calls = []
    layer._forward_core_decode_aiter = lambda **kw: layer.calls.append("aiter")
    layer._forward_core = lambda **kw: layer.calls.append("generic")
    layer.prepare_gdn_attention_core_inputs = lambda qkvz, ba, n: (
        torch.zeros(n, H * K * 2 + HV * V),
        torch.zeros(n, HV, V),
        torch.zeros(n, HV),
        torch.zeros(n, HV),
    )
    layer._forward_core_rocm = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_rocm, layer
    )
    return layer


def _run(layer, metadata: GDNAttentionMetadata) -> str:
    num_tokens = max(metadata.num_actual_tokens, 1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})
    with patch.object(
        qwen_gdn_linear_attn,
        "get_forward_context",
        return_value=context,
    ):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, 2 * H * K + 2 * HV * V),
            ba=torch.zeros(num_tokens, 2 * HV),
            z_out=torch.zeros(num_tokens, HV, V),
            core_attn_out=torch.zeros(num_tokens, HV, V),
        )
    assert len(layer.calls) == 1
    return layer.calls[0]


@pytest.mark.parametrize("supports_layout", [True, False])
def test_interleaved_always_takes_fast_path(supports_layout: bool) -> None:
    layer = _make_layer(True, supports_layout)
    assert _run(layer, _make_metadata()) == "aiter"


def test_flat_takes_fast_path_when_aiter_supports_layout() -> None:
    layer = _make_layer(False, True)
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        True,
    ):
        assert _run(layer, _make_metadata()) == "aiter"


def test_flat_falls_back_when_aiter_lacks_layout_support() -> None:
    layer = _make_layer(False, False)
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        False,
    ):
        assert _run(layer, _make_metadata()) == "generic"


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
@pytest.mark.parametrize(
    "metadata_kwargs",
    [
        {"num_prefills": 1},
        {"num_decodes": 0},
        {"spec_sequence_masks": torch.zeros(1, dtype=torch.bool)},
    ],
)
def test_non_pure_decode_uses_generic_path(
    gqa_interleaved_layout: bool,
    metadata_kwargs: dict,
) -> None:
    layer = _make_layer(gqa_interleaved_layout, True)
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        True,
    ):
        assert _run(layer, _make_metadata(**metadata_kwargs)) == "generic"


@pytest.mark.parametrize("qkvz_layout", ["flat", "interleaved"])
@pytest.mark.parametrize("supports_layout", [True, False])
def test_layout_kwarg_matches_capability(
    qkvz_layout: str,
    supports_layout: bool,
) -> None:
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        supports_layout,
    ):
        expected = {"qkvz_layout": qkvz_layout} if supports_layout else {}
        assert _resolve_aiter_conv_layout_kwargs(qkvz_layout) == expected

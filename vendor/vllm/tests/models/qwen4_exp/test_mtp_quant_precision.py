# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
"""No-device regressions for Qwen4Exp MTP packed-layer precision metadata."""

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.qwen4_exp.common.mtp import (
    QWEN4_EXP_MTP_PACKED_MODULES_MAPPING,
    remap_mtp_ignored_layers,
)


OFFICIAL_MTP_IGNORED = [
    "mtp.fc_embedding",
    "mtp.fc_hidden",
    "mtp.hyper_connection_mixer.input_mix_weight_down",
    "mtp.hyper_connection_mixer.input_mix_weight_up",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]


def _is_skipped(prefix: str, ignored: list[str]) -> bool:
    return is_layer_skipped(
        prefix,
        ignored,
        QWEN4_EXP_MTP_PACKED_MODULES_MAPPING,
        match_mode="exact",
    )


def test_official_mtp_ignored_layers_complete_every_runtime_fusion():
    ignored = remap_mtp_ignored_layers(
        OFFICIAL_MTP_IGNORED,
        mtp_start_layer_idx=48,
        packed_modules_mapping=QWEN4_EXP_MTP_PACKED_MODULES_MAPPING,
    )

    expected_skipped = {
        "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject",
        "mtp.layers.48.mlp.shared_expert.gate_up_proj",
        "mtp.layers.48.mlp_hyper_connection.input_mix_weight_down_block_inject",
        "mtp.layers.48.self_attn.qkv_proj",
    }
    assert expected_skipped <= set(ignored)
    assert not any(name.startswith("mtp.layers.0.") for name in ignored)

    # Audit every packed-module family that can occur in the one-layer MTP.
    # Full-attention and shared-expert fusions stay BF16; routed experts stay
    # block-FP8. GDN-only packed projections are not instantiated by this MTP.
    decisions = {
        "mtp.layers.48.self_attn.qkv_proj": True,
        "mtp.layers.48.mlp.shared_expert.gate_up_proj": True,
        "mtp.layers.48.mlp.experts.gate_up_proj": False,
        # The final mixer has use_combine=False and is not a merged module.
        "mtp.hyper_connection_mixer.input_mix_weight_down": True,
        "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject": True,
        "mtp.layers.48.mlp_hyper_connection.input_mix_weight_down_block_inject": True,
        "mtp.layers.48.linear_attn.in_proj_qkvz": False,
        "mtp.layers.48.linear_attn.in_proj_ba": False,
    }
    assert {prefix: _is_skipped(prefix, ignored) for prefix in decisions} == decisions


def test_partial_real_hc_shards_still_fail_closed():
    partial = [
        name
        for name in OFFICIAL_MTP_IGNORED
        if name != "mtp.layers.0.attn_hyper_connection.block_inject_weight"
    ]
    ignored = remap_mtp_ignored_layers(
        partial,
        mtp_start_layer_idx=48,
        packed_modules_mapping=QWEN4_EXP_MTP_PACKED_MODULES_MAPPING,
    )
    fused = "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject"
    assert fused not in ignored
    with pytest.raises(ValueError, match="some but not all shards"):
        _is_skipped(fused, ignored)


def test_serialized_fp8_constructs_layer48_merged_hc_as_bf16(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    quant_config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": OFFICIAL_MTP_IGNORED,
        }
    )
    quant_config.packed_modules_mapping = QWEN4_EXP_MTP_PACKED_MODULES_MAPPING
    quant_config.ignored_layers = remap_mtp_ignored_layers(
        quant_config.ignored_layers,
        mtp_start_layer_idx=48,
        packed_modules_mapping=quant_config.packed_modules_mapping,
    )

    for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
        prefix = (
            f"mtp.layers.48.{connection}."
            "input_mix_weight_down_block_inject"
        )
        layer = MergedColumnParallelLinear(
            10_240,
            [320, 4, 12],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=False,
            disable_tp=True,
        )
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)

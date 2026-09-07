# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.engine.arg_utils as arg_utils_module
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.transformers_utils.config as config_module
from transformers import PretrainedConfig
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader import get_model_loader
from vllm.transformers_utils.config import get_config_parser

import vllm_gguf_plugin.config_parser as gguf_config_parser_module
import vllm_gguf_plugin.quantization as gguf_quantization
import vllm_gguf_plugin.quantization.fused_moe as gguf_fused_moe_module
import vllm_gguf_plugin.quantization.linear as gguf_linear_module
import vllm_gguf_plugin.quantization.params as gguf_params_module
import vllm_gguf_plugin.quantization.route_profile as route_profile_module
import vllm_gguf_plugin.quantization.tiered_compaction as tiered_compaction_module
import vllm_gguf_plugin.quantization.tiered_experts as tiered_experts_module
import vllm_gguf_plugin.quantization.vocal_embeds as gguf_vocab_module
from vllm_gguf_plugin import OOTGGUFConfig, OOTGGUFModelLoader, register
from vllm_gguf_plugin.config_parser import GGUFConfigParser
from vllm_gguf_plugin.quantization import (
    GGUFUninitializedParameter,
    GGUFWeightParameter,
    GGUFWeightTypeParameter,
)


def _tiered_manifest(hot_ids: list[int]) -> dict:
    rank = {"hot_experts_by_layer": [hot_ids.copy() for _ in range(48)]}
    return {
        "version": 1,
        "num_layers": 48,
        "num_experts": 512,
        "ranks": {"0": rank, "1": rank},
    }


def test_tiered_expert_manifest_validation() -> None:
    num_experts, hot_lists = tiered_experts_module._validate_hot_lists(
        _tiered_manifest([7, 3, 11]), 1
    )

    assert num_experts == 512
    assert hot_lists[0] == [7, 3, 11]
    assert len(hot_lists) == 48


@pytest.mark.parametrize("hot_ids", ([1, 1], [-1], [512]))
def test_tiered_expert_manifest_rejects_invalid_ids(hot_ids) -> None:
    with pytest.raises(ValueError):
        tiered_experts_module._validate_hot_lists(_tiered_manifest(hot_ids), 0)


def test_tiered_reuse3_variant_is_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("QWEN38_TIERED_IQ_MOE_VARIANT", raising=False)
    assert gguf_fused_moe_module._tiered_iq_moe_variant() == ("generic", 0)

    monkeypatch.setenv("QWEN38_TIERED_IQ_MOE_VARIANT", "reuse3")
    assert gguf_fused_moe_module._tiered_iq_moe_variant() == ("reuse3", 30)

    monkeypatch.setenv("QWEN38_TIERED_IQ_MOE_VARIANT", "reuse3v2")
    assert gguf_fused_moe_module._tiered_iq_moe_variant() == ("reuse3v2", 31)


def test_tiered_prefill_group_size_is_strict(monkeypatch) -> None:
    monkeypatch.delenv("QWEN38_TIERED_PREFILL_GROUP_SIZE", raising=False)
    assert gguf_fused_moe_module._tiered_prefill_group_size() == 0

    for group_size in (4, 8, 16):
        monkeypatch.setenv("QWEN38_TIERED_PREFILL_GROUP_SIZE", str(group_size))
        assert gguf_fused_moe_module._tiered_prefill_group_size() == group_size

    monkeypatch.setenv("QWEN38_TIERED_PREFILL_GROUP_SIZE", "12")
    with pytest.raises(RuntimeError, match="must be 0, 4, 8, or 16"):
        gguf_fused_moe_module._tiered_prefill_group_size()


def test_tiered_cache_defaults_off_and_is_rank1_only(monkeypatch) -> None:
    monkeypatch.delenv("QWEN38_TIERED_EXPERT_CACHE_SLOTS", raising=False)
    monkeypatch.delenv("QWEN38_TIERED_EXPERT_CACHE_RANKS", raising=False)
    monkeypatch.delenv("QWEN38_TIERED_EXPERT_CACHE_ASYNC", raising=False)
    monkeypatch.delenv("QWEN38_TIERED_EXPERT_CACHE_POLICY", raising=False)
    assert tiered_experts_module._dynamic_cache_slots(0) == 0
    assert tiered_experts_module._dynamic_cache_slots(1) == 0
    assert tiered_experts_module._async_cache_enabled() is False
    assert tiered_experts_module._cache_policy() == "second_touch_rr"

    monkeypatch.setenv("QWEN38_TIERED_EXPERT_CACHE_SLOTS", "16")
    assert tiered_experts_module._dynamic_cache_slots(0) == 0
    assert tiered_experts_module._dynamic_cache_slots(1) == 16


def test_tiered_cache_rejects_out_of_bounds_and_bad_async(monkeypatch) -> None:
    monkeypatch.setenv("QWEN38_TIERED_EXPERT_CACHE_SLOTS", "17")
    with pytest.raises(ValueError, match="between 0 and 16"):
        tiered_experts_module._dynamic_cache_slots(1)
    monkeypatch.setenv("QWEN38_TIERED_EXPERT_CACHE_ASYNC", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        tiered_experts_module._async_cache_enabled()
    monkeypatch.setenv("QWEN38_TIERED_EXPERT_CACHE_POLICY", "lfu")
    with pytest.raises(ValueError, match="second_touch_rr or lru"):
        tiered_experts_module._cache_policy()


def test_tiered_cache_lru_is_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("QWEN38_TIERED_EXPERT_CACHE_POLICY", "lru")
    assert tiered_experts_module._cache_policy() == "lru"
    assert gguf_fused_moe_module._tiered_cache_policy() == "lru"


def test_register_overrides_gguf_config():
    register()

    quant_config = get_quantization_config("gguf")

    assert quant_config is OOTGGUFConfig


def test_register_overrides_gguf_loader():
    register()

    model_loader = get_model_loader(LoadConfig(load_format="gguf"))

    assert isinstance(model_loader, OOTGGUFModelLoader)


def test_register_is_idempotent():
    register()
    register()

    assert get_quantization_config("gguf") is OOTGGUFConfig
    assert isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), OOTGGUFModelLoader
    )
    assert isinstance(get_config_parser("gguf"), GGUFConfigParser)


def test_oot_config_reuses_in_tree_behavior():
    quant_config = OOTGGUFConfig.from_config({})

    assert isinstance(quant_config, OOTGGUFConfig)
    assert quant_config.get_name() == "gguf"
    assert repr(quant_config) == "GGUFConfig()"


def test_supported_act_dtypes_includes_bfloat16():
    quant_config = OOTGGUFConfig.from_config({})
    supported = quant_config.get_supported_act_dtypes()

    assert torch.half in supported
    assert torch.bfloat16 in supported
    assert torch.float32 in supported


def test_qwen38_q8_finite_guard_is_limited_to_known_rocm_shapes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gguf_linear_module.torch.version, "hip", "test")
    guarded_shapes = {
        (320, 10240),
        (2560, 640),
        (320, 2560),
    }

    for input_size, output_size in guarded_shapes:
        x = torch.empty(2, input_size)
        qweight = torch.empty(output_size, 34, dtype=torch.uint8)
        assert gguf_linear_module._needs_q8_finite_guard(x, qweight, 8)

    assert not gguf_linear_module._needs_q8_finite_guard(
        torch.empty(2, 2560),
        torch.empty(2560, 34, dtype=torch.uint8),
        8,
    )
    assert not gguf_linear_module._needs_q8_finite_guard(
        torch.empty(2, 320),
        torch.empty(10240, 34, dtype=torch.uint8),
        2,
    )


@pytest.mark.parametrize(
    ("qtype", "output_size", "input_size"),
    [
        (8, 10240, 2560),
        (8, 5120, 2560),
        (8, 2560, 6144),
        (8, 2560, 3072),
        (8, 6144, 2560),
        (8, 3072, 2560),
        (8, 12288, 2560),
        (12, 248320, 2560),
        (12, 124160, 2560),
        (14, 248320, 2560),
        (14, 124160, 2560),
    ],
)
def test_qwen38_reuse3_gate_accepts_only_exact_q4_target_shapes(
    monkeypatch, qtype, output_size, input_size
) -> None:
    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE3", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    x = torch.empty((3, input_size), dtype=torch.bfloat16)
    qweight = torch.empty((output_size, 1), dtype=torch.uint8)

    assert gguf_linear_module._use_qwen38_q4_reuse3(x, qweight, qtype)


def test_qwen38_reuse3_gate_is_default_off_m3_and_gfx1201_only(
    monkeypatch,
) -> None:
    x = torch.empty((3, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((5120, 1), dtype=torch.uint8)
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.delenv("QWEN38_USE_DENSE_MMVQ_REUSE3", raising=False)
    assert not gguf_linear_module._use_qwen38_q4_reuse3(x, qweight, 8)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE3", "1")
    assert not gguf_linear_module._use_qwen38_q4_reuse3(x[:2], qweight, 8)
    assert not gguf_linear_module._use_qwen38_q4_reuse3(
        x, torch.empty((4096, 1), dtype=torch.uint8), 8
    )
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: False)
    assert not gguf_linear_module._use_qwen38_q4_reuse3(x, qweight, 8)


def test_qwen38_reuse3_dispatch_calls_extension(monkeypatch) -> None:
    calls = []

    class FakeExtension:
        @staticmethod
        def dense_gemv_reuse3(qweight, x, qtype, rows):
            calls.append((tuple(qweight.shape), tuple(x.shape), qtype, rows))
            return torch.zeros((3, rows), dtype=x.dtype)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE3", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.setattr(gguf_linear_module, "_DENSE_MMVQ_HIP", FakeExtension())
    x = torch.empty((3, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((5120, 1), dtype=torch.uint8)

    output = gguf_linear_module._fused_mul_mat_gguf(x, qweight, 8)

    assert output.shape == (3, 5120)
    assert calls == [((5120, 1), (3, 2560), 8, 5120)]


@pytest.mark.parametrize("output_size", [8192, 6656])
def test_qwen38_q8_attention_m3_gate_accepts_only_exact_shapes(
    monkeypatch, output_size
) -> None:
    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    x = torch.empty((3, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((output_size, 1), dtype=torch.uint8)

    assert gguf_linear_module._use_qwen38_q8_attention_m3(x, qweight, 8)
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(x, qweight, 12)


def test_qwen38_q8_attention_m3_gate_is_default_off_and_strict(
    monkeypatch,
) -> None:
    x = torch.empty((3, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((8192, 1), dtype=torch.uint8)
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.delenv("QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3", raising=False)
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(x, qweight, 8)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3", "1")
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(x[:2], qweight, 8)
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(
        x, torch.empty((8192, 1), dtype=torch.int8), 8
    )
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(
        x, torch.empty((5120, 1), dtype=torch.uint8), 8
    )
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: False)
    assert not gguf_linear_module._use_qwen38_q8_attention_m3(x, qweight, 8)


def test_qwen38_q8_attention_m3_variant_is_validated(monkeypatch) -> None:
    monkeypatch.delenv("QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT", raising=False)
    assert gguf_linear_module._qwen38_q8_attention_m3_variant() == 2

    monkeypatch.setenv("QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT", "group4-w8")
    assert gguf_linear_module._qwen38_q8_attention_m3_variant() == 4

    monkeypatch.setenv("QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT", "not-a-variant")
    with pytest.raises(ValueError, match="must be one of"):
        gguf_linear_module._qwen38_q8_attention_m3_variant()


def test_qwen38_q8_attention_m3_dispatch_calls_extension(monkeypatch) -> None:
    calls = []

    class FakeExtension:
        @staticmethod
        def dense_gemv_q8_attention_m3(qweight, x, rows, variant):
            calls.append((tuple(qweight.shape), tuple(x.shape), rows, variant))
            return torch.zeros((3, rows), dtype=x.dtype)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3", "1")
    monkeypatch.setenv("QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT", "group4-w8")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.setattr(gguf_linear_module, "_DENSE_MMVQ_HIP", FakeExtension())
    x = torch.empty((3, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((6656, 1), dtype=torch.uint8)

    output = gguf_linear_module._fused_mul_mat_gguf(x, qweight, 8)

    assert output.shape == (3, 6656)
    assert calls == [((6656, 1), (3, 2560), 6656, 4)]


@pytest.mark.parametrize(
    ("qtype", "output_size", "input_size"),
    [
        (8, 10240, 2560),
        (8, 5120, 2560),
        (8, 2560, 6144),
        (8, 2560, 3072),
        (8, 6144, 2560),
        (8, 3072, 2560),
        (8, 12288, 2560),
        (12, 248320, 2560),
        (12, 124160, 2560),
        (14, 248320, 2560),
        (14, 124160, 2560),
    ],
)
def test_qwen38_reuse4_gate_accepts_only_exact_q4_target_shapes(
    monkeypatch, qtype, output_size, input_size
) -> None:
    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE4", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    x = torch.empty((4, input_size), dtype=torch.bfloat16)
    qweight = torch.empty((output_size, 1), dtype=torch.uint8)

    assert gguf_linear_module._use_qwen38_q4_reuse4(x, qweight, qtype)


def test_qwen38_reuse4_gate_is_default_off_m4_and_gfx1201_only(
    monkeypatch,
) -> None:
    x = torch.empty((4, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((5120, 1), dtype=torch.uint8)
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.delenv("QWEN38_USE_DENSE_MMVQ_REUSE4", raising=False)
    assert not gguf_linear_module._use_qwen38_q4_reuse4(x, qweight, 8)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE4", "1")
    assert not gguf_linear_module._use_qwen38_q4_reuse4(x[:3], qweight, 8)
    assert not gguf_linear_module._use_qwen38_q4_reuse4(
        x, torch.empty((4096, 1), dtype=torch.uint8), 8
    )
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: False)
    assert not gguf_linear_module._use_qwen38_q4_reuse4(x, qweight, 8)


def test_qwen38_reuse4_dispatch_calls_extension(monkeypatch) -> None:
    calls = []

    class FakeExtension:
        @staticmethod
        def dense_gemv_reuse4(qweight, x, qtype, rows):
            calls.append((tuple(qweight.shape), tuple(x.shape), qtype, rows))
            return torch.zeros((4, rows), dtype=x.dtype)

    monkeypatch.setenv("QWEN38_USE_DENSE_MMVQ_REUSE4", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.setattr(gguf_linear_module, "_DENSE_MMVQ_HIP", FakeExtension())
    x = torch.empty((4, 2560), dtype=torch.bfloat16)
    qweight = torch.empty((5120, 1), dtype=torch.uint8)

    output = gguf_linear_module._fused_mul_mat_gguf(x, qweight, 8)

    assert output.shape == (4, 5120)
    assert calls == [((5120, 1), (4, 2560), 8, 5120)]


def _qwen38_hc_up_inputs(tokens: int = 3):
    raw_with_padding = torch.empty((tokens, 336), dtype=torch.bfloat16)
    raw_lora = raw_with_padding[:, :320]
    xn = torch.empty((tokens, 10240), dtype=torch.bfloat16)
    qweight = torch.empty((10240, 340), dtype=torch.uint8)
    return raw_lora, xn, qweight


def test_qwen38_fused_hc_up_mix_is_strict_and_default_off(monkeypatch) -> None:
    raw_lora, xn, qweight = _qwen38_hc_up_inputs()
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.delenv("QWEN38_FUSED_HC_UP_MIX", raising=False)

    assert not gguf_linear_module._use_qwen38_fused_hc_up_mix(
        raw_lora, xn, qweight, 8, 4
    )

    monkeypatch.setenv("QWEN38_FUSED_HC_UP_MIX", "1")
    assert gguf_linear_module._use_qwen38_fused_hc_up_mix(raw_lora, xn, qweight, 8, 4)
    assert raw_lora.stride() == (336, 1)
    assert not gguf_linear_module._use_qwen38_fused_hc_up_mix(
        raw_lora, xn, qweight, 12, 4
    )
    assert not gguf_linear_module._use_qwen38_fused_hc_up_mix(
        raw_lora, xn, qweight, 8, 2
    )
    assert not gguf_linear_module._use_qwen38_fused_hc_up_mix(
        raw_lora[:, :319], xn, qweight, 8, 4
    )


def test_qwen38_fused_hc_up_mix_method_dispatches_custom_op(monkeypatch) -> None:
    raw_lora, xn, qweight = _qwen38_hc_up_inputs()
    calls = []

    def fake_fused(raw, normalized, weight, hc_count):
        calls.append((raw, normalized, weight, hc_count))
        return raw.new_zeros((raw.shape[0], normalized.shape[1] // hc_count))

    monkeypatch.setenv("QWEN38_FUSED_HC_UP_MIX", "1")
    monkeypatch.setattr(gguf_linear_module, "_is_gfx1201", lambda device: True)
    monkeypatch.setattr(
        gguf_linear_module,
        "fused_qwen38_hc_up_mix",
        fake_fused,
    )
    layer = SimpleNamespace(
        qweight=qweight,
        qweight_type=SimpleNamespace(weight_type=8),
    )
    method = gguf_linear_module.GGUFLinearMethod(None)

    output = method.apply_hc_up_mix(layer, raw_lora, xn, 4)

    assert output is not None
    assert output.shape == (3, 2560)
    assert len(calls) == 1
    assert calls[0][0].data_ptr() == raw_lora.data_ptr()
    assert calls[0][0].stride() == raw_lora.stride()
    assert calls[0][1].data_ptr() == xn.data_ptr()
    assert calls[0][2] is qweight
    assert calls[0][3] == 4


def test_qwen38_q8_vision_prefill_avoids_native_rocm_mmq(monkeypatch) -> None:
    monkeypatch.setattr(gguf_linear_module.torch.version, "hip", "test")
    native_calls = []

    def fake_dequantize(qweight, qweight_type, output_size, input_size, dtype):
        assert qweight_type == 8
        assert (input_size, output_size) == (1152, 2152)
        return torch.ones((output_size, input_size), dtype=dtype)

    monkeypatch.setattr(gguf_linear_module.ops, "ggml_dequantize", fake_dequantize)
    monkeypatch.setattr(
        gguf_linear_module.ops,
        "ggml_mul_mat_a8",
        lambda *args: native_calls.append(args),
    )
    x = torch.ones((4, 1152))
    qweight = torch.empty((2152, 1224), dtype=torch.uint8)

    result = gguf_linear_module._fused_mul_mat_gguf(x, qweight, 8)

    assert result.shape == (4, 2152)
    assert torch.all(result == 1152)
    assert not native_calls


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        (1152, 1152),
        (1152, 3456),
        (1152, 4304),
        (4304, 1152),
        (4608, 2560),
        (4608, 4608),
        (576, 1152),
        (1152, 1728),
        (1152, 2152),
        (2152, 1152),
        (2304, 2560),
        (4608, 2304),
    ],
)
def test_qwen38_q8_vision_fallback_covers_tp1_and_tp2_shapes(
    monkeypatch, input_size, output_size
) -> None:
    monkeypatch.setattr(gguf_linear_module.torch.version, "hip", "test")
    assert gguf_linear_module._needs_q8_vision_dequant_fallback(
        torch.empty((4, input_size)),
        torch.empty((output_size, 34), dtype=torch.uint8),
        8,
    )


def test_qwen38_q8_vision_fallback_keeps_other_quant_and_non_rocm_native(
    monkeypatch,
) -> None:
    x = torch.empty((3, 1152))
    qweight = torch.empty((2152, 34), dtype=torch.uint8)
    monkeypatch.setattr(gguf_linear_module.torch.version, "hip", "test")
    assert not gguf_linear_module._needs_q8_vision_dequant_fallback(x, qweight, 2)

    monkeypatch.setattr(gguf_linear_module.torch.version, "hip", None)
    assert not gguf_linear_module._needs_q8_vision_dequant_fallback(
        torch.empty((4, 1152)), qweight, 8
    )


def test_gguf_linear_uses_weight_loader_v2(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )

    assert "GGUFLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED
    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)
    assert layer.qweight.weight_loader.__name__.endswith("weight_loader_v2")

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(4, dtype=torch.uint8), 1)

    assert isinstance(layer.qweight, GGUFUninitializedParameter)
    assert len(layer.qweight.data_container) == 2
    assert isinstance(layer.qweight_type, GGUFUninitializedParameter)

    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shard_id == [0, 1]
    assert layer.qweight_type.shard_weight_type == {0: 3, 1: 4}


def test_gguf_packed_tuple_is_split_before_storage(monkeypatch) -> None:
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        gguf_params_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[2, 2, 4, 4],
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )
    packed_qkv = torch.arange(32, dtype=torch.uint8).reshape(8, 4)

    layer.qweight.weight_loader(layer.qweight, packed_qkv, (0, 1, 2))
    layer.qweight_type.weight_loader(layer.qweight_type, torch.tensor(3), (0, 1, 2))

    assert [tuple(shard.shape) for shard in layer.qweight.data_container] == [
        (2, 4),
        (2, 4),
        (4, 4),
    ]
    assert layer.qweight_type.shard_weight_type == {0: 3, 1: 3, 2: 3}


def test_gguf_row_parallel_weight_is_sharded_before_storage(monkeypatch) -> None:
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(gguf_params_module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        gguf_params_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    layer = RowParallelLinear(
        input_size=8,
        output_size=4,
        bias=False,
        input_is_parallel=True,
        quant_config=OOTGGUFConfig.from_config({}),
    )
    loaded = torch.arange(32, dtype=torch.uint8).reshape(4, 8)

    layer.qweight.weight_loader(layer.qweight, loaded)

    assert tuple(layer.qweight.shape) == (4, 4)
    assert torch.equal(layer.qweight, loaded[:, 4:])


def test_gguf_replicated_linear_materializes_uninitialized_weight() -> None:
    layer = ReplicatedLinear(
        input_size=8,
        output_size=4,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )
    loaded = torch.arange(16, dtype=torch.uint8).reshape(2, 8)

    layer.qweight.weight_loader(layer.qweight, loaded)
    layer.qweight_type.weight_loader(
        layer.qweight_type,
        torch.tensor(8, dtype=torch.uint8),
    )

    assert tuple(layer.qweight.shape) == (2, 8)
    assert torch.equal(layer.qweight, loaded)
    assert layer.qweight_type.weight_type == 8


def test_gguf_uva_materialization_retains_cpu_storage(monkeypatch) -> None:
    cpu_storage_ref = None

    def fake_accelerator_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal cpu_storage_ref
        cpu_storage_ref = weakref.ref(cpu_tensor)
        return torch.empty_like(cpu_tensor)

    monkeypatch.setattr(
        gguf_params_module,
        "get_accelerator_view_from_cpu_tensor",
        fake_accelerator_view,
    )
    param = GGUFUninitializedParameter(requires_grad=False)
    param._vllm_is_uva_offloaded = True
    param._vllm_uva_pin_memory = False

    gguf_params_module._materialize_parameter_data(param, (8, 4), torch.uint8)
    gc.collect()

    assert cpu_storage_ref is not None
    assert cpu_storage_ref() is param._vllm_uva_cpu_data
    assert tuple(param.shape) == (8, 4)


def _record_uva_host_allocations(monkeypatch) -> tuple[list, list]:
    """Capture the pinned allocations and accelerator views a UVA param makes."""
    monkeypatch.setenv("RADIANCE_UVA_HOST_COHERENCE", "default")
    monkeypatch.setenv("RADIANCE_UVA_HOST_NONCOHERENT", "0")
    pinned: list = []
    views: list = []

    def fake_pinned_empty(shape, dtype):
        pinned.append((tuple(shape), dtype))
        return torch.empty(tuple(shape), dtype=dtype)

    def fake_accelerator_view(cpu_tensor):
        views.append(cpu_tensor)
        return torch.empty_like(cpu_tensor)

    monkeypatch.setattr(gguf_params_module, "default_pinned_empty", fake_pinned_empty)
    monkeypatch.setattr(
        gguf_params_module,
        "get_accelerator_view_from_cpu_tensor",
        fake_accelerator_view,
    )
    return pinned, views


def test_gguf_tiered_expert_master_stays_pageable(monkeypatch) -> None:
    pinned, views = _record_uva_host_allocations(monkeypatch)

    param = GGUFUninitializedParameter(requires_grad=False)
    param._vllm_is_uva_offloaded = True
    param._vllm_uva_pin_memory = True
    setattr(param, tiered_compaction_module.MASTER_ATTR, True)

    gguf_params_module._materialize_parameter_data(param, (8, 4), torch.uint8)

    # The master is only a compaction source: no pinned allocation, and no
    # accelerator view either, since taking one pins unpinned input internally.
    assert pinned == []
    assert views == []
    assert param.device.type == "cpu"
    assert param.data_ptr() == param._vllm_uva_cpu_data.data_ptr()
    assert not param._vllm_uva_cpu_data.is_pinned()
    assert param._vllm_is_uva_offloaded is True


def test_gguf_uva_materialization_pins_unmarked_parameters(monkeypatch) -> None:
    pinned, views = _record_uva_host_allocations(monkeypatch)

    param = GGUFUninitializedParameter(requires_grad=False)
    param._vllm_is_uva_offloaded = True
    param._vllm_uva_pin_memory = True

    gguf_params_module._materialize_parameter_data(param, (8, 4), torch.uint8)

    assert pinned == [((8, 4), torch.uint8)]
    assert len(views) == 1
    assert tuple(param.shape) == (8, 4)


def test_gguf_uva_materialization_selects_noncoherent_storage(monkeypatch) -> None:
    allocated = []

    def fake_noncoherent_empty(shape, dtype):
        allocated.append((shape, dtype))
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setenv("RADIANCE_UVA_HOST_NONCOHERENT", "1")
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        gguf_params_module,
        "_hip_noncoherent_empty",
        fake_noncoherent_empty,
    )
    monkeypatch.setattr(
        gguf_params_module,
        "get_accelerator_view_from_cpu_tensor",
        torch.empty_like,
    )
    param = GGUFUninitializedParameter(requires_grad=False)
    param._vllm_is_uva_offloaded = True
    param._vllm_uva_pin_memory = True

    gguf_params_module._materialize_parameter_data(param, (8, 4), torch.uint8)

    assert allocated == [((8, 4), torch.uint8)]
    assert tuple(param._vllm_uva_cpu_data.shape) == (8, 4)
    assert param._vllm_uva_cpu_data.dtype == torch.uint8


@pytest.mark.parametrize(
    ("hip_version", "pin_memory"),
    [(None, True), ("test", False)],
)
def test_gguf_uva_noncoherent_selector_requires_rocm_and_pinning(
    monkeypatch, hip_version, pin_memory
) -> None:
    monkeypatch.setenv("RADIANCE_UVA_HOST_NONCOHERENT", "1")
    monkeypatch.setattr(torch.version, "hip", hip_version)
    param = GGUFUninitializedParameter(requires_grad=False)
    param._vllm_uva_pin_memory = pin_memory

    assert not gguf_params_module._use_noncoherent_uva_host_memory(param)


def test_gguf_uva_explicit_coherence_modes(monkeypatch) -> None:
    monkeypatch.setenv("RADIANCE_UVA_HOST_NONCOHERENT", "0")

    monkeypatch.setenv("RADIANCE_UVA_HOST_COHERENCE", "default")
    assert gguf_params_module._uva_host_coherence() == "default"

    monkeypatch.setenv("RADIANCE_UVA_HOST_COHERENCE", "coherent")
    assert gguf_params_module._uva_host_coherence() == "coherent"

    monkeypatch.setenv("RADIANCE_UVA_HOST_COHERENCE", "noncoherent")
    assert gguf_params_module._uva_host_coherence() == "noncoherent"


def test_gguf_uva_rejects_conflicting_coherence_modes(monkeypatch) -> None:
    monkeypatch.setenv("RADIANCE_UVA_HOST_NONCOHERENT", "1")
    monkeypatch.setenv("RADIANCE_UVA_HOST_COHERENCE", "coherent")

    with pytest.raises(ValueError, match="conflicts"):
        gguf_params_module._uva_host_coherence()


class _FakeHipHostApi:
    def __init__(self, malloc_status: int = 0) -> None:
        self.malloc_status = malloc_status
        self.allocations = {}
        self.flags = []
        self.freed = []

    def hipHostMalloc(self, pointer, size, flags):
        self.flags.append(flags.value)
        if self.malloc_status:
            return self.malloc_status
        owner = (ctypes.c_ubyte * size.value)()
        address = ctypes.addressof(owner)
        self.allocations[address] = owner
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
        return 0

    def hipHostFree(self, pointer):
        address = pointer.value
        self.freed.append(address)
        self.allocations.pop(address)
        return 0

    def hipGetErrorString(self, status):
        return f"fake HIP error {status}".encode()


def test_hip_noncoherent_empty_retains_owner_until_last_view(monkeypatch) -> None:
    api = _FakeHipHostApi()
    monkeypatch.setattr(gguf_params_module, "_hip_host_api", lambda: api)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

    tensor = gguf_params_module._hip_noncoherent_empty((4, 8), torch.bfloat16)
    view = tensor.view(-1)
    address = tensor.data_ptr()

    assert tensor.shape == (4, 8)
    assert tensor.dtype == torch.bfloat16
    assert api.flags == [gguf_params_module._HIP_HOST_MALLOC_NONCOHERENT]
    assert address in api.allocations

    del tensor
    gc.collect()
    assert api.freed == []

    del view
    gc.collect()
    assert api.freed == [address]


def test_hip_coherent_empty_uses_explicit_flag(monkeypatch) -> None:
    api = _FakeHipHostApi()
    monkeypatch.setattr(gguf_params_module, "_hip_host_api", lambda: api)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)

    tensor = gguf_params_module._hip_coherent_empty((8,), torch.uint8)

    assert api.flags == [gguf_params_module._HIP_HOST_MALLOC_COHERENT]
    del tensor
    gc.collect()
    assert len(api.freed) == 1


def test_hip_noncoherent_empty_handles_zero_size_without_hip(monkeypatch) -> None:
    monkeypatch.setattr(
        gguf_params_module,
        "_hip_host_api",
        lambda: pytest.fail("HIP API must not be loaded for an empty tensor"),
    )

    tensor = gguf_params_module._hip_noncoherent_empty((0, 4), torch.float32)

    assert tensor.shape == (0, 4)
    assert tensor.dtype == torch.float32


def test_hip_noncoherent_empty_reports_allocation_error(monkeypatch) -> None:
    api = _FakeHipHostApi(malloc_status=7)
    monkeypatch.setattr(gguf_params_module, "_hip_host_api", lambda: api)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    with pytest.raises(RuntimeError, match="fake HIP error 7"):
        gguf_params_module._hip_noncoherent_empty((8,), torch.uint8)


def test_hip_noncoherent_empty_rejects_unpinned_storage(monkeypatch) -> None:
    api = _FakeHipHostApi()
    monkeypatch.setattr(gguf_params_module, "_hip_host_api", lambda: api)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: False)

    with pytest.raises(RuntimeError, match="not recognized as pinned"):
        gguf_params_module._hip_noncoherent_empty((8,), torch.uint8)

    gc.collect()
    assert len(api.freed) == 1


def test_route_profiler_records_decode_events_without_prefill(tmp_path) -> None:
    profiler = route_profile_module._RouteProfiler(tmp_path, max_events=2, rank=0)
    prefill_ids = torch.zeros((4, 10), dtype=torch.int32)
    profiler.record("model.layers.0.mlp.experts", prefill_ids)

    for layer_id in range(48):
        ids = torch.tensor(
            [
                [(layer_id + offset) % 512 for offset in range(10)],
                [(layer_id + offset + 1) % 512 for offset in range(10)],
                [(layer_id + offset + 2) % 512 for offset in range(10)],
            ],
            dtype=torch.int32,
        )
        profiler.record(f"model.layers.{layer_id}.mlp.experts", ids)

    assert profiler.num_events == 1
    assert profiler.trace is not None
    assert torch.equal(
        profiler.trace[0, 47, 2],
        torch.tensor([(49 + offset) % 512 for offset in range(10)], dtype=torch.int16),
    )

    profiler.dump()
    output = next(tmp_path.glob("routes-rank0-pid*.npz"))
    profile = np.load(output)
    assert profile["routes"].shape == (1, 48, 3, 10)
    assert profile["rows"].tolist() == [3]


def test_route_profiler_mtp2_filter_auto_dumps_once(tmp_path) -> None:
    profiler = route_profile_module._RouteProfiler(
        tmp_path,
        max_events=1,
        rank=0,
        allowed_rows=frozenset((3,)),
        auto_dump=True,
    )

    for layer_id in range(48):
        profiler.record(
            f"model.layers.{layer_id}.mlp.experts",
            torch.full((1, 10), layer_id, dtype=torch.int32),
        )
    assert profiler.num_events == 0
    assert not list(tmp_path.glob("*.npz"))

    for layer_id in range(48):
        profiler.record(
            f"model.layers.{layer_id}.mlp.experts",
            torch.full((3, 10), layer_id, dtype=torch.int32),
        )

    output = next(tmp_path.glob("routes-rank0-pid*.npz"))
    with np.load(output) as profile:
        assert profile["routes"].shape == (1, 48, 3, 10)
        assert profile["rows"].tolist() == [3]
    assert profiler.complete

    for layer_id in range(48):
        profiler.record(
            f"model.layers.{layer_id}.mlp.experts",
            torch.zeros((3, 10), dtype=torch.int32),
        )
    assert list(tmp_path.glob("*.npz")) == [output]


def test_gguf_moe_sanitizes_invalid_expert_slots() -> None:
    ids = torch.tensor([[3, -1, 8], [-2, 0, 4]], dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])

    safe_ids, safe_weights = gguf_fused_moe_module._sanitize_moe_routing(
        ids,
        weights,
        num_experts=5,
    )

    assert torch.equal(
        safe_ids,
        torch.tensor([[3, 0, 0], [0, 0, 4]], dtype=torch.int32),
    )
    assert torch.equal(
        safe_weights,
        torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.1, 0.5]]),
    )


def test_gguf_moe_sanitizes_single_dummy_route() -> None:
    ids = torch.tensor([[-1, 8]], dtype=torch.int32)
    weights = torch.tensor([[0.4, 0.6]])

    safe_ids, safe_weights = gguf_fused_moe_module._sanitize_moe_routing(
        ids,
        weights,
        num_experts=5,
    )

    assert torch.equal(safe_ids, torch.zeros_like(ids))
    assert torch.equal(safe_weights, torch.zeros_like(weights))


def test_gguf_embedding_uses_plugin_weight_loader(monkeypatch):
    monkeypatch.delenv("GGUF_PLE_MMAP_PATH", raising=False)
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
        quant_config=OOTGGUFConfig.from_config({}),
    )

    loaded_qweight = torch.arange(60, dtype=torch.uint8).reshape(10, 6)
    layer.qweight.weight_loader(layer.qweight, loaded_qweight)
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(7, dtype=torch.uint8)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert isinstance(layer.qweight_type, GGUFWeightTypeParameter)
    assert layer.qweight.shape == (16, 6)
    assert torch.equal(layer.qweight[:10], loaded_qweight)
    assert torch.equal(layer.qweight[10:], torch.zeros((6, 6), dtype=torch.uint8))
    assert torch.equal(layer.qweight_type, torch.tensor([7], dtype=torch.uint8))
    assert layer.qweight_type.weight_type == 7


def test_gguf_embedding_can_bind_file_backed_weight(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setenv("GGUF_PLE_MMAP_TRIM_ROWS", "1")

    layer = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
        quant_config=OOTGGUFConfig.from_config({}),
    )
    loaded_qweight = torch.arange(60, dtype=torch.uint8).reshape(10, 6)
    mmap_path = tmp_path / "ple.bin"
    mmap_path.write_bytes(loaded_qweight.numpy().tobytes())
    monkeypatch.setenv("GGUF_PLE_MMAP_PATH", str(mmap_path))

    layer.qweight.weight_loader(layer.qweight, loaded_qweight)
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(7, dtype=torch.uint8)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, GGUFWeightParameter)
    assert layer.qweight.shape == loaded_qweight.shape
    assert torch.equal(layer.qweight, loaded_qweight)
    assert layer._vllm_gguf_mmap_path == str(mmap_path.resolve())
    assert layer._vllm_gguf_mmap_residency_mode == "ssd"

    gguf_params_module.maybe_trim_file_backed_embedding(layer, rows_read=1)
    assert layer._vllm_gguf_mmap_rows_since_trim == 0
    assert layer._vllm_gguf_mmap_trim_count == 1


def test_gguf_embedding_legacy_host_register_infers_pinned(monkeypatch):
    monkeypatch.delenv("VLLM_PLE_RESIDENCY_MODE", raising=False)
    monkeypatch.setenv("VLLM_PLE_MMAP_HOST_REGISTER", "1")

    assert gguf_params_module._ple_residency_mode() == "pinned"


def test_bounded_gguf_embedding_evicts_exact_packed_row_chunks(monkeypatch, tmp_path):
    mmap_path = tmp_path / "packed-rows.bin"
    mmap_path.write_bytes(bytes(100 * 90))
    qweight = torch.from_file(
        str(mmap_path),
        shared=False,
        size=100 * 90,
        dtype=torch.uint8,
    ).reshape(100, 90)
    advised_ranges = []
    fadvise_ranges = []
    monkeypatch.setattr(
        gguf_params_module,
        "_madvise_range",
        lambda address, nbytes, advice: advised_ranges.append(
            (address - qweight.data_ptr(), nbytes, advice)
        ),
    )
    monkeypatch.setattr(
        gguf_params_module.os,
        "posix_fadvise",
        lambda fd, offset, nbytes, advice: fadvise_ranges.append(
            (offset, nbytes, advice)
        ),
    )
    residency = gguf_params_module._BoundedMmapResidency(
        qweight,
        str(mmap_path),
        budget_bytes=8192,
        chunk_bytes=4096,
    )

    assert residency.row_nbytes == 90
    assert residency.prepare(torch.tensor([45])) == (2, 0, 0)
    assert residency.prepare(torch.tensor([99])) == (1, 1, 4096)

    assert advised_ranges == [(0, 4096, gguf_params_module._MADV_DONTNEED)]
    assert fadvise_ranges == [(0, 4096, os.POSIX_FADV_DONTNEED)]
    assert residency.tracked_bytes == 4904
    residency.close()


def test_mmap_row_readahead_merges_pages_for_packed_rows(monkeypatch, tmp_path):
    mmap_path = tmp_path / "readahead-packed-rows.bin"
    mmap_path.write_bytes(bytes(200 * 90))
    qweight = torch.from_file(
        str(mmap_path),
        shared=False,
        size=200 * 90,
        dtype=torch.uint8,
    ).reshape(200, 90)
    advised_ranges = []
    monkeypatch.setattr(gguf_params_module, "_native_mmap_readahead", lambda: None)
    monkeypatch.setattr(
        gguf_params_module,
        "_madvise_range",
        lambda address, nbytes, advice: advised_ranges.append(
            (address - qweight.data_ptr(), nbytes, advice)
        ),
    )
    readahead = gguf_params_module._MmapRowReadahead(qweight)

    pages, ranges, advised_bytes = readahead.prepare(torch.tensor([45, 45, 137]))

    assert (pages, ranges, advised_bytes) == (3, 2, 3 * 4096)
    assert advised_ranges == [
        (0, 8192, gguf_params_module._MADV_WILLNEED),
        (3 * 4096, 4096, gguf_params_module._MADV_WILLNEED),
    ]
    assert readahead.prepare_count == 1
    assert readahead.advised_pages == 3
    assert readahead.advised_ranges == 2
    assert readahead.advised_bytes == 3 * 4096


def test_mmap_row_readahead_rejects_invalid_row(tmp_path):
    mmap_path = tmp_path / "readahead-invalid-row.bin"
    mmap_path.write_bytes(bytes(100 * 90))
    qweight = torch.from_file(
        str(mmap_path),
        shared=False,
        size=100 * 90,
        dtype=torch.uint8,
    ).reshape(100, 90)
    readahead = gguf_params_module._MmapRowReadahead(qweight)

    with pytest.raises(IndexError, match=r"outside \[0, 100\)"):
        readahead.prepare(torch.tensor([100]))


def test_ssd_gguf_embedding_prepares_readahead(monkeypatch, tmp_path):
    mmap_path = tmp_path / "ssd-readahead.bin"
    mmap_path.write_bytes(bytes(200 * 90))
    qweight = torch.from_file(
        str(mmap_path),
        shared=False,
        size=200 * 90,
        dtype=torch.uint8,
    ).reshape(200, 90)
    advised_ranges = []
    monkeypatch.setattr(gguf_params_module, "_native_mmap_readahead", lambda: None)
    monkeypatch.setattr(
        gguf_params_module,
        "_madvise_range",
        lambda address, nbytes, advice: advised_ranges.append(
            (address - qweight.data_ptr(), nbytes, advice)
        ),
    )
    layer = SimpleNamespace(
        qweight=qweight,
        _vllm_gguf_mmap_path=str(mmap_path),
        _vllm_gguf_mmap_residency_mode="ssd",
        _vllm_gguf_mmap_readahead=True,
        _vllm_gguf_mmap_readahead_state=None,
    )

    gguf_params_module.prepare_file_backed_embedding_access(
        layer, torch.tensor([45, 137])
    )

    assert layer._vllm_gguf_mmap_last_readahead_pages == 3
    assert layer._vllm_gguf_mmap_last_readahead_ranges == 2
    assert layer._vllm_gguf_mmap_last_readahead_bytes == 3 * 4096
    assert advised_ranges == [
        (0, 8192, gguf_params_module._MADV_WILLNEED),
        (3 * 4096, 4096, gguf_params_module._MADV_WILLNEED),
    ]


def test_bounded_gguf_embedding_accepts_max_random_prefill_rows(monkeypatch, tmp_path):
    num_lookups = 16_384
    row_ids = torch.arange(num_lookups, dtype=torch.int64) * 46
    row_ids[0] = 45
    num_rows = int(row_ids.max().item()) + 1
    row_nbytes = 90
    mmap_path = tmp_path / "max-prefill-packed-rows.bin"
    with mmap_path.open("wb") as file_handle:
        file_handle.truncate(num_rows * row_nbytes)
    qweight = torch.from_file(
        str(mmap_path),
        shared=False,
        size=num_rows * row_nbytes,
        dtype=torch.uint8,
    ).reshape(num_rows, row_nbytes)
    monkeypatch.setattr(
        gguf_params_module,
        "_madvise_range",
        lambda *args: pytest.fail(f"first prefill unexpectedly evicted {args}"),
    )
    monkeypatch.setattr(
        gguf_params_module.os,
        "posix_fadvise",
        lambda *args: pytest.fail(f"first prefill unexpectedly evicted {args}"),
    )
    residency = gguf_params_module._BoundedMmapResidency(
        qweight,
        str(mmap_path),
        budget_bytes=4 * 1024**3,
        chunk_bytes=4096,
    )

    new_chunks, evicted_chunks, evicted_bytes = residency.prepare(row_ids)

    expected_chunks = set()
    for row in row_ids.tolist():
        row_start = row * row_nbytes
        expected_chunks.update(
            range(row_start // 4096, (row_start + row_nbytes - 1) // 4096 + 1)
        )
    assert new_chunks == len(expected_chunks)
    assert residency.tracked_chunks == len(expected_chunks)
    assert residency.tracked_bytes <= 2 * num_lookups * 4096
    assert evicted_chunks == 0
    assert evicted_bytes == 0
    residency.close()


def test_ssd_gguf_embedding_trim_drops_mapping_rss(tmp_path):
    mapping_bytes = 16 * 1024**2
    mmap_path = tmp_path / "ssd-rss.bin"
    with mmap_path.open("wb") as file_handle:
        file_handle.truncate(mapping_bytes)
    mapped = torch.from_file(
        str(mmap_path),
        shared=False,
        size=mapping_bytes,
        dtype=torch.uint8,
    )
    assert int(mapped[::4096].sum().item()) == 0
    rss_before = gguf_params_module._mapping_rss_bytes(mapped)

    gguf_params_module._madvise_tensor(mapped, gguf_params_module._MADV_DONTNEED)
    rss_after = gguf_params_module._mapping_rss_bytes(mapped)

    assert rss_before >= mapping_bytes
    assert rss_after < rss_before


def test_gguf_embedding_rejects_wrong_file_backed_weight(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
        quant_config=OOTGGUFConfig.from_config({}),
    )
    mmap_path = tmp_path / "ple.bin"
    mmap_path.write_bytes(b"too short")
    monkeypatch.setenv("GGUF_PLE_MMAP_PATH", str(mmap_path))

    with pytest.raises(ValueError, match="checkpoint embedding requires"):
        layer.qweight.weight_loader(
            layer.qweight,
            torch.arange(60, dtype=torch.uint8).reshape(10, 6),
        )


def test_gguf_embedding_cpu_path_bypasses_accelerator_custom_op(monkeypatch):
    method = gguf_vocab_module.GGUFEmbeddingMethod(None)
    method.params_dtype = torch.float32
    qweight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    qweight.tensor_shape = (3, 4)
    layer = SimpleNamespace(
        qweight=qweight,
        qweight_type=SimpleNamespace(weight_type=0),
    )

    def fail_accelerator_dispatch(*args, **kwargs):
        raise AssertionError("CPU embedding used the accelerator-only custom op")

    monkeypatch.setattr(
        gguf_quantization, "apply_gguf_embedding", fail_accelerator_dispatch
    )
    result = method.embedding(layer, torch.tensor([2, 0]))

    assert torch.equal(result, qweight[[2, 0]])


def test_gguf_linear_same_type_shards_skip_concat(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )
    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.qweight, torch.nn.Parameter)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_fused_mul_mat_gguf(x, qweight, qweight_type):
        calls.append((tuple(qweight.shape), qweight_type))
        return torch.zeros(
            (x.shape[0], qweight.shape[0]), dtype=x.dtype, device=x.device
        )

    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", fake_fused_mul_mat_gguf
    )
    out = layer.quant_method.apply(layer, torch.ones((2, 4), dtype=torch.float32))

    assert calls == [((8, 4), 3)]
    assert out.shape == (2, 8)


def test_gguf_linear_preserves_leading_dimensions(monkeypatch):
    method = gguf_linear_module.GGUFLinearMethod(None)
    qweight = torch.ones((8, 4), dtype=torch.uint8)
    qweight.shard_id = []
    layer = SimpleNamespace(
        qweight=qweight,
        qweight_type=SimpleNamespace(weight_type=8),
    )
    seen_shapes = []

    def fake_fused_mul_mat_gguf(x, qweight, qweight_type):
        seen_shapes.append(tuple(x.shape))
        return torch.zeros(
            (x.shape[0], qweight.shape[0]), dtype=x.dtype, device=x.device
        )

    monkeypatch.setattr(
        gguf_quantization, "fused_mul_mat_gguf", fake_fused_mul_mat_gguf
    )
    out = method.apply(layer, torch.ones((2, 3, 4), dtype=torch.float32))

    assert seen_shapes == [(6, 4)]
    assert out.shape == (2, 3, 8)


def test_gguf_config_parser_uses_parent_dir_for_local_file(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = {}

    def fake_parse(
        self, model, trust_remote_code, revision=None, code_revision=None, **kwargs
    ):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        return {}, PretrainedConfig(model_type="qwen3_moe")

    monkeypatch.setattr(
        gguf_config_parser_module.HFConfigParser,
        "parse",
        fake_parse,
    )
    monkeypatch.setattr(
        gguf_config_parser_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config: config,
    )

    config_dict, config = GGUFConfigParser().parse(gguf_path, trust_remote_code=False)

    assert calls["model"] == gguf_path.parent
    assert calls["trust_remote_code"] is False
    assert config_dict["norm_topk_prob"] is True
    assert config.architectures == ["Qwen3MoeForCausalLM"]


def test_register_sets_engine_args_for_gguf_model(monkeypatch):
    register()
    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(arg_utils_module, "ModelConfig", fake_model_config)
    engine_args = EngineArgs(model="/tmp/model.gguf", tokenizer="/tmp/tokenizer")

    engine_args.create_model_config()

    assert captured["config_format"] == "gguf"
    assert captured["model"] == "/tmp/tokenizer"
    assert captured["model_weights"] == "/tmp/model.gguf"
    assert captured["quantization"] == "gguf"
    assert engine_args.load_format == "gguf"


def test_register_skips_speculator_probe_for_gguf():
    register()

    model, tokenizer, speculative_config = (
        config_module.maybe_override_with_speculators(
            model="/tmp/model.gguf",
            tokenizer="/tmp/tokenizer",
            trust_remote_code=False,
            revision=None,
            vllm_speculative_config={"foo": "bar"},
            hf_token=None,
        )
    )

    assert model == "/tmp/model.gguf"
    assert tokenizer == "/tmp/tokenizer"
    assert speculative_config == {"foo": "bar"}


def test_gguf_qkv_shards_are_padded_in_qkv_order(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    layer = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
        disable_tp=True,
    )

    q = torch.full((4, 4), 1, dtype=torch.uint8)
    k = torch.full((2, 4), 2, dtype=torch.uint8)
    v = torch.full((2, 4), 3, dtype=torch.uint8)
    # Load out of canonical order to match GGUF tensor iteration order.
    layer.weight_loader_v2(layer.qweight, k, "k")
    layer.weight_loader_v2(layer.qweight, q, "q")
    layer.weight_loader_v2(layer.qweight, v, "v")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "k")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "q")
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), "v")

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.shard_id == ["q", "k", "v"]
    assert layer.qweight.shard_offset_map == {
        "q": (0, 4, 4),
        "k": (4, 6, 4),
        "v": (6, 8, 4),
    }
    assert torch.equal(layer.qweight[:4], q)
    assert torch.equal(layer.qweight[4:6], k)
    assert torch.equal(layer.qweight[6:8], v)


def _patch_tp(monkeypatch, tp_rank: int, tp_size: int):
    """Fake a tp_size-way TP group without initializing a process group."""
    for module in (linear_module, parameter_module, gguf_params_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: tp_size
        )


def _load_merged_column_shard(monkeypatch, tp_rank, loaded_weight):
    register()
    _patch_tp(monkeypatch, tp_rank, tp_size=2)
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
    )
    layer.weight_loader_v2(layer.qweight, loaded_weight, 0)

    return layer.qweight.data_container[0]


def _load_qkv_shard(monkeypatch, tp_rank, shard_id, loaded_weight):
    register()
    _patch_tp(monkeypatch, tp_rank, tp_size=2)
    layer = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=4,
        total_num_kv_heads=2,
        bias=False,
        quant_config=OOTGGUFConfig.from_config({}),
    )
    layer.weight_loader_v2(layer.qweight, loaded_weight, shard_id)

    return layer.qweight.data_container[0]


def test_gguf_merged_column_shard_follows_tp_rank(monkeypatch):
    # output_sizes=[4, 4] over tp_size=2 gives shard_size=2, so each rank must
    # take a different half of the 4 rows the GGUF file holds for shard 0.
    loaded_weight = torch.arange(16, dtype=torch.uint8).reshape(4, 4)

    rank0 = _load_merged_column_shard(monkeypatch, 0, loaded_weight)
    rank1 = _load_merged_column_shard(monkeypatch, 1, loaded_weight)

    assert torch.equal(rank0, loaded_weight[0:2])
    assert torch.equal(rank1, loaded_weight[2:4])


def test_gguf_qkv_shard_follows_tp_rank(monkeypatch):
    # total_num_heads=4 and total_num_kv_heads=2 over tp_size=2 gives
    # num_kv_head_replicas=1, q shard_size=4 of 8 rows, k shard_size=2 of 4.
    q = torch.arange(32, dtype=torch.uint8).reshape(8, 4)
    k = torch.arange(16, dtype=torch.uint8).reshape(4, 4)

    q_rank0 = _load_qkv_shard(monkeypatch, 0, "q", q)
    q_rank1 = _load_qkv_shard(monkeypatch, 1, "q", q)
    k_rank0 = _load_qkv_shard(monkeypatch, 0, "k", k)
    k_rank1 = _load_qkv_shard(monkeypatch, 1, "k", k)

    assert torch.equal(q_rank0, q[0:4])
    assert torch.equal(q_rank1, q[4:8])
    assert torch.equal(k_rank0, k[0:2])
    assert torch.equal(k_rank1, k[2:4])


def test_gguf_linear_preserves_cuda_weight_device(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    try:
        torch.empty(0, device="cuda")
    except RuntimeError:
        # ROCm builds can report logical devices in no-device build containers.
        return

    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    with torch.device("cuda"):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_sizes=[4, 4],
            bias=False,
            quant_config=OOTGGUFConfig.from_config({}),
            disable_tp=True,
        )

    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.device.type == "cuda"
    assert layer.qweight_type.device.type == "cuda"


def test_gguf_merged_column_releases_shards_after_concat(monkeypatch):
    register()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    quant_config = OOTGGUFConfig.from_config({})
    layer = MergedColumnParallelLinear(
        input_size=4,
        output_sizes=[4, 4],
        bias=False,
        quant_config=quant_config,
        disable_tp=True,
    )
    layer.weight_loader_v2(layer.qweight, torch.ones((4, 4), dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight, 2 * torch.ones((4, 4), dtype=torch.uint8), 1)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 0)
    layer.weight_loader_v2(layer.qweight_type, torch.tensor(3, dtype=torch.uint8), 1)

    # The loading path keeps the pre-materialization parameter alive past
    # process_weights_after_loading; hold it here so the shards can only be
    # freed if materialization handed its containers over instead of copying.
    source_param = layer.qweight
    shard_refs = [weakref.ref(shard) for shard in source_param.data_container]

    layer.quant_method.process_weights_after_loading(layer)
    gc.collect()

    assert layer.qweight is not source_param
    assert not source_param.data_container
    assert all(ref() is None for ref in shard_refs)

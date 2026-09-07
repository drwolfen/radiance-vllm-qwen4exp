# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import pytest
import torch

import vllm_gguf_plugin.weights_adapter.qwen4_exp as qwen4_exp_module
from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weights_adapter.qwen4_exp import (
    QWEN4_EXP_ARCHITECTURE,
    QWEN4_EXP_MTP_ARCHITECTURE,
    QWEN4_EXP_MULTIMODAL_ARCHITECTURE,
    Qwen4ExpGGUFAdapter,
    build_qwen4_exp_mtp_name_map,
    build_qwen4_exp_name_map,
    dequantize_qwen4_exp_packed_hc_down,
    merge_qwen4_exp_indexer_projections,
)


def test_qwen4_exp_adapter_registration_contract() -> None:
    config = SimpleNamespace(model_type="qwen4_exp")
    assert Qwen4ExpGGUFAdapter.matches(config)
    assert Qwen4ExpGGUFAdapter.architecture(config) == QWEN4_EXP_ARCHITECTURE

    mtp_config = SimpleNamespace(model_type="qwen4_exp_mtp")
    assert Qwen4ExpGGUFAdapter.matches(mtp_config)
    assert (
        Qwen4ExpGGUFAdapter.architecture(mtp_config)
        == QWEN4_EXP_MTP_ARCHITECTURE
    )


def test_qwen4_exp_adapter_selects_multimodal_architecture(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_GGUF_QWEN4_EXP_MULTIMODAL", "1")

    assert (
        Qwen4ExpGGUFAdapter.architecture(
            SimpleNamespace(model_type="qwen4_exp")
        )
        == QWEN4_EXP_MULTIMODAL_ARCHITECTURE
    )


def test_qwen4_exp_mtp_ignores_target_multimodal_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_GGUF_QWEN4_EXP_MULTIMODAL", "1")
    draft_config = SimpleNamespace(model_type="qwen4_exp_mtp")
    monkeypatch.setattr(
        qwen4_exp_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config, mmproj_path=None: config,
    )

    patched = Qwen4ExpGGUFAdapter().patch_hf_config(
        GGUFModelFiles(("mtp.gguf",)),
        draft_config,
    )

    assert patched is draft_config
    assert patched.architectures == [QWEN4_EXP_MTP_ARCHITECTURE]


def test_qwen4_exp_mtp_rejects_its_own_mmproj(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_GGUF_QWEN4_EXP_MULTIMODAL", "1")
    monkeypatch.setattr(
        qwen4_exp_module,
        "maybe_patch_hf_config_from_gguf",
        lambda *args, **kwargs: pytest.fail("draft projector reached config patching"),
    )

    with pytest.raises(RuntimeError, match="must not load an mmproj"):
        Qwen4ExpGGUFAdapter().patch_hf_config(
            GGUFModelFiles(("mtp.gguf",), "mmproj.gguf"),
            SimpleNamespace(model_type="qwen4_exp_mtp"),
        )


def test_qwen4_exp_target_still_requires_projector_with_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_GGUF_QWEN4_EXP_MULTIMODAL", "1")
    monkeypatch.setattr(
        qwen4_exp_module,
        "maybe_patch_hf_config_from_gguf",
        lambda *args, **kwargs: pytest.fail("invalid target reached config patching"),
    )

    with pytest.raises(RuntimeError, match="enabled without an mmproj"):
        Qwen4ExpGGUFAdapter().patch_hf_config(
            GGUFModelFiles(("target.gguf",)),
            SimpleNamespace(model_type="qwen4_exp"),
        )


def test_qwen4_exp_adapter_exposes_ple_offload_prefix() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(ple_layer_ids=[2])
        )
    )

    assert Qwen4ExpGGUFAdapter().get_ple_offload_prefixes(model_config) == (
        "model.layers.1.ple.ple_embedding.",
    )


def test_qwen4_exp_adapter_exposes_multimodal_ple_offload_prefix() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[QWEN4_EXP_MULTIMODAL_ARCHITECTURE],
            get_text_config=lambda: SimpleNamespace(ple_layer_ids=[2]),
        )
    )

    assert Qwen4ExpGGUFAdapter().get_ple_offload_prefixes(model_config) == (
        "model.language_model.layers.1.ple.ple_embedding.",
    )


def test_qwen4_exp_name_map_covers_hc_qsa_and_ple() -> None:
    names = {
        "token_embd.weight",
        "output_hc_norm.weight",
        "blk.3.hc_attn_norm.weight",
        "blk.3.hc_attn_down.weight",
        "blk.3.indexer.q_proj.weight",
        "blk.3.indexer.k_norm.weight",
        "blk.1.ple_key.weight",
        "per_layer_token_embd.weight",
    }
    name_map, unmapped = build_qwen4_exp_name_map(
        names,
        SimpleNamespace(ple_layer_ids=[2]),
    )

    assert not unmapped
    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert name_map["output_hc_norm.weight"] == (
        "model.hyper_connection_mixer.hc_norm.weight"
    )
    assert name_map["blk.3.hc_attn_norm.weight"] == (
        "model.layers.3.attn_hyper_connection.hc_norm.weight"
    )
    assert name_map["blk.3.hc_attn_down.weight"] == (
        "model.layers.3.attn_hyper_connection.input_mix_weight_down.weight"
    )
    assert name_map["blk.3.indexer.q_proj.weight"] == (
        "model.layers.3.self_attn.indexer.index_q_proj.weight"
    )
    assert name_map["blk.3.indexer.k_norm.weight"] == (
        "model.layers.3.self_attn.indexer.k_layernorm.weight"
    )
    assert name_map["blk.1.ple_key.weight"] == ("model.layers.1.ple.key_proj.weight")
    assert name_map["per_layer_token_embd.weight"] == (
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    )


def test_qwen4_exp_multimodal_name_map_covers_text_vision_and_ple() -> None:
    names = {
        "token_embd.weight",
        "blk.3.hc_attn_norm.weight",
        "per_layer_token_embd.weight",
        "v.blk.0.attn_qkv.weight",
        "v.patch_embd.weight.1",
        "mm.2.weight",
    }
    name_map, unmapped = build_qwen4_exp_name_map(
        names,
        SimpleNamespace(ple_layer_ids=[2]),
        is_multimodal=True,
    )

    assert not unmapped
    assert name_map["token_embd.weight"] == (
        "model.language_model.embed_tokens.weight"
    )
    assert name_map["blk.3.hc_attn_norm.weight"] == (
        "model.language_model.layers.3.attn_hyper_connection.hc_norm.weight"
    )
    assert name_map["per_layer_token_embd.weight"] == (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    )
    assert name_map["v.blk.0.attn_qkv.weight"] == (
        "model.visual.blocks.0.attn.qkv.weight"
    )
    assert name_map["v.patch_embd.weight.1"] == (
        "model.visual.patch_embed.proj.weight.1"
    )
    assert name_map["mm.2.weight"] == "model.visual.merger.linear_fc2.weight"


def test_qwen4_exp_name_map_rejects_ambiguous_ple_layers() -> None:
    with pytest.raises(ValueError, match="exactly one PLE layer"):
        build_qwen4_exp_name_map([], SimpleNamespace(ple_layer_ids=[2, 4]))


def test_qwen4_exp_mtp_name_map_uses_draft_prefixes() -> None:
    names = {
        "mtp.fc_embedding.weight",
        "mtp.hc_down.weight",
        "blk.48.ffn_down_exps.weight",
        "blk.48.indexer.q_proj.weight",
    }
    name_map, unmapped = build_qwen4_exp_mtp_name_map(
        names,
        SimpleNamespace(num_hidden_layers=48),
    )

    assert not unmapped
    assert name_map["mtp.fc_embedding.weight"] == "mtp.fc_embedding.weight"
    assert name_map["mtp.hc_down.weight"] == (
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
    )
    assert name_map["blk.48.ffn_down_exps.weight"] == (
        "mtp.layers.48.mlp.experts.0.down_proj.weight"
    )
    assert name_map["blk.48.indexer.q_proj.weight"] == (
        "mtp.layers.48.self_attn.indexer.index_q_proj.weight"
    )


def test_qwen4_exp_mtp_has_no_ple_offload_prefix() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen4_exp_mtp")
    )

    assert Qwen4ExpGGUFAdapter().get_ple_offload_prefixes(model_config) == ()


def test_merge_qwen4_exp_indexer_quantized_projections() -> None:
    prefix = "model.layers.3.self_attn.indexer"
    quant_type = torch.tensor(23)
    weights = [
        (f"{prefix}.index_q_proj.qweight_type", quant_type),
        (f"{prefix}.index_q_proj.qweight", torch.ones((4, 3))),
        (f"{prefix}.index_k_proj.qweight_type", quant_type.clone()),
        (f"{prefix}.index_k_proj.qweight", torch.full((2, 3), 2.0)),
    ]

    merged = dict(merge_qwen4_exp_indexer_projections(weights))

    assert torch.equal(merged[f"{prefix}.index_qk_proj.qweight_type"], quant_type)
    assert torch.equal(
        merged[f"{prefix}.index_qk_proj.qweight"],
        torch.cat((torch.ones((4, 3)), torch.full((2, 3), 2.0))),
    )


def test_merge_qwen4_exp_indexer_requires_matching_quant_types() -> None:
    prefix = "model.layers.3.self_attn.indexer"
    weights = [
        (f"{prefix}.index_q_proj.qweight_type", torch.tensor(23)),
        (f"{prefix}.index_k_proj.qweight_type", torch.tensor(24)),
    ]

    with pytest.raises(ValueError, match="different quantization types"):
        list(merge_qwen4_exp_indexer_projections(weights))


def test_qwen4_exp_marks_mixed_packed_hc_projection_unquantized(monkeypatch) -> None:
    prefix = "model.layers.3.attn_hyper_connection."
    adapter = Qwen4ExpGGUFAdapter()
    files = SimpleNamespace(backbone=("model.gguf",))
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.qwen4_exp."
        "get_gguf_unquantized_params",
        lambda files: [],
    )
    modules = adapter.get_additional_unquantized_modules(
        files,
        SimpleNamespace(),
        {
            "hc_down": f"{prefix}input_mix_weight_down.weight",
            "hc_inject": f"{prefix}block_inject_weight.weight",
        },
    )

    assert modules == (
        f"{prefix}_input_mix_padding",
        f"{prefix}input_mix_weight_down",
    )


def test_qwen4_exp_marks_fused_dense_indexer_projection_unquantized(
    monkeypatch,
) -> None:
    prefix = "model.layers.3.self_attn.indexer"
    name_map = {
        "blk.3.indexer.q_proj.weight": f"{prefix}.index_q_proj.weight",
        "blk.3.indexer.k_proj.weight": f"{prefix}.index_k_proj.weight",
    }
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.qwen4_exp."
        "get_gguf_unquantized_params",
        lambda files: list(name_map),
    )

    modules = Qwen4ExpGGUFAdapter().get_additional_unquantized_modules(
        SimpleNamespace(backbone=("model.gguf",)),
        SimpleNamespace(),
        name_map,
    )

    assert modules == (f"{prefix}.index_qk_proj",)


def test_qwen4_exp_rejects_mixed_indexer_projection_precision(monkeypatch) -> None:
    prefix = "model.layers.3.self_attn.indexer"
    q_raw = "blk.3.indexer.q_proj.weight"
    k_raw = "blk.3.indexer.k_proj.weight"
    name_map = {
        q_raw: f"{prefix}.index_q_proj.weight",
        k_raw: f"{prefix}.index_k_proj.weight",
    }
    monkeypatch.setattr(
        "vllm_gguf_plugin.weights_adapter.qwen4_exp."
        "get_gguf_unquantized_params",
        lambda files: [q_raw],
    )

    with pytest.raises(ValueError, match="mixed dense and quantized precision"):
        Qwen4ExpGGUFAdapter().get_additional_unquantized_modules(
            SimpleNamespace(backbone=("model.gguf",)),
            SimpleNamespace(),
            name_map,
        )


def test_dequantize_qwen4_exp_packed_hc_down(monkeypatch) -> None:
    prefix = "model.layers.3.attn_hyper_connection.input_mix_weight_down"
    quantized = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    dense = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    seen = {}

    def fake_dequantize(data, weight_type):
        seen["data"] = torch.from_numpy(data.copy())
        seen["weight_type"] = weight_type
        return dense.numpy()

    monkeypatch.setattr(gguf, "dequantize", fake_dequantize)
    weights = [
        (f"{prefix}.qweight_type", torch.tensor(int(gguf.GGMLQuantizationType.Q8_0))),
        (f"{prefix}.qweight", quantized),
    ]

    result = list(dequantize_qwen4_exp_packed_hc_down(weights))

    assert result[0][0] == f"{prefix}.weight"
    assert torch.equal(result[0][1], dense)
    assert torch.equal(seen["data"], quantized)
    assert seen["weight_type"] == gguf.GGMLQuantizationType.Q8_0

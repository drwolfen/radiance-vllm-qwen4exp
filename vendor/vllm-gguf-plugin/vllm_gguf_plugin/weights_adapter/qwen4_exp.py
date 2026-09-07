# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names, get_gguf_unquantized_params
from .base import GGUFWeight
from .qwen3_5 import (
    Qwen35GGUFAdapter,
    build_qwen35_vision_mapper,
    qwen35_layer_substr,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN4_EXP_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp")
QWEN4_EXP_ARCHITECTURE = "Qwen4ExpForCausalLM"
QWEN4_EXP_MULTIMODAL_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
QWEN4_EXP_MTP_ARCHITECTURE = "Qwen4ExpMTP"
QWEN4_EXP_MULTIMODAL_ENV = "VLLM_GGUF_QWEN4_EXP_MULTIMODAL"

QWEN4_EXP_SUBSTR: dict[str, str] = {
    "hc_attn_norm.": "attn_hyper_connection.hc_norm.",
    "hc_attn_down.": "attn_hyper_connection.input_mix_weight_down.",
    "hc_attn_up.": "attn_hyper_connection.input_mix_weight_up.",
    "hc_attn_inject.": "attn_hyper_connection.block_inject_weight.",
    "hc_ffn_norm.": "mlp_hyper_connection.hc_norm.",
    "hc_ffn_down.": "mlp_hyper_connection.input_mix_weight_down.",
    "hc_ffn_up.": "mlp_hyper_connection.input_mix_weight_up.",
    "hc_ffn_inject.": "mlp_hyper_connection.block_inject_weight.",
    "indexer.q_proj.": "self_attn.indexer.index_q_proj.",
    "indexer.k_proj.": "self_attn.indexer.index_k_proj.",
    "indexer.q_norm.": "self_attn.indexer.q_layernorm.",
    "indexer.k_norm.": "self_attn.indexer.k_layernorm.",
    "ple_key.": "ple.key_proj.",
    "ple_value.": "ple.value_proj.",
    "ple_norm_key.": "ple.norm_key.",
    "ple_norm_query.": "ple.norm_query.",
    "ple_norm_conv.": "ple.norm_conv.",
    "ple_conv1d.": "ple.conv1d.",
}

_INDEXER_PROJ_RE = re.compile(
    r"^(?P<prefix>.+\.self_attn\.indexer)\.index_(?P<part>[qk])_proj\."
    r"(?P<suffix>weight|qweight|qweight_type)$"
)
_PACKED_HC_DOWN_SUFFIX = "_hyper_connection.input_mix_weight_down"


def build_qwen4_exp_text_mapper(is_multimodal: bool = False) -> WeightsMapper:
    """Map llama.cpp Qwen4Exp GGUF names to upstream checkpoint names."""
    backbone_prefix = "model.language_model." if is_multimodal else "model."
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": backbone_prefix + "embed_tokens.",
            "blk.": backbone_prefix + "layers.",
            "output_hc_norm.": backbone_prefix + "hyper_connection_mixer.hc_norm.",
            "output_hc_down.": (
                backbone_prefix + "hyper_connection_mixer.input_mix_weight_down."
            ),
            "output_hc_up.": (
                backbone_prefix + "hyper_connection_mixer.input_mix_weight_up."
            ),
            "output.": "lm_head.",
        },
        # Qwen4Exp's HC names contain shorter Qwen3.5 substrings such as
        # ``attn_norm.``. Put the more-specific mappings first because
        # WeightsMapper applies substring replacements in insertion order.
        orig_to_new_substr=QWEN4_EXP_SUBSTR | qwen35_layer_substr(is_moe=True),
    )


def _mapped_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


def build_qwen4_exp_name_map(
    tensor_names: Iterable[str],
    text_config: PretrainedConfig,
    *,
    is_multimodal: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Build the Qwen4Exp map without instantiating the 176B HF model."""
    ple_layer_ids = tuple(int(layer_id) for layer_id in text_config.ple_layer_ids)
    if len(ple_layer_ids) != 1:
        raise ValueError(
            "Qwen4Exp GGUF loading currently requires exactly one PLE layer, "
            f"got {ple_layer_ids}"
        )
    ple_layer_index = ple_layer_ids[0] - 1
    if ple_layer_index < 0:
        raise ValueError(f"PLE layer ids are 1-based, got {ple_layer_ids[0]}")

    text_mapper = build_qwen4_exp_text_mapper(is_multimodal)
    vision_mapper = build_qwen35_vision_mapper()
    backbone_prefix = "model.language_model." if is_multimodal else "model."
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in sorted(tensor_names):
        if name == "per_layer_token_embd.weight":
            name_map[name] = (
                f"{backbone_prefix}layers.{ple_layer_index}.ple.ple_embedding."
                "ngram_embedding.weight"
            )
            continue
        mapper = (
            vision_mapper
            if is_multimodal and name.startswith(("v.", "mm."))
            else text_mapper
        )
        mapped = _mapped_name(mapper, name)
        if mapped is None:
            unmapped.append(name)
        else:
            name_map[name] = mapped
    return name_map, unmapped


def build_qwen4_exp_mtp_name_map(
    tensor_names: Iterable[str],
    text_config: PretrainedConfig,
) -> tuple[dict[str, str], list[str]]:
    """Map the standalone Qwen4Exp MTP GGUF into draft-model prefixes."""
    layer_index = int(text_config.num_hidden_layers)
    mapper = WeightsMapper(
        orig_to_new_prefix={
            f"blk.{layer_index}.": f"mtp.layers.{layer_index}.",
        },
        orig_to_new_substr=QWEN4_EXP_SUBSTR | qwen35_layer_substr(is_moe=True),
    )
    direct_names = {
        "mtp.fc_embedding.weight": "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight": "mtp.fc_hidden.weight",
        "mtp.hc_norm.weight": "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hc_down.weight": (
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
        ),
        "mtp.hc_up.weight": (
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
        ),
        "mtp.pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
    }
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in sorted(tensor_names):
        if name in direct_names:
            name_map[name] = direct_names[name]
            continue
        mapped = _mapped_name(mapper, name)
        if mapped is None:
            unmapped.append(name)
        else:
            name_map[name] = mapped
    return name_map, unmapped


def merge_qwen4_exp_indexer_projections(
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
    """Reassemble the GGUF indexer Q/K tensors into vLLM's fused projection."""
    pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, weight in weights:
        match = _INDEXER_PROJ_RE.match(name)
        if match is None:
            yield name, weight
            continue
        key = (match.group("prefix"), match.group("suffix"))
        part = match.group("part")
        if part in pending.setdefault(key, {}):
            raise ValueError(f"Duplicate Qwen4Exp indexer {part} tensor: {name}")
        pending[key][part] = weight

    for (prefix, suffix), parts in sorted(pending.items()):
        if set(parts) != {"q", "k"}:
            raise ValueError(
                "Qwen4Exp GGUF indexer projection is incomplete for "
                f"{prefix}.{suffix}: found {sorted(parts)}"
            )
        q_weight = parts["q"]
        k_weight = parts["k"]
        output_name = f"{prefix}.index_qk_proj.{suffix}"
        if suffix == "qweight_type":
            if not torch.equal(q_weight, k_weight):
                raise ValueError(
                    "Qwen4Exp GGUF indexer Q/K projections use different "
                    f"quantization types at {prefix}"
                )
            yield output_name, q_weight
        else:
            if q_weight.ndim != k_weight.ndim:
                raise ValueError(
                    "Qwen4Exp GGUF indexer Q/K ranks differ at "
                    f"{prefix}: {q_weight.ndim} != {k_weight.ndim}"
                )
            yield output_name, torch.cat((q_weight, k_weight), dim=0)


def dequantize_qwen4_exp_packed_hc_down(
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
    """Make the mixed-precision HC down/injection projection loadable.

    The published IQ4 GGUF stores the down shard as Q8_0 and the adjacent
    injection shard as F32. vLLM represents both as one fused linear, whose
    generic GGUF method requires one precision across all logical shards.
    Dequantize only the down shard; the separate HC up projections remain
    quantized.
    """
    pending_types: dict[str, int] = {}
    for name, weight in weights:
        if name.endswith(f"{_PACKED_HC_DOWN_SUFFIX}.qweight_type"):
            base = name.removesuffix(".qweight_type")
            pending_types[base] = int(weight.item())
            continue

        if name.endswith(f"{_PACKED_HC_DOWN_SUFFIX}.qweight"):
            base = name.removesuffix(".qweight")
            if base not in pending_types:
                raise ValueError(f"Missing GGUF quantization type for {name}")
            weight_type = gguf.GGMLQuantizationType(pending_types.pop(base))
            dense = gguf.dequantize(weight.contiguous().numpy(), weight_type)
            yield f"{base}.weight", torch.from_numpy(dense)
            continue

        yield name, weight

    if pending_types:
        raise ValueError(
            "Missing Qwen4Exp packed HC weight data for: "
            f"{sorted(pending_types)}"
        )


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Qwen3.8-Flash-Next text and multimodal GGUF adapter."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4_EXP_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        if config.model_type == "qwen4_exp_mtp":
            return QWEN4_EXP_MTP_ARCHITECTURE
        if os.getenv(QWEN4_EXP_MULTIMODAL_ENV) == "1":
            return QWEN4_EXP_MULTIMODAL_ARCHITECTURE
        return QWEN4_EXP_ARCHITECTURE

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        is_mtp_draft = hf_config.model_type == "qwen4_exp_mtp"
        multimodal_enabled = os.getenv(QWEN4_EXP_MULTIMODAL_ENV) == "1"

        # Qwen4ExpMTP consumes image embeddings produced by the target model;
        # it does not own a vision tower or projector.  The target and draft
        # loaders share this process-wide opt-in, so do not interpret it as a
        # requirement for a second mmproj on the draft.  Conversely, reject an
        # unexpected draft projector before config patching can turn the draft
        # into a projector-owning multimodal model.
        if is_mtp_draft:
            if files.mm_proj is not None:
                raise RuntimeError(
                    "Qwen4ExpMTP uses target-produced multimodal embeddings "
                    "and must not load an mmproj GGUF"
                )
        else:
            if files.mm_proj is not None and not multimodal_enabled:
                raise RuntimeError(
                    "Qwen4Exp mmproj loading requires "
                    f"{QWEN4_EXP_MULTIMODAL_ENV}=1"
                )
            if files.mm_proj is None and multimodal_enabled:
                raise RuntimeError(
                    "Qwen4Exp multimodal mode was enabled without an mmproj GGUF"
                )

        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        architecture = (
            QWEN4_EXP_MULTIMODAL_ARCHITECTURE
            if files.mm_proj is not None
            else self.architecture(hf_config)
        )
        patched.architectures = [architecture]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        tensor_names = get_gguf_tensor_names(files.all_files)
        text_config = model_config.hf_config.get_text_config()
        if getattr(model_config.hf_config, "model_type", None) == "qwen4_exp_mtp":
            name_map, unmapped = build_qwen4_exp_mtp_name_map(
                tensor_names,
                text_config,
            )
        else:
            name_map, unmapped = build_qwen4_exp_name_map(
                tensor_names,
                text_config,
                is_multimodal=files.mm_proj is not None,
            )
        if unmapped:
            logger.warning(
                "No HF name for %d Qwen4Exp GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def get_ple_offload_prefixes(
        self,
        model_config: ModelConfig,
    ) -> tuple[str, ...]:
        """Return the mapped Qwen4Exp n-gram table subtree."""
        if getattr(model_config.hf_config, "model_type", None) == "qwen4_exp_mtp":
            return ()
        text_config = model_config.hf_config.get_text_config()
        ple_layer_ids = tuple(int(layer_id) for layer_id in text_config.ple_layer_ids)
        if len(ple_layer_ids) != 1:
            raise ValueError(
                "Qwen4Exp GGUF loading currently requires exactly one PLE layer, "
                f"got {ple_layer_ids}"
            )
        architectures = getattr(model_config.hf_config, "architectures", ()) or ()
        backbone_prefix = (
            "model.language_model."
            if QWEN4_EXP_MULTIMODAL_ARCHITECTURE in architectures
            else "model."
        )
        return (
            f"{backbone_prefix}layers.{ple_layer_ids[0] - 1}.ple.ple_embedding.",
        )

    def get_additional_unquantized_modules(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
        name_map: dict[str, str],
    ) -> tuple[str, ...]:
        del model_config
        mapped_names = set(name_map.values())
        mapped_unquantized_names = {
            name_map[raw_name]
            for raw_name in get_gguf_unquantized_params(list(files.backbone))
            if raw_name in name_map
        }
        modules: set[str] = set()
        injection_suffix = "_hyper_connection.block_inject_weight.weight"
        for name in mapped_names:
            if not name.endswith(injection_suffix):
                continue
            prefix = name[: -len("block_inject_weight.weight")]
            down_name = f"{prefix}input_mix_weight_down.weight"
            if down_name not in mapped_names:
                raise ValueError(
                    "Qwen4Exp HC injection has no matching down projection: "
                    f"{name}"
                )
            modules.add(down_name.removesuffix(".weight"))
            modules.add(f"{prefix}_input_mix_padding")

        # The checkpoint exposes indexer Q and K separately, while vLLM
        # constructs one fused index_qk_proj module.  The generic GGUF scan
        # consequently marks only the pre-fusion names as dense.  Carry the
        # source tensors' precision decision onto the module that will
        # actually be instantiated.
        q_suffix = ".index_q_proj.weight"
        for q_name in mapped_names:
            if not q_name.endswith(q_suffix):
                continue
            prefix = q_name[: -len(q_suffix)]
            k_name = f"{prefix}.index_k_proj.weight"
            if k_name not in mapped_names:
                raise ValueError(
                    "Qwen4Exp indexer Q projection has no matching K projection: "
                    f"{q_name}"
                )
            q_is_unquantized = q_name in mapped_unquantized_names
            k_is_unquantized = k_name in mapped_unquantized_names
            if q_is_unquantized != k_is_unquantized:
                raise ValueError(
                    "Qwen4Exp indexer Q/K projections use mixed dense and "
                    f"quantized precision at {prefix}"
                )
            if q_is_unquantized:
                modules.add(f"{prefix}.index_qk_proj")
        return tuple(sorted(modules))

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        weights = dequantize_qwen4_exp_packed_hc_down(weights)
        merged = merge_qwen4_exp_indexer_projections(weights)
        yield from super().transform_weights(merged, model_config)

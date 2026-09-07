# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp MTP helpers."""

import re

import torch

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModalEmbeddings,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    _flatten_embeddings,
    _merge_multimodal_embeddings,
)


QWEN4_EXP_MTP_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "input_mix_weight_down_block_inject": [
        "input_mix_weight_down",
        "block_inject_weight",
        # Runtime-only zero rows used to align the merged skinny GEMM.  There
        # is intentionally no checkpoint tensor/config entry for this shard.
        "_input_mix_padding",
    ],
}


def remap_mtp_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
    packed_modules_mapping: dict[str, list[str]] | None = None,
) -> list[str]:
    """Remap MTP layer numbers and make fused precision decisions complete.

    Official Qwen4Exp checkpoints list the real checkpoint projections in
    ``modules_to_not_convert``.  Runtime can combine those projections into a
    single linear.  In particular, the HC down+inject fusion also contains a
    synthetic padding shard that can never appear in checkpoint metadata.  If
    all real shards of a runtime fusion are ignored, add the fused name itself;
    ``is_layer_skipped`` then applies one BF16 decision to the complete module,
    including synthetic shards.  A genuinely partial real-shard declaration is
    deliberately left partial so the quantization layer still fails closed.
    """

    mapping = packed_modules_mapping or QWEN4_EXP_MTP_PACKED_MODULES_MAPPING
    remapped: list[str] = []
    seen: set[str] = set()
    for name in ignored_layers:
        if name.startswith("mtp."):
            name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda match: str(mtp_start_layer_idx + int(match.group(0))),
                name,
            )
        if name not in seen:
            remapped.append(name)
            seen.add(name)

    additions: set[str] = set()
    mtp_names = {name for name in seen if name.startswith("mtp.")}
    for fused_name, shard_names in mapping.items():
        # Leading-underscore shards are runtime-only implementation details,
        # not checkpoint projections that can be present in an ignore list.
        real_shards = [name for name in shard_names if not name.startswith("_")]
        candidates: dict[str, set[str]] = {}
        for ignored_name in mtp_names:
            for shard_name in real_shards:
                suffix = f".{shard_name}"
                if ignored_name.endswith(suffix):
                    fused_prefix = f"{ignored_name[: -len(suffix)]}.{fused_name}"
                    candidates.setdefault(fused_prefix, set()).add(shard_name)
        required = set(real_shards)
        additions.update(
            fused_prefix
            for fused_prefix, present in candidates.items()
            if present == required
        )

    remapped.extend(sorted(additions - seen))
    return remapped


class Qwen4ExpMTPMultimodalMixin(SupportsMultiModalEmbeddings):
    """Merge target-produced multimodal embeddings into MTP inputs."""

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return self.model.embed_input_ids(input_ids)  # type: ignore[attr-defined]

        is_multimodal = _require_is_multimodal(is_multimodal)
        if is_multimodal.dtype != torch.bool:
            raise TypeError("is_multimodal must be a boolean tensor")
        if is_multimodal.shape != input_ids.shape:
            raise ValueError(
                "is_multimodal must have the same shape as input_ids; "
                f"got {tuple(is_multimodal.shape)} and {tuple(input_ids.shape)}"
            )

        mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
        num_mm_tokens = mm_embeds_flat.shape[0]
        num_placeholders = int(is_multimodal.sum().item())
        if num_mm_tokens != num_placeholders:
            raise ValueError(
                f"Attempted to assign {num_mm_tokens} multimodal tokens "
                f"to {num_placeholders} placeholders"
            )

        safe_input_ids = input_ids.masked_fill(
            is_multimodal.to(device=input_ids.device, non_blocking=True), 0
        )
        inputs_embeds = self.model.embed_input_ids(  # type: ignore[attr-defined]
            safe_input_ids
        )
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=mm_embeds_flat,
            is_multimodal=is_multimodal,
        )


__all__ = [
    "QWEN4_EXP_MTP_PACKED_MODULES_MAPPING",
    "Qwen4ExpMTPMultimodalMixin",
    "remap_mtp_ignored_layers",
]

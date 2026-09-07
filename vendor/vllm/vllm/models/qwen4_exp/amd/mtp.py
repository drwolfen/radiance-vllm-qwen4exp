# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - consumes external multi-modal embeddings produced by the target model,
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

import os
from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.mtp import Qwen4ExpMTPMultimodalMixin, remap_mtp_ignored_layers
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
    packed_modules_mapping: dict[str, list[str]] | None = None,
) -> list[str]:
    return remap_mtp_ignored_layers(
        ignored_layers,
        mtp_start_layer_idx,
        packed_modules_mapping,
    )


def _remap_mtp_weight_name(
    name: str,
    mtp_start_layer_idx: int = 0,
) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        name = name.replace("mtp.", "model.", 1)
        # Standalone MTP modules are held in a zero-based ModuleList even
        # when a GGUF uses the target model's absolute block numbers. Keep
        # the absolute prefix during initialization so GGUF quantization
        # metadata matches, then translate it only for parameter loading.
        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", name)
        if layer_match is not None:
            checkpoint_idx = int(layer_match.group(1))
            if checkpoint_idx >= mtp_start_layer_idx:
                local_idx = checkpoint_idx - mtp_start_layer_idx
                name = f"model.layers.{local_idx}.{layer_match.group(2)}"
        return name
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(
                    ignored_layers,
                    mtp_start_layer_idx,
                    draft_quant_config.packed_modules_mapping,
                ),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(
                    exclude_modules,
                    mtp_start_layer_idx,
                    draft_quant_config.packed_modules_mapping,
                ),
            )

    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
        parallel_config=speculative_config.draft_parallel_config,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


def _combine_mtp_fc_shards(
    local_embedding: torch.Tensor,
    local_hidden: torch.Tensor,
    *,
    gather_output: bool,
) -> torch.Tensor:
    """Add aligned column shards before the optional TP all-gather."""
    hidden_states = local_hidden + local_embedding.unsqueeze(-2)
    if gather_output:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)
    return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.mtp_owner_rank: int | None = None
        owner = os.environ.get("VLLM_QWEN4_EXP_MTP_OWNER_RANK")
        if owner is not None:
            owner_rank = int(owner)
            tp_size = get_tensor_model_parallel_world_size()
            speculative_config = vllm_config.speculative_config
            assert speculative_config is not None
            draft_tp_size = (
                speculative_config.draft_parallel_config.tensor_parallel_size
            )
            if get_pp_group().world_size != 1:
                raise NotImplementedError(
                    "single-rank Qwen4Exp MTP requires pipeline parallel size 1"
                )
            if draft_tp_size != 1:
                raise ValueError(
                    "VLLM_QWEN4_EXP_MTP_OWNER_RANK requires "
                    "draft_tensor_parallel_size=1"
                )
            if not 0 <= owner_rank < tp_size:
                raise ValueError(
                    f"MTP owner rank {owner_rank} is outside TP size {tp_size}"
                )
            self.mtp_owner_rank = owner_rank

        # With PP=1 the proposer always replaces these shared I/O modules with
        # the target model's TP-sharded embedding/head after the draft weights
        # load. Avoid allocating duplicate BF16 tables during draft startup.
        self.share_target_io = get_pp_group().world_size == 1
        self.embed_tokens = (
            PPMissingLayer()
            if self.share_target_io
            else VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        )
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            disable_tp = draft_vllm_config.parallel_config.tensor_parallel_size == 1
            fused_fc_gather = os.environ.get("VLLM_QWEN4_EXP_MTP_FUSED_FC_GATHER", "0")
            if fused_fc_gather not in {"0", "1"}:
                raise ValueError("VLLM_QWEN4_EXP_MTP_FUSED_FC_GATHER must be 0 or 1")
            self.fused_fc_gather = fused_fc_gather == "1" and not disable_tp
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=not self.fused_fc_gather,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
                disable_tp=disable_tp,
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=not self.fused_fc_gather,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
                disable_tp=disable_tp,
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            quant_config=draft_vllm_config.quant_config,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )
        if not self._owns_mtp_core():
            # A single-rank proposer never executes these modules on the other
            # target TP rank. Release their eagerly allocated CUDA storage and
            # retain only the module/parameter metadata needed by vLLM's model
            # interfaces. The non-owner forward returns after receiving the
            # owner's two broadcast tensors, so meta parameters are never read.
            for module in (
                self.fc_embedding,
                self.fc_hidden,
                self.layers,
                self.pre_fc_norm_embedding,
                self.pre_fc_norm_hidden,
                self.hyper_connection_mixer,
            ):
                # The GGUF loader constructs the inactive rank directly on
                # meta.  Its lazy GGUF parameters are intentionally
                # uninitialized, and ``to_empty(meta)`` tries to run
                # ``empty_like`` on them, which PyTorch rejects.  Only move a
                # module when it actually owns non-meta storage (the regular
                # safetensors/BF16 construction path).
                if any(param.device.type != "meta" for param in module.parameters()):
                    module.to_empty(device=torch.device("meta"))

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def _owns_mtp_core(self) -> bool:
        return self.mtp_owner_rank is None or (
            get_tensor_model_parallel_rank() == self.mtp_owner_rank
        )

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        if not self._owns_mtp_core():
            return

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        if not self._owns_mtp_core():
            return

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size

        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            if not self._owns_mtp_core():
                # A compile-safe all-reduce is used as a broadcast: only the
                # owner contributes data and inactive ranks contribute zeros.
                # Calling torch.distributed.broadcast here creates an
                # untraceable pybind BroadcastOptions object during graph
                # capture.
                sample_hidden_states = hidden_states.new_zeros(
                    hidden_states.shape[0], self.hidden_size
                )
                multi_hidden = torch.zeros_like(hidden_states)
                tp_group = get_tp_group()
                sample_hidden_states = tp_group.all_reduce(sample_hidden_states)
                multi_hidden = tp_group.all_reduce(multi_hidden)
                return sample_hidden_states, multi_hidden
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
            # the main model truly emits the pre-final-mixer multi stream
            # on the first step; subsequent steps reuse the prior draft
            # step's multi stream).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = self.fc_hidden(hidden_states)
            # The default gathers both projections before this add.  The
            # opt-in path keeps their aligned column shards local, adds first,
            # and gathers only the combined HC tensor, deleting the smaller
            # fc_embedding all-gather.
            hidden_states = _combine_mtp_fc_shards(
                inputs_embeds,
                hidden_states,
                gather_output=self.fused_fc_gather,
            )
            # Fold back to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
            hidden_states = hidden_states.flatten(-2)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            # As in the target model, PP carries a materialized tensor rather
            # than the delayed hidden/output/injection tuple.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        if self.mtp_owner_rank is not None:
            tp_group = get_tp_group()
            sample_hidden_states = tp_group.all_reduce(sample_hidden_states)
            multi_hidden = tp_group.all_reduce(multi_hidden)
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMTP(
    nn.Module,
    Qwen4ExpMTPMultimodalMixin,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if self.model.share_target_io:
            self.lm_head = PPMissingLayer()
        elif get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return greedy draft tokens without gathering full-vocabulary logits."""
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not self.model._owns_mtp_core():
            # Report the meta-only placeholders as satisfied without consuming
            # or materializing the MTP checkpoint on the non-owner rank.
            return {name for name, _ in self.named_parameters()}

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(
                    name,
                    self.model.mtp_start_layer_idx,
                )
                if remapped_name is not None:
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(remap_weight_names())
        # The proposer replaces these temporary TP-sharded modules with the
        # target model's embedding and LM head immediately after loading.  A
        # minimal standalone MTP checkpoint therefore intentionally omits them.
        loaded.update(
            name
            for name, _ in self.named_parameters()
            if name.startswith(("model.embed_tokens.", "lm_head."))
        )
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]

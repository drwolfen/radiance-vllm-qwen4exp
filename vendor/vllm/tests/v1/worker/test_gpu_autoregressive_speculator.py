# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _RecordingDraftModel(_DraftModel):
    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return super().forward(**kwargs)


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _HCMultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.last_inputs_embeds = None

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        return torch.zeros(input_ids.shape[0], self.embedding_size)

    def forward(self, *, hidden_states, inputs_embeds, **kwargs):
        self.last_inputs_embeds = inputs_embeds
        return hidden_states


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.uses_mrope = False
    speculator.mrope_positions = None
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


@pytest.mark.parametrize(
    ("draft_enforce_eager", "expected_modes"),
    [
        (False, [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY]),
        (True, [CUDAGraphMode.NONE, CUDAGraphMode.NONE]),
    ],
)
def test_draft_enforce_eager_only_disables_speculator_graphs(
    monkeypatch, draft_enforce_eager, expected_modes
):
    captured_modes = []

    def fake_graph_manager(vllm_config, device, mode, *args, **kwargs):
        captured_modes.append(mode)
        return Mock()

    monkeypatch.setattr(
        spec_module, "SpeculatorCudaGraphManager", fake_graph_manager
    )
    speculator = object.__new__(_TestSpeculator)
    speculator.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=draft_enforce_eager)
    )
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 2

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)

    assert captured_modes == expected_modes


@pytest.mark.parametrize(("hc_mult", "expected"), [(None, 64), (4, 256)])
def test_speculator_uses_draft_model_hidden_size(hc_mult, expected):
    hf_config = SimpleNamespace()
    if hc_mult is not None:
        hf_config.hc_mult = hc_mult
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        get_hidden_size=lambda: 64,
        get_inputs_embeds_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


def test_speculator_allocates_noncontiguous_mrope_buffer():
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        uses_mrope=True,
        get_hidden_size=lambda: 64,
        get_inputs_embeds_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.uses_mrope
    assert speculator.mrope_positions is not None
    assert speculator.mrope_positions.shape == (3, 9)
    assert speculator.mrope_positions[:, :8].stride() == (9, 1)


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.inputs_embeds_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.inputs_embeds_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_hc_multimodal_embedding_buffer_keeps_base_width(monkeypatch):
    base_hidden_size = 2560
    hc_mult = 4
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(hc_mult=hc_mult),
        get_hidden_size=lambda: base_hidden_size,
        get_inputs_embeds_size=lambda: base_hidden_size,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=2,
        ),
        model_config=SimpleNamespace(
            max_model_len=16,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )
    model = _HCMultimodalDraftModel(base_hidden_size)
    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))
    speculator.test_draft_model = model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.hidden_states.shape == (2, base_hidden_size * hc_mult)
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (2, base_hidden_size)

    speculator._run_model(
        2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        mm_inputs=(
            [torch.zeros(1, base_hidden_size)],
            torch.tensor([False, True]),
        ),
    )

    assert model.last_inputs_embeds is not None
    assert model.last_inputs_embeds.shape == (2, base_hidden_size)


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_run_model_passes_exact_three_axis_mrope_positions(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)
    model = _RecordingDraftModel(hidden)
    speculator.model = model
    speculator.uses_mrope = True
    speculator.mrope_positions = torch.tensor(
        [
            [5, 6, 7, 8, 0],
            [5, 6, 9, 9, 0],
            [5, 7, 7, 9, 0],
        ]
    )

    speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    positions = model.last_kwargs["positions"]
    assert positions.shape == (3, 4)
    torch.testing.assert_close(positions, speculator.mrope_positions[:, :4])
    assert positions.stride(0) == 5


def test_target_mrope_positions_come_from_model_rope_state():
    speculator = object.__new__(_TestSpeculator)
    speculator.uses_mrope = True
    expected = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 8, 8],
            [1, 7, 7, 8],
        ]
    )
    rope_state = SimpleNamespace(
        num_dims=3,
        get_positions=lambda num_tokens: expected[:, :num_tokens],
    )
    speculator.model_state = SimpleNamespace(rope_state=rope_state)

    actual = speculator._get_target_mrope_positions(3)

    torch.testing.assert_close(actual, expected[:, :3])


@pytest.mark.parametrize(
    "rope_state",
    [None, SimpleNamespace(num_dims=2)],
)
def test_target_mrope_positions_reject_missing_or_wrong_rope_state(rope_state):
    speculator = object.__new__(_TestSpeculator)
    speculator.uses_mrope = True
    speculator.model_state = SimpleNamespace(rope_state=rope_state)

    with pytest.raises(RuntimeError, match="must be bound before profile/propose"):
        speculator._get_target_mrope_positions(3)


def test_profile_time_propose_uses_early_bound_target_mrope_state(monkeypatch):
    class _StopAfterInputPreparation(Exception):
        pass

    speculator = object.__new__(_TestSpeculator)
    speculator.uses_mrope = True
    speculator.mrope_positions = torch.zeros(3, 5, dtype=torch.int64)
    speculator.max_model_len = 16
    speculator.max_num_reqs = 1
    speculator.num_speculative_steps = 2
    speculator.method = "mtp"
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.last_token_indices = torch.zeros(1, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0, dtype=torch.int64)
    speculator.input_buffers = SimpleNamespace()
    speculator._copy_request_inputs = lambda *args, **kwargs: None

    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 6, 6, 7],
            [1, 5, 7, 7],
        ],
        dtype=torch.int64,
    )
    rope_state = SimpleNamespace(
        num_dims=3,
        get_positions=lambda num_tokens: expected_positions[:, :num_tokens],
    )
    # Exercise the same early binding GPUModelRunner.load_model performs before
    # profile_run, rather than relying on set_attn (which runs later).
    runner = object.__new__(GPUModelRunner)
    runner.speculator = speculator
    runner.model_state = SimpleNamespace(rope_state=rope_state)
    runner._bind_speculator_model_state()

    captured = {}

    def stop_after_input_preparation(*args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterInputPreparation

    monkeypatch.setattr(
        spec_module,
        "prepare_prefill_inputs",
        stop_after_input_preparation,
    )
    input_batch = SimpleNamespace(
        num_tokens=4,
        num_tokens_after_padding=4,
        num_reqs=1,
        num_scheduled_tokens=torch.tensor([4]),
        seq_lens_cpu_upper_bound=torch.tensor([4]),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
    )

    with pytest.raises(_StopAfterInputPreparation):
        speculator.propose(
            input_batch=input_batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.zeros(4, 3),
            aux_hidden_states=None,
            num_sampled=torch.ones(1, dtype=torch.int32),
            num_rejected=torch.zeros(1, dtype=torch.int32),
            last_sampled=torch.zeros(1, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
            temperature=torch.zeros(1),
            seeds=torch.zeros(1, dtype=torch.int64),
            dummy_run=True,
            is_profile=True,
        )

    assert captured["draft_mrope_positions"] is speculator.mrope_positions
    torch.testing.assert_close(
        captured["target_mrope_positions"], expected_positions
    )


def test_compact_mrope_positions_keeps_each_requests_last_token():
    speculator = object.__new__(_TestSpeculator)
    speculator.uses_mrope = True
    speculator.mrope_positions = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0],
            [0, 1, 8, 8, 4, 9, 0],
            [0, 7, 7, 8, 4, 9, 0],
        ]
    )
    expected = speculator.mrope_positions[:, [2, 5]].clone()

    speculator._compact_mrope_positions(torch.tensor([2, 5]), num_reqs=2)

    torch.testing.assert_close(speculator.mrope_positions[:, :2], expected)


@pytest.mark.parametrize(
    "kernel",
    [
        spec_module._prepare_decode_inputs_kernel,
        spec_module._update_draft_inputs_kernel,
    ],
)
def test_mrope_decode_kernels_advance_each_axis_independently(kernel):
    # The target's last image-prefill position can have three distinct axes.
    # Guard against loading only axis 0 and broadcasting it to every axis.
    source = inspect.getsource(kernel.fn)
    per_axis_load = (
        "mrope_positions_ptr\n"
        "                    + dim * mrope_positions_stride\n"
        "                    + req_idx"
    )

    assert "for dim in tl.static_range(NUM_MROPE_DIMS)" in source
    assert f"mrope_position = tl.load(\n                    {per_axis_load}" in source
    assert (
        "mrope_position = tl.load(mrope_positions_ptr + req_idx)" not in source
    )


@pytest.mark.parametrize(
    ("draft_positions", "target_positions", "error", "match"),
    [
        (
            torch.zeros(3, 5, dtype=torch.int64),
            None,
            ValueError,
            "provided together",
        ),
        (
            torch.zeros(2, 5, dtype=torch.int64),
            torch.zeros(3, 5, dtype=torch.int64),
            ValueError,
            r"shape \[3, tokens\]",
        ),
        (
            torch.zeros(3, 5, dtype=torch.int32),
            torch.zeros(3, 5, dtype=torch.int64),
            TypeError,
            "int64",
        ),
        (
            torch.zeros(3, 3, dtype=torch.int64),
            torch.zeros(3, 5, dtype=torch.int64),
            ValueError,
            "at least 4 tokens",
        ),
    ],
)
def test_prepare_prefill_rejects_malformed_mrope_buffers_before_launch(
    draft_positions,
    target_positions,
    error,
    match,
):
    input_batch = SimpleNamespace(num_reqs=1, num_tokens_after_padding=4)

    with pytest.raises(error, match=match):
        spec_module.prepare_prefill_inputs(
            last_token_indices=None,
            current_draft_step=None,
            input_buffers=None,
            input_batch=input_batch,
            num_sampled=None,
            num_rejected=None,
            last_sampled=None,
            next_prefill_tokens=None,
            max_num_reqs=1,
            draft_mrope_positions=draft_positions,
            target_mrope_positions=target_positions,
        )


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None

# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import torch

from vllm_gguf_plugin import weight_utils


class _FakeTensor:
    def __init__(
        self,
        name: str,
        *,
        payload_allowed: bool,
        tensor_type=None,
    ) -> None:
        self.name = name
        self.tensor_type = tensor_type or SimpleNamespace(name="F32")
        self._payload_allowed = payload_allowed

    @property
    def data(self) -> np.ndarray:
        if not self._payload_allowed:
            raise AssertionError(f"payload for {self.name} was materialized")
        return np.ones((1,), dtype=np.float32)


def test_gguf_prefix_filter_runs_before_payload_materialization(monkeypatch) -> None:
    reader = SimpleNamespace(
        tensors=[
            _FakeTensor("ple.weight", payload_allowed=False),
            _FakeTensor("model.weight", payload_allowed=True),
        ]
    )
    monkeypatch.setattr(weight_utils.gguf, "GGUFReader", lambda _: reader)

    weights = list(
        weight_utils.gguf_quant_weights_iterator_multi(
            ["unused.gguf"],
            {
                "ple.weight": "model.layers.1.ple.ple_embedding.weight",
                "model.weight": "model.layers.1.mlp.weight",
            },
            exclude_prefixes=("model.layers.1.ple.ple_embedding.",),
        )
    )

    assert [name for name, _ in weights] == ["model.layers.1.mlp.weight"]


def test_quantized_weight_rewrites_only_terminal_weight_component(monkeypatch) -> None:
    class _QuantType(IntEnum):
        Q4_0 = 2

    reader = SimpleNamespace(
        tensors=[
            _FakeTensor(
                "output_hc_down.weight",
                payload_allowed=True,
                tensor_type=_QuantType.Q4_0,
            )
        ]
    )
    monkeypatch.setattr(weight_utils.gguf, "GGUFReader", lambda _: reader)

    weights = list(
        weight_utils.gguf_quant_weights_iterator_multi(
            ["unused.gguf"],
            {
                "output_hc_down.weight": (
                    "model.hyper_connection_mixer.input_mix_weight_down.weight"
                )
            },
        )
    )

    assert [name for name, _ in weights] == [
        "model.hyper_connection_mixer.input_mix_weight_down.qweight_type",
        "model.hyper_connection_mixer.input_mix_weight_down.qweight",
    ]


def test_weight_iterator_uses_numpy_payload_as_staging_view(monkeypatch) -> None:
    payload = np.ones((2,), dtype=np.float32)
    tensor = SimpleNamespace(
        name="model.weight",
        tensor_type=SimpleNamespace(name="F32"),
        data=payload,
    )
    reader = SimpleNamespace(tensors=[tensor])
    monkeypatch.setattr(weight_utils.gguf, "GGUFReader", lambda _: reader)

    [(name, loaded)] = list(
        weight_utils.gguf_quant_weights_iterator_multi(
            ["unused.gguf"], {"model.weight": "model.weight"}
        )
    )
    payload[0] = 7.0

    assert name == "model.weight"
    assert torch.equal(loaded, torch.tensor([7.0, 1.0]))

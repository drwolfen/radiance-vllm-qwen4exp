# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.amd.hyperconnection import _try_fused_hc_up_mix


def test_hc_up_mix_delegates_to_quant_method() -> None:
    calls = []

    class FakeMethod:
        @staticmethod
        def apply_hc_up_mix(layer, raw_lora, xn, hc_count):
            calls.append((layer, raw_lora, xn, hc_count))
            return xn.new_empty((xn.shape[0], xn.shape[1] // hc_count))

    projection = SimpleNamespace(quant_method=FakeMethod())
    raw_lora = torch.empty((3, 320), dtype=torch.bfloat16)
    xn = torch.empty((3, 10240), dtype=torch.bfloat16)

    output = _try_fused_hc_up_mix(projection, raw_lora, xn, 4)

    assert output is not None
    assert output.shape == (3, 2560)
    assert len(calls) == 1
    assert calls[0][0] is projection
    assert calls[0][1] is raw_lora
    assert calls[0][2] is xn
    assert calls[0][3] == 4


def test_hc_up_mix_falls_back_without_quant_hook() -> None:
    projection = SimpleNamespace(quant_method=object())
    raw_lora = torch.empty((1, 320), dtype=torch.bfloat16)
    xn = torch.empty((1, 10240), dtype=torch.bfloat16)

    assert _try_fused_hc_up_mix(projection, raw_lora, xn, 4) is None

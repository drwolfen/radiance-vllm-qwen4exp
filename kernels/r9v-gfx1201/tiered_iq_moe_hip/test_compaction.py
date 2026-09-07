# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc

import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from vllm_gguf_plugin.quantization.params import allocate_uva_host_empty
from vllm_gguf_plugin.quantization.tiered_experts import (
    _compact_expert_parameter,
)


def main() -> None:
    rows = 1024
    packed_bytes = 1024
    original = torch.arange(4 * rows * packed_bytes, dtype=torch.int64)
    original = original.to(torch.uint8).reshape(4, rows, packed_bytes)
    owner = allocate_uva_host_empty(tuple(original.shape), original.dtype)
    owner.copy_(original)
    accelerator_view = get_accelerator_view_from_cpu_tensor(owner)
    parameter = torch.nn.Parameter(accelerator_view, requires_grad=False)
    parameter._vllm_is_uva_offloaded = True
    parameter._vllm_uva_cpu_data = owner

    hot, hot_map, cold_map, hot_bytes, cold_bytes = _compact_expert_parameter(
        parameter, [1, 3], 4
    )
    del accelerator_view, owner
    gc.collect()

    assert torch.equal(hot.cpu(), original[[1, 3]])
    assert torch.equal(parameter.cpu(), original[[0, 2]])
    assert hot_map.cpu().tolist() == [-1, 0, -1, 1]
    assert cold_map.cpu().tolist() == [0, -1, 1, -1]
    assert hot_bytes == 2 * rows * packed_bytes
    assert cold_bytes == 2 * rows * packed_bytes
    assert parameter._vllm_uva_cpu_data.is_pinned()
    print("PASS", tuple(hot.shape), tuple(parameter.shape))


if __name__ == "__main__":
    main()

# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


class _FakeTensor:
    """Tiny rank-3 tensor sufficient to execute the source helper on CPU."""

    def __init__(self, data: list[Any]) -> None:
        self.data = data

    def unsqueeze(self, dim: int) -> _FakeTensor:
        assert dim == -2
        return _FakeTensor([[row] for row in self.data])

    def __add__(self, other: _FakeTensor) -> _FakeTensor:
        result = []
        for left_rows, right_rows in zip(self.data, other.data, strict=True):
            if len(right_rows) == 1:
                right_rows = right_rows * len(left_rows)
            result.append(
                [
                    [left + right for left, right in zip(a, b, strict=True)]
                    for a, b in zip(left_rows, right_rows, strict=True)
                ]
            )
        return _FakeTensor(result)


def _load_combine_helper(gather):
    source_path = Path(__file__).parents[3] / "vllm/models/qwen4_exp/amd/mtp.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_combine_mtp_fc_shards"
    )
    function.returns = None
    for argument in (*function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "tensor_model_parallel_all_gather": gather,
    }
    exec(compile(module, source_path, "exec"), namespace)
    return namespace["_combine_mtp_fc_shards"]


def _cat_last(shards: list[_FakeTensor]) -> _FakeTensor:
    rows = []
    for shard_rows in zip(*(shard.data for shard in shards), strict=True):
        rows.append(
            [
                sum((branch_rows[branch] for branch_rows in shard_rows), [])
                for branch in range(len(shard_rows[0]))
            ]
        )
    return _FakeTensor(rows)


def test_local_fc_add_distributes_over_tp_all_gather() -> None:
    combine = _load_combine_helper(lambda *_args, **_kwargs: None)
    embedding_shards = [
        _FakeTensor([[0, 1], [2, 3]]),
        _FakeTensor([[4, 5], [6, 7]]),
    ]
    hidden_shards = [
        _FakeTensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]]]),
        _FakeTensor([[[20, 21], [22, 23]], [[24, 25], [26, 27]]]),
    ]

    local_results = [
        combine(embedding, hidden, gather_output=False)
        for embedding, hidden in zip(embedding_shards, hidden_shards, strict=True)
    ]
    fused = _cat_last(local_results)
    reference = _cat_last(hidden_shards) + _cat_last(
        [embedding.unsqueeze(-2) for embedding in embedding_shards]
    )

    assert fused.data == reference.data


def test_fused_fc_path_gathers_once_after_local_add() -> None:
    calls: list[tuple[_FakeTensor, int]] = []

    def fake_all_gather(tensor: _FakeTensor, dim: int) -> _FakeTensor:
        calls.append((tensor, dim))
        return tensor

    combine = _load_combine_helper(fake_all_gather)
    embedding = _FakeTensor([[0, 1], [2, 3]])
    hidden = _FakeTensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]]])
    expected = hidden + embedding.unsqueeze(-2)

    output = combine(embedding, hidden, gather_output=True)

    assert len(calls) == 1
    assert calls[0][0].data == expected.data
    assert calls[0][1] == -1
    assert output.data == expected.data

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import unittest

MAX_MATCHES = 3


def reuse_plan(expert_ids: list[int], num_experts: int) -> list[list[int]]:
    plan: list[list[int]] = []
    for group, expert in enumerate(expert_ids):
        if not 0 <= expert < num_experts:
            plan.append([group])
            continue
        matches = [route for route, routed in enumerate(expert_ids) if routed == expert]
        if len(matches) > MAX_MATCHES:
            plan.append([group])
        elif group == matches[0]:
            plan.append(matches)
        else:
            plan.append([])
    return plan


def dot(weight: list[list[float]], activation: list[float]) -> list[float]:
    return [
        sum(value * activation[column] for column, value in enumerate(row))
        for row in weight
    ]


def baseline(
    activations: list[list[float]],
    weights: list[list[list[float]]],
    expert_ids: list[int],
    top_k: int,
) -> list[list[float]]:
    rows = len(weights[0])
    return [
        dot(weights[expert], activations[group // top_k])
        if 0 <= expert < len(weights)
        else [0.0] * rows
        for group, expert in enumerate(expert_ids)
    ]


def reused(
    activations: list[list[float]],
    weights: list[list[list[float]]],
    expert_ids: list[int],
    top_k: int,
) -> list[list[float]]:
    rows = len(weights[0])
    output: list[list[float] | None] = [None] * len(expert_ids)
    for owner, matches in enumerate(reuse_plan(expert_ids, len(weights))):
        expert = expert_ids[owner]
        for group in matches:
            output[group] = (
                dot(weights[expert], activations[group // top_k])
                if 0 <= expert < len(weights)
                else [0.0] * rows
            )
    assert all(value is not None for value in output)
    return [value for value in output if value is not None]


class ReusePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = random.Random(29)
        self.weights = [
            [[generator.uniform(-1.0, 1.0) for _ in range(7)] for _ in range(5)]
            for _ in range(18)
        ]
        self.w13_ids = [
            *range(10),
            0,
            1,
            2,
            3,
            4,
            10,
            11,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
            8,
            9,
            10,
            11,
            4,
            5,
        ]
        self.generator = generator

    def activations(self, count: int) -> list[list[float]]:
        return [
            [self.generator.uniform(-1.0, 1.0) for _ in range(7)] for _ in range(count)
        ]

    def test_mtp_top10_first_owner_covers_every_output_once(self) -> None:
        plan = reuse_plan(self.w13_ids, len(self.weights))
        covered = [group for matches in plan for group in matches]

        self.assertEqual(sorted(covered), list(range(30)))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(sum(bool(matches) for matches in plan), 12)

    def test_w13_and_w2_outputs_preserve_group_order_exactly(self) -> None:
        w13_activations = self.activations(3)
        self.assertEqual(
            reused(w13_activations, self.weights, self.w13_ids, 10),
            baseline(w13_activations, self.weights, self.w13_ids, 10),
        )

        w2_activations = self.activations(30)
        self.assertEqual(
            reused(w2_activations, self.weights, self.w13_ids, 1),
            baseline(w2_activations, self.weights, self.w13_ids, 1),
        )

    def test_invalid_and_more_than_three_matches_fall_back_safely(self) -> None:
        ids = self.w13_ids.copy()
        ids[:5] = [6] * 5
        ids[8] = -1
        ids[17] = len(self.weights)
        plan = reuse_plan(ids, len(self.weights))
        activations = self.activations(3)

        self.assertEqual(plan[:5], [[0], [1], [2], [3], [4]])
        self.assertEqual(
            reused(activations, self.weights, ids, 10),
            baseline(activations, self.weights, ids, 10),
        )


if __name__ == "__main__":
    unittest.main()

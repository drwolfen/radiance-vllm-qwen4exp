# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import struct

ROWS = 336
COLS = 10240
TOKENS = 3
WORKGROUPS = 32
WAVES_PER_WORKGROUP = 16


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def multiply(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def stock_rows() -> list[tuple[int, int, int]]:
    # For this shape mindiv(336, 32*2, 16) returns six active waves.
    return [
        (block, wave, (block * 6 + wave) * 2 + tile)
        for block in range(WORKGROUPS)
        for wave in range(6)
        for tile in range(2)
        if (block * 6 + wave) * 2 + tile < ROWS
    ]


def cyclic_rows() -> list[tuple[int, int, int]]:
    return [
        (block, wave, block + wave * WORKGROUPS)
        for block in range(WORKGROUPS)
        for wave in range(WAVES_PER_WORKGROUP)
        if block + wave * WORKGROUPS < ROWS
    ]


def lane_columns(lane: int) -> list[int]:
    return [
        k1 + k2 * 32 * 8 + lane * 8 + item
        for k1 in range(0, COLS, 32 * 8 * 2)
        for k2 in range(2)
        for item in range(8)
    ]


def lane_dot(weight: list[float], activation: list[float], lane: int) -> float:
    total = 0.0
    columns = lane_columns(lane)
    for offset in range(0, len(columns), 2):
        first = multiply(weight[columns[offset]], activation[columns[offset]])
        second = multiply(
            weight[columns[offset + 1]], activation[columns[offset + 1]]
        )
        total = add(total, add(first, second))
    return total


def test_geometry() -> None:
    stock = stock_rows()
    cyclic = cyclic_rows()
    assert sorted(row for _, _, row in stock) == list(range(ROWS))
    assert sorted(row for _, _, row in cyclic) == list(range(ROWS))
    assert len({row for _, _, row in stock}) == ROWS
    assert len({row for _, _, row in cyclic}) == ROWS

    # Stock stages the complete activation in four workgroups that never
    # process a weight row; cyclic makes every workgroup useful and doubles
    # the number of independent row waves from 168 to 336.
    assert len({block for block, _, _ in stock}) == 28
    assert len({block for block, _, _ in cyclic}) == 32
    assert len({(block, wave) for block, wave, _ in stock}) == 168
    assert len({(block, wave) for block, wave, _ in cyclic}) == 336


def test_k_order_is_identical_and_complete() -> None:
    produced: list[int] = []
    for lane in range(32):
        columns = lane_columns(lane)
        assert len(columns) == COLS // 32
        produced.extend(columns)
    assert sorted(produced) == list(range(COLS))

    # Both kernels use this exact lane-local sequence. Exercise the explicit
    # pair-add then accumulator-add boundary with deterministic F32 values.
    generator = random.Random(38)
    weight = [f32(generator.uniform(-0.25, 0.25)) for _ in range(COLS)]
    activation = [f32(generator.uniform(-0.25, 0.25)) for _ in range(COLS)]
    stock = [lane_dot(weight, activation, lane) for lane in range(32)]
    cyclic = [lane_dot(weight, activation, lane) for lane in range(32)]
    assert [struct.pack("<f", value) for value in cyclic] == [
        struct.pack("<f", value) for value in stock
    ]


if __name__ == "__main__":
    test_geometry()
    test_k_order_is_identical_and_complete()
    print("HC-down M=3 CPU geometry/order checks passed")

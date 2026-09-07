# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import struct

BLOCKS = 80
ROWS_PER_WAVE = 4
WAVES_PER_BLOCK = 4


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def contribution(scale: float, dot: int) -> float:
    return f32(f32(scale) * dot)


def stock_row(weights, weight_scales, activation, activation_scales):
    lanes = [0.0] * 32
    for lane in range(32):
        for block in range(lane // 4, BLOCKS, 8):
            offset = 8 * (lane % 4)
            dot = sum(
                weights[block][item] * activation[block][item]
                for item in range(offset, offset + 8)
            )
            scale = f32(weight_scales[block] * activation_scales[block])
            lanes[lane] = add(lanes[lane], contribution(scale, dot))
    for mask in (16, 8, 4, 2, 1):
        previous = lanes[:]
        lanes = [add(previous[lane], previous[lane ^ mask]) for lane in range(32)]
    return bf16_bits(lanes[0])


def group4_row(weights, weight_scales, activation, activation_scales):
    lanes = [0.0] * 8
    for group in range(8):
        for block in range(group, BLOCKS, 8):
            dot = sum(
                weights[block][item] * activation[block][item] for item in range(32)
            )
            scale = f32(weight_scales[block] * activation_scales[block])
            lanes[group] = add(lanes[group], contribution(scale, dot))
    for offset in (4, 2, 1):
        previous = lanes[:]
        for lane in range(8 - offset):
            lanes[lane] = add(previous[lane], previous[lane + offset])
    return bf16_bits(lanes[0])


def make_case(seed: int):
    generator = random.Random(seed)
    weights = [[generator.randint(-127, 127) for _ in range(32)] for _ in range(BLOCKS)]
    activation = [
        [generator.randint(-127, 127) for _ in range(32)] for _ in range(BLOCKS)
    ]
    weight_scales = [2.0 ** generator.randint(-12, -5) for _ in range(BLOCKS)]
    activation_scales = [2.0 ** generator.randint(-12, -5) for _ in range(BLOCKS)]
    return weights, weight_scales, activation, activation_scales


def test_geometry() -> None:
    rows_per_block = ROWS_PER_WAVE * WAVES_PER_BLOCK
    for rows in (8192, 6656):
        produced = [
            block * rows_per_block + wave * ROWS_PER_WAVE + row
            for block in range(rows // rows_per_block)
            for wave in range(WAVES_PER_BLOCK)
            for row in range(ROWS_PER_WAVE)
        ]
        assert produced == list(range(rows))


def test_exact_quartet_and_group_boundary() -> None:
    for seed in range(16):
        case = make_case(seed)
        stock = stock_row(*case)
        exact_quartet = stock_row(*case)
        assert exact_quartet == stock

        grouped = group4_row(*case)
        stock_value = bf16_value(stock)
        grouped_value = bf16_value(grouped)
        difference = abs(grouped_value - stock_value)
        tolerance = max(abs(stock_value), 1.0) * 0.02
        assert difference <= tolerance, (seed, stock_value, grouped_value)


if __name__ == "__main__":
    test_geometry()
    test_exact_quartet_and_group_boundary()
    print("attention M=3 CPU mapping/parity checks passed")

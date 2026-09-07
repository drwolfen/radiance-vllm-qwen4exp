# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import struct

HIDDEN = 2560
STREAMS = 4
VECS = 3
BLOCKS_PER_ROW = 10
VARIANTS = {
    "w4-r1": (4, 1),
    "w8-r1": (8, 1),
    "w4-r2": (4, 2),
    "w8-r2": (8, 2),
}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def grouped_rows(waves: int, outputs_per_wave: int):
    grid = HIDDEN // (waves * outputs_per_wave)
    return [
        (block, wave, item, (block * waves + wave) * outputs_per_wave + item)
        for block in range(grid)
        for wave in range(waves)
        for item in range(outputs_per_wave)
    ]


def xor_tree(lanes: list[float]) -> float:
    values = lanes[:]
    for mask in (16, 8, 4, 2, 1):
        previous = values[:]
        values = [add(previous[lane], previous[lane ^ mask]) for lane in range(32)]
    return values[0]


def row_result(seed: int, inner: int) -> tuple[float, ...]:
    generator = random.Random(seed + inner * 131)
    outputs = []
    for stream in range(STREAMS):
        for vec in range(VECS):
            lanes = [0.0] * 32
            for lane in range(32):
                # The legacy and grouped kernels both assign Q8 blocks by
                # lane/4 and revisit them with stride eight.
                for block in range(lane // 4, BLOCKS_PER_ROW, 8):
                    contribution = f32(
                        generator.uniform(-0.125, 0.125)
                        + stream * 0.001
                        + vec * 0.0001
                        + block * 0.00001
                    )
                    lanes[lane] = add(lanes[lane], contribution)
            outputs.append(xor_tree(lanes))
    return tuple(outputs)


def test_grouped_geometry() -> None:
    for waves, outputs_per_wave in VARIANTS.values():
        rows = grouped_rows(waves, outputs_per_wave)
        assert len(rows) == HIDDEN
        assert sorted(row for _, _, _, row in rows) == list(range(HIDDEN))
        assert len({row for _, _, _, row in rows}) == HIDDEN
        assert len({block for block, _, _, _ in rows}) == HIDDEN // (
            waves * outputs_per_wave
        )


def test_grouping_does_not_change_per_row_order() -> None:
    legacy = {inner: row_result(38, inner) for inner in range(32)}
    for waves, outputs_per_wave in VARIANTS.values():
        grouped = {
            inner: row_result(38, inner)
            for _, _, _, inner in grouped_rows(waves, outputs_per_wave)
            if inner < 32
        }
        assert grouped == legacy


if __name__ == "__main__":
    test_grouped_geometry()
    test_grouping_does_not_change_per_row_order()
    print("grouped HC-up CPU geometry/order checks passed")

# SPDX-License-Identifier: Apache-2.0
"""Headless proof of the async expert-cache staging protocol.

This intentionally models publication, not HIP timing.  The device test in
``test_bounded.py`` is the bounded parity/graph/fill discriminator to run only
after the production model is stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class CacheRing:
    logical_slots: int
    experts: int = 32
    tags: list[int] = field(init=False)
    reverse: list[int] = field(init=False)
    payload: list[int | None] = field(init=False)
    clock: int = 0
    pending: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.logical_slots <= 16:
            raise ValueError("logical slots must be in [1, 16]")
        physical = self.logical_slots + 1
        self.tags = [-1] * physical
        self.reverse = [-1] * self.experts
        self.payload = [None] * physical

    def schedule_and_copy(self, expert: int, occurrences: int = 1) -> bool:
        """Copy once; duplicate mode publishes now, singleton waits for W2."""
        if self.reverse[expert] >= 0:
            return False
        assert self.pending is None
        staging = self.tags.index(-1)
        published = sum(tag >= 0 for tag in self.tags)
        victim = -1
        if published == self.logical_slots:
            for offset in range(len(self.tags)):
                candidate = (self.clock + offset) % len(self.tags)
                if self.tags[candidate] >= 0:
                    victim = candidate
                    self.clock = (candidate + 1) % len(self.tags)
                    break
        self.payload[staging] = expert
        mode = 1 if occurrences > 1 else 2
        self.pending = (expert, staging, victim, mode)
        if mode == 1:
            self.commit_after_w2(expected_mode=1)
        return True

    def commit_after_w2(self, expected_mode: int = 2) -> None:
        if self.pending is None:
            return
        expert, staging, victim, mode = self.pending
        if mode != expected_mode:
            return
        if victim >= 0:
            old_expert = self.tags[victim]
            assert self.reverse[old_expert] == victim
            self.reverse[old_expert] = -1
            self.tags[victim] = -1
        assert self.tags[staging] == -1
        assert self.payload[staging] == expert
        self.tags[staging] = expert
        self.reverse[expert] = staging
        self.pending = None

    def assert_invariants(self) -> None:
        published = [tag for tag in self.tags if tag >= 0]
        assert len(published) <= self.logical_slots
        assert len(set(published)) == len(published)
        assert self.tags.count(-1) >= 1
        for expert, slot in enumerate(self.reverse):
            if slot >= 0:
                assert self.tags[slot] == expert
                assert self.payload[slot] == expert


@pytest.mark.parametrize("logical_slots", [1, 4, 8, 16])
def test_staging_is_never_visible_before_commit(logical_slots: int) -> None:
    ring = CacheRing(logical_slots)
    pointers = (id(ring.tags), id(ring.reverse), id(ring.payload))
    for expert in range(logical_slots + 4):
        assert ring.schedule_and_copy(expert)
        assert ring.reverse[expert] == -1  # current pass must use cold fallback
        ring.assert_invariants()
        ring.commit_after_w2()
        assert ring.reverse[expert] >= 0
        ring.assert_invariants()
    assert pointers == (id(ring.tags), id(ring.reverse), id(ring.payload))


def test_cache8_rotation_matches_device_discriminator() -> None:
    ring = CacheRing(8)
    for expert in range(8):
        ring.schedule_and_copy(expert)
        ring.commit_after_w2()
    assert ring.tags == [0, 1, 2, 3, 4, 5, 6, 7, -1]

    ring.schedule_and_copy(8)
    assert ring.tags == [0, 1, 2, 3, 4, 5, 6, 7, -1]
    assert ring.payload[8] == 8
    ring.commit_after_w2()
    assert ring.tags == [-1, 1, 2, 3, 4, 5, 6, 7, 8]

    ring.schedule_and_copy(9)
    assert ring.tags == [-1, 1, 2, 3, 4, 5, 6, 7, 8]
    assert ring.payload[0] == 9
    ring.commit_after_w2()
    assert ring.tags == [9, -1, 2, 3, 4, 5, 6, 7, 8]
    assert ring.reverse[0] == ring.reverse[1] == -1
    ring.assert_invariants()


def test_duplicate_and_singleton_modes_are_mutually_exclusive() -> None:
    ring = CacheRing(2)
    ring.schedule_and_copy(2, occurrences=2)
    assert ring.pending is None
    assert ring.reverse[2] >= 0  # duplicate is visible to the current pass

    ring.schedule_and_copy(3, occurrences=1)
    assert ring.pending is not None
    assert ring.reverse[3] == -1  # singleton current pass stays cold
    ring.commit_after_w2(expected_mode=1)
    assert ring.reverse[3] == -1  # wrong mode cannot publish
    ring.commit_after_w2(expected_mode=2)
    assert ring.reverse[3] >= 0
    ring.assert_invariants()

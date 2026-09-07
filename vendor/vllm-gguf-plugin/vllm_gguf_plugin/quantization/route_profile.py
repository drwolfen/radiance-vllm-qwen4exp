# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import functools
import logging
import os
import re
from pathlib import Path

import torch
from vllm.distributed import get_tensor_model_parallel_rank

logger = logging.getLogger(__name__)

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PROFILE_DIR_ENV = "RADIANCE_ROUTE_PROFILE_DIR"
_PROFILE_MAX_EVENTS_ENV = "RADIANCE_ROUTE_PROFILE_MAX_EVENTS"
_PROFILE_RANKS_ENV = "RADIANCE_ROUTE_PROFILE_RANKS"
_PROFILE_ROWS_ENV = "RADIANCE_ROUTE_PROFILE_ROWS"
_PROFILE_AUTO_DUMP_ENV = "RADIANCE_ROUTE_PROFILE_AUTO_DUMP"


class _RouteProfiler:
    def __init__(
        self,
        output_dir: Path,
        max_events: int,
        rank: int,
        allowed_rows: frozenset[int] = frozenset((1, 2, 3)),
        auto_dump: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.max_events = max_events
        self.rank = rank
        self.allowed_rows = allowed_rows
        self.auto_dump = auto_dump
        self.trace: torch.Tensor | None = None
        self.rows = torch.zeros(max_events, dtype=torch.int8, device="cpu")
        self.num_events = 0
        self.current_event: int | None = None
        self.complete = False
        atexit.register(self.dump)
        logger.warning(
            "Qwen eager route profiling enabled for TP rank %d with %d events "
            "and row counts %s",
            rank,
            max_events,
            sorted(allowed_rows),
        )

    def _start_event(self, topk_ids: torch.Tensor) -> None:
        num_rows = topk_ids.shape[0]
        if (
            self.complete
            or num_rows not in self.allowed_rows
            or self.num_events >= self.max_events
        ):
            self.current_event = None
            return
        if self.trace is None:
            self.trace = torch.full(
                (self.max_events, 48, 3, 10),
                -1,
                dtype=torch.int16,
                device=topk_ids.device,
            )
        self.current_event = self.num_events
        self.rows[self.num_events] = num_rows
        self.num_events += 1

    def record(self, layer_name: str, topk_ids: torch.Tensor) -> None:
        if self.complete:
            return
        match = _LAYER_PATTERN.search(layer_name)
        if match is None:
            return
        layer_id = int(match.group(1))
        if not 0 <= layer_id < 48:
            return
        if layer_id == 0:
            self._start_event(topk_ids)
        event = self.current_event
        if event is None or self.trace is None:
            return

        num_rows = int(self.rows[event])
        if topk_ids.shape != (num_rows, 10):
            logger.warning(
                "Skipping inconsistent route profile event %d at layer %d: %s",
                event,
                layer_id,
                tuple(topk_ids.shape),
            )
            self.current_event = None
            return
        self.trace[event, layer_id, :num_rows].copy_(topk_ids)
        if layer_id == 47:
            self.current_event = None
            if self.auto_dump and self.num_events == self.max_events:
                self.dump()

    def dump(self) -> None:
        if self.trace is None or self.num_events == 0:
            return
        try:
            import numpy as np

            self.output_dir.mkdir(parents=True, exist_ok=True)
            routes = self.trace[: self.num_events].cpu().numpy()
            rows = self.rows[: self.num_events].numpy()
            output = self.output_dir / f"routes-rank{self.rank}-pid{os.getpid()}.npz"
            np.savez_compressed(output, routes=routes, rows=rows)
            logger.info(
                "Wrote %d Qwen route profile events to %s",
                self.num_events,
                output,
            )
        except Exception:
            logger.exception("Failed to write Qwen route profile")
        finally:
            self.trace = None
            self.num_events = 0
            self.current_event = None
            self.complete = True


@functools.cache
def _get_route_profiler() -> _RouteProfiler | None:
    output_dir = os.environ.get(_PROFILE_DIR_ENV)
    if not output_dir:
        return None
    rank = get_tensor_model_parallel_rank()
    ranks = {
        int(value)
        for value in os.environ.get(_PROFILE_RANKS_ENV, "0").split(",")
        if value.strip()
    }
    if rank not in ranks:
        return None
    max_events = int(os.environ.get(_PROFILE_MAX_EVENTS_ENV, "256"))
    if max_events <= 0:
        raise ValueError(f"{_PROFILE_MAX_EVENTS_ENV} must be positive")
    try:
        allowed_rows = frozenset(
            int(value.strip())
            for value in os.environ.get(_PROFILE_ROWS_ENV, "1,2,3").split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError(
            f"{_PROFILE_ROWS_ENV} must be a comma-separated subset of 1,2,3"
        ) from error
    if not allowed_rows or not allowed_rows <= {1, 2, 3}:
        raise ValueError(
            f"{_PROFILE_ROWS_ENV} must be a comma-separated subset of 1,2,3"
        )
    auto_dump_raw = os.environ.get(_PROFILE_AUTO_DUMP_ENV, "0")
    if auto_dump_raw not in {"0", "1"}:
        raise ValueError(f"{_PROFILE_AUTO_DUMP_ENV} must be 0 or 1")
    return _RouteProfiler(
        Path(output_dir),
        max_events,
        rank,
        allowed_rows,
        auto_dump=auto_dump_raw == "1",
    )


def record_route_profile(layer_name: str, topk_ids: torch.Tensor) -> None:
    profiler = _get_route_profiler()
    if profiler is not None:
        profiler.record(layer_name, topk_ids)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a concurrency schedule.jsonl emitted by agentic_code_gen synthesize.

The on-disk format is deliberately minimal — one JSON object per line with
exactly ``time_sec`` and ``concurrency`` fields. Interpolation and noise have
already been baked into the stream by the generator; this loader's only job
is to validate the envelope (monotonic timestamps, starts at zero, positive
concurrency) and hand back a ready-to-consume list of ticks.
"""

from __future__ import annotations

from pathlib import Path

import orjson
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class ScheduleTick(AIPerfBaseModel):
    """A single (time, concurrency) point on the schedule."""

    time_sec: float = Field(ge=0.0, description="Seconds from benchmark start.")
    concurrency: int = Field(
        ge=1, description="Target session concurrency at time_sec."
    )


def load_schedule(path: Path) -> list[ScheduleTick]:
    """Read and validate a schedule.jsonl file.

    Raises ValueError on empty files, non-zero first timestamp, or any
    non-strictly-monotonic segment.
    """
    lines = [line for line in path.read_bytes().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"schedule file is empty: {path}")

    ticks = [ScheduleTick.model_validate(orjson.loads(line)) for line in lines]

    if ticks[0].time_sec != 0:
        raise ValueError(f"schedule must start at time_sec=0, got {ticks[0].time_sec}")

    for prev, curr in zip(ticks, ticks[1:], strict=False):
        if curr.time_sec <= prev.time_sec:
            raise ValueError(
                "schedule ticks must be strictly monotonic in time_sec "
                f"(tick at {curr.time_sec} followed {prev.time_sec})"
            )

    return ticks

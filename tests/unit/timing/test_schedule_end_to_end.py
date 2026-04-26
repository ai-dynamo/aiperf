# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: agentic_code_gen emits schedule.jsonl -> aiperf loads it ->
Ramper + ScheduleFollowerStrategy drive set_session_limit."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.dataset.agentic_code_gen.models import (
    ConcurrencyScheduleAnchor,
    ConcurrencyScheduleConfig,
    SessionDistributionConfig,
)
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import write_dataset
from aiperf.dataset.loader.schedule import load_schedule
from aiperf.plugin.enums import RampType
from aiperf.timing.ramping import RampConfig, Ramper


@pytest.mark.asyncio
async def test_schedule_round_trip_drives_setter_in_order(tmp_path: Path) -> None:
    """Generator -> schedule.jsonl -> loader -> Ramper produces setter calls
    that follow the tick stream's concurrency values in order."""
    base_config = SessionDistributionConfig()
    config = base_config.model_copy(
        update={
            "concurrency_schedule": ConcurrencyScheduleConfig(
                interpolation="linear",
                tick_sec=0.01,
                noise_sigma=0.0,
                anchors=[
                    ConcurrencyScheduleAnchor(time_sec=0, concurrency=5),
                    ConcurrencyScheduleAnchor(time_sec=0.05, concurrency=10),
                ],
            )
        }
    )
    synth = SessionSynthesizer(config, seed=42)
    sessions = synth.synthesize_sessions(2)
    _, _, _, schedule_path = write_dataset(sessions, tmp_path / "run", config, seed=42)
    assert schedule_path is not None

    ticks = load_schedule(schedule_path)
    tick_tuples = tuple((t.time_sec, t.concurrency) for t in ticks)

    applied: list[int] = []

    def setter(v: float) -> None:
        applied.append(int(v))

    ramp_config = RampConfig(
        ramp_type=RampType.SCHEDULE_FOLLOWER,
        start=float(ticks[0].concurrency),
        target=float(ticks[-1].concurrency),
        duration_sec=max(ticks[-1].time_sec, 1e-6),
        schedule_ticks=tick_tuples,
    )
    ramper = Ramper(setter=setter, config=ramp_config)
    task = ramper.start()
    await asyncio.wait_for(task, timeout=5.0)

    # First call is Ramper's initial setter(start). Subsequent calls come from
    # next_step. After dedup, the sequence must contain the full tick trajectory.
    assert applied, "setter should have been called"
    assert applied[0] == ticks[0].concurrency
    # Every tick's concurrency must appear at least once in order.
    tick_values = [t.concurrency for t in ticks]
    # applied starts with the initial call duplicating tick[0], followed by each tick in order
    after_initial = applied[1:]
    assert after_initial == tick_values

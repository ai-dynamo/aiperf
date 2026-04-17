# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ScheduleFollowerStrategy ramp strategy."""

from __future__ import annotations

import pytest

from aiperf.plugin.enums import RampType
from aiperf.timing.ramping import RampConfig, ScheduleFollowerStrategy


def _config(ticks: tuple[tuple[float, int], ...]) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.SCHEDULE_FOLLOWER,
        start=float(ticks[0][1]),
        target=float(ticks[-1][1]),
        duration_sec=max(ticks[-1][0], 1e-6),
        schedule_ticks=ticks,
    )


class TestScheduleFollowerStrategy:
    def test_missing_ticks_raises(self) -> None:
        config = RampConfig(
            ramp_type=RampType.SCHEDULE_FOLLOWER,
            start=1.0,
            target=10.0,
            duration_sec=10.0,
        )
        with pytest.raises(ValueError, match="schedule_ticks"):
            ScheduleFollowerStrategy(config)

    def test_next_step_walks_ticks_in_order(self) -> None:
        ticks = ((0.0, 10), (5.0, 15), (10.0, 20))
        strat = ScheduleFollowerStrategy(_config(ticks))
        assert strat.next_step(0.0, 0.0) == (0.0, 10.0)
        assert strat.next_step(10.0, 0.0) == (5.0, 15.0)
        assert strat.next_step(15.0, 5.0) == (5.0, 20.0)
        assert strat.next_step(20.0, 10.0) is None

    def test_next_step_clamps_negative_delay_to_zero(self) -> None:
        """If the runner is running behind schedule, delays clamp to 0."""
        ticks = ((0.0, 10), (5.0, 15))
        strat = ScheduleFollowerStrategy(_config(ticks))
        delay, _ = strat.next_step(0.0, 100.0)
        assert delay == 0.0

    def test_value_at_returns_step_function(self) -> None:
        ticks = ((0.0, 10), (5.0, 15), (10.0, 20))
        strat = ScheduleFollowerStrategy(_config(ticks))
        assert strat.value_at(0.0) == 10.0
        assert strat.value_at(3.0) == 10.0
        assert strat.value_at(5.0) == 15.0
        assert strat.value_at(7.5) == 15.0
        assert strat.value_at(10.0) is None

    def test_start_and_target_reflect_tick_endpoints(self) -> None:
        ticks = ((0.0, 10), (100.0, 200))
        strat = ScheduleFollowerStrategy(_config(ticks))
        assert strat.start == 10.0
        assert strat.target == 200.0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle, PhaseState


def cfg(dur: float | None = None, grace: float | None = None) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        expected_duration_sec=dur,
        grace_period_sec=grace,
    )


class TestPhaseLifecycle:
    def test_full_lifecycle(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        assert lc.is_started and not lc.is_sending_complete and not lc.is_complete
        lc.mark_sending_complete()
        assert lc.is_started and lc.is_sending_complete and not lc.is_complete
        lc.mark_complete()
        assert lc.is_started and lc.is_sending_complete and lc.is_complete

    def test_cannot_start_twice(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        with pytest.raises(ValueError, match="Credit phase already started"):
            lc.start()

    def test_cannot_mark_sending_complete_before_start(self) -> None:
        lc = PhaseLifecycle(cfg())
        with pytest.raises(ValueError, match="Credit phase not started"):
            lc.mark_sending_complete()

    def test_timeout_triggered_flag(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        lc.mark_sending_complete(timeout_triggered=True)
        assert lc.timeout_triggered is True

    def test_grace_period_triggered_flag(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        lc.mark_sending_complete()
        lc.mark_complete(grace_period_triggered=True)
        assert lc.grace_period_triggered is True

    def test_cancelled_phase_can_still_complete(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        lc.cancel()
        lc.mark_sending_complete()
        lc.mark_complete()
        assert lc.was_cancelled is True and lc.state == PhaseState.COMPLETE

    def test_time_left_returns_none_without_duration(self) -> None:
        lc = PhaseLifecycle(cfg())
        lc.start()
        assert lc.time_left_in_seconds() is None

    def test_time_left_returns_full_duration_at_start(self) -> None:
        lc = PhaseLifecycle(cfg(dur=60.0, grace=10.0))
        lc.start()
        t = lc.time_left_in_seconds()
        # Use wider tolerance for CI variance (allow up to 100ms elapsed)
        assert t is not None and 59.9 <= t <= 60.1

    def test_time_left_with_grace_period(self) -> None:
        lc = PhaseLifecycle(cfg(dur=60.0, grace=10.0))
        lc.start()
        without = lc.time_left_in_seconds(include_grace_period=False)
        with_grace = lc.time_left_in_seconds(include_grace_period=True)
        assert without is not None and with_grace is not None
        # Grace period is exactly 10s, use wider tolerance for CI variance
        assert with_grace > without and with_grace - without >= 9.8

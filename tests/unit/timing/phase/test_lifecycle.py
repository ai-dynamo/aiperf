# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhaseLifecycle state machine."""

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle, PhaseState


@pytest.fixture
def minimal_config() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
    )


@pytest.fixture
def config_with_duration() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        expected_duration_sec=60.0,
        grace_period_sec=10.0,
    )


class TestPhaseLifecycleTransitions:
    def test_full_lifecycle(self, minimal_config: CreditPhaseConfig) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.is_started
        assert not lifecycle.is_sending_complete
        assert not lifecycle.is_complete
        lifecycle.mark_sending_complete()
        assert lifecycle.is_started
        assert lifecycle.is_sending_complete
        assert not lifecycle.is_complete
        lifecycle.mark_complete()
        assert lifecycle.is_started
        assert lifecycle.is_sending_complete
        assert lifecycle.is_complete

    def test_cannot_start_twice(self, minimal_config: CreditPhaseConfig) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        with pytest.raises(ValueError, match="Credit phase already started"):
            lifecycle.start()

    def test_cannot_mark_sending_complete_before_start(
        self, minimal_config: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        with pytest.raises(ValueError, match="Credit phase not started"):
            lifecycle.mark_sending_complete()


class TestPhaseLifecycleFlags:
    def test_timeout_triggered_flag(self, minimal_config: CreditPhaseConfig) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete(timeout_triggered=True)
        assert lifecycle.timeout_triggered is True

    def test_grace_period_triggered_flag(
        self, minimal_config: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete(grace_period_triggered=True)
        assert lifecycle.grace_period_triggered is True


class TestPhaseLifecycleCancellation:
    def test_cancelled_phase_can_still_complete(
        self, minimal_config: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        lifecycle.cancel()
        lifecycle.mark_sending_complete()
        lifecycle.mark_complete()
        assert lifecycle.was_cancelled is True
        assert lifecycle.state == PhaseState.COMPLETE


class TestPhaseLifecycleTimeLeft:
    def test_time_left_returns_none_without_duration(
        self, minimal_config: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(minimal_config)
        lifecycle.start()
        assert lifecycle.time_left_in_seconds() is None

    def test_time_left_returns_full_duration_at_start(
        self, config_with_duration: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()
        time_left = lifecycle.time_left_in_seconds()
        assert time_left is not None
        assert time_left <= 60.0
        assert time_left >= 59.9

    def test_time_left_with_grace_period(
        self, config_with_duration: CreditPhaseConfig
    ) -> None:
        lifecycle = PhaseLifecycle(config_with_duration)
        lifecycle.start()
        without_grace = lifecycle.time_left_in_seconds(include_grace_period=False)
        with_grace = lifecycle.time_left_in_seconds(include_grace_period=True)
        assert without_grace is not None
        assert with_grace is not None
        assert with_grace > without_grace
        assert with_grace - without_grace >= 9.9

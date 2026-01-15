# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StopConditionChecker and individual stop conditions."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.stop_conditions import (
    DurationStopCondition,
    LifecycleStopCondition,
    RequestCountStopCondition,
    SessionCountStopCondition,
    StopConditionChecker,
)


def make_config(
    total_expected_requests: int | None = None,
    expected_num_sessions: int | None = None,
    expected_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=total_expected_requests,
        expected_num_sessions=expected_num_sessions,
        expected_duration_sec=expected_duration_sec,
    )


def make_mock_lifecycle(
    was_cancelled: bool = False,
    is_sending_complete: bool = False,
    time_left: float = 10.0,
) -> MagicMock:
    lifecycle = MagicMock(spec=PhaseLifecycle)
    lifecycle.was_cancelled = was_cancelled
    lifecycle.is_sending_complete = is_sending_complete
    lifecycle.time_left_in_seconds = MagicMock(return_value=time_left)
    return lifecycle


def make_mock_counter(
    requests_sent: int = 0,
    sent_sessions: int = 0,
    total_session_turns: int = 0,
) -> MagicMock:
    counter = MagicMock(spec=CreditCounter)
    counter.requests_sent = requests_sent
    counter.sent_sessions = sent_sessions
    counter.total_session_turns = total_session_turns
    return counter


class TestLifecycleStopCondition:
    def test_should_use_always_returns_true(self) -> None:
        assert LifecycleStopCondition.should_use(make_config()) is True

    def test_can_send_when_not_cancelled_and_not_complete(self) -> None:
        lifecycle = make_mock_lifecycle(was_cancelled=False, is_sending_complete=False)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)
        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_cancelled(self) -> None:
        lifecycle = make_mock_lifecycle(was_cancelled=True, is_sending_complete=False)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)
        assert condition.can_send_any_turn() is False

    def test_cannot_send_when_sending_complete(self) -> None:
        lifecycle = make_mock_lifecycle(was_cancelled=False, is_sending_complete=True)
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)
        assert condition.can_send_any_turn() is False

    def test_can_start_new_session_returns_true(self) -> None:
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter()
        condition = LifecycleStopCondition(make_config(), lifecycle, counter)
        assert condition.can_start_new_session() is True


class TestRequestCountStopCondition:
    def test_should_use_when_request_count_configured(self) -> None:
        assert (
            RequestCountStopCondition.should_use(
                make_config(total_expected_requests=100)
            )
            is True
        )

    def test_should_not_use_when_no_request_count(self) -> None:
        assert (
            RequestCountStopCondition.should_use(
                make_config(total_expected_requests=None)
            )
            is False
        )

    # fmt: skip
    @pytest.mark.parametrize(
        "sent,limit,expected",
        [
            (0, 1, True),
            (0, 100, True),
            (99, 100, True),
            (100, 100, False),
            (150, 100, False),
        ],
    )
    def test_request_count_scenarios(
        self, sent: int, limit: int, expected: bool
    ) -> None:
        config = make_config(total_expected_requests=limit)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(requests_sent=sent)
        condition = RequestCountStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is expected


class TestSessionCountStopCondition:
    def test_should_use_when_session_count_configured(self) -> None:
        assert (
            SessionCountStopCondition.should_use(make_config(expected_num_sessions=10))
            is True
        )

    def test_should_not_use_when_no_session_count(self) -> None:
        assert (
            SessionCountStopCondition.should_use(
                make_config(expected_num_sessions=None)
            )
            is False
        )

    def test_can_send_when_sessions_under_limit(self) -> None:
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(sent_sessions=5)
        condition = SessionCountStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is True

    def test_can_send_when_sessions_at_limit_but_turns_remaining(self) -> None:
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=15, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is True

    def test_cannot_send_when_all_session_turns_complete(self) -> None:
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=20, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is False

    def test_can_start_new_session_when_under_limit(self) -> None:
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(sent_sessions=5)
        condition = SessionCountStopCondition(config, lifecycle, counter)
        assert condition.can_start_new_session() is True

    def test_cannot_start_new_session_when_at_limit(self) -> None:
        config = make_config(expected_num_sessions=10)
        lifecycle = make_mock_lifecycle()
        counter = make_mock_counter(
            sent_sessions=10, requests_sent=5, total_session_turns=20
        )
        condition = SessionCountStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is True
        assert condition.can_start_new_session() is False


class TestDurationStopCondition:
    def test_should_use_when_duration_configured(self) -> None:
        assert (
            DurationStopCondition.should_use(make_config(expected_duration_sec=60.0))
            is True
        )

    def test_should_not_use_when_no_duration(self) -> None:
        assert (
            DurationStopCondition.should_use(make_config(expected_duration_sec=None))
            is False
        )

    # fmt: skip
    @pytest.mark.parametrize(
        "time_left,expected",
        [(30.0, True), (0.001, True), (0.0, False), (-5.0, False)],
    )
    def test_duration_scenarios(self, time_left: float, expected: bool) -> None:
        config = make_config(expected_duration_sec=60.0)
        lifecycle = make_mock_lifecycle(time_left=time_left)
        counter = make_mock_counter()
        condition = DurationStopCondition(config, lifecycle, counter)
        assert condition.can_send_any_turn() is expected


class TestStopConditionCheckerConfiguration:
    def test_lifecycle_always_included(self) -> None:
        checker = StopConditionChecker(
            make_config(), make_mock_lifecycle(), make_mock_counter()
        )
        assert len(checker._stop_conditions) >= 1

    def test_request_count_condition_included_when_configured(self) -> None:
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(),
            make_mock_counter(),
        )
        condition_types = [type(c).__name__ for c in checker._stop_conditions]
        assert "LifecycleStopCondition" in condition_types
        assert "RequestCountStopCondition" in condition_types

    def test_all_conditions_included_when_all_configured(self) -> None:
        checker = StopConditionChecker(
            make_config(
                total_expected_requests=100,
                expected_num_sessions=10,
                expected_duration_sec=60.0,
            ),
            make_mock_lifecycle(),
            make_mock_counter(),
        )
        condition_types = [type(c).__name__ for c in checker._stop_conditions]
        assert "LifecycleStopCondition" in condition_types
        assert "RequestCountStopCondition" in condition_types
        assert "SessionCountStopCondition" in condition_types
        assert "DurationStopCondition" in condition_types


class TestStopConditionCheckerCanSendAnyTurn:
    def test_can_send_when_all_conditions_pass(self) -> None:
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(was_cancelled=False, is_sending_complete=False),
            make_mock_counter(requests_sent=50),
        )
        assert checker.can_send_any_turn() is True

    def test_cannot_send_when_lifecycle_fails(self) -> None:
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(was_cancelled=True),
            make_mock_counter(requests_sent=50),
        )
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_request_count_reached(self) -> None:
        checker = StopConditionChecker(
            make_config(total_expected_requests=100),
            make_mock_lifecycle(),
            make_mock_counter(requests_sent=100),
        )
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_duration_expired(self) -> None:
        checker = StopConditionChecker(
            make_config(expected_duration_sec=60.0),
            make_mock_lifecycle(time_left=0.0),
            make_mock_counter(),
        )
        assert checker.can_send_any_turn() is False


class TestStopConditionCheckerCanStartNewSession:
    def test_can_start_session_when_all_conditions_pass(self) -> None:
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(sent_sessions=5),
        )
        assert checker.can_start_new_session() is True

    def test_cannot_start_session_when_general_condition_fails(self) -> None:
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(was_cancelled=True),
            make_mock_counter(sent_sessions=5),
        )
        assert checker.can_send_any_turn() is False
        assert checker.can_start_new_session() is False

    def test_cannot_start_session_when_session_limit_reached(self) -> None:
        checker = StopConditionChecker(
            make_config(expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(
                sent_sessions=10, requests_sent=5, total_session_turns=20
            ),
        )
        assert checker.can_send_any_turn() is True
        assert checker.can_start_new_session() is False


class TestStopConditionCheckerEdgeCases:
    def test_empty_config_only_lifecycle(self) -> None:
        checker = StopConditionChecker(
            make_config(),
            make_mock_lifecycle(),
            make_mock_counter(requests_sent=1_000_000),
        )
        assert checker.can_send_any_turn() is True
        assert checker.can_start_new_session() is True

    def test_first_condition_failure_short_circuits(self) -> None:
        lifecycle = make_mock_lifecycle(was_cancelled=True)
        counter = make_mock_counter()
        checker = StopConditionChecker(
            make_config(total_expected_requests=100, expected_duration_sec=60.0),
            lifecycle,
            counter,
        )
        assert checker.can_send_any_turn() is False

    # fmt: skip
    @pytest.mark.parametrize(
        "requests,sessions,turns,expected_any,expected_new",
        [
            (5, 5, 20, True, True),
            (99, 5, 20, True, True),
            (100, 5, 20, False, False),
            (5, 9, 20, True, True),
            (5, 10, 20, True, False),
            (20, 10, 20, False, False),
        ],
    )
    def test_boundary_conditions(
        self,
        requests: int,
        sessions: int,
        turns: int,
        expected_any: bool,
        expected_new: bool,
    ) -> None:
        checker = StopConditionChecker(
            make_config(total_expected_requests=100, expected_num_sessions=10),
            make_mock_lifecycle(),
            make_mock_counter(
                requests_sent=requests,
                sent_sessions=sessions,
                total_session_turns=turns,
            ),
        )
        assert checker.can_send_any_turn() is expected_any
        assert checker.can_start_new_session() is expected_new

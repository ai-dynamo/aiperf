# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
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


def cfg(
    reqs: int | None = None, sessions: int | None = None, dur: float | None = None
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
        expected_duration_sec=dur,
    )


def lc(
    cancelled: bool = False, sending_complete: bool = False, time_left: float = 10.0
) -> MagicMock:
    m = MagicMock(spec=PhaseLifecycle)
    m.was_cancelled = cancelled
    m.is_sending_complete = sending_complete
    m.time_left_in_seconds = MagicMock(return_value=time_left)
    return m


def ctr(sent: int = 0, sessions: int = 0, turns: int = 0) -> MagicMock:
    m = MagicMock(spec=CreditCounter)
    m.requests_sent = sent
    m.sent_sessions = sessions
    m.total_session_turns = turns
    return m


class TestLifecycleStopCondition:
    def test_should_use_always_true(self) -> None:
        assert LifecycleStopCondition.should_use(cfg()) is True

    def test_can_send_when_not_cancelled_and_not_complete(self) -> None:
        cond = LifecycleStopCondition(
            cfg(), lc(cancelled=False, sending_complete=False), ctr()
        )
        assert cond.can_send_any_turn() is True

    def test_cannot_send_when_cancelled(self) -> None:
        cond = LifecycleStopCondition(cfg(), lc(cancelled=True), ctr())
        assert cond.can_send_any_turn() is False

    def test_cannot_send_when_sending_complete(self) -> None:
        cond = LifecycleStopCondition(cfg(), lc(sending_complete=True), ctr())
        assert cond.can_send_any_turn() is False

    def test_can_start_new_session_returns_true(self) -> None:
        cond = LifecycleStopCondition(cfg(), lc(), ctr())
        assert cond.can_start_new_session() is True


class TestRequestCountStopCondition:
    def test_should_use_when_configured(self) -> None:
        assert RequestCountStopCondition.should_use(cfg(reqs=100)) is True

    def test_should_not_use_when_not_configured(self) -> None:
        assert RequestCountStopCondition.should_use(cfg(reqs=None)) is False

    # fmt: off
    @pytest.mark.parametrize("sent,limit,expected", [(0, 1, True), (0, 100, True), (99, 100, True), (100, 100, False), (150, 100, False)])
    def test_request_count_scenarios(self, sent: int, limit: int, expected: bool) -> None:
        cond = RequestCountStopCondition(cfg(reqs=limit), lc(), ctr(sent=sent))
        assert cond.can_send_any_turn() is expected
    # fmt: on


class TestSessionCountStopCondition:
    def test_should_use_when_configured(self) -> None:
        assert SessionCountStopCondition.should_use(cfg(sessions=10)) is True

    def test_should_not_use_when_not_configured(self) -> None:
        assert SessionCountStopCondition.should_use(cfg(sessions=None)) is False

    def test_can_send_when_under_limit(self) -> None:
        cond = SessionCountStopCondition(cfg(sessions=10), lc(), ctr(sessions=5))
        assert cond.can_send_any_turn() is True

    def test_can_send_when_at_limit_but_turns_remaining(self) -> None:
        cond = SessionCountStopCondition(
            cfg(sessions=10), lc(), ctr(sessions=10, sent=15, turns=20)
        )
        assert cond.can_send_any_turn() is True

    def test_cannot_send_when_all_turns_complete(self) -> None:
        cond = SessionCountStopCondition(
            cfg(sessions=10), lc(), ctr(sessions=10, sent=20, turns=20)
        )
        assert cond.can_send_any_turn() is False

    def test_can_start_new_session_when_under_limit(self) -> None:
        cond = SessionCountStopCondition(cfg(sessions=10), lc(), ctr(sessions=5))
        assert cond.can_start_new_session() is True

    def test_cannot_start_new_session_at_limit(self) -> None:
        cond = SessionCountStopCondition(
            cfg(sessions=10), lc(), ctr(sessions=10, sent=5, turns=20)
        )
        assert (
            cond.can_send_any_turn() is True and cond.can_start_new_session() is False
        )


class TestDurationStopCondition:
    def test_should_use_when_configured(self) -> None:
        assert DurationStopCondition.should_use(cfg(dur=60.0)) is True

    def test_should_not_use_when_not_configured(self) -> None:
        assert DurationStopCondition.should_use(cfg(dur=None)) is False

    # fmt: off
    @pytest.mark.parametrize("time_left,expected", [(30.0, True), (0.001, True), (0.0, False), (-5.0, False)])
    def test_duration_scenarios(self, time_left: float, expected: bool) -> None:
        cond = DurationStopCondition(cfg(dur=60.0), lc(time_left=time_left), ctr())
        assert cond.can_send_any_turn() is expected
    # fmt: on


class TestStopConditionChecker:
    def test_can_send_when_all_pass(self) -> None:
        checker = StopConditionChecker(cfg(reqs=100), lc(), ctr(sent=50))
        assert checker.can_send_any_turn() is True

    def test_cannot_send_when_lifecycle_fails(self) -> None:
        checker = StopConditionChecker(cfg(reqs=100), lc(cancelled=True), ctr(sent=50))
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_request_count_reached(self) -> None:
        checker = StopConditionChecker(cfg(reqs=100), lc(), ctr(sent=100))
        assert checker.can_send_any_turn() is False

    def test_cannot_send_when_duration_expired(self) -> None:
        checker = StopConditionChecker(cfg(dur=60.0), lc(time_left=0.0), ctr())
        assert checker.can_send_any_turn() is False

    def test_can_start_session_when_all_pass(self) -> None:
        checker = StopConditionChecker(cfg(sessions=10), lc(), ctr(sessions=5))
        assert checker.can_start_new_session() is True

    def test_cannot_start_session_when_general_fails(self) -> None:
        checker = StopConditionChecker(
            cfg(sessions=10), lc(cancelled=True), ctr(sessions=5)
        )
        assert (
            checker.can_send_any_turn() is False
            and checker.can_start_new_session() is False
        )

    def test_cannot_start_session_when_limit_reached(self) -> None:
        checker = StopConditionChecker(
            cfg(sessions=10), lc(), ctr(sessions=10, sent=5, turns=20)
        )
        assert (
            checker.can_send_any_turn() is True
            and checker.can_start_new_session() is False
        )

    def test_empty_config_only_lifecycle(self) -> None:
        checker = StopConditionChecker(cfg(), lc(), ctr(sent=1_000_000))
        assert (
            checker.can_send_any_turn() is True
            and checker.can_start_new_session() is True
        )

    # fmt: off
    @pytest.mark.parametrize("sent,sessions,turns,exp_any,exp_new", [
        (5, 5, 20, True, True), (99, 5, 20, True, True), (100, 5, 20, False, False),
        (5, 9, 20, True, True), (5, 10, 20, True, False), (20, 10, 20, False, False),
    ])
    def test_boundary_conditions(self, sent: int, sessions: int, turns: int, exp_any: bool, exp_new: bool) -> None:
        checker = StopConditionChecker(cfg(reqs=100, sessions=10), lc(), ctr(sent=sent, sessions=sessions, turns=turns))
        assert checker.can_send_any_turn() is exp_any and checker.can_start_new_session() is exp_new
    # fmt: on

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
    CancellationStopCondition,
    DurationStopCondition,
    RequestCountStopCondition,
    SendingCompleteStopCondition,
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


def ctr(
    sent: int = 0,
    sessions: int = 0,
    turns: int = 0,
    root_sent: int | None = None,
) -> MagicMock:
    m = MagicMock(spec=CreditCounter)
    m.requests_sent = sent
    # No DAG children in most unit cases, so root_requests_sent tracks
    # requests_sent unless a test explicitly drives them apart.
    m.root_requests_sent = sent if root_sent is None else root_sent
    m.sent_sessions = sessions
    m.total_session_turns = turns
    return m


class TestCancellationStopCondition:
    def test_should_use_always_true(self) -> None:
        assert CancellationStopCondition.should_use(cfg()) is True

    def test_can_send_when_not_cancelled(self) -> None:
        cond = CancellationStopCondition(cfg(), lc(cancelled=False), ctr())
        assert cond.can_send_any_turn() is True

    def test_cannot_send_when_cancelled(self) -> None:
        cond = CancellationStopCondition(cfg(), lc(cancelled=True), ctr())
        assert cond.can_send_any_turn() is False

    def test_ignores_sending_complete_flag(self) -> None:
        """Sending-complete is a separate concern — see
        SendingCompleteStopCondition. Cancellation alone gates here."""
        cond = CancellationStopCondition(
            cfg(), lc(cancelled=False, sending_complete=True), ctr()
        )
        assert cond.can_send_any_turn() is True

    def test_applies_to_dag_children(self) -> None:
        assert CancellationStopCondition.applies_to_dag_children is True


class TestSendingCompleteStopCondition:
    def test_should_use_always_true(self) -> None:
        assert SendingCompleteStopCondition.should_use(cfg()) is True

    def test_can_send_when_not_sending_complete(self) -> None:
        cond = SendingCompleteStopCondition(cfg(), lc(sending_complete=False), ctr())
        assert cond.can_send_any_turn() is True

    def test_cannot_send_when_sending_complete(self) -> None:
        cond = SendingCompleteStopCondition(cfg(), lc(sending_complete=True), ctr())
        assert cond.can_send_any_turn() is False

    def test_ignores_cancellation(self) -> None:
        """Cancellation is a separate concern — see CancellationStopCondition."""
        cond = SendingCompleteStopCondition(
            cfg(), lc(cancelled=True, sending_complete=False), ctr()
        )
        assert cond.can_send_any_turn() is True

    def test_does_not_apply_to_dag_children(self) -> None:
        """The whole reason this condition is split from CancellationStopCondition:
        DAG children must bypass the root-sampler-done signal to drain."""
        assert SendingCompleteStopCondition.applies_to_dag_children is False


class TestRequestCountStopCondition:
    def test_should_use_when_configured(self) -> None:
        assert RequestCountStopCondition.should_use(cfg(reqs=100)) is True

    def test_should_not_use_when_not_configured(self) -> None:
        assert RequestCountStopCondition.should_use(cfg(reqs=None)) is False

    def test_applies_to_dag_children_invariant(self) -> None:
        # INVARIANT (exact-cutoff "N means N"): --request-count is a literal wire
        # cap honored by DAG children, so this MUST inherit the base True. Pinned
        # by name on purpose — re-adding `applies_to_dag_children = False` here
        # (the regression that has happened 3x) fails loudly. Paired with
        # SessionCountStopCondition staying False (see TestSessionCountStopCondition).
        assert RequestCountStopCondition.applies_to_dag_children is True

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

    def test_dag_children_do_not_close_gate_on_unsent_root_turns(self) -> None:
        """Regression: DAG children inflate ``requests_sent`` but inherit the
        parent's session slot and add no root turns. The gate must compare
        ``root_requests_sent`` (not ``requests_sent``) against
        ``total_session_turns`` so a multi-turn root's remaining continuations
        still dispatch. Pre-fix, ``requests_sent (20) >= total_session_turns
        (20)`` closed the gate while ``root_requests_sent (5) < 20`` meant
        ``is_final_credit`` never fired -> dropped root turns / hang."""
        cond = SessionCountStopCondition(
            cfg(sessions=10),
            lc(),
            ctr(sessions=10, sent=20, turns=20, root_sent=5),
        )
        assert cond.can_send_any_turn() is True

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


class TestStopConditionCheckerChildTurns:
    """``can_send_child_turn`` is what ``CreditIssuer`` consults for DAG
    children. Children HONOR cancellation, duration timeout, AND the
    ``--request-count`` wire cap ("N means N"); they BYPASS only
    ``SendingCompleteStopCondition`` (root-sampler-done) and
    ``SessionCountStopCondition`` (``--num-conversations`` = run full trees).
    """

    def test_child_can_send_past_sending_complete(self) -> None:
        """The one condition children are supposed to bypass: the
        phase's ``is_sending_complete`` flag flips the instant root
        sampling finishes, but the DAG may still have in-flight
        descendants that need to run."""
        checker = StopConditionChecker(cfg(), lc(sending_complete=True), ctr())
        assert checker.can_send_any_turn() is False  # roots stopped
        assert checker.can_send_child_turn() is True  # children continue

    def test_child_still_honors_cancellation(self) -> None:
        """Regression guard: user Ctrl-C / explicit abort must stop
        DAG children too, even though children bypass
        is_sending_complete."""
        checker = StopConditionChecker(cfg(), lc(cancelled=True), ctr())
        assert checker.can_send_any_turn() is False
        assert checker.can_send_child_turn() is False

    def test_child_still_honors_duration_timeout(self) -> None:
        """Children must stop when the benchmark duration expires —
        we promised the user a time-bounded run."""
        checker = StopConditionChecker(cfg(dur=60.0), lc(time_left=-1.0), ctr())
        assert checker.can_send_any_turn() is False
        assert checker.can_send_child_turn() is False

    def test_child_honors_request_count_limit(self) -> None:
        """``--request-count N`` is a literal cap on total wire requests
        ("N means N"), honored by EVERY credit including DAG children. At the
        cap a child is refused; the issuance chokepoint routes that refusal
        through ``on_child_stopped`` so the parent's join drains.
        """
        checker = StopConditionChecker(cfg(reqs=1), lc(), ctr(sent=1))
        assert checker.can_send_any_turn() is False  # roots stopped
        assert (
            checker.can_send_child_turn() is False
        )  # children ALSO stopped (hard cap)

    def test_child_bypasses_session_count_limit(self) -> None:
        """Same rationale as request count: ``--conversation-num`` caps
        the sampler's session plan, not reactive DAG offspring."""
        checker = StopConditionChecker(cfg(sessions=1), lc(), ctr(sessions=1))
        assert checker.can_send_any_turn() is False
        assert checker.can_send_child_turn() is True

    def test_child_honors_both_cancellation_and_sending_complete_combined(self) -> None:
        """When both signals are set (cancel during DAG drain),
        children must stop — cancellation dominates."""
        checker = StopConditionChecker(
            cfg(), lc(cancelled=True, sending_complete=True), ctr()
        )
        assert checker.can_send_any_turn() is False
        assert checker.can_send_child_turn() is False

    def test_child_allowed_when_all_conditions_happy(self) -> None:
        checker = StopConditionChecker(cfg(reqs=100, dur=60.0), lc(), ctr(sent=5))
        assert checker.can_send_any_turn() is True
        assert checker.can_send_child_turn() is True

    # fmt: off
    @pytest.mark.parametrize("sent,sessions,turns,exp_any,exp_new", [
        (5, 5, 20, True, True), (99, 5, 20, True, True), (100, 5, 20, False, False),
        (5, 9, 20, True, True), (5, 10, 20, True, False), (20, 10, 20, False, False),
    ])
    def test_boundary_conditions(self, sent: int, sessions: int, turns: int, exp_any: bool, exp_new: bool) -> None:
        checker = StopConditionChecker(cfg(reqs=100, sessions=10), lc(), ctr(sent=sent, sessions=sessions, turns=turns))
        assert checker.can_send_any_turn() is exp_any and checker.can_start_new_session() is exp_new
    # fmt: on

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.credit.structs import TurnToSend
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


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


def turn(conv: str = "c1", idx: int = 0, num: int = 1, corr: str = "x1") -> TurnToSend:
    return TurnToSend(
        conversation_id=conv, turn_index=idx, num_turns=num, x_correlation_id=corr
    )


class TestCreditCounter:
    def test_initial_state_counters_zero(self) -> None:
        c = CreditCounter(cfg())
        assert (
            c.requests_sent
            == c.requests_completed
            == c.requests_cancelled
            == c.request_errors
            == 0
        )
        assert (
            c.sent_sessions
            == c.completed_sessions
            == c.cancelled_sessions
            == c.total_session_turns
            == 0
        )
        assert (
            c.prefills_released
            == c.in_flight
            == c.in_flight_sessions
            == c.in_flight_prefills
            == 0
        )

    def test_initial_state_finals_none(self) -> None:
        c = CreditCounter(cfg())
        assert (
            c.final_requests_sent
            is c.final_requests_completed
            is c.final_requests_cancelled
            is None
        )
        assert (
            c.final_request_errors
            is c.final_sent_sessions
            is c.final_completed_sessions
            is None
        )
        assert c.final_cancelled_sessions is None

    def test_increment_sent_returns_sequential_index(self) -> None:
        c = CreditCounter(cfg())
        for i in range(10):
            idx, _ = c.increment_sent(turn(idx=0))
            assert idx == i
        assert c.requests_sent == 10

    def test_first_turn_increments_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        assert c.sent_sessions == 1 and c.total_session_turns == 3
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=2, num=3))
        assert c.sent_sessions == 1 and c.requests_sent == 3

    def test_total_session_turns_accumulates(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=2, num=3))
        c.increment_sent(turn(idx=0, num=5))
        assert c.total_session_turns == 8

    # fmt: off
    @pytest.mark.parametrize("reqs,expected_finals", [(3, [False, False, True]), (1, [True])])
    def test_final_when_request_count_reached(self, reqs: int, expected_finals: list[bool]) -> None:
        c = CreditCounter(cfg(reqs=reqs))
        for exp in expected_finals:
            _, is_final = c.increment_sent(turn())
            assert is_final == exp
    # fmt: on

    def test_not_final_without_request_limit(self) -> None:
        c = CreditCounter(cfg(reqs=None))
        for _ in range(100):
            _, is_final = c.increment_sent(turn())
            assert not is_final

    def test_final_when_sessions_complete(self) -> None:
        c = CreditCounter(cfg(sessions=2))
        finals = []
        for _ in range(2):
            for t in range(2):
                _, f = c.increment_sent(turn(idx=t, num=2))
                finals.append(f)
        assert finals == [False, False, False, True]

    def test_not_final_until_all_session_turns_sent(self) -> None:
        c = CreditCounter(cfg(sessions=2))
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=1, num=2))
        _, is_final = c.increment_sent(turn(idx=2, num=3))
        assert is_final

    def test_increment_returned_completed(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.requests_completed == 1 and c.requests_cancelled == 0

    def test_increment_returned_cancelled(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.requests_cancelled == 1 and c.requests_completed == 0

    def test_increment_returned_final_turn_increments_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.completed_sessions == 0
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.completed_sessions == 1

    def test_increment_returned_cancelled_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=1))
        c.increment_returned(is_final_turn=True, cancelled=True)
        assert c.cancelled_sessions == 1 and c.completed_sessions == 0

    def test_increment_returned_all_done(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_sent(turn())
        c.freeze_sent_counts()
        assert not c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.increment_returned(is_final_turn=True, cancelled=False)

    def test_in_flight_tracking(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(3):
            c.increment_sent(turn())
        assert c.in_flight == 3
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.in_flight == 2
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.in_flight == 1

    def test_in_flight_sessions(self) -> None:
        c = CreditCounter(cfg())
        for x in "abc":
            c.increment_sent(turn(conv=x, idx=0, num=2))
        assert c.in_flight_sessions == 3
        c.increment_sent(turn(conv="a", idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.in_flight_sessions == 2
        c.increment_returned(is_final_turn=True, cancelled=True)
        assert c.in_flight_sessions == 1

    def test_in_flight_prefills(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(3):
            c.increment_sent(turn())
        assert c.in_flight_prefills == 3
        c.increment_prefill_released()
        c.increment_prefill_released()
        c.increment_prefill_released()
        assert c.in_flight_prefills == 0

    def test_freeze_sent_counts(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=1, num=3))
        c.freeze_sent_counts()
        assert c.final_requests_sent == 2 and c.final_sent_sessions == 1
        c.increment_sent(turn(idx=2, num=3))
        assert c.final_requests_sent == 2 and c.requests_sent == 3

    def test_freeze_completed_counts(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=2))
        c.freeze_sent_counts()
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.freeze_completed_counts()
        assert (
            c.final_requests_completed == 1
            and c.final_requests_cancelled == 0
            and c.final_completed_sessions == 0
        )

    def test_check_all_returned_requires_frozen(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert not c.check_all_returned_or_cancelled()
        c.increment_sent(turn())
        c.freeze_sent_counts()
        assert not c.check_all_returned_or_cancelled()
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.check_all_returned_or_cancelled()

    def test_single_request_limit_is_final(self) -> None:
        c = CreditCounter(cfg(reqs=1))
        _, is_final = c.increment_sent(turn())
        assert is_final

    def test_single_session_single_turn_is_final(self) -> None:
        c = CreditCounter(cfg(sessions=1))
        _, is_final = c.increment_sent(turn(idx=0, num=1))
        assert is_final

    def test_mixed_completed_and_cancelled(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(5):
            c.increment_sent(turn())
        c.freeze_sent_counts()
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=True)
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert (
            c.requests_completed == 2 and c.requests_cancelled == 2 and c.in_flight == 1
        )
        assert not c.check_all_returned_or_cancelled()
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.check_all_returned_or_cancelled()

    # fmt: off
    @pytest.mark.parametrize("num_sessions,turns_per_session", [(1, 1), (1, 5), (5, 1), (3, 4), (10, 10)])
    def test_session_completion_various_configs(self, num_sessions: int, turns_per_session: int) -> None:
        c = CreditCounter(cfg(sessions=num_sessions))
        final_detected = False
        for s in range(num_sessions):
            for t in range(turns_per_session):
                _, is_final = c.increment_sent(turn(idx=t, num=turns_per_session, conv=f"c{s}"))
                if is_final:
                    final_detected = True
        total = num_sessions * turns_per_session
        assert c.requests_sent == total and c.sent_sessions == num_sessions and c.total_session_turns == total
        assert final_detected
    # fmt: on

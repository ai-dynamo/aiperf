# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
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
    def test_initial_state(self) -> None:
        c = CreditCounter(cfg())
        # All progress counters start at zero
        assert c.requests_sent == 0
        assert c.requests_completed == 0
        assert c.requests_cancelled == 0
        assert c.request_errors == 0
        assert c.sent_sessions == 0
        assert c.completed_sessions == 0
        assert c.cancelled_sessions == 0
        assert c.total_session_turns == 0
        assert c.prefills_released == 0
        # Derived counters also zero
        assert c.in_flight == 0
        assert c.in_flight_sessions == 0
        assert c.in_flight_prefills == 0
        # Final counts are None until frozen
        assert c.final_requests_sent is None
        assert c.final_requests_completed is None
        assert c.final_requests_cancelled is None
        assert c.final_request_errors is None
        assert c.final_sent_sessions is None
        assert c.final_completed_sessions is None
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

    def test_single_session_single_turn_is_final(self) -> None:
        c = CreditCounter(cfg(sessions=1))
        _, is_final = c.increment_sent(turn(idx=0, num=1))
        assert is_final

    def test_mixed_completed_and_cancelled_with_all_done_check(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(5):
            c.increment_sent(turn())
        c.freeze_sent_counts()
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=True)
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.requests_completed == 2
        assert c.requests_cancelled == 2
        assert not c.check_all_returned_or_cancelled()
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.check_all_returned_or_cancelled()

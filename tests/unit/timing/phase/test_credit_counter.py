# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CreditCounter lock-free credit tracking."""

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.credit.structs import TurnToSend
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


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


def make_turn(
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    x_correlation_id: str = "corr1",
) -> TurnToSend:
    return TurnToSend(
        conversation_id=conversation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        x_correlation_id=x_correlation_id,
    )


class TestCreditCounterInitialState:
    def test_counters_start_at_zero(self) -> None:
        counter = CreditCounter(make_config())
        assert counter.requests_sent == 0
        assert counter.requests_completed == 0
        assert counter.requests_cancelled == 0
        assert counter.request_errors == 0
        assert counter.sent_sessions == 0
        assert counter.completed_sessions == 0
        assert counter.cancelled_sessions == 0
        assert counter.total_session_turns == 0
        assert counter.prefills_released == 0

    def test_final_counts_start_as_none(self) -> None:
        counter = CreditCounter(make_config())
        assert counter.final_requests_sent is None
        assert counter.final_requests_completed is None
        assert counter.final_requests_cancelled is None
        assert counter.final_request_errors is None
        assert counter.final_sent_sessions is None
        assert counter.final_completed_sessions is None
        assert counter.final_cancelled_sessions is None

    def test_derived_properties_start_at_zero(self) -> None:
        counter = CreditCounter(make_config())
        assert counter.in_flight == 0
        assert counter.in_flight_sessions == 0
        assert counter.in_flight_prefills == 0


class TestAtomicIncrementSent:
    def test_returns_zero_index_for_first_credit(self) -> None:
        counter = CreditCounter(make_config())
        index, is_final = counter.increment_sent(make_turn())
        assert index == 0

    def test_increments_index_sequentially(self) -> None:
        counter = CreditCounter(make_config())
        for expected_index in range(10):
            index, _ = counter.increment_sent(make_turn(turn_index=0))
            assert index == expected_index

    def test_increments_sent_count(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        assert counter.requests_sent == 1
        counter.increment_sent(make_turn())
        assert counter.requests_sent == 2

    def test_first_turn_increments_session_count(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.sent_sessions == 1
        assert counter.total_session_turns == 3

    def test_subsequent_turn_does_not_increment_session_count(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.sent_sessions == 1
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.sent_sessions == 1
        assert counter.requests_sent == 3

    def test_total_session_turns_tracks_expected_turns(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.total_session_turns == 3
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.total_session_turns == 3
        counter.increment_sent(make_turn(turn_index=0, num_turns=5))
        assert counter.total_session_turns == 8


class TestAtomicIncrementSentFinalDetection:
    def test_final_when_request_count_reached(self) -> None:
        counter = CreditCounter(make_config(total_expected_requests=3))
        _, is_final = counter.increment_sent(make_turn())
        assert not is_final
        _, is_final = counter.increment_sent(make_turn())
        assert not is_final
        _, is_final = counter.increment_sent(make_turn())
        assert is_final

    def test_not_final_without_request_count_limit(self) -> None:
        counter = CreditCounter(make_config(total_expected_requests=None))
        for _ in range(100):
            _, is_final = counter.increment_sent(make_turn())
            assert not is_final

    def test_final_when_sessions_complete(self) -> None:
        counter = CreditCounter(make_config(expected_num_sessions=2))
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert is_final

    def test_not_final_until_all_session_turns_sent(self) -> None:
        counter = CreditCounter(make_config(expected_num_sessions=2))
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert is_final


class TestAtomicIncrementReturned:
    def test_increments_completed_for_success(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.requests_completed == 1
        assert counter.requests_cancelled == 0

    def test_increments_cancelled_for_cancellation(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_returned(is_final_turn=False, cancelled=True)
        assert counter.requests_cancelled == 1
        assert counter.requests_completed == 0

    def test_increments_completed_sessions_on_final_turn(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.completed_sessions == 0
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.completed_sessions == 1

    def test_increments_cancelled_sessions_on_final_cancelled_turn(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=1))
        counter.increment_returned(is_final_turn=True, cancelled=True)
        assert counter.cancelled_sessions == 1
        assert counter.completed_sessions == 0

    def test_returns_false_when_more_credits_in_flight(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()
        all_done = counter.increment_returned(is_final_turn=False, cancelled=False)
        assert not all_done
        assert counter.in_flight == 1

    def test_returns_true_when_all_returned(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()
        counter.increment_returned(is_final_turn=False, cancelled=False)
        all_done = counter.increment_returned(is_final_turn=True, cancelled=False)
        assert all_done


class TestInFlightProperties:
    def test_in_flight_equals_sent_minus_returned(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        assert counter.in_flight == 3
        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.in_flight == 2
        counter.increment_returned(is_final_turn=False, cancelled=True)
        assert counter.in_flight == 1

    def test_in_flight_sessions(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(
            make_turn(turn_index=0, num_turns=2, conversation_id="a")
        )
        counter.increment_sent(
            make_turn(turn_index=0, num_turns=2, conversation_id="b")
        )
        counter.increment_sent(
            make_turn(turn_index=0, num_turns=2, conversation_id="c")
        )
        assert counter.in_flight_sessions == 3
        counter.increment_sent(
            make_turn(turn_index=1, num_turns=2, conversation_id="a")
        )
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.in_flight_sessions == 2
        counter.increment_returned(is_final_turn=True, cancelled=True)
        assert counter.in_flight_sessions == 1

    def test_in_flight_prefills(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        assert counter.in_flight_prefills == 3
        counter.increment_prefill_released()
        assert counter.in_flight_prefills == 2
        counter.increment_prefill_released()
        counter.increment_prefill_released()
        assert counter.in_flight_prefills == 0


class TestFreezeCounts:
    def test_freeze_sent_counts(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        counter.freeze_sent_counts()
        assert counter.final_requests_sent == 2
        assert counter.final_sent_sessions == 1
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.final_requests_sent == 2
        assert counter.requests_sent == 3

    def test_freeze_completed_counts(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        counter.freeze_sent_counts()
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.freeze_completed_counts()
        assert counter.final_requests_completed == 1
        assert counter.final_requests_cancelled == 0
        assert counter.final_completed_sessions == 0

    def test_check_all_returned_requires_frozen_sent(self) -> None:
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert not counter.check_all_returned_or_cancelled()
        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()
        assert not counter.check_all_returned_or_cancelled()
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.check_all_returned_or_cancelled()


class TestCreditCounterEdgeCases:
    def test_single_request_limit(self) -> None:
        counter = CreditCounter(make_config(total_expected_requests=1))
        _, is_final = counter.increment_sent(make_turn())
        assert is_final

    def test_single_session_single_turn(self) -> None:
        counter = CreditCounter(make_config(expected_num_sessions=1))
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=1))
        assert is_final

    def test_mixed_completed_and_cancelled(self) -> None:
        counter = CreditCounter(make_config())
        for _ in range(5):
            counter.increment_sent(make_turn())
        counter.freeze_sent_counts()
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.increment_returned(is_final_turn=False, cancelled=True)
        counter.increment_returned(is_final_turn=False, cancelled=True)
        assert counter.requests_completed == 2
        assert counter.requests_cancelled == 2
        assert counter.in_flight == 1
        assert not counter.check_all_returned_or_cancelled()
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.check_all_returned_or_cancelled()

    # fmt: skip
    @pytest.mark.parametrize(
        "num_sessions,turns_per_session",
        [(1, 1), (1, 5), (5, 1), (3, 4), (10, 10)],
    )
    def test_session_completion_various_configs(
        self, num_sessions: int, turns_per_session: int
    ) -> None:
        counter = CreditCounter(make_config(expected_num_sessions=num_sessions))
        total_turns = num_sessions * turns_per_session
        final_detected = False
        for session in range(num_sessions):
            for turn in range(turns_per_session):
                _, is_final = counter.increment_sent(
                    make_turn(
                        turn_index=turn,
                        num_turns=turns_per_session,
                        conversation_id=f"conv{session}",
                    )
                )
                if is_final:
                    final_detected = True
        assert counter.requests_sent == total_turns
        assert counter.sent_sessions == num_sessions
        assert counter.total_session_turns == total_turns
        assert final_detected, "Final credit should have been detected"

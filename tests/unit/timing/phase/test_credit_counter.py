# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CreditCounter lock-free credit tracking.

Tests cover:
- Initial state verification
- Atomic increment operations
- Property calculations (in-flight, prefills, etc.)
- Freezing counts at phase transitions
- Final turn detection based on config limits
- Edge cases and boundary conditions
"""

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.credit.structs import TurnToSend
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter

# =============================================================================
# Helpers
# =============================================================================


def make_config(
    total_expected_requests: int | None = None,
    expected_num_sessions: int | None = None,
    expected_duration_sec: float | None = None,
) -> CreditPhaseConfig:
    """Create a CreditPhaseConfig for testing."""
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
    """Create a TurnToSend for testing."""
    return TurnToSend(
        conversation_id=conversation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        x_correlation_id=x_correlation_id,
    )


# =============================================================================
# Initial State Tests
# =============================================================================


class TestCreditCounterInitialState:
    """Test CreditCounter initialization."""

    def test_counters_start_at_zero(self):
        """All counters should start at zero."""
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

    def test_final_counts_start_as_none(self):
        """Final counts should be None initially."""
        counter = CreditCounter(make_config())
        assert counter.final_requests_sent is None
        assert counter.final_requests_completed is None
        assert counter.final_requests_cancelled is None
        assert counter.final_request_errors is None
        assert counter.final_sent_sessions is None
        assert counter.final_completed_sessions is None
        assert counter.final_cancelled_sessions is None

    def test_derived_properties_start_at_zero(self):
        """Derived properties should start at zero."""
        counter = CreditCounter(make_config())
        assert counter.in_flight == 0
        assert counter.in_flight_sessions == 0
        assert counter.in_flight_prefills == 0


# =============================================================================
# Atomic Increment Sent Tests
# =============================================================================


class TestAtomicIncrementSent:
    """Test increment_sent behavior."""

    def test_returns_zero_index_for_first_credit(self):
        """First credit should have index 0."""
        counter = CreditCounter(make_config())
        index, is_final = counter.increment_sent(make_turn())
        assert index == 0

    def test_increments_index_sequentially(self):
        """Credit indices should increment sequentially."""
        counter = CreditCounter(make_config())

        for expected_index in range(10):
            index, _ = counter.increment_sent(make_turn(turn_index=0))
            assert index == expected_index

    def test_increments_sent_count(self):
        """requests_sent should increment on each call."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn())
        assert counter.requests_sent == 1

        counter.increment_sent(make_turn())
        assert counter.requests_sent == 2

    def test_first_turn_increments_session_count(self):
        """First turn (turn_index=0) should increment sent_sessions."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.sent_sessions == 1
        assert counter.total_session_turns == 3

    def test_subsequent_turn_does_not_increment_session_count(self):
        """Subsequent turns (turn_index > 0) should not increment sent_sessions."""
        counter = CreditCounter(make_config())

        # First turn
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.sent_sessions == 1

        # Subsequent turns
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.sent_sessions == 1  # Unchanged
        assert counter.requests_sent == 3

    def test_total_session_turns_tracks_expected_turns(self):
        """total_session_turns should sum num_turns from first turns only."""
        counter = CreditCounter(make_config())

        # First session: 3 turns
        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert counter.total_session_turns == 3

        # Send remaining turns of first session
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.total_session_turns == 3  # Unchanged

        # Second session: 5 turns
        counter.increment_sent(make_turn(turn_index=0, num_turns=5))
        assert counter.total_session_turns == 8  # 3 + 5


class TestAtomicIncrementSentFinalDetection:
    """Test final credit detection in increment_sent."""

    def test_final_when_request_count_reached(self):
        """Should detect final credit when request count reached."""
        counter = CreditCounter(make_config(total_expected_requests=3))

        _, is_final = counter.increment_sent(make_turn())
        assert not is_final

        _, is_final = counter.increment_sent(make_turn())
        assert not is_final

        _, is_final = counter.increment_sent(make_turn())
        assert is_final

    def test_not_final_without_request_count_limit(self):
        """Without request count limit, should never be final based on count."""
        counter = CreditCounter(make_config(total_expected_requests=None))

        for _ in range(100):
            _, is_final = counter.increment_sent(make_turn())
            assert not is_final

    def test_final_when_sessions_complete(self):
        """Should detect final when session count AND all turns sent."""
        counter = CreditCounter(make_config(expected_num_sessions=2))

        # Session 1: 2 turns
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert not is_final

        # Session 2: 2 turns
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final
        # Final turn of final session
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert is_final

    def test_not_final_until_all_session_turns_sent(self):
        """Session count alone doesn't make it final - must finish turns."""
        counter = CreditCounter(make_config(expected_num_sessions=2))

        # Session 1 first turn
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        assert not is_final

        # Session 2 first turn - sessions reached but turns not complete
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        assert not is_final

        # Session 1 second turn
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=3))
        assert not is_final

        # Session 2 second (final) turn
        _, is_final = counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        assert not is_final  # Still have session 1's third turn

        # Session 1 final turn
        _, is_final = counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert is_final


# =============================================================================
# Atomic Increment Returned Tests
# =============================================================================


class TestAtomicIncrementReturned:
    """Test increment_returned behavior."""

    def test_increments_completed_for_success(self):
        """Successful returns should increment requests_completed."""
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())

        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.requests_completed == 1
        assert counter.requests_cancelled == 0

    def test_increments_cancelled_for_cancellation(self):
        """Cancelled returns should increment requests_cancelled."""
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())

        counter.increment_returned(is_final_turn=False, cancelled=True)
        assert counter.requests_cancelled == 1
        assert counter.requests_completed == 0

    def test_increments_completed_sessions_on_final_turn(self):
        """Final turn completion should increment completed_sessions."""
        counter = CreditCounter(make_config())
        # Send 2 turns for session
        counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        counter.increment_sent(make_turn(turn_index=1, num_turns=2))

        # Return first turn (not final)
        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.completed_sessions == 0

        # Return final turn
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.completed_sessions == 1

    def test_increments_cancelled_sessions_on_final_cancelled_turn(self):
        """Cancelled final turn should increment cancelled_sessions."""
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn(turn_index=0, num_turns=1))

        counter.increment_returned(is_final_turn=True, cancelled=True)
        assert counter.cancelled_sessions == 1
        assert counter.completed_sessions == 0

    def test_returns_false_when_more_credits_in_flight(self):
        """Should return False if credits still in flight."""
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()

        all_done = counter.increment_returned(is_final_turn=False, cancelled=False)
        assert not all_done
        assert counter.in_flight == 1

    def test_returns_true_when_all_returned(self):
        """Should return True when all sent credits are returned."""
        counter = CreditCounter(make_config())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()

        counter.increment_returned(is_final_turn=False, cancelled=False)
        all_done = counter.increment_returned(is_final_turn=True, cancelled=False)
        assert all_done


# =============================================================================
# In-Flight Property Tests
# =============================================================================


class TestInFlightProperties:
    """Test in-flight counting properties."""

    def test_in_flight_equals_sent_minus_returned(self):
        """in_flight should be sent - (completed + cancelled)."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        counter.increment_sent(make_turn())
        assert counter.in_flight == 3

        counter.increment_returned(is_final_turn=False, cancelled=False)
        assert counter.in_flight == 2

        counter.increment_returned(is_final_turn=False, cancelled=True)
        assert counter.in_flight == 1

    def test_in_flight_sessions(self):
        """in_flight_sessions should track active conversations."""
        counter = CreditCounter(make_config())

        # Start 3 sessions
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

        # Complete one session
        counter.increment_sent(
            make_turn(turn_index=1, num_turns=2, conversation_id="a")
        )
        counter.increment_returned(is_final_turn=False, cancelled=False)  # turn 0
        counter.increment_returned(is_final_turn=True, cancelled=False)  # turn 1
        assert counter.in_flight_sessions == 2

        # Cancel one session
        counter.increment_returned(is_final_turn=True, cancelled=True)
        assert counter.in_flight_sessions == 1

    def test_in_flight_prefills(self):
        """in_flight_prefills should track requests awaiting TTFT."""
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


# =============================================================================
# Freeze Counts Tests
# =============================================================================


class TestFreezeCounts:
    """Test freezing counts at phase transitions."""

    def test_freeze_sent_counts(self):
        """freeze_sent_counts should capture current sent values."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn(turn_index=0, num_turns=3))
        counter.increment_sent(make_turn(turn_index=1, num_turns=3))

        counter.freeze_sent_counts()

        assert counter.final_requests_sent == 2
        assert counter.final_sent_sessions == 1

        # Further sends don't affect final counts
        counter.increment_sent(make_turn(turn_index=2, num_turns=3))
        assert counter.final_requests_sent == 2  # Still 2
        assert counter.requests_sent == 3  # But live count updates

    def test_freeze_completed_counts(self):
        """freeze_completed_counts should capture current completed values."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn(turn_index=0, num_turns=2))
        counter.increment_sent(make_turn(turn_index=1, num_turns=2))
        counter.freeze_sent_counts()

        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.freeze_completed_counts()

        assert counter.final_requests_completed == 1
        assert counter.final_requests_cancelled == 0
        assert counter.final_completed_sessions == 0  # No final turns completed yet

    def test_check_all_returned_requires_frozen_sent(self):
        """check_all_returned_or_cancelled requires frozen sent counts."""
        counter = CreditCounter(make_config())

        counter.increment_sent(make_turn())
        counter.increment_returned(is_final_turn=True, cancelled=False)

        # Without freeze, always returns False
        assert not counter.check_all_returned_or_cancelled()

        counter.increment_sent(make_turn())
        counter.freeze_sent_counts()

        # Now it should work
        assert not counter.check_all_returned_or_cancelled()  # 1 returned, 2 sent
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.check_all_returned_or_cancelled()


# =============================================================================
# Edge Cases
# =============================================================================


class TestCreditCounterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_request_limit(self):
        """Single request limit means first credit is final."""
        counter = CreditCounter(make_config(total_expected_requests=1))
        _, is_final = counter.increment_sent(make_turn())
        assert is_final

    def test_single_session_single_turn(self):
        """Single session with single turn should be final immediately."""
        counter = CreditCounter(make_config(expected_num_sessions=1))
        _, is_final = counter.increment_sent(make_turn(turn_index=0, num_turns=1))
        assert is_final

    def test_mixed_completed_and_cancelled(self):
        """Verify mixed completed and cancelled are tracked correctly."""
        counter = CreditCounter(make_config())

        # Send 5 credits
        for _ in range(5):
            counter.increment_sent(make_turn())
        counter.freeze_sent_counts()

        # Complete 2, cancel 2
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.increment_returned(is_final_turn=False, cancelled=False)
        counter.increment_returned(is_final_turn=False, cancelled=True)
        counter.increment_returned(is_final_turn=False, cancelled=True)

        assert counter.requests_completed == 2
        assert counter.requests_cancelled == 2
        assert counter.in_flight == 1
        assert not counter.check_all_returned_or_cancelled()

        # Final return
        counter.increment_returned(is_final_turn=True, cancelled=False)
        assert counter.check_all_returned_or_cancelled()

    @pytest.mark.parametrize(
        "num_sessions,turns_per_session",
        [
            (1, 1),
            (1, 5),
            (5, 1),
            (3, 4),
            (10, 10),
        ],
    )
    def test_session_completion_various_configs(self, num_sessions, turns_per_session):
        """Test session completion detection with various configurations."""
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

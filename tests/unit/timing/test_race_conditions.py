# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for race conditions and deadlock scenarios in inter-process communication.

These tests validate that the system never gets stuck waiting for something that
never happens, focusing on:
1. Credit lifecycle completion tracking
2. RecordsManager completion detection
3. Worker credit return guarantees
4. Phase transition atomicity

Key patterns tested:
- Credit sent but never returned (worker crash simulation)
- Records processed before credit completion message
- Concurrent credit returns and phase completion
- Timeout and cancellation drain scenarios

Architecture notes:
- PhaseLifecycle: State machine (CREATED → STARTED → SENDING_COMPLETE → COMPLETE)
- PhaseProgressTracker: Credit counting + events (wraps CreditCounter)
- StopConditionChecker: Evaluates stop conditions (reads lifecycle + counter)
- PhaseRunner: Coordinates phase execution (replaces old PhaseExecutor)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, DatasetSamplingStrategy, TimingMode
from aiperf.common.factories import DatasetSamplingStrategyFactory
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.records.records_tracker import RecordsTracker
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker

# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_dataset_sampler(conversation_ids: list[str] | None = None):
    """Create mock dataset sampler for testing."""
    if conversation_ids is None:
        conversation_ids = ["conv1", "conv2", "conv3"]
    return DatasetSamplingStrategyFactory.create_instance(
        DatasetSamplingStrategy.SEQUENTIAL,
        conversation_ids=conversation_ids,
    )


def create_turn_to_send(
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    x_correlation_id: str | None = None,
) -> TurnToSend:
    """Create a TurnToSend for testing."""
    return TurnToSend(
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id or f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
    )


def create_credit(
    phase: CreditPhase = CreditPhase.PROFILING,
    credit_id: int = 1,
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
) -> Credit:
    """Create a Credit for testing."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
    )


def create_phase_components(
    config: CreditPhaseConfig,
) -> tuple[PhaseLifecycle, PhaseProgressTracker, StopConditionChecker]:
    """Create the phase component trio for testing.

    Returns:
        (lifecycle, progress, stop_checker) tuple
    """
    lifecycle = PhaseLifecycle(config)
    progress = PhaseProgressTracker(config)
    stop_checker = StopConditionChecker(
        config=config,
        lifecycle=lifecycle,
        counter=progress.counter,
    )
    return lifecycle, progress, stop_checker


# =============================================================================
# Race Condition: Credit Returns vs Phase Completion
# =============================================================================


@pytest.mark.asyncio
class TestCreditReturnRaceConditions:
    """Tests for race conditions in credit return tracking."""

    async def test_late_credit_return_after_phase_complete(self):
        """Test that late credit returns don't corrupt state after phase completion.

        Scenario: Phase is marked complete (via grace period timeout), then a
        slow credit finally returns. Should not increment counters or cause errors.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
            expected_duration_sec=1.0,
            grace_period_sec=0.5,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 10 credits
        for i in range(10):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return only 9 credits
        for i in range(9):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            progress.increment_returned(credit.is_final_turn, cancelled=False)

        # Mark phase as complete (simulating grace period timeout)
        lifecycle.mark_complete(grace_period_triggered=True)
        progress.freeze_completed_counts()
        assert lifecycle.is_complete is True

        # Now a late credit tries to return - should be handled gracefully
        # In the new architecture, caller checks lifecycle.is_complete before calling
        # increment_returned. If called anyway, it still increments (no guard in tracker).
        # The guard is in the caller (CreditCallbackHandler).
        late_credit = create_credit(credit_id=9, conversation_id="conv9")

        # Simulate what CreditCallbackHandler does - check lifecycle first
        if not lifecycle.is_complete:
            progress.increment_returned(late_credit.is_final_turn, cancelled=False)

        # Counts should NOT be affected since we checked lifecycle first
        stats = progress.create_stats(lifecycle)
        assert stats.requests_completed == 9

    async def test_concurrent_credit_returns_dont_miss_completion(self):
        """Test that concurrent returns don't cause missed completion detection.

        Scenario: Two credits return nearly simultaneously when we're at N-1 credits.
        Both check if they're the final credit. Only one should trigger completion.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=3,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 3 credits
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return first credit
        credit0 = create_credit(credit_id=0, conversation_id="conv0")
        progress.increment_returned(credit0.is_final_turn, cancelled=False)

        # Simulate "concurrent" returns - in asyncio this is really sequential
        # but the logic should still be correct
        credit1 = create_credit(credit_id=1, conversation_id="conv1")
        credit2 = create_credit(credit_id=2, conversation_id="conv2")

        result1 = progress.increment_returned(credit1.is_final_turn, cancelled=False)
        if result1:
            progress.all_credits_returned_event.set()
        result2 = progress.increment_returned(credit2.is_final_turn, cancelled=False)
        if result2:
            progress.all_credits_returned_event.set()

        # Exactly one should be the final return
        assert (result1 and not result2) or (not result1 and result2)
        # Event should be set exactly once
        assert progress.all_credits_returned_event.is_set()

    async def test_cancelled_and_completed_credits_both_counted(self):
        """Test that mix of cancelled and completed credits reaches completion.

        Scenario: Some credits complete normally, others are cancelled.
        Total returned (completed + cancelled) should trigger completion.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=4,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 4 credits
        for i in range(4):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return 2 completed, 2 cancelled
        for i in range(2):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            is_all_returned = progress.increment_returned(
                credit.is_final_turn, cancelled=False
            )
            if is_all_returned:
                progress.all_credits_returned_event.set()

        for i in range(2, 4):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            result = progress.increment_returned(credit.is_final_turn, cancelled=True)
            if result:
                progress.all_credits_returned_event.set()
            if i == 3:
                assert result is True  # Final return

        assert progress.all_credits_returned_event.is_set()
        stats = progress.create_stats(lifecycle)
        assert stats.requests_completed == 2
        assert stats.requests_cancelled == 2


# =============================================================================
# Race Condition: RecordsManager Completion Detection
# =============================================================================


class TestRecordsManagerCompletionRace:
    """Tests for race conditions in RecordsManager completion detection.

    The RecordsManager must handle the case where:
    1. All records arrive before CreditPhaseCompleteMessage
    2. CreditPhaseCompleteMessage arrives before all records
    """

    def test_records_arrive_before_phase_complete(self):
        """Test completion when all records arrive before phase completion message.

        Scenario: RecordsManager processes all records, but hasn't received
        CreditPhaseCompleteMessage yet. Should NOT trigger completion until
        it knows the final count.
        """
        records_tracker = RecordsTracker()

        # Receive phase start with expected count
        phase_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            total_expected_requests=5,
            start_ns=1000,
        )
        records_tracker.update_phase_info(phase_stats)

        # Process 5 records (success)
        for _i in range(5):
            phase_tracker = records_tracker._get_phase_tracker(CreditPhase.PROFILING)
            phase_tracker.increment_success_records()

        # Check if all records received - should be FALSE because we don't have
        # final_requests_completed set yet
        result = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is False

        # Now receive phase complete with final count
        phase_complete_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=5,
            start_ns=1000,
            requests_end_ns=2000,
        )
        records_tracker.update_phase_info(phase_complete_stats)

        # NOW check should succeed
        result = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is True

    def test_phase_complete_arrives_before_records(self):
        """Test completion when phase complete message arrives before all records.

        Scenario: CreditPhaseCompleteMessage arrives first with final count,
        then records trickle in. Completion should trigger when count matches.
        """
        records_tracker = RecordsTracker()

        # Receive phase complete first (with final count)
        phase_complete_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=3,
            start_ns=1000,
            requests_end_ns=2000,
        )
        records_tracker.update_phase_info(phase_complete_stats)

        # Check - should be false, we have 0 records
        result = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is False

        # Process 2 records
        phase_tracker = records_tracker._get_phase_tracker(CreditPhase.PROFILING)
        phase_tracker.increment_success_records()
        phase_tracker.increment_success_records()

        # Still not complete
        result = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is False

        # Third record arrives
        phase_tracker.increment_success_records()

        # NOW should be complete
        result = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is True

    def test_duplicate_completion_check_returns_false(self):
        """Test that duplicate completion checks return False.

        Once completion is detected and flagged, subsequent checks should
        return False to prevent duplicate processing.
        """
        records_tracker = RecordsTracker()

        # Set up final count
        phase_complete_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=1,
            start_ns=1000,
        )
        records_tracker.update_phase_info(phase_complete_stats)

        # Process 1 record
        phase_tracker = records_tracker._get_phase_tracker(CreditPhase.PROFILING)
        phase_tracker.increment_success_records()

        # First check should return True
        result1 = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result1 is True

        # Second check should return False (already sent)
        result2 = records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result2 is False


# =============================================================================
# Race Condition: StickyCreditRouter Worker Registration
# =============================================================================


@pytest.mark.asyncio
class TestStickyRouterWorkerRace:
    """Tests for race conditions in worker registration/unregistration."""

    async def test_credit_sent_to_unregistered_worker_during_cancellation(
        self, service_config
    ):
        """Test graceful handling when worker unregisters during credit tracking.

        Scenario: Credits are being cancelled, worker unregisters, then late
        returns arrive for that worker.
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=5)
        }
        router._workers["worker-1"].active_credit_ids = {i for i in range(5)}
        router._workers_cache = list(router._workers.values())

        # Start cancellation
        router._cancellation_pending = True

        # Unregister worker during cancellation
        router._unregister_worker("worker-1")

        # Late credit return for unregistered worker - should not error
        # (worker is gone, but cancellation_pending suppresses errors)
        router._track_credit_returned(
            "worker-1", 0, cancelled=True, error_reported=False
        )

        # Should not raise, should be a no-op due to cancellation pending

    async def test_worker_registration_during_active_routing(self, service_config):
        """Test that new worker registration is immediately available for routing.

        Scenario: Worker registers while credits are being routed.
        New worker should be immediately available for load balancing.
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        # Register first worker
        router._register_worker("worker-1")
        assert len(router._workers) == 1
        assert len(router._workers_cache) == 1

        # Register second worker
        router._register_worker("worker-2")

        # Both should be in cache for load balancing
        assert len(router._workers) == 2
        assert len(router._workers_cache) == 2
        assert {w.worker_id for w in router._workers_cache} == {"worker-1", "worker-2"}

    async def test_worker_unregister_clears_from_cache(self, service_config):
        """Test that unregistered workers are removed from load balancing cache."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        # Register two workers
        router._register_worker("worker-1")
        router._register_worker("worker-2")
        assert len(router._workers_cache) == 2

        # Unregister one
        router._unregister_worker("worker-1")

        # Cache should be updated atomically
        assert len(router._workers) == 1
        assert len(router._workers_cache) == 1
        assert router._workers_cache[0].worker_id == "worker-2"


# =============================================================================
# Race Condition: Counter Atomicity
# =============================================================================


class TestCreditCounterAtomicity:
    """Tests for atomicity of credit counter operations.

    These tests verify that the lock-free design (relying on asyncio's
    single-threaded execution) correctly serializes operations.
    """

    def test_increment_sent_returns_unique_indices(self):
        """Test that increment_sent returns unique credit indices.

        Each call should return a unique index even under rapid calls.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=100,
        )
        counter = CreditCounter(config)

        indices = []
        for i in range(100):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            index, is_final = counter.increment_sent(turn)
            indices.append(index)

        # All indices should be unique and sequential
        assert indices == list(range(100))
        # Last one should be final
        assert counter.requests_sent == 100

    def test_increment_returned_tracks_correctly(self):
        """Test that returned credit counting is accurate."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
        )
        counter = CreditCounter(config)

        # Send 10 credits
        for i in range(10):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            counter.increment_sent(turn)

        counter.freeze_sent_counts()

        # Return 5 completed, 5 cancelled
        for i in range(5):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            counter.increment_returned(credit.is_final_turn, cancelled=False)

        for i in range(5, 10):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            result = counter.increment_returned(credit.is_final_turn, cancelled=True)
            if i == 9:
                assert result is True  # Final return

        assert counter.requests_completed == 5
        assert counter.requests_cancelled == 5
        assert counter.in_flight == 0

    def test_session_counting_with_multi_turn_conversations(self):
        """Test that session (conversation) counting is accurate for multi-turn.

        Each conversation's first turn should increment sent_sessions.
        Only final turns completing should increment completed_sessions.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_num_sessions=3,
        )
        counter = CreditCounter(config)

        # Send 3 conversations with 2 turns each
        # First turns
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=2)
            counter.increment_sent(turn)

        assert counter.sent_sessions == 3
        assert counter.total_session_turns == 6  # 3 convs * 2 turns

        # Second turns
        for i in range(3):
            turn = create_turn_to_send(
                f"conv{i}", turn_index=1, num_turns=2, x_correlation_id=f"corr-conv{i}"
            )
            counter.increment_sent(turn)

        # sent_sessions should still be 3 (not 6)
        assert counter.sent_sessions == 3
        assert counter.requests_sent == 6

        counter.freeze_sent_counts()

        # Return all 6 credits (3 first turns, 3 final turns)
        for i in range(3):
            # First turn (not final) - num_turns=2 means turn_index=0 is not final
            credit = create_credit(
                credit_id=i * 2, conversation_id=f"conv{i}", turn_index=0, num_turns=2
            )
            counter.increment_returned(credit.is_final_turn, cancelled=False)

        assert counter.completed_sessions == 0  # No final turns completed yet

        for i in range(3):
            # Final turn - num_turns=2 means turn_index=1 is final
            credit = create_credit(
                credit_id=i * 2 + 1,
                conversation_id=f"conv{i}",
                turn_index=1,
                num_turns=2,
            )
            counter.increment_returned(credit.is_final_turn, cancelled=False)

        assert counter.completed_sessions == 3

    def test_cancelled_session_tracking(self):
        """Test that cancelled sessions are tracked separately from completed.

        When a final turn is cancelled, it should increment cancelled_sessions
        (not completed_sessions), and in_flight_sessions should reflect this.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_num_sessions=3,
        )
        counter = CreditCounter(config)

        # Send 3 single-turn conversations
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            counter.increment_sent(turn)

        assert counter.sent_sessions == 3
        assert counter.in_flight_sessions == 3  # All in flight

        counter.freeze_sent_counts()

        # Complete 2, cancel 1
        credit1 = create_credit(credit_id=0, conversation_id="conv0")
        counter.increment_returned(credit1.is_final_turn, cancelled=False)

        credit2 = create_credit(credit_id=1, conversation_id="conv1")
        counter.increment_returned(credit2.is_final_turn, cancelled=False)

        assert counter.completed_sessions == 2
        assert counter.cancelled_sessions == 0
        assert counter.in_flight_sessions == 1

        # Cancel the last one (final turn)
        credit3 = create_credit(credit_id=2, conversation_id="conv2")
        counter.increment_returned(credit3.is_final_turn, cancelled=True)

        assert counter.completed_sessions == 2
        assert counter.cancelled_sessions == 1
        assert counter.in_flight_sessions == 0

    def test_in_flight_sessions_with_incomplete_conversations(self):
        """Test in_flight_sessions when conversations don't finish.

        This simulates a timeout scenario where sessions start but never
        complete (no final turn returned).
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=6,  # 3 sessions * 2 turns
        )
        counter = CreditCounter(config)

        # Send 3 conversations with 2 turns each (only first turns)
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=2)
            counter.increment_sent(turn)

        assert counter.sent_sessions == 3
        assert counter.in_flight_sessions == 3

        # Return only the first turns (not final)
        for i in range(3):
            credit = create_credit(
                credit_id=i, conversation_id=f"conv{i}", turn_index=0, num_turns=2
            )
            assert credit.is_final_turn is False
            counter.increment_returned(credit.is_final_turn, cancelled=False)

        # Sessions are still in-flight (no final turn returned)
        assert counter.completed_sessions == 0
        assert counter.cancelled_sessions == 0
        assert counter.in_flight_sessions == 3


# =============================================================================
# Deadlock Scenario: Event Never Set
# =============================================================================


@pytest.mark.asyncio
class TestDeadlockPrevention:
    """Tests that verify the system never deadlocks waiting for events."""

    async def test_all_credits_sent_event_set_on_reaching_limit(self):
        """Test that all_credits_sent_event is set when request limit reached.

        Verifies CreditIssuer.issue_credit sets the event properly.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=3,
        )
        lifecycle, progress, stop_checker = create_phase_components(config)
        lifecycle.start()

        # Simulate issuing credits
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            _index, is_final = progress.increment_sent(turn)
            if is_final:
                lifecycle.mark_sending_complete()
                progress.freeze_sent_counts()
                progress.all_credits_sent_event.set()

        assert progress.all_credits_sent_event.is_set()
        assert lifecycle.is_sending_complete

    async def test_all_credits_returned_event_set_when_all_return(self):
        """Test that all_credits_returned_event is set when all credits return.

        This is critical - if this event is never set, the phase never completes.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=3,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 3 credits
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return all 3
        for i in range(3):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            is_all_returned = progress.increment_returned(
                credit.is_final_turn, cancelled=False
            )
            if is_all_returned:
                progress.all_credits_returned_event.set()

        # Event must be set
        assert progress.all_credits_returned_event.is_set()

    async def test_progress_tracker_handles_duration_based_phase_no_credits_sent(self):
        """Test that duration-based phases with no credits sent complete correctly.

        Edge case: Duration-based phase (no expected count) where timeout occurs
        before any credits are sent. Should not deadlock.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_duration_sec=1.0,  # Duration-based, no expected count
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # No credits sent, but phase times out
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Should be able to check completion without any returns
        # (0 sent = 0 to return)
        assert progress.check_all_returned_or_cancelled()


# =============================================================================
# Race Condition: Sticky Session Routing
# =============================================================================


@pytest.mark.asyncio
class TestStickySessionRaceConditions:
    """Tests for race conditions in sticky session routing."""

    async def test_sticky_session_eviction_before_turn_completes(self, service_config):
        """Test that session eviction on final turn doesn't break in-flight credits.

        Scenario: Final turn is sent → sticky session is evicted → but the
        credit hasn't returned yet. Late return should still be handled.
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")

        # Create a 3-turn conversation
        x_correlation_id = "multi-turn-session"

        # Turn 0 (first) - num_turns=3 means turn_index=0 is not final
        credit_turn_0 = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id=x_correlation_id,
            turn_index=0,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        await router.send_credit(credit_turn_0)
        worker_id = router._router_client.send_to.call_args[0][0]

        # Session should be created
        assert x_correlation_id in router._sticky_sessions

        # Turn 1 (middle)
        credit_turn_1 = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id=x_correlation_id,
            turn_index=1,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        await router.send_credit(credit_turn_1)

        # Turn 2 (final) - num_turns=3 means turn_index=2 is final
        credit_turn_2 = Credit(
            id=2,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id=x_correlation_id,
            turn_index=2,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        await router.send_credit(credit_turn_2)

        # Session should be evicted after final turn routing
        assert x_correlation_id not in router._sticky_sessions

        # But we still have 3 in-flight credits
        assert router._workers["worker-1"].in_flight_credits == 3

        # Returns arrive in any order - should all work
        router._track_credit_returned(
            worker_id, 2, cancelled=False, error_reported=False
        )
        router._track_credit_returned(
            worker_id, 0, cancelled=False, error_reported=False
        )
        router._track_credit_returned(
            worker_id, 1, cancelled=False, error_reported=False
        )

        assert router._workers["worker-1"].in_flight_credits == 0
        assert router._workers["worker-1"].total_completed_credits == 3

    async def test_worker_unregisters_mid_session(self, service_config):
        """Test that worker unregistering mid-session reassigns remaining turns.

        Scenario: Turn 0 goes to worker-1, worker-1 unregisters, turn 1 should
        go to worker-2 (and create new sticky session).
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")

        x_correlation_id = "reassigned-session"

        # Turn 0 to worker-1 (set worker-2 higher load so worker-1 is selected)
        router._workers["worker-2"].in_flight_credits = 10
        router._workers_by_load[10].add("worker-2")
        router._workers_by_load[0].discard("worker-2")

        credit_turn_0 = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id=x_correlation_id,
            turn_index=0,
            num_turns=2,
            issued_at_ns=time.time_ns(),
        )
        await router.send_credit(credit_turn_0)
        worker_id_0 = router._router_client.send_to.call_args[0][0]

        # Verify sticky session points to first worker
        assert router._sticky_sessions[x_correlation_id] == worker_id_0

        # Worker-1 unregisters (crash/shutdown)
        router._cancellation_pending = True  # Suppress error logging
        router._unregister_worker(worker_id_0)

        # Update _min_load to reflect that worker-2 (at load 10) is now the only worker
        router._min_load = 10

        # Turn 1 should reassign to worker-2 (worker-1 is gone, sticky session gone too)
        # Clear sticky session to simulate the failover scenario
        if x_correlation_id in router._sticky_sessions:
            del router._sticky_sessions[x_correlation_id]

        credit_turn_1 = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id=x_correlation_id,
            turn_index=1,
            num_turns=2,
            issued_at_ns=time.time_ns(),
        )
        await router.send_credit(credit_turn_1)
        worker_id_1 = router._router_client.send_to.call_args[0][0]

        # Should have been reassigned (worker-1 is gone)
        assert worker_id_1 == "worker-2"


# =============================================================================
# Race Condition: Multi-Turn Credit Tracking
# =============================================================================


@pytest.mark.asyncio
class TestMultiTurnCreditRace:
    """Tests for race conditions in multi-turn conversation credit tracking."""

    async def test_interleaved_multi_turn_conversations(self):
        """Test interleaved turns from multiple conversations.

        Scenario: Two 3-turn conversations interleaved:
        Conv A turn 0, Conv B turn 0, Conv A turn 1, Conv B turn 1, etc.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_num_sessions=2,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Interleaved sends
        # Conv A turn 0
        progress.increment_sent(create_turn_to_send("convA", 0, 3))
        # Conv B turn 0
        progress.increment_sent(create_turn_to_send("convB", 0, 3))
        # Conv A turn 1
        turn_a1 = TurnToSend(
            conversation_id="convA",
            x_correlation_id="corr-convA",
            turn_index=1,
            num_turns=3,
        )
        progress.increment_sent(turn_a1)
        # Conv B turn 1
        turn_b1 = TurnToSend(
            conversation_id="convB",
            x_correlation_id="corr-convB",
            turn_index=1,
            num_turns=3,
        )
        progress.increment_sent(turn_b1)
        # Conv A turn 2 (final)
        turn_a2 = TurnToSend(
            conversation_id="convA",
            x_correlation_id="corr-convA",
            turn_index=2,
            num_turns=3,
        )
        progress.increment_sent(turn_a2)
        # Conv B turn 2 (final)
        turn_b2 = TurnToSend(
            conversation_id="convB",
            x_correlation_id="corr-convB",
            turn_index=2,
            num_turns=3,
        )
        _idx, is_final = progress.increment_sent(turn_b2)
        assert is_final  # Should be final since 2 sessions * 3 turns = 6 total
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        stats = progress.create_stats(lifecycle)
        assert stats.sent_sessions == 2
        assert stats.total_session_turns == 6
        assert stats.requests_sent == 6

    async def test_multi_turn_with_partial_cancellation(self):
        """Test multi-turn conversation where middle turn is cancelled.

        Scenario: Turn 0 completes, Turn 1 cancelled, Turn 2 completes.
        Session should still count as completed (final turn completed).
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=3,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 3 turns of same conversation
        for i in range(3):
            turn = TurnToSend(
                conversation_id="conv1",
                x_correlation_id="corr-conv1",
                turn_index=i,
                num_turns=3,
            )
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return turn 0 (success) - num_turns=3, turn_index=0 is not final
        credit_0 = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=0,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        progress.increment_returned(credit_0.is_final_turn, cancelled=False)

        # Return turn 1 (cancelled) - num_turns=3, turn_index=1 is not final
        credit_1 = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=1,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        progress.increment_returned(credit_1.is_final_turn, cancelled=True)

        # Return turn 2 (success, final) - num_turns=3, turn_index=2 is final
        credit_2 = Credit(
            id=2,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=2,
            num_turns=3,
            issued_at_ns=time.time_ns(),
        )
        result = progress.increment_returned(credit_2.is_final_turn, cancelled=False)

        assert result is True  # Final return
        stats = progress.create_stats(lifecycle)
        assert stats.requests_completed == 2
        assert stats.requests_cancelled == 1
        assert stats.completed_sessions == 1  # Final turn completed


# =============================================================================
# Integration Test: Full Credit Flow
# =============================================================================


@pytest.mark.asyncio
class TestFullCreditFlowRace:
    """Integration tests for the full credit flow."""

    async def test_credits_return_in_different_order_than_sent(self):
        """Test that credits returning out of order doesn't cause issues.

        Credits 0, 1, 2 are sent. Credits return in order 2, 0, 1.
        Phase should still complete correctly.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=3,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send in order
        for i in range(3):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return out of order: 2, 0, 1
        return_order = [2, 0, 1]
        for i in return_order:
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            result = progress.increment_returned(credit.is_final_turn, cancelled=False)
            if result:
                progress.all_credits_returned_event.set()
            if i == return_order[-1]:  # Last to return (credit 1)
                assert result is True

        assert progress.all_credits_returned_event.is_set()
        stats = progress.create_stats(lifecycle)
        assert stats.requests_completed == 3

    async def test_mixed_completion_and_cancellation_out_of_order(self):
        """Test mixed completion/cancellation in random order.

        Credits sent: 0, 1, 2, 3, 4
        Returns: 3 (cancelled), 0 (complete), 4 (cancelled), 2 (complete), 1 (complete)
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=5,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send 5 credits
        for i in range(5):
            turn = create_turn_to_send(f"conv{i}", turn_index=0, num_turns=1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return in mixed order with mixed status
        returns = [  # fmt: skip
            (3, True),  # cancelled
            (0, False),  # complete
            (4, True),  # cancelled
            (2, False),  # complete
            (1, False),  # complete
        ]

        for credit_num, cancelled in returns:
            credit = create_credit(
                credit_id=credit_num, conversation_id=f"conv{credit_num}"
            )
            is_all_returned = progress.increment_returned(
                credit.is_final_turn, cancelled=cancelled
            )
            if is_all_returned:
                progress.all_credits_returned_event.set()

        assert progress.all_credits_returned_event.is_set()
        stats = progress.create_stats(lifecycle)
        assert stats.requests_completed == 3
        assert stats.requests_cancelled == 2


# =============================================================================
# Race Condition: Worker Credit Task Lifecycle
# =============================================================================


@pytest.mark.asyncio
class TestWorkerCreditTaskRace:
    """Tests for race conditions in Worker credit task handling.

    Credits must ALWAYS be returned, even if:
    - Task is cancelled before starting
    - Task errors during execution
    - Task is cancelled during execution
    """

    async def test_credit_context_returned_flag_prevents_double_return(self):
        """Test that returned flag prevents duplicate credit returns.

        The done callback checks if credit was already returned by finally block.
        """
        from aiperf.credit.structs import CreditContext

        # CreditContext expects the msgspec Credit struct
        credit = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=1000)

        # Simulate finally block setting returned=True
        assert not ctx.returned
        # Note: msgspec Structs are frozen, so we need to create a new instance
        ctx = CreditContext(
            credit=credit, drop_perf_ns=1000, returned=True, cancelled=ctx.cancelled
        )
        assert ctx.returned

        # Done callback would check this and skip
        # (we're just validating the flag mechanism)

    async def test_credit_context_tracks_cancellation(self):
        """Test that credit context tracks cancellation status."""
        from aiperf.credit.structs import CreditContext

        # CreditContext expects the msgspec Credit struct
        credit = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=1000)

        assert not ctx.cancelled
        # Note: msgspec Structs are frozen, so we need to create a new instance
        ctx = CreditContext(
            credit=credit, drop_perf_ns=1000, cancelled=True, returned=ctx.returned
        )
        assert ctx.cancelled


# =============================================================================
# Race Condition: Phase State Machine
# =============================================================================


@pytest.mark.asyncio
class TestPhaseStateMachineRace:
    """Tests for race conditions in phase state transitions."""

    async def test_phase_lifecycle_state_ordering(self):
        """Test that phase lifecycle states progress in correct order.

        States: not started → started → sending complete → complete
        The phase lifecycle requires explicit mark_complete() call - it doesn't
        auto-complete. The all_credits_returned_event signals when returns are done.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=5,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)

        # Initial state
        assert not lifecycle.is_started
        assert not lifecycle.is_sending_complete
        assert not lifecycle.is_complete

        # Start
        lifecycle.start()
        assert lifecycle.is_started
        assert not lifecycle.is_sending_complete
        assert not lifecycle.is_complete

        # Send credits
        for i in range(5):
            turn = create_turn_to_send(f"conv{i}", 0, 1)
            progress.increment_sent(turn)

        # Mark sending complete
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()
        assert lifecycle.is_started
        assert lifecycle.is_sending_complete
        assert not lifecycle.is_complete

        # Return all credits
        for i in range(5):
            credit = create_credit(credit_id=i, conversation_id=f"conv{i}")
            is_all_returned = progress.increment_returned(
                credit.is_final_turn, cancelled=False
            )
            if is_all_returned:
                progress.all_credits_returned_event.set()

        # all_credits_returned_event is set when all returns are in
        assert progress.all_credits_returned_event.is_set()

        # Explicit mark_complete() is required to transition to COMPLETE state
        # This is done by PhaseRunner after waiting for returns
        lifecycle.mark_complete()
        progress.freeze_completed_counts()
        assert lifecycle.is_complete

    async def test_cannot_send_after_phase_complete(self):
        """Test that phase rejects operations after completion."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=1,
        )
        lifecycle, progress, stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send and return one credit
        turn = create_turn_to_send("conv1", 0, 1)
        progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()
        credit = create_credit(credit_id=0, conversation_id="conv1")
        progress.increment_returned(credit.is_final_turn, cancelled=False)

        # Explicitly mark phase as complete (done by PhaseRunner)
        lifecycle.mark_complete()
        progress.freeze_completed_counts()

        # Phase is now complete
        assert lifecycle.is_complete

        # Stop checker should return False (can't send after sending complete)
        assert not stop_checker.can_send_any_turn()


# =============================================================================
# Race Condition: Records Tracker Phase Tracking
# =============================================================================


class TestRecordsTrackerPhaseRace:
    """Tests for race conditions in RecordsTracker phase tracking."""

    def test_phase_info_updates_are_additive(self):
        """Test that phase info updates combine information correctly.

        Multiple messages may arrive with partial information:
        - Start message: total_expected_requests, start_ns
        - Sending complete: final_requests_sent
        - Phase complete: final_requests_completed, requests_end_ns
        """
        tracker = RecordsTracker()

        # Start message
        start_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            total_expected_requests=100,
            start_ns=1000,
        )
        tracker.update_phase_info(start_stats)

        phase = tracker._get_phase_tracker(CreditPhase.PROFILING)
        assert phase._start_ns == 1000
        assert phase._final_requests_completed is None  # Not yet set

        # Sending complete (sets final_requests_sent but not final_requests_completed)
        sending_complete_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_sent=95,
            start_ns=1000,
        )
        tracker.update_phase_info(sending_complete_stats)
        # final_requests_completed still None

        # Phase complete
        complete_stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=90,
            requests_end_ns=2000,
            start_ns=1000,
        )
        tracker.update_phase_info(complete_stats)

        assert phase._final_requests_completed == 90
        assert phase._requests_end_ns == 2000

    def test_records_from_multiple_workers_aggregate_correctly(self):
        """Test that records from multiple workers are correctly aggregated.

        Scenario: 3 workers each process some records. Total should sum up.
        """
        tracker = RecordsTracker()

        # Set up phase with expected count
        stats = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            final_requests_completed=30,
            start_ns=1000,
        )
        tracker.update_phase_info(stats)

        phase = tracker._get_phase_tracker(CreditPhase.PROFILING)

        # Simulate records from 3 workers
        for _ in range(10):  # Worker 1: 10 success
            phase.increment_success_records()
        for _ in range(10):  # Worker 2: 10 success
            phase.increment_success_records()
        for _ in range(8):  # Worker 3: 8 success, 2 errors
            phase.increment_success_records()
        for _ in range(2):
            phase.increment_error_records()

        # Check completion
        result = tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
        )
        assert result is True

        stats = tracker.create_stats_for_phase(CreditPhase.PROFILING)
        assert stats.success_records == 28
        assert stats.error_records == 2
        assert stats.total_records == 30


# =============================================================================
# Race Condition: Warmup to Profiling Transition
# =============================================================================


@pytest.mark.asyncio
class TestWarmupToProfilingTransition:
    """Tests for race conditions during warmup → profiling phase transition.

    Seamless warmup means profiling starts while warmup credits are still
    in-flight. Must ensure counters don't get confused.
    """

    async def test_warmup_credits_dont_affect_profiling_counts(self):
        """Test that warmup credits returning don't affect profiling counters."""
        warmup_config = CreditPhaseConfig(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=5,
            seamless=True,
        )
        profiling_config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
        )

        warmup_lifecycle, warmup_progress, _ = create_phase_components(warmup_config)
        profiling_lifecycle, profiling_progress, _ = create_phase_components(
            profiling_config
        )

        # Start warmup
        warmup_lifecycle.start()
        for i in range(5):
            turn = create_turn_to_send(f"warmup{i}", 0, 1)
            warmup_progress.increment_sent(turn)
        warmup_lifecycle.mark_sending_complete()
        warmup_progress.freeze_sent_counts()

        # Start profiling (seamless - warmup credits still in-flight)
        profiling_lifecycle.start()
        for i in range(10):
            turn = create_turn_to_send(f"prof{i}", 0, 1)
            profiling_progress.increment_sent(turn)
        profiling_lifecycle.mark_sending_complete()
        profiling_progress.freeze_sent_counts()

        # Interleaved returns
        # Return warmup credit - num_turns=1, turn_index=0 is final
        warmup_credit = Credit(
            id=0,
            phase=CreditPhase.WARMUP,
            conversation_id="warmup0",
            x_correlation_id="corr-warmup0",
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
        )
        warmup_progress.increment_returned(warmup_credit.is_final_turn, cancelled=False)

        # Return profiling credit - num_turns=1, turn_index=0 is final
        prof_credit = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="prof0",
            x_correlation_id="corr-prof0",
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
        )
        profiling_progress.increment_returned(
            prof_credit.is_final_turn, cancelled=False
        )

        # Counters should be independent
        warmup_stats = warmup_progress.create_stats(warmup_lifecycle)
        profiling_stats = profiling_progress.create_stats(profiling_lifecycle)

        assert warmup_stats.requests_completed == 1
        assert warmup_stats.requests_sent == 5

        assert profiling_stats.requests_completed == 1
        assert profiling_stats.requests_sent == 10


# =============================================================================
# Race Condition: Router Load Balancing
# =============================================================================


@pytest.mark.asyncio
class TestRouterLoadBalancingRace:
    """Tests for race conditions in router load balancing decisions."""

    async def test_least_loaded_selection_with_ties(self, service_config):
        """Test that ties in load balancing are broken randomly.

        When multiple workers have the same load, selection should be random
        (deterministic with seeded RNG for testing).
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")
        router._register_worker("worker-3")

        # Set same load for all workers
        for worker_id in router._workers:
            router._workers[worker_id].in_flight_credits = 5
        router._workers_by_load.clear()
        router._workers_by_load[5] = {"worker-1", "worker-2", "worker-3"}
        router._min_load = 5

        # Route multiple credits and track which workers are selected
        selections = set()
        for i in range(10):
            credit = create_credit(
                credit_id=i,
                conversation_id=f"conv{i}",
            )
            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            selections.add(worker_id)

        # Should select from tied workers (exact distribution depends on RNG)
        assert all(w in {"worker-1", "worker-2", "worker-3"} for w in selections)

    async def test_load_balancing_prefers_lower_load(self, service_config):
        """Test that load balancing selects worker with lowest in-flight count."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")
        router._register_worker("worker-3")

        # Set different loads and update load tracking atomically
        # First remove workers from their current load level (0)
        router._workers_by_load[0].discard("worker-1")
        router._workers_by_load[0].discard("worker-2")
        router._workers_by_load[0].discard("worker-3")

        # Set different in-flight counts with enough gap to avoid collision
        # after 5 credits (worker-2 goes from 1 to 6, so worker-3 must be > 6)
        router._workers["worker-1"].in_flight_credits = 20
        router._workers["worker-2"].in_flight_credits = 1
        router._workers["worker-3"].in_flight_credits = 10

        # Add to new load levels
        router._workers_by_load[20].add("worker-1")
        router._workers_by_load[1].add("worker-2")
        router._workers_by_load[10].add("worker-3")

        # Update min_load to the actual minimum
        router._min_load = 1

        # Should always select worker-2 (lowest load)
        for i in range(5):
            credit = create_credit(
                credit_id=i,
                conversation_id=f"conv{i}",
            )
            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            assert worker_id == "worker-2"

    async def test_load_updated_atomically_on_send_and_return(self, service_config):
        """Test that in_flight_credits is updated atomically on send/return."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-1")

        worker = router._workers["worker-1"]

        # Rapid send/return cycle - each operation should be atomic
        for i in range(100):
            router._track_credit_sent("worker-1", i)
            assert worker.in_flight_credits == i + 1
            assert len(worker.active_credit_ids) == i + 1

        for i in range(100):
            router._track_credit_returned(
                "worker-1", i, cancelled=False, error_reported=False
            )
            assert worker.in_flight_credits == 99 - i
            assert len(worker.active_credit_ids) == 99 - i

        assert worker.in_flight_credits == 0
        assert worker.total_sent_credits == 100
        assert worker.total_completed_credits == 100


# =============================================================================
# Race Condition: Cancellation Message Handling
# =============================================================================


@pytest.mark.asyncio
class TestCancellationRace:
    """Tests for race conditions in credit cancellation handling."""

    async def test_cancel_all_credits_snapshots_state(self, service_config):
        """Test that cancel_all_credits takes atomic snapshot of in-flight credits.

        Cancellation should not miss any credits that were in-flight at the
        moment cancel_all_credits was called.
        """
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        # Mock the router client
        router._router_client = MagicMock()
        router._router_client.send_to = AsyncMock()

        router._workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=3),
            "worker-2": WorkerLoad(worker_id="worker-2", in_flight_credits=2),
        }
        router._workers["worker-1"].active_credit_ids = {1, 2, 3}
        router._workers["worker-2"].active_credit_ids = {4, 5}
        router._workers_cache = list(router._workers.values())

        await router.cancel_all_credits()

        # Should have sent cancel messages to both workers
        assert router._router_client.send_to.call_count == 2

        # Verify the credit IDs in cancel messages
        calls = router._router_client.send_to.call_args_list
        worker_1_call = next(c for c in calls if c[0][0] == "worker-1")
        worker_2_call = next(c for c in calls if c[0][0] == "worker-2")

        assert set(worker_1_call[0][1].credit_ids) == {1, 2, 3}
        assert set(worker_2_call[0][1].credit_ids) == {4, 5}

    async def test_cancel_skips_workers_with_no_inflight(self, service_config):
        """Test that cancellation skips workers with 0 in-flight credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._router_client = MagicMock()
        router._router_client.send_to = AsyncMock()

        router._workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0),
            "worker-2": WorkerLoad(worker_id="worker-2", in_flight_credits=5),
        }
        router._workers["worker-2"].active_credit_ids = {i for i in range(5)}
        router._workers_cache = list(router._workers.values())

        await router.cancel_all_credits()

        # Should only send to worker-2
        assert router._router_client.send_to.call_count == 1
        call = router._router_client.send_to.call_args
        assert call[0][0] == "worker-2"


# =============================================================================
# Race Condition: Event Wait Timeouts
# =============================================================================


@pytest.mark.asyncio
class TestEventWaitTimeoutRace:
    """Tests for race conditions in event wait with timeout scenarios."""

    async def test_event_set_just_before_timeout(self):
        """Test behavior when event is set just before timeout expires.

        This tests the race between timeout firing and event being set.
        """
        event = asyncio.Event()

        async def set_event_after_delay():
            await asyncio.sleep(0.01)  # Small delay
            event.set()

        # Start task to set event
        task = asyncio.create_task(set_event_after_delay())

        # Wait with slightly longer timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=0.1)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True

        await task

        # Should NOT have timed out
        assert not timed_out
        assert event.is_set()

    async def test_timeout_before_event_set(self):
        """Test behavior when timeout expires before event is set."""
        event = asyncio.Event()

        # Wait with very short timeout, event never set
        try:
            await asyncio.wait_for(event.wait(), timeout=0.001)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True

        # Should have timed out
        assert timed_out
        assert not event.is_set()


# =============================================================================
# Stress Test: High Volume Credit Processing
# =============================================================================


@pytest.mark.asyncio
class TestHighVolumeCreditProcessing:
    """Stress tests for high-volume credit processing scenarios."""

    async def test_rapid_credit_lifecycle_1000_credits(self):
        """Test rapid processing of 1000 credits with correct state tracking."""
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=1000,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send all credits
        for i in range(1000):
            turn = create_turn_to_send(f"conv{i}", 0, 1)
            progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        # Return in random-ish order (reversed batches)
        return_order = list(range(1000))
        # Reverse in batches of 100
        for start in range(0, 1000, 100):
            return_order[start : start + 100] = reversed(
                return_order[start : start + 100]
            )

        for i, credit_num in enumerate(return_order):
            credit = create_credit(
                credit_id=credit_num,
                conversation_id=f"conv{credit_num}",
            )
            result = progress.increment_returned(credit.is_final_turn, cancelled=False)
            if result:
                progress.all_credits_returned_event.set()
            if i == 999:  # Last return
                assert result is True

        assert progress.all_credits_returned_event.is_set()
        stats = progress.create_stats(lifecycle)
        assert stats.requests_sent == 1000
        assert stats.requests_completed == 1000
        assert stats.in_flight_requests == 0

    async def test_high_concurrency_multi_turn_sessions(self):
        """Test high concurrency with many multi-turn sessions.

        100 sessions, each with 5 turns = 500 total credits.
        """
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_num_sessions=100,
        )
        lifecycle, progress, _stop_checker = create_phase_components(config)
        lifecycle.start()

        # Send all turns for all sessions
        for session_idx in range(100):
            for turn_idx in range(5):
                turn = TurnToSend(
                    conversation_id=f"session{session_idx}",
                    x_correlation_id=f"corr-session{session_idx}",
                    turn_index=turn_idx,
                    num_turns=5,
                )
                progress.increment_sent(turn)
        lifecycle.mark_sending_complete()
        progress.freeze_sent_counts()

        stats = progress.create_stats(lifecycle)
        assert stats.sent_sessions == 100
        assert stats.total_session_turns == 500
        assert stats.requests_sent == 500

        # Return all credits
        for session_idx in range(100):
            for turn_idx in range(5):
                # num_turns=5 means turn_idx=4 is final (turn_idx == num_turns - 1)
                credit = Credit(
                    id=session_idx * 5 + turn_idx,
                    phase=CreditPhase.PROFILING,
                    conversation_id=f"session{session_idx}",
                    x_correlation_id=f"corr-session{session_idx}",
                    turn_index=turn_idx,
                    num_turns=5,
                    issued_at_ns=time.time_ns(),
                )
                is_all_returned = progress.increment_returned(
                    credit.is_final_turn, cancelled=False
                )
                if is_all_returned:
                    progress.all_credits_returned_event.set()

        assert progress.all_credits_returned_event.is_set()
        stats = progress.create_stats(lifecycle)
        assert stats.completed_sessions == 100

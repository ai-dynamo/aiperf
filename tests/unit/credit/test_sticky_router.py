# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import FirstToken
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit
from tests.unit.timing.conftest import make_credit


class TestStickyCreditRouterFairLoadBalancing:
    """Test fair load balancing for first turns."""

    async def test_routes_to_least_loaded_worker(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")
        router._register_worker("worker-3")

        router._workers["worker-1"].in_flight_credits = 5
        router._workers["worker-2"].in_flight_credits = 2
        router._workers["worker-3"].in_flight_credits = 8

        router._workers_by_load.clear()
        router._workers_by_load[5].add("worker-1")
        router._workers_by_load[2].add("worker-2")
        router._workers_by_load[8].add("worker-3")
        router._min_load = 2

        credit = make_credit(
            id=1,
            conv_id="session-1",
            turn=0,
            corr_id="test-corr-id-1",
            num_turns=3,
        )

        await router.send_credit(credit)

        router._router_client.send_to.assert_called_once()
        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-2"
        assert len(router._sticky_sessions) == 1
        assert list(router._sticky_sessions.values())[0] == "worker-2"

    async def test_creates_conversation_assignment(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")

        credit = make_credit(id=1, corr_id="test-corr-id", turn=0, num_turns=3)

        await router.send_credit(credit)

        assert len(router._sticky_sessions) == 1
        assert router._sticky_sessions["test-corr-id"] == "worker-A"

    async def test_error_if_no_workers_available(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        credit = make_credit()

        with pytest.raises(RuntimeError, match="No workers available"):
            await router.send_credit(credit)


class TestStickyCreditRouterStickyRouting:
    """Test sticky routing for subsequent turns."""

    async def test_routes_to_assigned_worker(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")
        router._register_worker("worker-B")

        instance_id = "test-instance-123"
        router._sticky_sessions[instance_id] = "worker-A"

        credit = make_credit(
            id=2,
            conv_id="session-123",
            turn=1,
            corr_id=instance_id,
            num_turns=5,
        )

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-A"
        assert router._sticky_sessions[instance_id] == "worker-A"

    async def test_cleans_up_assignment_on_final_turn(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")

        instance_id = "test-instance-456"
        router._sticky_sessions[instance_id] = "worker-A"

        credit = make_credit(
            id=5,
            conv_id="session-456",
            turn=4,
            corr_id=instance_id,
            num_turns=5,
        )

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-A"
        assert instance_id not in router._sticky_sessions

    async def test_fallback_to_fair_load_if_assignment_missing(
        self, service_config
    ) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")
        router._register_worker("worker-B")

        router._workers["worker-A"].in_flight_credits = 5
        router._workers["worker-B"].in_flight_credits = 2
        router._workers_by_load.clear()
        router._workers_by_load[5].add("worker-A")
        router._workers_by_load[2].add("worker-B")
        router._min_load = 2

        credit = make_credit(
            id=10,
            conv_id="session-999",
            turn=1,
            corr_id="test-corr-id",
            num_turns=3,
        )

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-B"


class TestStickyCreditRouterLoadTracking:
    """Test worker load tracking."""

    async def test_track_credit_sent(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-1")

        assert router._workers["worker-1"].in_flight_credits == 0

        router._track_credit_sent("worker-1", 1)
        assert router._workers["worker-1"].in_flight_credits == 1
        assert router._workers["worker-1"].total_sent_credits == 1

        router._track_credit_sent("worker-1", 2)
        assert router._workers["worker-1"].in_flight_credits == 2
        assert router._workers["worker-1"].total_sent_credits == 2

    async def test_track_credit_returned(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-1")

        router._workers["worker-1"].in_flight_credits = 5
        router._workers["worker-1"].active_credit_ids.add(1)
        router._workers["worker-1"].active_credit_ids.add(2)
        router._workers_by_load[0].discard("worker-1")
        router._workers_by_load[5].add("worker-1")
        router._min_load = 5

        router._track_credit_returned(
            "worker-1", 1, cancelled=False, error_reported=False
        )
        assert router._workers["worker-1"].in_flight_credits == 4
        assert router._workers["worker-1"].total_completed_credits == 1

        router._track_credit_returned(
            "worker-1", 2, cancelled=False, error_reported=False
        )
        assert router._workers["worker-1"].in_flight_credits == 3
        assert router._workers["worker-1"].total_completed_credits == 2

    async def test_register_worker(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-A")

        assert "worker-A" in router._workers
        assert router._workers["worker-A"].in_flight_credits == 0
        assert router._workers["worker-A"].total_completed_credits == 0

    async def test_unregister_worker(self, service_config) -> None:
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-A")
        router._unregister_worker("worker-A")

        assert "worker-A" not in router._workers


class TestStickyCreditRouterCompleteScenario:
    """Test complete routing scenario with multiple conversations."""

    async def test_five_turn_conversation(self, service_config) -> None:
        """Test routing a complete 5-turn conversation."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-A")
        router._register_worker("worker-B")
        router._register_worker("worker-C")

        instance_id = "test-corr-id"
        num_turns = 5

        # Turn 1 (first turn, fair load)
        credit1 = make_credit(
            id=1,
            conv_id="session-test",
            turn=0,
            corr_id=instance_id,
            num_turns=num_turns,
        )

        await router.send_credit(credit1)
        worker1 = router._router_client.send_to.call_args[0][0]
        assert worker1 in ["worker-A", "worker-B", "worker-C"]

        # Turns 2-5 (sticky)
        for turn_idx in range(1, 5):
            credit = make_credit(
                id=turn_idx + 1,
                conv_id="session-test",
                turn=turn_idx,
                corr_id=instance_id,
                num_turns=num_turns,
            )
            await router.send_credit(credit)
            worker = router._router_client.send_to.call_args[0][0]
            assert worker == worker1

        # Assignment should be cleaned up after final turn
        assert instance_id not in router._sticky_sessions

    async def test_multiple_conversations_balanced(self, service_config) -> None:
        """Test that multiple conversations are balanced across workers."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        for i in range(3):
            router._register_worker(f"worker-{i}")

        # Route first turns of 9 conversations (multi-turn to create sticky sessions)
        instance_ids = []
        for i in range(9):
            instance_id = f"instance-{i}"
            credit = make_credit(
                id=i,
                conv_id=f"session-{i}",
                turn=0,
                corr_id=instance_id,
                num_turns=3,  # Multi-turn so it creates sticky sessions
            )

            await router.send_credit(credit)
            instance_ids.append(instance_id)

        # Each worker should get 3 conversations (balanced)
        assert all(w.in_flight_credits == 3 for w in router._workers.values())

        # Route second turns (should be sticky)
        for i, instance_id in enumerate(instance_ids):
            expected_worker = router._sticky_sessions[instance_id]
            credit = make_credit(
                id=100 + i,
                conv_id="session-test",
                turn=1,
                corr_id=instance_id,
                num_turns=3,
            )

            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            assert worker_id == expected_worker


class TestStickyCreditRouterEdgeCases:
    """Test edge cases and error handling."""

    async def test_single_worker(self, service_config) -> None:
        """Test with only one worker."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("only-worker")

        for i in range(10):
            credit = make_credit(
                id=i,
                conv_id=f"session-{i}",
                turn=0,
                corr_id=f"test-corr-id-{i}",
                num_turns=1,  # Single-turn (final) conversations
            )

            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            assert worker_id == "only-worker"

    async def test_unequal_worker_loads(self, service_config) -> None:
        """Test fair load balancing with significantly unequal loads."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-overloaded")
        router._register_worker("worker-idle")

        router._workers["worker-overloaded"].in_flight_credits = 100
        router._workers["worker-idle"].in_flight_credits = 0

        router._workers_by_load.clear()
        router._workers_by_load[100].add("worker-overloaded")
        router._workers_by_load[0].add("worker-idle")
        router._min_load = 0

        credit = make_credit()

        await router.send_credit(credit)
        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-idle"

    async def test_worker_registration_idempotent(self, service_config) -> None:
        """Test that re-registering worker is safe."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._workers["worker-1"].in_flight_credits = 5

        router._register_worker("worker-1")
        assert router._workers["worker-1"].in_flight_credits == 5


class TestStickyCreditRouterFirstToken:
    """Test FirstToken message handling."""

    async def test_first_token_callback_called(self, service_config) -> None:
        """Test that first token callback is invoked."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        callback_received = []

        async def on_first_token(message: FirstToken) -> None:
            callback_received.append(message)

        router.set_first_token_callback(on_first_token)
        router._register_worker("worker-1")

        first_token = FirstToken(
            credit_id=42,
            phase=CreditPhase.PROFILING,
            ttft_ns=150_000_000,  # 150ms
        )

        await router._handle_router_message("worker-1", first_token)

        assert len(callback_received) == 1
        assert callback_received[0].credit_id == 42
        assert callback_received[0].phase == CreditPhase.PROFILING
        assert callback_received[0].ttft_ns == 150_000_000

    async def test_first_token_no_callback_does_not_error(self, service_config) -> None:
        """Test that FirstToken without callback set doesn't cause error."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-1")

        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.WARMUP,
            ttft_ns=100_000_000,
        )

        # Should not raise
        await router._handle_router_message("worker-1", first_token)

    async def test_first_token_warmup_phase(self, service_config) -> None:
        """Test that FirstToken works for warmup phase."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        received_phases = []

        async def on_first_token(message: FirstToken) -> None:
            received_phases.append(message.phase)

        router.set_first_token_callback(on_first_token)
        router._register_worker("worker-1")

        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.WARMUP,
            ttft_ns=50_000_000,
        )

        await router._handle_router_message("worker-1", first_token)

        assert received_phases == [CreditPhase.WARMUP]


class TestStickyCreditRouterLateJoiningWorker:
    """Test fair handling of late-joining workers (thundering herd prevention)."""

    async def test_late_joiner_gets_average_virtual_credits(
        self, service_config
    ) -> None:
        """Late-joining worker should initialize with average virtual_sent_credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        router._workers["worker-1"].virtual_sent_credits = 50
        router._workers["worker-2"].virtual_sent_credits = 50
        router._workers["worker-1"].total_sent_credits = 50
        router._workers["worker-2"].total_sent_credits = 50

        router._workers_cache = list(router._workers.values())

        router._register_worker("worker-3")

        assert router._workers["worker-3"].virtual_sent_credits == 50
        assert router._workers["worker-3"].total_sent_credits == 0

    async def test_late_joiner_not_preferred_over_existing(
        self, service_config
    ) -> None:
        """Late-joining worker should not get all requests due to zero credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        for w in ["worker-1", "worker-2"]:
            router._workers[w].virtual_sent_credits = 100
            router._workers[w].total_sent_credits = 100
        router._workers_cache = list(router._workers.values())

        router._register_worker("worker-3")
        assert router._workers["worker-3"].virtual_sent_credits == 100

        credits_per_worker = {"worker-1": 0, "worker-2": 0, "worker-3": 0}

        for i in range(30):
            credit = make_credit(
                id=i,
                corr_id=f"session-{i}",
                num_turns=1,  # Single turn - no sticky session
            )
            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            credits_per_worker[worker_id] += 1

        assert credits_per_worker["worker-1"] == 10
        assert credits_per_worker["worker-2"] == 10
        assert credits_per_worker["worker-3"] == 10

    async def test_late_joiner_without_existing_workers(self, service_config) -> None:
        """First worker to join should have virtual_sent_credits = 0."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        assert router._workers["worker-1"].virtual_sent_credits == 0

        router._register_worker("worker-2")
        assert router._workers["worker-2"].virtual_sent_credits == 0

    async def test_virtual_credits_vs_total_credits_semantics(
        self, service_config
    ) -> None:
        """Verify virtual and total credits track independently."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")

        router._workers["worker-1"].virtual_sent_credits = 100
        router._workers["worker-1"].total_sent_credits = 100
        router._workers_cache = list(router._workers.values())

        router._register_worker("worker-2")

        assert router._workers["worker-2"].virtual_sent_credits == 100
        assert router._workers["worker-2"].total_sent_credits == 0

        credit = make_credit(id=1, corr_id="s1", num_turns=1)
        router._workers_by_load[0] = {"worker-2"}  # Force selection
        router._min_load = 0
        await router.send_credit(credit)

        assert router._workers["worker-2"].virtual_sent_credits == 101
        assert router._workers["worker-2"].total_sent_credits == 1


class TestStickyCreditRouterCancellation:
    """Test credit cancellation behavior."""

    async def test_cancel_all_credits_sends_to_workers_with_in_flight(
        self, service_config
    ) -> None:
        """Test that cancel_all_credits sends to workers with in-flight credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # worker-1 has 3 in-flight credits
        router._workers["worker-1"].in_flight_credits = 3
        router._workers["worker-1"].active_credit_ids = {1, 2, 3}
        # worker-2 has 0 in-flight credits
        router._workers["worker-2"].in_flight_credits = 0

        await router.cancel_all_credits()

        # Should only send to worker-1
        assert router._router_client.send_to.call_count == 1
        call_args = router._router_client.send_to.call_args[0]
        assert call_args[0] == "worker-1"
        assert call_args[1].credit_ids == {1, 2, 3}

    async def test_cancel_all_credits_no_workers_with_in_flight(
        self, service_config
    ) -> None:
        """Test cancel_all_credits when no workers have in-flight credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._workers["worker-1"].in_flight_credits = 0

        await router.cancel_all_credits()

        # Should not send any messages
        router._router_client.send_to.assert_not_called()

    async def test_cancel_sets_cancellation_pending_flag(self, service_config) -> None:
        """Test that cancel_all_credits sets _cancellation_pending flag."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        assert router._cancellation_pending is False

        await router.cancel_all_credits()

        assert router._cancellation_pending is True


class TestStickyCreditRouterWorkerUnregistration:
    """Test worker unregistration edge cases."""

    async def test_unregister_with_active_sessions_clears_sticky(
        self, service_config
    ) -> None:
        """Test that unregistering worker clears sticky sessions."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._workers["worker-1"].active_sessions = 2
        router._workers["worker-1"].active_session_ids = {"session-1", "session-2"}
        router._sticky_sessions = {"session-1": "worker-1", "session-2": "worker-1"}

        router._unregister_worker("worker-1")

        # Sticky sessions should be cleared
        assert "session-1" not in router._sticky_sessions
        assert "session-2" not in router._sticky_sessions

    async def test_unregister_during_cancellation_suppresses_warning(
        self, service_config
    ) -> None:
        """Test that unregistering with in-flight during cancellation doesn't warn."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._cancellation_pending = True

        router._register_worker("worker-1")
        router._workers["worker-1"].in_flight_credits = 5

        # Should not raise or warn excessively
        router._unregister_worker("worker-1")

        assert "worker-1" not in router._workers

    async def test_unregister_unknown_worker_is_safe(self, service_config) -> None:
        """Test that unregistering unknown worker doesn't crash."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        # Should not raise
        router._unregister_worker("never-registered")


class TestStickyCreditRouterMinLoadTracking:
    """Test minimum load tracking for fair load balancing."""

    async def test_min_load_updates_after_credit_sent(self, service_config) -> None:
        """Test that min_load updates correctly when credit sent."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        assert router._min_load == 0

        router._track_credit_sent("worker-1", 1)
        # worker-2 still at 0, so min_load stays 0
        assert router._min_load == 0

        router._track_credit_sent("worker-2", 2)
        # Both workers now at 1, min_load should be 1
        assert router._min_load == 1

    async def test_min_load_updates_after_credit_returned(self, service_config) -> None:
        """Test that min_load updates correctly when credit returned."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Send credits to both workers
        router._track_credit_sent("worker-1", 1)
        router._track_credit_sent("worker-2", 2)
        router._track_credit_sent("worker-1", 3)

        # worker-1: 2 in-flight, worker-2: 1 in-flight
        assert router._min_load == 1

        # Return from worker-2, now at 0
        router._track_credit_returned(
            "worker-2", 2, cancelled=False, error_reported=False
        )
        assert router._min_load == 0

    async def test_min_load_recalculates_after_unregister_last_at_min(
        self, service_config
    ) -> None:
        """Test min_load recalculation when last worker at min is unregistered."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        router._workers["worker-1"].in_flight_credits = 5
        router._workers["worker-2"].in_flight_credits = 2

        router._workers_by_load.clear()
        router._workers_by_load[5].add("worker-1")
        router._workers_by_load[2].add("worker-2")
        router._min_load = 2

        # Unregister worker-2 (the only one at min_load)
        router._unregister_worker("worker-2")

        # min_load should be recalculated to 5
        assert router._min_load == 5


class TestStickyCreditRouterErrorTracking:
    """Test error tracking in credit returns."""

    async def test_track_credit_returned_with_error(self, service_config) -> None:
        """Test that error_reported flag increments error counter."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._track_credit_sent("worker-1", 1)

        router._track_credit_returned(
            "worker-1", 1, cancelled=False, error_reported=True
        )

        assert router._workers["worker-1"].total_errors_reported == 1
        assert router._workers["worker-1"].total_completed_credits == 1

    async def test_track_credit_cancelled_with_error(self, service_config) -> None:
        """Test that cancelled credit with error tracks both."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._track_credit_sent("worker-1", 1)

        router._track_credit_returned(
            "worker-1", 1, cancelled=True, error_reported=True
        )

        assert router._workers["worker-1"].total_cancelled_credits == 1
        assert router._workers["worker-1"].total_errors_reported == 1
        assert router._workers["worker-1"].total_completed_credits == 0


class TestStickyCreditRouterCreditValidation:
    """Test credit validation."""

    async def test_send_credit_with_empty_correlation_id_raises(
        self, service_config
    ) -> None:
        """Test that send_credit raises if x_correlation_id is empty string."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._register_worker("worker-1")

        # Create credit with empty x_correlation_id (bypasses the "if not" check)
        # Note: Credit is a frozen msgspec Struct so we create directly
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv-1",
            x_correlation_id="",  # Empty string triggers validation
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
        )

        with pytest.raises(RuntimeError, match="x_correlation_id must be set"):
            await router.send_credit(credit)


class TestStickyCreditRouterTieBreaking:
    """Test tie-breaking behavior when multiple workers at min load."""

    async def test_prefers_worker_with_fewer_active_sessions(
        self, service_config
    ) -> None:
        """Test tie-breaking prefers workers with fewer active sessions."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Both at same load and virtual credits, but worker-2 has more sessions
        router._workers["worker-1"].active_sessions = 1
        router._workers["worker-2"].active_sessions = 5
        router._workers["worker-1"].virtual_sent_credits = 100
        router._workers["worker-2"].virtual_sent_credits = 100
        # Make last_sent_at_ns equal to eliminate that as tie-breaker
        router._workers["worker-1"].last_sent_at_ns = 1000
        router._workers["worker-2"].last_sent_at_ns = 1000

        credit = make_credit(id=1, corr_id="test-session", num_turns=1)

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-1"

    async def test_prefers_worker_with_fewer_virtual_credits(
        self, service_config
    ) -> None:
        """Test tie-breaking prefers workers with fewer virtual credits."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Same active sessions, but different virtual credits
        router._workers["worker-1"].active_sessions = 2
        router._workers["worker-2"].active_sessions = 2
        router._workers["worker-1"].virtual_sent_credits = 50
        router._workers["worker-2"].virtual_sent_credits = 100
        router._workers["worker-1"].last_sent_at_ns = 1000
        router._workers["worker-2"].last_sent_at_ns = 1000

        credit = make_credit(id=1, corr_id="test-session", num_turns=1)

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-1"

    async def test_prefers_worker_with_older_last_sent(self, service_config) -> None:
        """Test tie-breaking prefers workers with older last_sent_at_ns."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Same sessions and virtual credits, but different last_sent times
        router._workers["worker-1"].active_sessions = 0
        router._workers["worker-2"].active_sessions = 0
        router._workers["worker-1"].virtual_sent_credits = 100
        router._workers["worker-2"].virtual_sent_credits = 100
        router._workers["worker-1"].last_sent_at_ns = 1000  # Earlier (preferred)
        router._workers["worker-2"].last_sent_at_ns = 2000

        credit = make_credit(id=1, corr_id="test-session", num_turns=1)

        await router.send_credit(credit)

        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-1"


class TestStickyCreditRouterMarkComplete:
    """Test mark_credits_complete behavior."""

    async def test_mark_complete_suppresses_orphan_warnings(
        self, service_config
    ) -> None:
        """Test that mark_credits_complete suppresses orphan session warnings."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._register_worker("worker-1")
        router._workers["worker-1"].active_sessions = 2
        router._workers["worker-1"].active_session_ids = {"s1", "s2"}
        router._sticky_sessions = {"s1": "worker-1", "s2": "worker-1"}

        router.mark_credits_complete()

        # Should not warn when unregistering
        router._unregister_worker("worker-1")

        assert router._credits_complete is True


class TestStickyCreditRouterStickySessionReassignment:
    """Test sticky session reassignment when worker becomes unavailable."""

    async def test_reassigns_to_new_worker_if_sticky_worker_gone(
        self, service_config
    ) -> None:
        """Test that sticky session is reassigned if assigned worker is gone."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )
        router._router_client.send_to = AsyncMock()

        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Create sticky session to worker-1
        router._sticky_sessions["session-X"] = "worker-1"

        # Unregister worker-1
        router._unregister_worker("worker-1")

        # Now send a credit for this session - should fall back to fair load
        credit = make_credit(
            id=1,
            corr_id="session-X",
            turn=1,
            num_turns=3,
        )

        await router.send_credit(credit)

        # Should route to worker-2 (only remaining worker)
        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-2"

        # New sticky session should be created
        assert router._sticky_sessions["session-X"] == "worker-2"

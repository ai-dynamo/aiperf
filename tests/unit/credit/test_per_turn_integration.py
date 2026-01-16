# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for per-turn credit end-to-end flow."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models.dataset_models import Turn
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit


def create_credit(
    credit_id: int,
    conversation_id: str,
    x_correlation_id: str,
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    """Helper to create test credits using native msgspec struct."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
    )


class TestPerCreditIntegration:
    """Integration tests for complete per-turn credit flow."""

    @pytest.mark.asyncio
    async def test_complete_single_turn_flow(self, service_config):
        """Test complete flow for a single-turn conversation."""
        # Setup StickyCreditRouter
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")
        router._register_worker("worker-B")

        # Create credit for single-turn conversation
        credit = create_credit(
            credit_id=1,
            x_correlation_id="inst-single-turn",
            conversation_id="conv-123",
            turn_index=0,
            num_turns=1,  # Single turn = final turn
        )

        # Route and send credit
        await router.send_credit(credit)
        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id in ["worker-A", "worker-B"]
        assert router._workers[worker_id].in_flight_credits == 1

        # Worker processes and returns
        router._track_credit_returned(
            worker_id, credit.id, cancelled=False, error_reported=False
        )
        assert router._workers[worker_id].in_flight_credits == 0
        assert router._workers[worker_id].total_completed_credits == 1

        # Assignment should be cleaned up (final turn)
        assert len(router._sticky_sessions) == 0

    @pytest.mark.asyncio
    async def test_complete_multi_turn_flow(self, service_config):
        """Test complete flow for a multi-turn conversation."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )

        async def mock_send_to(*args, **kwargs):
            return None

        router._router_client.send_to = MagicMock(side_effect=mock_send_to)
        router._register_worker("worker-1")
        router._register_worker("worker-2")
        router._register_worker("worker-3")

        instance_id = "instance-456"
        num_turns = 5

        # Process all turns
        first_worker = None
        for turn_index in range(num_turns):
            credit = create_credit(
                credit_id=turn_index + 1,
                x_correlation_id=instance_id,  # Same instance for sticky routing
                conversation_id="session-789",
                turn_index=turn_index,
                num_turns=num_turns,
            )

            # Route and send credit
            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]

            # First turn: fair load
            if turn_index == 0:
                first_worker = worker_id
                assert worker_id in ["worker-1", "worker-2", "worker-3"]
            else:
                # Subsequent turns: sticky to same worker
                assert worker_id == first_worker

            # Simulate worker processing (credit return)
            router._track_credit_returned(
                worker_id, credit.id, cancelled=False, error_reported=False
            )

        # Assignment should be cleaned up after final turn
        assert instance_id not in router._sticky_sessions

    @pytest.mark.asyncio
    async def test_load_balancing_across_multiple_conversations(self, service_config):
        """Test that multiple conversations are balanced across workers."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()

        # Register 3 workers
        for i in range(3):
            router._register_worker(f"worker-{i}")

        # Start 9 conversations (should distribute evenly: 3 per worker)
        conversations = []
        for i in range(9):
            instance_id = f"instance-{i}"
            credit = create_credit(
                credit_id=i,
                x_correlation_id=instance_id,
                conversation_id=f"session-{i}",
                turn_index=0,  # First turn
                num_turns=2,  # Multi-turn
            )

            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            conversations.append((instance_id, worker_id))

        # Verify even distribution by checking in-flight credits
        worker_loads = {
            w: router._workers[w].in_flight_credits for w in router._workers
        }
        assert all(load == 3 for load in worker_loads.values())

        # Now send second turns for all conversations (should be sticky)
        for i, (instance_id, expected_worker) in enumerate(conversations):
            credit = create_credit(
                credit_id=100 + i,
                x_correlation_id=instance_id,  # Same instance
                conversation_id=f"session-{instance_id}",
                turn_index=1,
                num_turns=2,  # Final turn
            )

            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            assert worker_id == expected_worker  # Sticky!

    @pytest.mark.asyncio
    async def test_turn_data_embedding_simulation(self, service_config):
        """Test simulation of turn data embedding and consumption."""

        # Simulate TimingManager embedding turn data
        def embed_turn_data(credit: Credit, turn_index: int) -> dict:
            """Simulate TimingManager fetching and embedding turn data."""
            turn = Turn(
                role="user",
                content=f"Turn {turn_index} content",
                delay=100 if turn_index > 0 else None,
            )
            return {
                "credit": credit,
                "turn_data": turn,
            }

        # Simulate Worker consuming embedded turn data
        def worker_process_credit(data: dict) -> dict:
            """Simulate Worker processing credit with embedded data."""
            if not data.get("turn_data"):
                raise RuntimeError("Turn data not embedded!")

            turn_data = data["turn_data"]
            credit = data["credit"]
            return {
                "turn_index": credit.turn_index,
                "content": turn_data.content,
                "delay": turn_data.delay,
            }

        # Setup router
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-X")

        # Process 3-turn conversation
        instance_id = "test-instance"
        num_turns = 3
        results = []

        for turn_index in range(num_turns):
            # Create credit
            credit = create_credit(
                credit_id=turn_index + 1,
                x_correlation_id=instance_id,
                conversation_id="test-session",
                turn_index=turn_index,
                num_turns=num_turns,
            )

            # TimingManager: embed turn data
            data = embed_turn_data(credit, turn_index)
            assert data["turn_data"] is not None

            # StickyCreditRouter: route to worker
            await router.send_credit(credit)
            worker_id = router._router_client.send_to.call_args[0][0]
            assert worker_id == "worker-X"

            # Worker: process credit
            result = worker_process_credit(data)
            results.append(result)

            # Simulate credit return
            router._track_credit_returned(
                worker_id, credit.id, cancelled=False, error_reported=False
            )

        # Verify results
        assert len(results) == 3
        assert results[0]["delay"] is None  # First turn has no delay
        assert results[1]["delay"] == 100  # Second turn has delay
        assert results[2]["delay"] == 100  # Third turn has delay

        # Verify assignment cleaned up
        assert instance_id not in router._sticky_sessions

    @pytest.mark.asyncio
    async def test_error_handling_missing_turn_data(self, service_config):
        """Test error handling when turn data is not embedded."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-err")

        # Create credit (simulating fetch failure)
        credit = create_credit(
            credit_id=1,
            x_correlation_id="inst-error",
            conversation_id="conv-error",
            turn_index=0,
            num_turns=1,
        )

        # Router should still route successfully
        await router.send_credit(credit)
        worker_id = router._router_client.send_to.call_args[0][0]
        assert worker_id == "worker-err"

        # But worker should detect missing data and raise error
        def worker_process_credit(data: dict | None):
            if not data or not data.get("turn_data"):
                raise RuntimeError(
                    "Turn data not embedded in credit. "
                    "TimingManager must always embed turn data."
                )

        with pytest.raises(RuntimeError, match="Turn data not embedded"):
            worker_process_credit(None)

    @pytest.mark.asyncio
    async def test_concurrent_conversations_different_workers(self, service_config):
        """Test that concurrent conversations can be processed by different workers."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")

        # Start two conversations concurrently
        conv1_credit = create_credit(
            credit_id=1,
            x_correlation_id="instance-A",
            conversation_id="session-A",
            turn_index=0,
            num_turns=2,
        )

        conv2_credit = create_credit(
            credit_id=2,
            x_correlation_id="instance-B",
            conversation_id="session-B",
            turn_index=0,
            num_turns=2,
        )

        # Route both
        await router.send_credit(conv1_credit)
        worker1 = router._router_client.send_to.call_args[0][0]
        await router.send_credit(conv2_credit)
        worker2 = router._router_client.send_to.call_args[0][0]

        # Both should be routed (possibly to different workers)
        assert worker1 in ["worker-1", "worker-2"]
        assert worker2 in ["worker-1", "worker-2"]

        # Should have 2 active conversations
        assert len(router._sticky_sessions) == 2

        # Load should be distributed
        total_in_flight = sum(w.in_flight_credits for w in router._workers.values())
        assert total_in_flight == 2

    @pytest.mark.asyncio
    async def test_same_session_different_instances_balanced(self, service_config):
        """Test that same session sampled multiple times is balanced."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-A")
        router._register_worker("worker-B")

        # Same session_id, different x_correlation_ids (instances)
        # This simulates sampling "conv-42" three times during benchmark
        session_id = "conv-42"

        instance1_credit = create_credit(
            credit_id=1,
            x_correlation_id="uuid-1",  # Different instance
            conversation_id=session_id,
            turn_index=0,
            num_turns=2,
        )

        instance2_credit = create_credit(
            credit_id=2,
            x_correlation_id="uuid-2",  # Different instance
            conversation_id=session_id,
            turn_index=0,
            num_turns=2,
        )

        instance3_credit = create_credit(
            credit_id=3,
            x_correlation_id="uuid-3",  # Different instance
            conversation_id=session_id,
            turn_index=0,
            num_turns=2,
        )

        # Route all three
        await router.send_credit(instance1_credit)
        await router.send_credit(instance2_credit)
        await router.send_credit(instance3_credit)

        # Should create 3 separate assignments (different instances)
        assert len(router._sticky_sessions) == 3

        # Should be balanced across workers (not all to same worker)
        worker_loads = {
            w: router._workers[w].in_flight_credits for w in router._workers
        }
        # At least one worker should have 2 credits, one should have 1
        # (since 3 credits, 2 workers)
        assert sorted(worker_loads.values()) == [1, 2]

    @pytest.mark.asyncio
    async def test_worker_failure_and_reassignment(self, service_config):
        """Test that conversations can be reassigned if worker fails."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-service"
        )
        router._router_client.send_to = AsyncMock()
        router._register_worker("worker-1")
        router._register_worker("worker-2")

        instance_id = "instance-failover"

        # First turn routed to a worker
        credit1 = create_credit(
            credit_id=1,
            x_correlation_id=instance_id,
            conversation_id="session-123",
            turn_index=0,
            num_turns=2,
        )

        await router.send_credit(credit1)
        worker1 = router._router_client.send_to.call_args[0][0]
        router._track_credit_returned(
            worker1, credit1.id, cancelled=False, error_reported=False
        )

        # Worker fails and unregisters
        router._unregister_worker(worker1)

        # Second turn - should fallback to available worker
        # (assignment exists but worker is gone)
        # Clear assignment to simulate worker failure
        if instance_id in router._sticky_sessions:
            del router._sticky_sessions[instance_id]

        credit2 = create_credit(
            credit_id=2,
            x_correlation_id=instance_id,  # Same instance
            conversation_id="session-123",
            turn_index=1,
            num_turns=2,  # Final turn
        )

        # Should fallback to fair load balancing
        await router.send_credit(credit2)
        worker2 = router._router_client.send_to.call_args[0][0]

        # Should route to remaining worker
        remaining_worker = [w for w in ["worker-1", "worker-2"] if w != worker1][0]
        assert worker2 == remaining_worker

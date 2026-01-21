# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test concurrent operations in StickyCreditRouter.

These tests validate that the sticky router maintains consistent state when handling
concurrent credit tracking operations. The tests focus on verifying that asyncio's
single-threaded execution model properly serializes operations without explicit locking.

Note: router_with_worker fixture is provided by conftest.py.
"""

from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad


class TestConcurrentCreditTracking:
    """Test that concurrent track_credit_sent/returned maintain consistent state."""

    async def test_rapid_send_return_cycles(self, router_with_worker):
        """Test rapid send/return cycles maintain correct counts."""
        router = router_with_worker

        # Simulate rapid send/return cycles
        credit_nums = list(range(100))

        # Send all credits
        for credit_num in credit_nums:
            router._track_credit_sent("worker-1", credit_num)

        # Verify all credits are in-flight
        worker_load = router._workers["worker-1"]
        assert worker_load.in_flight_credits == 100
        assert len(worker_load.active_credit_ids) == 100
        assert worker_load.total_sent_credits == 100

        # Return half
        for credit_num in credit_nums[:50]:
            router._track_credit_returned(
                "worker-1", credit_num, cancelled=False, error_reported=False
            )

        assert worker_load.in_flight_credits == 50
        assert len(worker_load.active_credit_ids) == 50
        assert worker_load.total_completed_credits == 50
        assert worker_load.total_cancelled_credits == 0

        # Cancel remaining half
        for credit_num in credit_nums[50:]:
            router._track_credit_returned(
                "worker-1", credit_num, cancelled=True, error_reported=False
            )

        assert worker_load.in_flight_credits == 0
        assert len(worker_load.active_credit_ids) == 0
        assert worker_load.total_completed_credits == 50
        assert worker_load.total_cancelled_credits == 50

    async def test_interleaved_send_and_return(self, router_with_worker):
        """Test interleaved send/return operations maintain correct counts."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Interleave sends and returns
        for i in range(50):
            # Send two credits
            router._track_credit_sent("worker-1", i * 2)
            router._track_credit_sent("worker-1", i * 2 + 1)

            # Return one credit
            router._track_credit_returned(
                "worker-1", i * 2, cancelled=False, error_reported=False
            )

        # At end: sent 100, returned 50, in-flight should be 50
        assert worker_load.total_sent_credits == 100
        assert worker_load.total_completed_credits == 50
        assert worker_load.in_flight_credits == 50
        assert len(worker_load.active_credit_ids) == 50

        # Verify the correct credits are still in-flight (odd numbered)
        expected_in_flight = {i * 2 + 1 for i in range(50)}
        assert worker_load.active_credit_ids == expected_in_flight

    async def test_track_credit_sent_increments_all_counters(self, router_with_worker):
        """Test that track_credit_sent increments all relevant counters atomically."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Initial state
        assert worker_load.total_sent_credits == 0
        assert worker_load.in_flight_credits == 0
        assert len(worker_load.active_credit_ids) == 0

        # Send a credit
        router._track_credit_sent("worker-1", 1)

        # All counters should be updated
        assert worker_load.total_sent_credits == 1
        assert worker_load.in_flight_credits == 1
        assert 1 in worker_load.active_credit_ids

    async def test_track_credit_returned_decrements_correctly(self, router_with_worker):
        """Test that track_credit_returned decrements in-flight counter atomically."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Send and return a credit
        router._track_credit_sent("worker-1", 1)
        router._track_credit_returned(
            "worker-1", 1, cancelled=False, error_reported=False
        )

        # Verify state after return
        assert worker_load.total_sent_credits == 1
        assert worker_load.total_completed_credits == 1
        assert worker_load.in_flight_credits == 0
        assert 1 not in worker_load.active_credit_ids

    async def test_track_credit_cancelled_updates_correct_counter(
        self, router_with_worker
    ):
        """Test that cancelled credits update the cancelled counter, not returned."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Send and cancel a credit
        router._track_credit_sent("worker-1", 1)
        router._track_credit_returned(
            "worker-1", 1, cancelled=True, error_reported=False
        )

        # Verify cancelled counter is updated, not returned
        assert worker_load.total_sent_credits == 1
        assert worker_load.total_completed_credits == 0
        assert worker_load.total_cancelled_credits == 1
        assert worker_load.in_flight_credits == 0


class TestWorkerLoadInvariants:
    """Test that worker load invariants are maintained across operations."""

    async def test_in_flight_equals_active_credit_ids_count(self, router_with_worker):
        """Test that in_flight_credits always equals len(active_credit_ids)."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Test invariant through various operations
        for i in range(10):
            router._track_credit_sent("worker-1", i)
            assert worker_load.in_flight_credits == len(worker_load.active_credit_ids)

        for i in range(5):
            router._track_credit_returned(
                "worker-1", i, cancelled=False, error_reported=False
            )
            assert worker_load.in_flight_credits == len(worker_load.active_credit_ids)

        for i in range(5, 10):
            router._track_credit_returned(
                "worker-1", i, cancelled=True, error_reported=False
            )
            assert worker_load.in_flight_credits == len(worker_load.active_credit_ids)

    async def test_total_sent_equals_returned_plus_cancelled_plus_in_flight(
        self, router_with_worker
    ):
        """Test that total_sent = returned + cancelled + in_flight."""
        router = router_with_worker
        worker_load = router._workers["worker-1"]

        # Send 20 credits
        for i in range(20):
            router._track_credit_sent("worker-1", i)

        # Return 10, cancel 5, leave 5 in-flight
        for i in range(10):
            router._track_credit_returned(
                "worker-1", i, cancelled=False, error_reported=False
            )
        for i in range(10, 15):
            router._track_credit_returned(
                "worker-1", i, cancelled=True, error_reported=False
            )

        # Verify invariant
        assert (
            worker_load.total_sent_credits
            == worker_load.total_completed_credits
            + worker_load.total_cancelled_credits
            + worker_load.in_flight_credits
        )
        assert worker_load.total_sent_credits == 20
        assert worker_load.total_completed_credits == 10
        assert worker_load.total_cancelled_credits == 5
        assert worker_load.in_flight_credits == 5


class TestMultipleWorkerConcurrency:
    """Test concurrent operations across multiple workers."""

    async def test_multiple_workers_independent_tracking(self, service_config):
        """Test that multiple workers maintain independent state correctly."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        # Register three workers
        router._workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0),
            "worker-2": WorkerLoad(worker_id="worker-2", in_flight_credits=0),
            "worker-3": WorkerLoad(worker_id="worker-3", in_flight_credits=0),
        }

        # Send credits to each worker (use offset ranges to avoid collisions)
        for i in range(10):
            router._track_credit_sent("worker-1", i)
            router._track_credit_sent("worker-2", 100 + i)
            router._track_credit_sent("worker-3", 200 + i)

        # Verify each worker has correct state
        for worker_id in ["worker-1", "worker-2", "worker-3"]:
            worker_load = router._workers[worker_id]
            assert worker_load.total_sent_credits == 10
            assert worker_load.in_flight_credits == 10
            assert len(worker_load.active_credit_ids) == 10

        # Return credits from worker-1, cancel from worker-2, leave worker-3 alone
        for i in range(10):
            router._track_credit_returned(
                "worker-1", i, cancelled=False, error_reported=False
            )
            router._track_credit_returned(
                "worker-2", 100 + i, cancelled=True, error_reported=False
            )

        # Verify independent state
        assert router._workers["worker-1"].in_flight_credits == 0
        assert router._workers["worker-1"].total_completed_credits == 10
        assert router._workers["worker-2"].in_flight_credits == 0
        assert router._workers["worker-2"].total_cancelled_credits == 10
        assert router._workers["worker-3"].in_flight_credits == 10
        assert router._workers["worker-3"].total_completed_credits == 0

    async def test_concurrent_operations_different_workers(self, service_config):
        """Test that operations on different workers don't interfere."""
        router = StickyCreditRouter(
            service_config=service_config, service_id="test-router"
        )

        router._workers = {
            "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0),
            "worker-2": WorkerLoad(worker_id="worker-2", in_flight_credits=0),
        }

        # Interleave operations across workers (use offset ranges)
        for i in range(50):
            router._track_credit_sent("worker-1", i)
            router._track_credit_sent("worker-2", 100 + i)

            if i > 0:
                router._track_credit_returned(
                    "worker-1", i - 1, cancelled=False, error_reported=False
                )
                router._track_credit_returned(
                    "worker-2", 100 + i - 1, cancelled=False, error_reported=False
                )

        # Worker-1 should have 1 in-flight (last one), 49 returned
        assert router._workers["worker-1"].total_sent_credits == 50
        assert router._workers["worker-1"].total_completed_credits == 49
        assert router._workers["worker-1"].in_flight_credits == 1

        # Worker-2 should have same
        assert router._workers["worker-2"].total_sent_credits == 50
        assert router._workers["worker-2"].total_completed_credits == 49
        assert router._workers["worker-2"].in_flight_credits == 1

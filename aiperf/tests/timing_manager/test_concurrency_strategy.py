# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the ConcurrencyStrategy class.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.messages import CreditReturnMessage
from aiperf.services.timing_manager.concurrency_strategy import ConcurrencyStrategy
from aiperf.tests.utils.async_test_utils import MockSemaphore


def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds."""
    return int(seconds * NANOS_PER_SECOND)


@pytest.mark.asyncio
class TestConcurrencyStrategy:
    """Tests for the ConcurrencyStrategy class."""

    @patch("time.time_ns")
    async def test_start_sets_start_time_and_launches_tasks(
        self, mock_time_ns, config, mock_credit_manager
    ):
        """Test that start() sets start time and launches background tasks."""
        mock_time_ns.return_value = seconds_to_ns(1)

        strategy = ConcurrencyStrategy(config, mock_credit_manager)

        # Mock the background task methods to prevent actual execution
        strategy._progress_report_loop = AsyncMock()
        strategy._credit_drop_loop = AsyncMock()

        await strategy.start()

        assert strategy.start_time_ns == seconds_to_ns(1)
        # Verify tasks were created (we can't easily verify they were started without more complex mocking)
        assert len(strategy.tasks) == 2

    async def test_credit_drop_loop_sends_all_credits(
        self, config, mock_credit_manager
    ):
        """Test that credit drop loop sends all credits respecting concurrency."""

        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy._total_credits = 3
        strategy._concurrency = 2

        # Mock the semaphore to avoid deadlock
        strategy._semaphore = MockSemaphore()

        # Run the credit drop loop
        await strategy._credit_drop_loop()

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        # Verify all credits were sent
        assert strategy._sent_credits == 3
        assert strategy._semaphore.acquire_count == 3
        assert len(mock_credit_manager.dropped_credits) == 3

    async def test_credit_drop_loop_with_zero_credits(
        self, config, mock_credit_manager
    ):
        """Test credit drop loop behavior with zero total credits."""
        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy._total_credits = 0
        strategy._concurrency = 10

        await strategy._credit_drop_loop()

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        assert strategy._sent_credits == 0
        assert len(mock_credit_manager.dropped_credits) == 0

    @patch("time.time_ns")
    async def test_on_credit_return_basic_functionality(
        self, mock_time_ns, config, mock_credit_manager
    ):
        """Test basic credit return processing."""
        mock_time_ns.return_value = seconds_to_ns(2)

        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy.start_time_ns = seconds_to_ns(1)
        strategy._semaphore = MockSemaphore()

        message = CreditReturnMessage(service_id="test-service")

        await strategy.on_credit_return(message)

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        assert strategy._completed_credits == 1
        assert strategy._semaphore.release_count == 1

    @patch("time.time_ns")
    async def test_on_credit_return_triggers_credits_complete(
        self, mock_time_ns, config, mock_credit_manager
    ):
        """Test that credit return triggers credits complete when all credits are returned."""
        mock_time_ns.return_value = seconds_to_ns(2)

        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy._total_credits = 2
        strategy._concurrency = 1
        strategy._semaphore = MockSemaphore()

        strategy.start_time_ns = seconds_to_ns(1)
        strategy._completed_credits = 1  # One credit already completed

        message = CreditReturnMessage(service_id="test-service")

        await strategy.on_credit_return(message)

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        assert strategy._completed_credits == 2
        assert len(mock_credit_manager.credits_complete_calls) == 1
        assert strategy._semaphore.release_count == 1
        assert mock_credit_manager.credits_complete_calls[0]["cancelled"] is False

    @patch("time.time_ns")
    async def test_full_credit_lifecycle(
        self, mock_time_ns, config, mock_credit_manager
    ):
        """Test a complete credit lifecycle from start to finish."""
        mock_time_ns.return_value = seconds_to_ns(1)

        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy._total_credits = 2
        strategy._concurrency = 1
        strategy._semaphore = MockSemaphore()

        # Mock the background loops to prevent them from running indefinitely
        strategy._progress_report_loop = AsyncMock()
        strategy._credit_drop_loop = AsyncMock()

        # Start the strategy
        await strategy.start()

        # Simulate credit returns
        message1 = CreditReturnMessage(service_id="test-service")
        message2 = CreditReturnMessage(service_id="test-service")

        await strategy.on_credit_return(message1)
        await strategy.on_credit_return(message2)

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        # Verify final state
        assert strategy._completed_credits == 2
        assert len(mock_credit_manager.credits_complete_calls) == 1
        assert mock_credit_manager.credits_complete_calls[0]["cancelled"] is False

    async def test_credit_drop_and_return_flow(self, config, mock_credit_manager):
        """Test coordinated credit drop and return flow without deadlocks."""
        strategy = ConcurrencyStrategy(config, mock_credit_manager)
        strategy._total_credits = 3
        strategy._concurrency = 2
        strategy._semaphore = MockSemaphore()

        # Run credit drop loop
        await strategy._credit_drop_loop()

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        # Verify credits were sent
        assert strategy._sent_credits == 3
        assert strategy._semaphore.acquire_count == 3
        assert len(mock_credit_manager.dropped_credits) == 3

        # Simulate credit returns
        message1 = CreditReturnMessage(service_id="test-service")
        message2 = CreditReturnMessage(service_id="test-service")
        message3 = CreditReturnMessage(service_id="test-service")

        await strategy.on_credit_return(message1)
        await strategy.on_credit_return(message2)
        await strategy.on_credit_return(message3)

        # Wait for the pending tasks to complete before checking the results
        await asyncio.gather(*strategy.tasks)

        # Verify credits were returned
        assert strategy._completed_credits == 3
        assert strategy._semaphore.release_count == 3
        assert len(mock_credit_manager.credits_complete_calls) == 1

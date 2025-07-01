# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager fixed schedule strategy.
"""

from unittest.mock import patch

import pytest

from aiperf.services.timing_manager.timing_manager import FixedScheduleStrategy
from aiperf.tests.utils.async_test_utils import async_fixture


@pytest.mark.asyncio
class TestFixedScheduleStrategy:
    """
    Tests for the fixed schedule strategy.
    """

    async def test_start(self):
        """Test that credits are dropped according to schedule with credit returns."""

        fixed_schedule_strategy = FixedScheduleStrategy()

        fixed_schedule_strategy._schedule = [
            (0, "0"),
            (500000000, "1"),
            (1000000000, "2"),
            (1500000000, "3"),
        ]

        await fixed_schedule_strategy.start()

        # 3. Mock the time function
        mock_time = 1000000000  # Starting time

        def mock_time_ns():
            nonlocal mock_time
            return mock_time

        # 4. Create collectors for messages
        pushed_messages = []

        # 5. Create a function to simulate credit returns
        async def simulate_credit_return(topic, message):
            # First collect the pushed message
            pushed_messages.append((topic, message))

            # Create and process a return message
            return_message = CreditReturnMessage(service_id="test-consumer", amount=1)
            await service._on_credit_return(return_message)

        # 6. Mock necessary functions
        with (
            patch("time.time_ns", side_effect=mock_time_ns),
            patch("asyncio.sleep", return_value=None),
            patch.object(service.comms, "push", side_effect=simulate_credit_return),
        ):
            # 7. Set initial time
            service._start_time_ns = mock_time

            # 8. Run the credit drop method
            await service._issue_credit_drops()

            # 9. Verify all 4 credits were processed
            assert len(pushed_messages) == 4, (
                "Should have dropped 4 credits with returns"
            )

            # 10. Check details of the pushed messages
            for topic, message in pushed_messages:
                assert topic == Topic.CREDIT_DROP
                assert message.service_id == service.service_id
                assert message.amount == 1
                # You could also verify the timestamp corresponds to the schedule
                # if the implementation preserves that information

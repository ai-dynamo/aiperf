# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager fixed schedule strategy.
"""

from unittest.mock import patch

import pytest

from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.messages.base_messages import MessageType
from aiperf.timing import FixedScheduleStrategy, TimingManagerConfig
from tests.timing_manager.conftest import MockCreditManager


@pytest.mark.skip(reason="TODO: This test is broken")
@pytest.mark.asyncio
class TestFixedScheduleStrategy:
    """
    Tests for the fixed schedule strategy.
    """

    async def test_start(self):
        """Test that credits are dropped according to schedule with credit returns."""

        fixed_schedule_strategy = FixedScheduleStrategy(
            config=TimingManagerConfig(),
            credit_manager=MockCreditManager(),
            schedule=[
                (0, "0"),
                (500000000, "1"),
                (1000000000, "2"),
                (1500000000, "3"),
            ],
        )

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
            return_message = CreditReturnMessage(
                service_id="test-consumer",
                phase=CreditPhase.PROFILING,
                delayed_ns=None,
            )
            await fixed_schedule_strategy._on_credit_return(return_message)

        # 6. Mock necessary functions
        with (
            patch("time.time_ns", side_effect=mock_time_ns),
            patch("asyncio.sleep", return_value=None),
            patch.object(
                fixed_schedule_strategy.credit_manager,
                "publish_progress",
                side_effect=simulate_credit_return,
            ),
        ):
            # 7. Set initial time
            fixed_schedule_strategy.start_time_ns = mock_time

            # 8. Run the credit drop method
            await fixed_schedule_strategy._execute_phases()

            # 9. Verify all 4 credits were processed
            assert len(pushed_messages) == 4, (
                "Should have dropped 4 credits with returns"
            )

            # 10. Check details of the pushed messages
            for topic, message in pushed_messages:
                assert topic == MessageType.CREDIT_DROP
                assert message.service_id == "test-consumer"
                assert message.phase == CreditPhase.PROFILING
                assert message.conversation_id == "0"
                # You could also verify the timestamp corresponds to the schedule
                # if the implementation preserves that information

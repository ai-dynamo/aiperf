# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager service.
"""

import io
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from aiperf.common.enums import CommandType, ServiceType, Topic
from aiperf.common.messages import CommandMessage, CreditReturnMessage
from aiperf.common.service.base_service import BaseService
from aiperf.services.timing_manager.timing_manager import TimingManager
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class TimingManagerTestConfig(BaseModel):
    """Configuration model for timing manager tests."""

    input_file_path: str = "test_input_file.jsonl"


@pytest.mark.asyncio
class TestTimingManager(BaseTestComponentService):
    """
    Tests for the timing manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding timing manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return TimingManager

    @pytest.fixture
    def timing_config(self) -> TimingManagerTestConfig:
        """
        Return a test configuration for the timing manager.
        """
        return TimingManagerTestConfig()

    async def test_timing_manager_initialization(
        self, initialized_service: TimingManager
    ) -> None:
        """
        Test that the timing manager initializes with the correct service type.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.TIMING_MANAGER

    async def test_timing_manager_configure(
        self, initialized_service: TimingManager
    ) -> None:
        """
        Test that the timing manager can be configured with a message.

        This tests the @on_configure handler functionality.
        """

        # Define the mock file content
        mock_file_content = """
        {"timestamp": 0, "request": {"model":"facebook/opt-125m","messages":[{"role":"user","content":"This is my first prompt."}],"max_completion_tokens":1}}
        {"timestamp": 500000000, "request": {"model":"facebook/opt-125m","messages":[{"role":"user","content":"This is my second prompt."}],"max_completion_tokens":1}}
        {"timestamp": 1000000000, "request": {"model":"facebook/opt-125m","messages":[{"role":"user","content":"This is my third prompt."}],"max_completion_tokens":1}}
        {"timestamp": 1500000000, "request": {"model":"facebook/opt-125m","messages":[{"role":"user","content":"This is my fourth prompt."}],"max_completion_tokens":1}}
        """

        # Create the configuration model
        config_data = TimingManagerTestConfig(input_file_path="test_input_file.jsonl")

        # Create a Message object with the test file path
        config_message = CommandMessage(
            service_id="test-service-id",  # Required - who is sending the command
            command=CommandType.CONFIGURE,  # Required - what command to execute
            target_service_id="target-service",  # Optional - who should receive it
            require_response=True,  # Optional - default is False
            data=config_data,  # Using TimingManagerTestConfig as data
        )

        # Mock the open function when called with that specific path
        mock_file = io.StringIO(mock_file_content)

        # Use patch to replace the built-in open function
        with patch("builtins.open", return_value=mock_file):
            service = await async_fixture(initialized_service)

            # Configure the service
            await service._configure(config_message)

            assert service.schedule == [
                0,
                500000000,
                1000000000,
                1500000000,
            ], "The schedule should be populated with the correct timestamps."

            # For now, we can just verify the service is still in the correct state
            assert service.service_type == ServiceType.TIMING_MANAGER

    async def test_timing_manager_credit_drops(
        self, initialized_service: TimingManager
    ):
        """Test that credits are dropped according to schedule with credit returns."""
        service = await async_fixture(initialized_service)

        # 1. Setup test schedule
        service.schedule = [0, 500000000, 1000000000, 1500000000]

        # 2. Reset credits to 1 (should be default, but let's be explicit)
        service._credits_available = 1

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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Base test class for controller services.
"""

from unittest.mock import MagicMock

from aiperf.common.enums import (
    CommandType,
    Topic,
)
from aiperf.common.models import CommandPayload
from aiperf.common.service.base_controller_service import BaseControllerService
from aiperf.tests.base_test_service import BaseTestService, async_fixture


class BaseTestControllerService(BaseTestService):
    """
    Base class for testing controller services.

    This extends BaseTestService with specific tests for controller service
    functionality such as command sending, service registration handling,
    and monitoring of component services.
    """

    async def test_controller_command_publishing(
        self, initialized_service: BaseControllerService, mock_communication: MagicMock
    ) -> None:
        """
        Test that the controller can publish command messages.

        Verifies the controller can send properly formatted commands to components.
        """
        service = await async_fixture(initialized_service)

        # Create a test command message
        test_service_id = "test_service_123"
        command = CommandType.START

        # Create a command message
        command_payload = CommandPayload(
            command=command,
            target_service_id=test_service_id,
        )
        command_message = service.create_message(command_payload)

        # Publish the command
        await service.comms.publish(Topic.COMMAND, command_message)

        # Check that the command was published
        assert Topic.COMMAND in mock_communication.mock_data.published_messages
        assert len(mock_communication.mock_data.published_messages[Topic.COMMAND]) == 1

        # Verify command message
        published_cmd = mock_communication.mock_data.published_messages[Topic.COMMAND][
            0
        ]
        assert published_cmd.service_id == service.service_id
        assert published_cmd.payload.command == command
        assert published_cmd.payload.target_service_id == test_service_id

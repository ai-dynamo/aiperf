# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Base test class for component services.
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import (
    ServiceState,
    Topic,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.service.base_service import BaseService
from aiperf.tests.base_test_service import BaseTestService, async_fixture


class BaseTestComponentService(BaseTestService):
    """
    Base class for testing component services.

    This extends BaseTestService with specific tests for the component service
    functionality such as heartbeat, registration, and status updates.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """
        Return the service class to test.

        Returns:
            The BaseComponentService class for testing
        """
        return BaseComponentService

    async def test_service_heartbeat(
        self, initialized_service: BaseComponentService, mock_communication: MagicMock
    ) -> None:
        """
        Test that the service sends heartbeat messages correctly.

        Verifies:
        1. The service generates and sends a valid heartbeat message
        2. The message contains the correct service information
        """
        service = await async_fixture(initialized_service)

        # Directly send a heartbeat instead of waiting for the task
        await service.send_heartbeat()

        # Check that a heartbeat message was published
        assert Topic.HEARTBEAT in mock_communication.mock_data.published_messages
        assert len(mock_communication.mock_data.published_messages[Topic.HEARTBEAT]) > 0

        # Verify heartbeat message contents
        heartbeat_msg = mock_communication.mock_data.published_messages[
            Topic.HEARTBEAT
        ][0]
        assert heartbeat_msg.service_id == service.service_id
        assert heartbeat_msg.payload.service_type == service.service_type

    async def test_service_registration(
        self, initialized_service: BaseComponentService, mock_communication: MagicMock
    ) -> None:
        """
        Test that the service registers with the system controller.

        Verifies:
        1. The service sends a registration message to the controller
        2. The registration message contains the correct service information
        """
        service = await async_fixture(initialized_service)

        # Register the service
        await service.register()

        # Check that a registration message was published
        assert Topic.REGISTRATION in mock_communication.mock_data.published_messages

        # Verify registration message contents
        registration_msg = mock_communication.mock_data.published_messages[
            Topic.REGISTRATION
        ][0]
        assert registration_msg.service_id == service.service_id
        assert registration_msg.payload.service_type == service.service_type

    async def test_service_status_update(
        self, initialized_service: BaseComponentService, mock_communication: MagicMock
    ) -> None:
        """
        Test that the service updates its status correctly.

        Verifies:
        1. The service publishes status messages when state changes
        2. The status message contains the correct state and service information
        """
        service = await async_fixture(initialized_service)

        # Update the service status
        await service.set_state(ServiceState.READY)

        # Check that a status message was published
        assert Topic.STATUS in mock_communication.mock_data.published_messages

        # Verify status message contents
        status_msg = mock_communication.mock_data.published_messages[Topic.STATUS][0]
        assert status_msg.service_id == service.service_id
        assert status_msg.payload.service_type == service.service_type
        assert status_msg.payload.state == ServiceState.READY

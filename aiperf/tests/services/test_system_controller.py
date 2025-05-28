# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the system controller service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from aiperf.common.enums import (
    ServiceType,
    Topic,
)
from aiperf.common.service.base_service import BaseService
from aiperf.services.service_manager.multiprocess import (
    MultiProcessServiceManager,
)
from aiperf.services.system_controller.system_controller import SystemController
from aiperf.tests.base_test_controller_service import BaseTestControllerService
from aiperf.tests.base_test_service import async_fixture


class SystemControllerTestConfig(BaseModel):
    """Configuration model for system controller tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestSystemController(BaseTestControllerService):
    """
    Tests for the system controller service.

    This test class extends BaseTestControllerService to leverage common
    controller service tests while adding system controller specific tests.
    Tests include service lifecycle management, message handling, and coordination.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the class to test."""
        return SystemController

    @pytest.fixture
    def controller_config(self) -> SystemControllerTestConfig:
        """Return a test configuration for the system controller."""
        return SystemControllerTestConfig()

    async def test_controller_subscriptions(
        self, initialized_service: SystemController, mock_communication: MagicMock
    ) -> None:
        """Verifies the controller sets up subscriptions to receive messages from components."""

        await async_fixture(initialized_service)

        # A SystemController should subscribe to registration, status, and heartbeat topics
        expected_topics = [Topic.REGISTRATION, Topic.STATUS, Topic.HEARTBEAT]

        for topic in expected_topics:
            assert topic in mock_communication.mock_data.subscriptions
            assert callable(mock_communication.mock_data.subscriptions[topic])

    @pytest.fixture(autouse=True)
    def service_manager_with_multiprocess(
        self, monkeypatch, service_config
    ) -> MultiProcessServiceManager:
        """
        Return a test service manager with multiprocess support.

        This fixture mocks the initialization methods to avoid actual process creation.

        Args:
            monkeypatch: Pytest monkeypatch fixture for patching functions

        Returns:
            A MultiProcessServiceManager instance configured for testing
        """
        # Create a proper async mock for the service methods
        async_mock = AsyncMock(return_value=None)

        monkeypatch.setattr(
            MultiProcessServiceManager, "wait_for_all_services_registration", async_mock
        )
        monkeypatch.setattr(
            MultiProcessServiceManager, "initialize_all_services", async_mock
        )
        monkeypatch.setattr(MultiProcessServiceManager, "stop_all_services", async_mock)
        monkeypatch.setattr(
            MultiProcessServiceManager, "wait_for_all_services_start", async_mock
        )

        multiprocess_manager = MultiProcessServiceManager(
            required_service_types=[ServiceType.TEST],
            config=service_config,
        )

        return multiprocess_manager

    async def test_service_run_does_start(
        self, initialized_service: SystemController
    ) -> None:
        """
        Test that the service run method starts the service (added by BaseControllerService using @on_run).
        """
        service = await async_fixture(initialized_service)

        with (
            patch(
                "aiperf.services.system_controller.system_controller.SystemController._forever_loop",
                return_value=None,
            ) as mock_forever_loop,
            patch(
                "aiperf.services.system_controller.system_controller.SystemController.start",
                return_value=None,
            ) as mock_start,
        ):
            await service.run_forever()
            mock_forever_loop.assert_called_once()
            mock_start.assert_called_once()

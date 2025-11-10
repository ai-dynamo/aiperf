# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf controller.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType
from aiperf.common.messages import CommandErrorResponse
from aiperf.common.models import ErrorDetails
from aiperf.controller.system_controller import SystemController


class MockTestException(Exception):
    """Mock test exception."""


@pytest.fixture
def mock_service_manager() -> AsyncMock:
    """Mock service manager."""
    mock_manager = AsyncMock()
    mock_manager.service_id_map = {"test_service_1": MagicMock()}
    return mock_manager


@pytest.fixture
def system_controller(
    service_config: ServiceConfig,
    user_config: UserConfig,
    mock_service_manager: AsyncMock,
) -> SystemController:
    """Create a SystemController instance with mocked dependencies."""
    with (
        patch("aiperf.controller.system_controller.ServiceManagerFactory") as mock_factory,
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch("aiperf.controller.system_controller.AIPerfUIFactory") as mock_ui_factory,
        patch("aiperf.common.factories.CommunicationFactory") as mock_comm_factory,
    ):  # fmt: skip
        mock_factory.create_instance.return_value = mock_service_manager
        mock_proxy.return_value = AsyncMock()
        mock_ui_factory.create_instance.return_value = AsyncMock()

        # Mock the communication factory to return a mock communication object
        mock_comm = AsyncMock()
        mock_comm_factory.get_or_create_instance.return_value = mock_comm

        controller = SystemController(
            user_config=user_config,
            service_config=service_config,
            service_id="test_controller",
        )
        # Mock the stop method to avoid actual shutdown
        controller.stop = AsyncMock()
        return controller


@pytest.fixture
def mock_exception() -> MockTestException:
    """Mock the exception."""
    return MockTestException("Test error")


@pytest.fixture
def error_details(mock_exception: MockTestException) -> ErrorDetails:
    """Mock the error details."""
    return ErrorDetails.from_exception(mock_exception)


@pytest.fixture
def error_response(error_details: ErrorDetails) -> CommandErrorResponse:
    """Mock the command responses."""
    return CommandErrorResponse(
        service_id="test_service_1",
        command=CommandType.PROFILE_CONFIGURE,
        command_id="test_command_id",
        error=error_details,
    )

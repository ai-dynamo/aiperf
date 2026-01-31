# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    mock_ui = AsyncMock()
    mock_comm = AsyncMock()

    def mock_get_class(protocol, name):
        if protocol == "service_manager":
            return lambda **kwargs: mock_service_manager
        if protocol == "ui":
            return lambda **kwargs: mock_ui
        if protocol == "communication":
            return lambda **kwargs: mock_comm
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.controller.system_controller.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            side_effect=mock_get_class,
        ),
    ):  # fmt: skip
        mock_proxy.return_value = AsyncMock()

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

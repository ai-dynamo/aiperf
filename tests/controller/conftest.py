# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf controller.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, ServiceType
from aiperf.common.messages import CommandErrorResponse
from aiperf.common.models import ErrorDetails
from aiperf.controller.multiprocess_service_manager import AsyncSubprocessRunInfo
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


@pytest.fixture
def mock_subprocess_process() -> MagicMock:
    """Create a mock asyncio subprocess process."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.returncode = None
    mock_process.wait = AsyncMock(return_value=0)
    mock_process.terminate = MagicMock()
    mock_process.kill = MagicMock()
    mock_process.stdout = AsyncMock()
    mock_process.stderr = AsyncMock()
    return mock_process


@pytest.fixture
def create_subprocess_info():
    """Factory fixture to create AsyncSubprocessRunInfo instances."""

    def _create(
        service_type: ServiceType = ServiceType.DATASET_MANAGER,
        service_id: str = "test_service",
        process: MagicMock | None = None,
        user_config_file: Path | None = None,
        service_config_file: Path | None = None,
    ) -> AsyncSubprocessRunInfo:
        if process is None:
            return AsyncSubprocessRunInfo(
                service_type=service_type,
                service_id=service_id,
                user_config_file=user_config_file,
                service_config_file=service_config_file,
            )
        return AsyncSubprocessRunInfo.model_construct(
            process=process,
            service_type=service_type,
            service_id=service_id,
            user_config_file=user_config_file,
            service_config_file=service_config_file,
        )

    return _create

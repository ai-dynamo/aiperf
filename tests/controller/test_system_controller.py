# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CommandType
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.messages.command_messages import CommandErrorResponse
from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.controller.system_controller import SystemController
from tests.controller.conftest import MockTestException


def assert_exit_error(
    system_controller: SystemController,
    expected_error_or_exception: ErrorDetails | LifecycleOperationError,
    operation: str,
    service_id: str | None,
) -> None:
    """Assert that an exit error was recorded with the proper details."""
    assert len(system_controller._exit_errors) == 1
    exit_error = system_controller._exit_errors[0]
    assert isinstance(exit_error, ExitErrorInfo)

    # Handle both ErrorDetails objects and LifecycleOperationError objects
    if isinstance(expected_error_or_exception, ErrorDetails):
        expected_error_details = expected_error_or_exception
    else:
        expected_error_details = ErrorDetails.from_exception(
            expected_error_or_exception
        )

    assert exit_error.error_details == expected_error_details
    assert exit_error.operation == operation
    assert exit_error.service_id == service_id


class TestSystemController:
    """Test SystemController."""

    @pytest.mark.asyncio
    async def test_system_controller_no_error_on_initialize_success(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """Test that SystemController does not exit when initialize succeeds."""
        mock_service_manager.initialize.return_value = None
        await system_controller._initialize_system_controller()
        # Verify that no exit errors were recorded
        assert len(system_controller._exit_errors) == 0

    @pytest.mark.asyncio
    async def test_system_controller_no_error_on_start_success(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """Test that SystemController does not exit when start services succeeds."""
        mock_service_manager.start.return_value = None
        mock_service_manager.wait_for_all_services_registration.return_value = None
        system_controller._start_profiling_all_services = AsyncMock(return_value=None)
        system_controller._profile_configure_all_services = AsyncMock(return_value=None)

        await system_controller._start_services()
        # Verify that no exit errors were recorded
        assert len(system_controller._exit_errors) == 0

        assert mock_service_manager.start.called
        assert mock_service_manager.wait_for_all_services_registration.called
        assert system_controller._start_profiling_all_services.called
        assert system_controller._profile_configure_all_services.called


class TestSystemControllerExitScenarios:
    """Test exit scenarios for the SystemController."""

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_profile_configure_error_response(
        self,
        system_controller: SystemController,
        mock_exception: MockTestException,
        error_response: CommandErrorResponse,
    ):
        """Test that SystemController exits when receiving a CommandErrorResponse for profile_configure."""
        error_responses = [
            error_response.model_copy(
                deep=True, update={"command": CommandType.PROFILE_CONFIGURE}
            )
        ]
        # Mock the command responses
        system_controller.send_command_and_wait_for_all_responses = AsyncMock(
            return_value=error_responses
        )

        with pytest.raises(
            LifecycleOperationError,
            match="Failed to perform operation 'Configure Profiling'",
        ):
            await system_controller._profile_configure_all_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            error_response.error,
            "Configure Profiling",
            error_responses[0].service_id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_profile_start_error_response(
        self,
        system_controller: SystemController,
        mock_exception: MockTestException,
        error_response: CommandErrorResponse,
    ):
        """Test that SystemController exits when receiving a CommandErrorResponse for profile_start."""
        error_responses = [
            error_response.model_copy(
                deep=True, update={"command": CommandType.PROFILE_START}
            )
        ]
        # Mock the command responses
        system_controller.send_command_and_wait_for_all_responses = AsyncMock(
            return_value=error_responses
        )

        with pytest.raises(
            LifecycleOperationError,
            match="Failed to perform operation 'Start Profiling'",
        ):
            await system_controller._start_profiling_all_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            error_response.error,
            "Start Profiling",
            error_responses[0].service_id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_service_manager_initialize_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when the service manager initialize fails."""
        mock_service_manager.initialize.side_effect = mock_exception
        with pytest.raises(LifecycleOperationError, match=str(mock_exception)):
            await system_controller._initialize_system_controller()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            mock_exception,
            "Initialize Service Manager",
            system_controller.id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_service_manager_start_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when the service manager start fails."""
        mock_service_manager.start.side_effect = LifecycleOperationError(
            operation="Start Service",
            original_exception=mock_exception,
            lifecycle_id=system_controller.id,
        )
        with pytest.raises(LifecycleOperationError, match="Test error"):
            await system_controller._start_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            LifecycleOperationError(
                operation="Start Service",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            ),
            "Start Service Manager",
            system_controller.id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_wait_for_all_services_registration_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when the service manager wait_for_all_services_registration fails."""
        mock_service_manager.start.return_value = None
        mock_service_manager.wait_for_all_services_registration.side_effect = (
            LifecycleOperationError(
                operation="Register Service",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            )
        )
        with pytest.raises(LifecycleOperationError, match="Test error"):
            await system_controller._start_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            LifecycleOperationError(
                operation="Register Service",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            ),
            "Register Services",
            system_controller.id,
        )

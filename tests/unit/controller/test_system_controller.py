# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.messages.command_messages import CommandErrorResponse
from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.controller.system_controller import SystemController
from tests.unit.controller.conftest import MockTestException


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
        # Mock the command responses (using fail-fast method)
        system_controller.send_command_and_wait_until_first_error = AsyncMock(
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
        # Mock the command responses (using fail-fast method)
        system_controller.send_command_and_wait_until_first_error = AsyncMock(
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


# =============================================================================
# Signal Handling Tests (Two-Stage Ctrl+C)
# =============================================================================


class TestSignalHandling:
    """Tests for two-stage Ctrl+C signal handling."""

    @pytest.mark.asyncio
    async def test_first_signal_calls_cancel_profiling(
        self, system_controller: SystemController
    ):
        """First Ctrl+C calls _cancel_profiling for graceful shutdown."""
        system_controller._cancel_profiling = AsyncMock()
        system_controller._kill = AsyncMock()

        # First signal - should trigger graceful cancel
        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._cancel_profiling.assert_called_once()
        system_controller._kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_signal_calls_kill(self, system_controller: SystemController):
        """Second Ctrl+C calls _kill for immediate termination."""
        system_controller._cancel_profiling = AsyncMock()
        system_controller._kill = AsyncMock()
        system_controller._was_cancelled = (
            True  # Simulate first Ctrl+C already happened
        )

        # Second signal - should trigger force quit
        with patch.object(system_controller, "_print_force_quit_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._kill.assert_called_once()
        system_controller._cancel_profiling.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_signal_sets_was_cancelled_flag(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """First Ctrl+C sets _was_cancelled flag via _cancel_profiling."""
        # Mock the command response
        system_controller.send_command_and_wait_for_all_responses = AsyncMock(
            return_value=[]
        )
        system_controller.stop = AsyncMock()  # Prevent actual stop

        assert system_controller._was_cancelled is False

        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        assert system_controller._was_cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_profiling_sends_profile_cancel_command(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """_cancel_profiling sends ProfileCancelCommand to all services."""
        system_controller.send_command_and_wait_for_all_responses = AsyncMock(
            return_value=[]
        )
        system_controller.stop = AsyncMock()

        await system_controller._cancel_profiling()

        # Verify ProfileCancelCommand was sent
        system_controller.send_command_and_wait_for_all_responses.assert_called_once()
        call_args = system_controller.send_command_and_wait_for_all_responses.call_args
        assert call_args[0][0].command == CommandType.PROFILE_CANCEL

    def test_print_cancel_warning_uses_console(
        self, system_controller: SystemController
    ):
        """_print_cancel_warning prints to console."""
        with patch("aiperf.controller.system_controller.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            system_controller._print_cancel_warning()

            # Should have printed something
            assert mock_console.print.call_count >= 2  # Panel and newlines
            mock_console.file.flush.assert_called_once()

    def test_print_force_quit_warning_uses_console(
        self, system_controller: SystemController
    ):
        """_print_force_quit_warning prints to console."""
        with patch("aiperf.controller.system_controller.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            system_controller._print_force_quit_warning()

            # Should have printed something
            assert mock_console.print.call_count >= 2  # Panel and newlines
            mock_console.file.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_signals_go_graceful_then_force(
        self, system_controller: SystemController
    ):
        """Sequential signals: first graceful cancel, second force quit."""

        # Mock _cancel_profiling to set _was_cancelled flag (mimicking real behavior)
        async def cancel_side_effect():
            system_controller._was_cancelled = True

        system_controller._cancel_profiling = AsyncMock(side_effect=cancel_side_effect)
        system_controller._kill = AsyncMock()

        # First signal
        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        assert system_controller._was_cancelled is True
        system_controller._cancel_profiling.assert_called_once()
        system_controller._kill.assert_not_called()

        # Second signal
        with patch.object(system_controller, "_print_force_quit_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._kill.assert_called_once()


class TestSSLVerificationWarning:
    """Test SSL verification warning in SystemController."""

    @pytest.mark.asyncio
    async def test_warning_logged_when_ssl_verify_disabled(
        self, system_controller: SystemController, monkeypatch
    ):
        """Test that a warning is logged when SSL verification is disabled."""
        monkeypatch.setattr(Environment.HTTP, "SSL_VERIFY", False)

        system_controller.send_command_and_wait_until_first_error = AsyncMock(
            return_value=[]
        )

        with patch.object(system_controller, "warning") as mock_warning:
            await system_controller._profile_configure_all_services()

            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "SSL certificate verification is DISABLED" in warning_message

    @pytest.mark.asyncio
    async def test_no_warning_logged_when_ssl_verify_enabled(
        self, system_controller: SystemController, monkeypatch
    ):
        """Test that no warning is logged when SSL verification is enabled."""
        monkeypatch.setattr(Environment.HTTP, "SSL_VERIFY", True)

        system_controller.send_command_and_wait_until_first_error = AsyncMock(
            return_value=[]
        )

        with patch.object(system_controller, "warning") as mock_warning:
            await system_controller._profile_configure_all_services()

            mock_warning.assert_not_called()

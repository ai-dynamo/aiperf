# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock, patch

import pytest

from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ExitErrorInfo


class TestAIPerfLifecycleContextManager:
    """Test suite for the try_operation_or_stop context manager in AIPerfLifecycleMixin."""

    @pytest.fixture
    def lifecycle_component(self):
        """Create a minimal lifecycle component for testing."""
        component = AIPerfLifecycleMixin()
        component._exit_errors = []
        return component

    @pytest.mark.asyncio
    async def test_try_operation_or_stop_success(self, lifecycle_component):
        """Test that the context manager works correctly when no exception occurs."""
        operation_executed = False

        async with lifecycle_component.try_operation_or_stop("Test Operation"):
            operation_executed = True
            await asyncio.sleep(0.001)

        assert operation_executed
        assert len(lifecycle_component._exit_errors) == 0

    @pytest.mark.parametrize(
        "exception_class,message",
        [
            (ValueError, "Something went wrong"),
            (RuntimeError, "Runtime error occurred"),
            (ConnectionError, "Connection failed"),
            (TypeError, "Type error"),
        ],
    )
    @pytest.mark.asyncio
    async def test_try_operation_or_stop_with_exceptions(
        self, lifecycle_component, exception_class, message
    ):
        """Test that the context manager properly handles various exceptions."""
        operation_name = "Test Operation"
        original_exception = exception_class(message)

        with patch.object(lifecycle_component, "error") as mock_error:
            with pytest.raises(LifecycleOperationError) as exc_info:
                async with lifecycle_component.try_operation_or_stop(operation_name):
                    raise original_exception

            # Verify the raised exception is properly wrapped
            raised_exception = exc_info.value
            assert isinstance(raised_exception, LifecycleOperationError)
            assert raised_exception.operation == operation_name
            assert raised_exception.original_exception == original_exception
            assert raised_exception.lifecycle_id == lifecycle_component.id
            assert raised_exception.__cause__ == original_exception

            # Verify error logging was called
            mock_error.assert_called_once_with(
                f"Failed to {operation_name.lower()}: {original_exception}"
            )

            # Verify exit error was added
            assert len(lifecycle_component._exit_errors) == 1

    @pytest.mark.parametrize(
        "operation_name",
        ["UPPERCASE", "MixedCase", "lowercase", "Multi Word Operation"],
    )
    @pytest.mark.asyncio
    async def test_operation_name_case_handling(
        self, lifecycle_component, operation_name
    ):
        """Test that operation names are properly lowercased in error messages."""
        with patch.object(lifecycle_component, "error") as mock_error:
            with pytest.raises(LifecycleOperationError):
                async with lifecycle_component.try_operation_or_stop(operation_name):
                    raise Exception("Test error")

            expected_message = f"Failed to {operation_name.lower()}: Test error"
            mock_error.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_multiple_context_manager_uses(self, lifecycle_component):
        """Test that the context manager can be used multiple times."""
        # Two successful operations
        async with lifecycle_component.try_operation_or_stop("First Operation"):
            pass
        async with lifecycle_component.try_operation_or_stop("Second Operation"):
            pass

        # One failed operation
        with pytest.raises(LifecycleOperationError):
            async with lifecycle_component.try_operation_or_stop("Failed Operation"):
                raise Exception("Test failure")

        # Only one exit error from the failed operation
        assert len(lifecycle_component._exit_errors) == 1

    @pytest.mark.parametrize(
        "failing_step,expected_operation",
        [
            ("initialize", "Initialize"),
            ("start", "Start"),
        ],
    )
    @pytest.mark.asyncio
    async def test_initialize_and_start_failures(
        self, lifecycle_component, failing_step, expected_operation
    ):
        """Test that initialize_and_start handles failures in different steps."""
        test_exception = RuntimeError(f"{failing_step.title()} failed")

        with (
            patch.object(lifecycle_component, "initialize") as mock_init,
            patch.object(lifecycle_component, "start") as mock_start,
            patch.object(lifecycle_component, "error") as mock_error,
        ):
            # Set up the failure
            if failing_step == "initialize":
                mock_init.side_effect = test_exception
            else:
                mock_start.side_effect = test_exception

            with pytest.raises(LifecycleOperationError) as exc_info:
                await lifecycle_component.initialize_and_start()

            # Verify proper error handling
            raised_exception = exc_info.value
            assert raised_exception.operation == expected_operation
            assert raised_exception.original_exception == test_exception

            # Verify error logging
            expected_log = f"Failed to {expected_operation.lower()}: {test_exception}"
            mock_error.assert_called_once_with(expected_log)

    @pytest.mark.asyncio
    async def test_initialize_and_start_success(self, lifecycle_component):
        """Test that initialize_and_start works correctly when both steps succeed."""
        with (
            patch.object(lifecycle_component, "initialize") as mock_init,
            patch.object(lifecycle_component, "start") as mock_start,
        ):
            await lifecycle_component.initialize_and_start()

            mock_init.assert_called_once()
            mock_start.assert_called_once()
            assert len(lifecycle_component._exit_errors) == 0

    @pytest.mark.asyncio
    async def test_exit_error_info_creation(self, lifecycle_component):
        """Test that ExitErrorInfo is properly created from LifecycleOperationError."""
        with patch(
            "aiperf.common.models.ExitErrorInfo.from_lifecycle_operation_error"
        ) as mock_from_error:
            mock_exit_error = Mock(spec=ExitErrorInfo)
            mock_from_error.return_value = mock_exit_error

            with pytest.raises(LifecycleOperationError):
                async with lifecycle_component.try_operation_or_stop("Test Operation"):
                    raise ValueError("Test error")

            # Verify ExitErrorInfo.from_lifecycle_operation_error was called
            mock_from_error.assert_called_once()
            called_error = mock_from_error.call_args[0][0]
            assert isinstance(called_error, LifecycleOperationError)
            assert mock_exit_error in lifecycle_component._exit_errors

    @pytest.mark.asyncio
    async def test_lifecycle_operation_error_triggers_stop_via_fail(
        self, lifecycle_component
    ):
        """Test that LifecycleOperationError from context manager triggers stop() when caught by lifecycle system."""
        # Simulate the lifecycle system catching the LifecycleOperationError and calling _fail
        original_exception = RuntimeError("Simulated operation failure")

        # First raise the LifecycleOperationError from the context manager
        lifecycle_operation_error = None
        try:
            async with lifecycle_component.try_operation_or_stop("Test Operation"):
                raise original_exception
        except LifecycleOperationError as e:
            lifecycle_operation_error = e

        # Verify we got the expected LifecycleOperationError
        assert lifecycle_operation_error is not None
        assert lifecycle_operation_error.operation == "Test Operation"
        assert lifecycle_operation_error.original_exception == original_exception

        # Now simulate the lifecycle system catching this error and calling _fail
        # This simulates what happens in _execute_state_transition when an exception occurs
        with (
            patch.object(lifecycle_component, "stop") as mock_stop,
            patch.object(lifecycle_component, "_set_state") as mock_set_state,
        ):
            # Set a non-stopping state to ensure stop() gets called
            await lifecycle_component._set_state("running")

            with pytest.raises(asyncio.CancelledError):
                await lifecycle_component._fail(lifecycle_operation_error)

            # Verify that stop() was called as part of the _fail process
            mock_stop.assert_called_once()

            # Verify that the state was set to FAILED
            mock_set_state.assert_called_with("failed")

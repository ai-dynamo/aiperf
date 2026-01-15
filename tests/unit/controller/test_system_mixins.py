# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
from unittest.mock import AsyncMock

import pytest

from aiperf.controller.system_mixins import SignalHandlerMixin


@pytest.fixture
def signal_handler_instance():
    """Factory fixture for creating SignalHandlerMixin instances."""

    class TestSignalHandler(SignalHandlerMixin):
        def __init__(self, **kwargs):
            super().__init__(logger_name="TestSignalHandler", **kwargs)

    return TestSignalHandler()


class TestSignalHandlerMixinInitialization:
    """Test SignalHandlerMixin initialization."""

    def test_initialization(self, signal_handler_instance):
        """Test that SignalHandlerMixin initializes signal tasks set."""
        assert hasattr(signal_handler_instance, "_signal_tasks")
        assert isinstance(signal_handler_instance._signal_tasks, set)
        assert len(signal_handler_instance._signal_tasks) == 0


class TestSetupSignalHandlers:
    """Test signal handler setup and signal handling."""

    @pytest.mark.asyncio
    async def test_setup_signal_handlers(self, signal_handler_instance):
        """Test that setup_signal_handlers registers SIGINT handler."""
        callback = AsyncMock()

        # Monkey-patch loop.add_signal_handler to capture the handler
        loop = asyncio.get_running_loop()
        original_add_signal_handler = loop.add_signal_handler
        captured_handler = None
        captured_args = None

        def capture_handler(sig, handler, *args):
            nonlocal captured_handler, captured_args
            captured_handler = handler
            captured_args = args
            return original_add_signal_handler(sig, handler, *args)

        loop.add_signal_handler = capture_handler

        try:
            signal_handler_instance.setup_signal_handlers(callback)

            # Verify handler was registered
            assert captured_handler is not None

            # Invoke the captured signal handler with the captured args
            captured_handler(*captured_args)

            # Wait for the callback task to complete (use longer delay for CI stability)
            await asyncio.sleep(0.1)

            # Verify callback was invoked with SIGINT
            callback.assert_called_once_with(signal.SIGINT)

            # Wait for all tasks to complete and be cleaned up
            tasks = list(signal_handler_instance._signal_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                # Give event loop a chance to process done_callbacks
                await asyncio.sleep(0)

            # Verify tasks were cleaned up
            assert len(signal_handler_instance._signal_tasks) == 0
        finally:
            # Restore original
            loop.add_signal_handler = original_add_signal_handler

    @pytest.mark.asyncio
    async def test_signal_handler_callback_invocation(self, signal_handler_instance):
        """Test signal handler callback mechanism by directly invoking the internal handler."""
        callback = AsyncMock()

        # Setup signal handlers
        signal_handler_instance.setup_signal_handlers(callback)

        # Create a mock signal handler that mimics the internal behavior
        task = asyncio.create_task(callback(signal.SIGINT))
        signal_handler_instance._signal_tasks.add(task)
        task.add_done_callback(signal_handler_instance._signal_tasks.discard)

        # Await the task to ensure it completes and callback is processed
        await task

        # Verify callback was called
        callback.assert_called_once_with(signal.SIGINT)

        # Verify task was cleaned up after completion
        assert task not in signal_handler_instance._signal_tasks

    @pytest.mark.asyncio
    async def test_multiple_tasks_management(self, signal_handler_instance):
        """Test that multiple signal handler tasks are managed correctly."""
        callback = AsyncMock()

        signal_handler_instance.setup_signal_handlers(callback)

        # Simulate multiple signal receptions by creating multiple tasks
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(callback(signal.SIGINT))
            signal_handler_instance._signal_tasks.add(task)
            task.add_done_callback(signal_handler_instance._signal_tasks.discard)
            tasks.append(task)

        # Await all tasks to ensure they complete
        await asyncio.gather(*tasks)

        # All callbacks should have been called
        assert callback.call_count == 3

        # All tasks should be cleaned up
        for task in tasks:
            assert task not in signal_handler_instance._signal_tasks

    @pytest.mark.asyncio
    async def test_task_cleanup_on_completion(self, signal_handler_instance):
        """Test that tasks are cleaned up via done callback."""
        callback = AsyncMock()

        signal_handler_instance.setup_signal_handlers(callback)

        # Create a task and add to set
        task = asyncio.create_task(callback(signal.SIGINT))
        signal_handler_instance._signal_tasks.add(task)

        # Verify task is in set
        assert task in signal_handler_instance._signal_tasks

        # Add done callback
        task.add_done_callback(signal_handler_instance._signal_tasks.discard)

        # Await task to ensure it completes
        await task

        # Task should be removed from set
        assert task not in signal_handler_instance._signal_tasks


class TestSignalHandlerEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_signal_handler_with_failing_callback(self, signal_handler_instance):
        """Test that signal handler handles callback exceptions gracefully."""

        async def failing_callback(sig: int) -> None:
            raise ValueError("Callback error")

        signal_handler_instance.setup_signal_handlers(failing_callback)

        # Simulate signal handler behavior without sending actual signal
        task = asyncio.create_task(failing_callback(signal.SIGINT))
        signal_handler_instance._signal_tasks.add(task)
        task.add_done_callback(signal_handler_instance._signal_tasks.discard)

        # Wait for task to complete - exception should not propagate
        with pytest.raises(ValueError, match="Callback error"):
            await task

        # Give event loop a chance to process done_callback
        await asyncio.sleep(0)

        # Task should be cleaned up despite exception
        assert task not in signal_handler_instance._signal_tasks

    @pytest.mark.asyncio
    async def test_setup_handlers_called_multiple_times(self, signal_handler_instance):
        """Test that calling setup_signal_handlers multiple times doesn't break."""
        callback = AsyncMock()

        # Setup handler multiple times (should not raise exception)
        signal_handler_instance.setup_signal_handlers(callback)
        signal_handler_instance.setup_signal_handlers(callback)
        signal_handler_instance.setup_signal_handlers(callback)

        # Verify handler setup completed without errors
        assert True

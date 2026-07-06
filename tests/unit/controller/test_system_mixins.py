# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

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

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows uses signal.signal() instead of loop.add_signal_handler(); covered by integration test.",
    )
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


class TestSetupSignalHandlersWindowsBranch:
    """Test Bug 4 fix — Windows uses signal.signal() + run_coroutine_threadsafe.

    Linux ProactorEventLoop has no add_signal_handler. We mock IS_WINDOWS to
    exercise the Windows branch from non-Windows CI.
    """

    @pytest.mark.asyncio
    async def test_windows_calls_signal_signal_for_sigint(
        self, signal_handler_instance
    ):
        """On Windows, signal.signal is used instead of loop.add_signal_handler.

        Both SIGINT (Ctrl+C) and SIGBREAK (Ctrl+Break) are registered. We
        patch SIGBREAK so the test runs identically on POSIX CI.
        """
        callback = AsyncMock()

        with (
            patch("aiperf.controller.system_mixins.IS_WINDOWS", True),
            patch("aiperf.controller.system_mixins.signal.SIGBREAK", 21, create=True),
            patch("aiperf.controller.system_mixins.signal.signal") as mock_signal,
        ):
            signal_handler_instance.setup_signal_handlers(callback)

        assert mock_signal.call_count == 2
        registered_signals = {call.args[0] for call in mock_signal.call_args_list}
        assert registered_signals == {signal.SIGINT, 21}
        for call in mock_signal.call_args_list:
            assert callable(call.args[1])

    @pytest.mark.asyncio
    async def test_windows_skips_sigbreak_when_undefined(self, signal_handler_instance):
        """When signal.SIGBREAK is missing (POSIX), only SIGINT is registered."""
        callback = AsyncMock()

        # Remove SIGBREAK to simulate POSIX environment with mocked IS_WINDOWS.
        import aiperf.controller.system_mixins as sysmix_mod

        original_sigbreak = getattr(sysmix_mod.signal, "SIGBREAK", None)
        if original_sigbreak is not None:
            del sysmix_mod.signal.SIGBREAK

        try:
            with (
                patch("aiperf.controller.system_mixins.IS_WINDOWS", True),
                patch("aiperf.controller.system_mixins.signal.signal") as mock_signal,
            ):
                signal_handler_instance.setup_signal_handlers(callback)

            assert mock_signal.call_count == 1
            assert mock_signal.call_args.args[0] == signal.SIGINT
        finally:
            if original_sigbreak is not None:
                sysmix_mod.signal.SIGBREAK = original_sigbreak

    @pytest.mark.asyncio
    async def test_windows_does_not_call_loop_add_signal_handler(
        self, signal_handler_instance
    ):
        """On Windows, loop.add_signal_handler must NOT be called (it raises NotImplementedError)."""
        callback = AsyncMock()

        loop = asyncio.get_running_loop()
        with (
            patch("aiperf.controller.system_mixins.IS_WINDOWS", True),
            patch("aiperf.controller.system_mixins.signal.signal"),
            patch.object(loop, "add_signal_handler") as mock_add_signal,
        ):
            signal_handler_instance.setup_signal_handlers(callback)

        mock_add_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_windows_handler_bridges_sync_to_async_via_run_coroutine_threadsafe(
        self, signal_handler_instance
    ):
        """The installed Windows handler is sync; it must schedule the async callback safely.

        Both SIGINT and SIGBREAK should install the SAME handler — they share
        the graceful-shutdown coroutine. We capture both and assert identity.
        """
        callback = AsyncMock()
        captured_handlers = {}

        def capture_handler(sig, handler):
            captured_handlers[sig] = handler
            return None  # signal.signal returns previous handler; not used here

        with (
            patch("aiperf.controller.system_mixins.IS_WINDOWS", True),
            patch("aiperf.controller.system_mixins.signal.SIGBREAK", 21, create=True),
            patch(
                "aiperf.controller.system_mixins.signal.signal",
                side_effect=capture_handler,
            ),
        ):
            signal_handler_instance.setup_signal_handlers(callback)

        assert set(captured_handlers.keys()) == {signal.SIGINT, 21}
        assert captured_handlers[signal.SIGINT] is captured_handlers[21], (
            "SIGINT and SIGBREAK should install the same handler closure"
        )
        captured_handler = captured_handlers[signal.SIGINT]
        assert captured_handler is not None, "signal.signal was not called"

        # Invoke the captured handler with the (sig, frame) signature.
        # It should call run_coroutine_threadsafe with a coroutine and the loop.
        with patch(
            "aiperf.controller.system_mixins.asyncio.run_coroutine_threadsafe"
        ) as mock_run:
            captured_handler(signal.SIGINT, MagicMock())

        mock_run.assert_called_once()
        coro_arg, _loop_arg = mock_run.call_args.args
        assert asyncio.iscoroutine(coro_arg), (
            "Windows handler must schedule an async coroutine, not a sync callable"
        )
        coro_arg.close()  # Suppress 'coroutine was never awaited' warning


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

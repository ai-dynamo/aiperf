# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import concurrent.futures
import signal
from collections.abc import Callable, Coroutine

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.mixins import AIPerfLoggerMixin


class SignalHandlerMixin(AIPerfLoggerMixin):
    """Mixin for services that need to handle system signals."""

    def __init__(self, **kwargs) -> None:
        # Set to store signal handler tasks to prevent them from being garbage collected
        self._signal_tasks = set()
        super().__init__(**kwargs)

    def setup_signal_handlers(self, callback: Callable[[int], Coroutine]) -> None:
        """Set up signal handler for the SIGINT signal to trigger graceful shutdown.

        Args:
            callback: The callback to call when a signal is received
        """
        loop = asyncio.get_running_loop()
        self.debug(f"Setting up SIGINT handler on loop {loop}")

        # Windows ProactorEventLoop does not implement add_signal_handler.
        # Fall back to signal.signal(), which Windows supports for SIGINT.
        # The handler is dispatched on the main thread and re-enters the loop
        # via run_coroutine_threadsafe to invoke the async callback. The
        # scheduled task is held by the loop, so no _signal_tasks tracking
        # is needed on Windows.
        #
        # Limitation: on Windows the handler runs in main-thread Python-level
        # interrupt context. CPython on Windows cannot interrupt a blocking C
        # extension call (e.g. zmq.poll, aiohttp blocking I/O) until that
        # call returns control to Python, so the user may observe sub-second
        # Ctrl+C lag during heavy I/O. This is a CPython-on-Windows constraint,
        # not a bug in the handler.
        if IS_WINDOWS:

            def windows_signal_handler(sig: int, _frame: object) -> None:
                self.warning(f"Signal {sig} received, initiating graceful shutdown")
                fut = asyncio.run_coroutine_threadsafe(callback(sig), loop)

                def _on_callback_done(f: "concurrent.futures.Future") -> None:
                    # ``run_coroutine_threadsafe`` returns a future whose
                    # result is never awaited. If the wrapped coroutine
                    # raises, the exception is otherwise silently swallowed
                    # (Python emits "exception never retrieved" on GC, into
                    # stderr that may already be redirected to NUL by
                    # ``_redirect_stdio_to_devnull``). Surface failures
                    # explicitly via the service logger. Skip CancelledError
                    # — that's the normal loop-teardown path during shutdown,
                    # not a real failure to surface.
                    exc = f.exception()
                    if exc is not None and not isinstance(exc, asyncio.CancelledError):
                        self.error(f"Signal-handler callback raised: {exc!r}")

                fut.add_done_callback(_on_callback_done)

            signal.signal(signal.SIGINT, windows_signal_handler)
            # SIGBREAK (Ctrl+Break / CTRL_BREAK_EVENT) is distinct from SIGINT
            # on Windows. Some scripted runs, CI runners, and remote consoles
            # send CTRL_BREAK_EVENT instead of CTRL_C_EVENT — without this
            # handler the process dies without graceful shutdown. Guarded
            # by getattr because signal.SIGBREAK is undefined on POSIX and
            # this branch is unit-tested on Linux/macOS via IS_WINDOWS mock.
            sigbreak = getattr(signal, "SIGBREAK", None)
            if sigbreak is not None:
                signal.signal(sigbreak, windows_signal_handler)
        else:

            def signal_handler(sig: int) -> None:
                self.warning(f"Signal {sig} received, initiating graceful shutdown")
                task = asyncio.create_task(callback(sig))
                self._signal_tasks.add(task)
                task.add_done_callback(self._signal_tasks.discard)

            loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT)
        self.debug("SIGINT handler installed successfully")

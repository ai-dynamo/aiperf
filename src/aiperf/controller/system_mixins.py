# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
from collections.abc import Callable, Coroutine

from aiperf.common.mixins import AIPerfLoggerMixin


class SignalHandlerMixin(AIPerfLoggerMixin):
    """Mixin for services that need to handle system signals."""

    def __init__(self, **kwargs) -> None:
        # Set to store signal handler tasks to prevent them from being garbage collected
        self._signal_tasks = set()
        super().__init__(**kwargs)

    def setup_signal_handlers(self, callback: Callable[[int], Coroutine]) -> None:
        """This method will set up signal handlers for the SIGTERM and SIGINT signals
        in order to trigger a graceful shutdown of the service.

        Args:
            callback: The callback to call when a signal is received
        """
        loop = asyncio.get_running_loop()

        def signal_handler(sig: int) -> None:
            task = asyncio.create_task(callback(sig))
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT)

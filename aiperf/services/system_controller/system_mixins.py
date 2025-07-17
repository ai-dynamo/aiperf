# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import signal
from collections.abc import Callable, Coroutine
from typing import Any


class SignalHandlerMixin:
    """Mixin for services that need to handle system signals."""

    def __init__(self, *args, **kwargs) -> None:
        # Set to store signal handler tasks to prevent them from being garbage collected
        self._signal_tasks = set()
        self.logger = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

    def setup_signal_handlers(
        self, callback: Callable[[int], Coroutine[Any, Any, None]]
    ) -> None:
        """This method will set up signal handlers for the SIGTERM and SIGINT signals
        in order to trigger a graceful shutdown of the service.

        Args:
            callback: The callback to call when a signal is received
        """
        loop = asyncio.get_running_loop()

        def signal_handler(sig: int) -> None:
            # Create a task and store it so it doesn't get garbage collected
            task = asyncio.create_task(callback(sig))

            # Store the task somewhere to prevent it from being garbage collected
            # before it completes
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

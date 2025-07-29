# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
from collections.abc import Callable

from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin


class SignalHandlerMixin(AIPerfLoggerMixin):
    """Mixin for services that need to handle system signals."""

    # TODO: Need to address this code, as the shutdown logic is not working as expected.

    def setup_signal_handlers(self, callback: Callable[[int], None]) -> None:
        """This method will set up signal handlers for the SIGTERM and SIGINT signals
        in order to trigger a graceful shutdown of the service.

        Args:
            callback: The callback to call when a signal is received
        """
        loop = asyncio.get_running_loop()

        def signal_handler(sig: int) -> None:
            callback(sig)

        loop.add_signal_handler(signal.SIGINT, lambda: signal_handler(signal.SIGINT))

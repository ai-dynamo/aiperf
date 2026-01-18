# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event loop health monitoring utility class.

Provides a background task that monitors event loop responsiveness and logs
warnings when the event loop is blocked for longer than a configurable threshold.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin


class EventLoopMonitor(AIPerfLoggerMixin):
    """Utility class that monitors event loop health and logs warnings when blocked.

    This utility class adds a background task that periodically checks if the event loop
    is responsive by sleeping for a known interval and measuring actual elapsed time.
    If the delta exceeds the configured threshold, it indicates blocking operations.

    Configurable via Environment.SERVICE:
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED: Enable/disable monitoring (default: True)
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL: Sleep interval in seconds (default: 0.25)
    - AIPERF_SERVICE_EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS: Warning threshold in ms (default: 10)
    """

    def __init__(self, service_id: str, **kwargs) -> None:
        super().__init__(service_id=service_id, **kwargs)
        self._service_id = service_id
        self._event_loop_health_task = None
        self._stop_requested = False
        self._callback: Callable[[float], Awaitable] | None = None

    def set_callback(self, callback: Callable[[float], Awaitable]) -> None:
        """Set the callback to be called when the event loop is blocked."""
        self._callback = callback

    def start(self) -> None:
        """Start the event loop health task."""
        self._stop_requested = False
        if self._event_loop_health_task is None:
            self._event_loop_health_task = asyncio.create_task(
                self._monitor_event_loop()
            )

    def stop(self) -> None:
        """Stop the event loop health task."""
        if self._stop_requested:
            return
        self._stop_requested = True
        if self._event_loop_health_task is not None:
            self._event_loop_health_task.cancel()
        self._event_loop_health_task = None

    async def _monitor_event_loop(self) -> None:
        """Monitor event loop health and log warnings when latency exceeds threshold.

        This task detects blocked event loops by sleeping for a known interval
        and measuring actual elapsed time. If the delta exceeds the configured
        threshold, it indicates the event loop was blocked by other tasks.
        """
        if not Environment.SERVICE.EVENT_LOOP_HEALTH_ENABLED:
            return

        interval_sec = Environment.SERVICE.EVENT_LOOP_HEALTH_INTERVAL
        threshold_ns = (
            Environment.SERVICE.EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS * NANOS_PER_MILLIS
        )
        expected_ns = round(interval_sec * NANOS_PER_SECOND)

        while not self._stop_requested:
            start_perf_ns = time.perf_counter_ns()
            await asyncio.sleep(interval_sec)
            elapsed_ns = time.perf_counter_ns() - start_perf_ns
            delta_ns = elapsed_ns - expected_ns
            if self.is_trace_enabled:
                self.trace(
                    f"Event loop health check: expected {interval_sec * MILLIS_PER_SECOND:.1f}ms, actual {elapsed_ns / NANOS_PER_MILLIS:.2f}ms, delta {delta_ns / NANOS_PER_MILLIS:.2f}ms"
                )
            if delta_ns > threshold_ns:
                self.warning(
                    f"Event loop for {self._service_id} is taking too long to run. Overhead: {delta_ns / NANOS_PER_MILLIS:,.2f}ms"
                )
                if self._callback is not None:
                    await self._callback(delta_ns / NANOS_PER_MILLIS)

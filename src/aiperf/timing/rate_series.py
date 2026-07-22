# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuous request-rate series controller."""

from __future__ import annotations

import asyncio
import bisect
import logging
import time
from collections.abc import Callable

from aiperf.config.rate_series import RateSeriesConfig

logger = logging.getLogger(__name__)


class RateSeriesController:
    """Apply piecewise-linear request-rate updates until stopped."""

    def __init__(
        self,
        setter: Callable[[float], None],
        config: RateSeriesConfig,
        update_interval: float,
        start_delay: float = 0.0,
    ) -> None:
        self._setter = setter
        self._config = config
        self._update_interval = update_interval
        self._start_delay = start_delay
        self._times = [point.time_s for point in config.points]
        self._qps = [point.qps for point in config.points]
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Return True if the rate-series controller task is currently running."""
        return self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task:
        """Start request-rate series updates in a background task."""
        self._task = asyncio.create_task(self._run())
        return self._task

    def stop(self) -> None:
        """Stop rate-series updates early."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def value_at(self, elapsed_sec: float) -> float:
        """Return the interpolated request rate at elapsed phase time."""
        if elapsed_sec <= self._times[0]:
            return self._qps[0]
        if elapsed_sec >= self._times[-1]:
            return self._qps[-1]

        right = bisect.bisect_right(self._times, elapsed_sec)
        left = right - 1
        left_time = self._times[left]
        right_time = self._times[right]
        progress = (elapsed_sec - left_time) / (right_time - left_time)
        return self._qps[left] + (self._qps[right] - self._qps[left]) * progress

    async def _run(self) -> None:
        try:
            if self._start_delay > 0:
                await asyncio.sleep(self._start_delay)

            series_start = time.perf_counter()
            if not self._set_rate(self.value_at(0.0)):
                return

            while True:
                await asyncio.sleep(self._update_interval)
                elapsed = time.perf_counter() - series_start
                if elapsed >= self._times[-1]:
                    self._set_rate(self._qps[-1])
                    break
                if not self._set_rate(self.value_at(elapsed)):
                    return
        except asyncio.CancelledError:
            pass

    def _set_rate(self, rate: float) -> bool:
        try:
            self._setter(rate)
        except Exception:
            logger.exception("Request-rate series update failed for rate=%s", rate)
            return False
        return True

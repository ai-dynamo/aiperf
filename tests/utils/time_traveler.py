# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Time traveler for use in tests.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from pytest import approx

from aiperf.common.constants import NANOS_PER_SECOND
from tests.conftest import real_sleep


class TimeTraveler:
    def __init__(self):
        self._real_sleep = real_sleep
        self.original_time_ns = time.time_ns()
        self.original_perf_counter_ns = time.perf_counter_ns()
        self.original_monotonic_ns = time.monotonic_ns()
        self.current_time_ns = self.original_time_ns
        self.current_perf_counter_ns = self.original_perf_counter_ns
        self.current_monotonic_ns = self.original_monotonic_ns
        self.patches = []

    def advance_time(self, seconds: float):
        """Advance the time by the specified number of seconds."""
        self.current_time_ns += int(seconds * NANOS_PER_SECOND)
        self.current_perf_counter_ns += int(seconds * NANOS_PER_SECOND)
        self.current_monotonic_ns += int(seconds * NANOS_PER_SECOND)

    async def sleep(self, delay: float):
        """Mock asyncio.sleep that advances internal clock and yields control."""
        if delay > 0:
            self.advance_time(delay)
        # Important: Yield control to the event loop so that other tasks can run. If we don't do this,
        # the event loop will be starved and the test will hang.
        await self.yield_to_event_loop()

    def time_ns(self) -> int:
        return self.current_time_ns

    def time(self) -> float:
        return self.current_time_ns / NANOS_PER_SECOND

    def perf_counter(self) -> float:
        return self.current_perf_counter_ns / NANOS_PER_SECOND

    def perf_counter_ns(self) -> int:
        return self.current_perf_counter_ns

    def monotonic(self) -> float:
        return self.current_monotonic_ns / NANOS_PER_SECOND

    def monotonic_ns(self) -> int:
        return self.current_monotonic_ns

    def start_traveling(self):
        patches_config = {
            "asyncio.sleep": self.sleep,
            "time.time_ns": self.time_ns,
            "time.time": self.time,
            "time.perf_counter": self.perf_counter,
            "time.perf_counter_ns": self.perf_counter_ns,
            "time.monotonic": self.monotonic,
            "time.monotonic_ns": self.monotonic_ns,
        }

        for patch_target, side_effect in patches_config.items():
            self.patches.append(patch(patch_target, side_effect=side_effect))

        for patch_obj in self.patches:
            patch_obj.start()

    def stop_traveling(self):
        for patch_obj in self.patches:
            patch_obj.stop()
        self.patches.clear()

    async def real_sleep(self, seconds: float) -> None:
        await self._real_sleep(seconds)

    async def yield_to_event_loop(self) -> None:
        await self._real_sleep(0)

    @contextmanager
    def sleeps_for(
        self, expected_seconds: float, tolerance: float = 0.001
    ) -> Generator[None, Any, None]:
        """Assert that the code block sleeps for the expected duration.

        Args:
            expected_seconds: Expected sleep duration in seconds
            tolerance: Tolerance for pytest.approx comparison (default: 0.001 seconds)

        Usage:
            with time_traveler.sleeps_for(0.2):
                await some_async_operation()
        """
        start_time = self.perf_counter()
        yield
        actual_sleep_duration = self.perf_counter() - start_time
        assert actual_sleep_duration == approx(expected_seconds, abs=tolerance), (
            f"Expected code to sleep for {expected_seconds} seconds, got {actual_sleep_duration} seconds"
        )


@pytest.fixture
def time_traveler():
    """
    Provides the ability to travel through time by calling `advance_time` and `sleep`.

    The internal clock advances when `asyncio.sleep` is called, allowing tests to run
    quickly while maintaining proper timing relationships. The mocked `asyncio.sleep`
    calls real `asyncio.sleep(0)` to yield control to the event loop.
    """
    time_traveler = TimeTraveler()
    time_traveler.start_traveling()
    yield time_traveler
    time_traveler.stop_traveling()

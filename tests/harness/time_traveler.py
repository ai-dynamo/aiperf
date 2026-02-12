# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
TimeTraveler - Lightweight utility for virtual time testing.

Provides:
1. Patches time.* functions to return virtual time (synced with event loop)
2. Utilities for timing assertions (sleeps_for, etc.)
3. Easy access to virtual time values
"""

import asyncio
import time
from contextlib import contextmanager
from unittest.mock import patch

from aiperf.common.constants import NANOS_PER_SECOND

# Save real time functions before any patching happens
# These are captured at module import time, before pytest runs
_REAL_TIME_NS = time.time_ns
_REAL_TIME = time.time
_REAL_PERF_COUNTER_NS = time.perf_counter_ns
_REAL_PERF_COUNTER = time.perf_counter
_REAL_MONOTONIC_NS = time.monotonic_ns
_REAL_MONOTONIC = time.monotonic
_REAL_SLEEP = asyncio.sleep


class TimeTraveler:
    """
    Lightweight utility for virtual time testing.

    Patches time.* functions to return virtual time synced with the event loop.
    Looptime (enabled via pytest.mark.looptime) handles the fake event loop time.
    TimeTraveler makes the standard library's time functions return virtual time
    synced with the fake event loop.

    Example:
    ```python
        async def test_something(time_traveler):
            start = time_traveler.time()
            await asyncio.sleep(10.0)  # Instant in real time!
            elapsed = time_traveler.time() - start
            assert elapsed >= 10.0  # Virtual time advanced

        async def test_with_assertion(time_traveler):
            async with time_traveler.sleeps_for(5.0):
                await asyncio.sleep(5.0)  # Asserts exactly 5s virtual time
    ```
    """

    def __init__(self, patch_sleep: bool = True):
        """Initialize TimeTraveler by capturing the initial time offsets."""
        # Capture the "real" time at initialization
        # Virtual time will be: original_time + loop.time()
        self.original_time_ns = _REAL_TIME_NS()
        self.original_perf_counter_ns = _REAL_PERF_COUNTER_NS()
        self.original_monotonic_ns = _REAL_MONOTONIC_NS()
        self.offset_ns = 0
        self.patches = []
        self.patch_sleep = patch_sleep

    def _get_loop_time_ns(self) -> int:
        """Get event loop's virtual time in nanoseconds.

        When looptime is disabled (e.g. during fixture setup/teardown),
        we must NOT call loop.time() because BaseEventLoop.time() calls
        time.monotonic(), which we've patched to call loop.time() again,
        creating infinite recursion.
        """
        try:
            loop = asyncio.get_running_loop()
            if hasattr(loop, "looptime_on") and not loop.looptime_on:
                return 0
            return int(loop.time() * NANOS_PER_SECOND)
        except RuntimeError:
            # No running loop - return 0 (tests that aren't async)
            return 0

    # -------------------------------------------------------------------------
    # Virtual time accessors (these are patched into time.* functions)
    # -------------------------------------------------------------------------

    @property
    def current_time_ns(self) -> int:
        """Current virtual time_ns (wall-clock equivalent)."""
        return self.original_time_ns + self.offset_ns + self._get_loop_time_ns()

    @property
    def current_perf_counter_ns(self) -> int:
        """Current virtual perf_counter_ns (high-res timer equivalent)."""
        return self.original_perf_counter_ns + self.offset_ns + self._get_loop_time_ns()

    @property
    def current_monotonic_ns(self) -> int:
        """Current virtual monotonic_ns (monotonic timer equivalent)."""
        return self.original_monotonic_ns + self.offset_ns + self._get_loop_time_ns()

    def time_ns(self) -> int:
        """Get virtual time.time_ns() value."""
        return self.current_time_ns

    def time(self) -> float:
        """Get virtual time.time() value in seconds."""
        return self.current_time_ns / NANOS_PER_SECOND

    def perf_counter_ns(self) -> int:
        """Get virtual time.perf_counter_ns() value."""
        return self.current_perf_counter_ns

    def perf_counter(self) -> float:
        """Get virtual time.perf_counter() value in seconds."""
        return self.current_perf_counter_ns / NANOS_PER_SECOND

    def monotonic_ns(self) -> int:
        """Get virtual time.monotonic_ns() value."""
        return self.current_monotonic_ns

    def monotonic(self) -> float:
        """Get virtual time.monotonic() value in seconds."""
        return self.current_monotonic_ns / NANOS_PER_SECOND

    async def sleep(self, delay: float) -> None:
        """Get virtual asyncio.sleep() value."""
        self.offset_ns += int(delay * NANOS_PER_SECOND)
        # Make sure to yield to the event loop to allow other coroutines to run
        await self.real_sleep(0)

    def advance_time(self, delay: float) -> None:
        """Advance virtual time by the given delay."""
        self.offset_ns += int(delay * NANOS_PER_SECOND)

    def rewind_time(self, delay: float) -> None:
        """Rewind virtual time by the given delay."""
        self.offset_ns -= int(delay * NANOS_PER_SECOND)

    # -------------------------------------------------------------------------
    # Real time accessors (unpatched system time)
    # -------------------------------------------------------------------------

    def real_time_ns(self) -> int:
        """Get real wall-clock time in nanoseconds (unpatched)."""
        return _REAL_TIME_NS()

    def real_time(self) -> float:
        """Get real wall-clock time in seconds (unpatched)."""
        return _REAL_TIME()

    def real_perf_counter_ns(self) -> int:
        """Get real performance counter in nanoseconds (unpatched)."""
        return _REAL_PERF_COUNTER_NS()

    def real_perf_counter(self) -> float:
        """Get real performance counter in seconds (unpatched)."""
        return _REAL_PERF_COUNTER()

    def real_monotonic_ns(self) -> int:
        """Get real monotonic time in nanoseconds (unpatched)."""
        return _REAL_MONOTONIC_NS()

    def real_monotonic(self) -> float:
        """Get real monotonic time in seconds (unpatched)."""
        return _REAL_MONOTONIC()

    async def real_sleep(self, delay: float) -> None:
        """Get real asyncio.sleep() value."""
        return await _REAL_SLEEP(delay)

    # -------------------------------------------------------------------------
    # Timing assertion utilities
    # -------------------------------------------------------------------------

    @contextmanager
    def sleeps_for(self, expected_duration: float, tolerance: float = 0.001):
        """Assert that code block sleeps for exactly expected_duration (virtual time).
        This only include asyncio.sleep, and not other calls like asyncio.wait_for, etc.

        Args:
            expected_duration: Expected virtual time duration in seconds
            tolerance: Acceptable difference in seconds (default: 1ms)

        Raises:
            AssertionError: If actual duration doesn't match expected

        Example:
            async with time_traveler.sleeps_for(5.0):
                await asyncio.sleep(5.0)  # Pass

            async with time_traveler.sleeps_for(5.0):
                await asyncio.sleep(3.0)  # Fail: expected 5s, got 3s
        """
        yield
        sleep_duration = self.offset_ns / NANOS_PER_SECOND
        assert abs(sleep_duration - expected_duration) <= tolerance, (
            f"Expected to sleep for {expected_duration}s (±{tolerance}s), "
            f"but slept for {sleep_duration:.6f}s (diff: {abs(sleep_duration - expected_duration):.6f}s)"
        )

    @contextmanager
    def travels_for(self, expected_duration: float, tolerance: float = 0.001):
        """Assert that code block travels forward in time by exactly expected_duration (virtual time).
        This includes all code that affects the virtual time, such as asyncio.sleep, asyncio.wait_for, etc.

        Args:
            expected_duration: Expected virtual time duration in seconds
            tolerance: Acceptable difference in seconds (default: 1ms)

        Raises:
            AssertionError: If actual duration doesn't match expected

        Example:
            async with time_traveler.travels_for(5.0):
                await asyncio.sleep(5.0)  # Pass
        """
        # Use loop.time() for relative elapsed time (works with LoopScheduler)
        start_loop_time = self._get_loop_time_ns() / NANOS_PER_SECOND
        start_offset = self.offset_ns / NANOS_PER_SECOND
        yield
        end_loop_time = self._get_loop_time_ns() / NANOS_PER_SECOND
        end_offset = self.offset_ns / NANOS_PER_SECOND
        # Elapsed time is delta in loop time + delta in offset (from manual time advances)
        elapsed = (end_loop_time - start_loop_time) + (end_offset - start_offset)
        assert abs(elapsed - expected_duration) <= tolerance, (
            f"Expected to travel for {expected_duration}s (±{tolerance}s), "
            f"but traveled for {elapsed:.6f}s (diff: {abs(elapsed - expected_duration):.6f}s)"
        )

    # -------------------------------------------------------------------------
    # Lifecycle management
    # -------------------------------------------------------------------------

    def start_traveling(self):
        """Activate time.* patching.

        Patches the standard library's time functions to return virtual time.
        Looptime (enabled via pytest.mark.looptime) handles the actual time magic.
        """
        # Patch time.* functions to return virtual time
        self.patches = [
            patch("time.time_ns", self.time_ns),
            patch("time.time", self.time),
            patch("time.perf_counter_ns", self.perf_counter_ns),
            patch("time.perf_counter", self.perf_counter),
            patch("time.monotonic_ns", self.monotonic_ns),
            patch("time.monotonic", self.monotonic),
        ]

        # Only patch asyncio.sleep when patch_sleep=True.
        # When patch_sleep=False, let looptime handle asyncio.sleep for virtual time.
        if self.patch_sleep:
            self.patches.append(patch("asyncio.sleep", self.sleep))

        for p in self.patches:
            p.start()

    def stop_traveling(self):
        """Cleanup time.* patches."""
        for p in self.patches:
            p.stop()
        self.patches.clear()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the time_traveler fixture."""

import asyncio
import time

import pytest

from aiperf.common.constants import NANOS_PER_SECOND

EPSILON = 1e-9
"""Tolerance for floating point comparisons."""


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_YEAR = 365 * SECONDS_PER_DAY


@pytest.mark.asyncio
class TestTimeTraveler:
    """Tests demonstrating the time_traveler fixture."""

    async def test_time_traveler_advances_with_sleep(self, time_traveler):
        """Test that the time traveler advances when asyncio.sleep is called."""
        initial_time = time_traveler.monotonic()

        # Sleep for 1 second - this should advance the time_traveler by 1 second
        await asyncio.sleep(1.0)

        after_sleep_time = time_traveler.monotonic()
        assert abs(after_sleep_time - initial_time) - 1.0 < EPSILON

        initial_time = time_traveler.perf_counter_ns()
        # Sleep for a day - this would normally take hours, but its a walk in the park for the time traveler!
        await asyncio.sleep(SECONDS_PER_DAY)

        final_time = time_traveler.perf_counter_ns()
        assert (
            abs(final_time - initial_time) - (SECONDS_PER_DAY * NANOS_PER_SECOND)
            < EPSILON
        )

        initial_time = time_traveler.time_ns()
        # Sleep for 30 years - this would normally take decades, but the time traveler can do it in a flash!
        await asyncio.sleep(30 * SECONDS_PER_YEAR)

        final_time = time_traveler.time_ns()
        assert (
            abs(final_time - initial_time) - (30 * SECONDS_PER_YEAR * NANOS_PER_SECOND)
            < EPSILON
        )

    async def test_all_time_functions_use_time_traveler_time(self, time_traveler):
        """Test that all time functions return the time traveler's time."""
        # Advance time manually
        time_traveler.advance_time(10.0)

        # All time functions should return the same mock time
        expected_time_ns = time_traveler.time_ns()
        expected_time_s = time_traveler.time()
        expected_time_perf_counter_ns = time_traveler.perf_counter_ns()
        expected_time_perf_counter = time_traveler.perf_counter()
        expected_time_monotonic_ns = time_traveler.monotonic_ns()
        expected_time_monotonic = time_traveler.monotonic()

        assert time.time_ns() == expected_time_ns
        assert abs(time.time() - expected_time_s) < EPSILON
        assert time.perf_counter_ns() == expected_time_perf_counter_ns
        assert abs(time.perf_counter() - expected_time_perf_counter) < EPSILON
        assert time.monotonic_ns() == expected_time_monotonic_ns
        assert abs(time.monotonic() - expected_time_monotonic) < EPSILON

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aiperf.common.event_loop's platform-conditional policy switch.

Bug 5: pyzmq's async sockets call loop.add_reader() / add_writer(), which the
default Windows ProactorEventLoop does not implement. AIPerf forces a
selector-based loop on Windows before the loop is created.

On Windows with Python < 3.13, time.monotonic() (and so the stock loop's
clock) advances in 15.625 ms ticks, so that loop also gets a perf_counter
clock. These tests mock IS_WINDOWS and the version gate to exercise every
branch from non-Windows CI.
"""

from __future__ import annotations

import asyncio
import math
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiperf.common.event_loop import (
    PerfCounterSelectorEventLoop,
    PerfCounterSelectorEventLoopPolicy,
    aiperf_event_loop_policy,
    configure_event_loop_policy_for_platform,
)

_WINDOWS_TICK = 0.015625


class TestConfigureEventLoopPolicyForPlatform:
    """Verify the platform-conditional event-loop-policy setup."""

    def test_sets_event_loop_policy_when_windows(self) -> None:
        """On Windows with a precise monotonic clock (Python 3.13+),
        asyncio.set_event_loop_policy must be called once with a
        WindowsSelectorEventLoopPolicy instance."""
        # WindowsSelectorEventLoopPolicy doesn't exist on non-Windows Python,
        # so patch with create=True so the attribute lookup succeeds.
        with (
            patch("aiperf.common.event_loop.IS_WINDOWS", True),
            patch("aiperf.common.event_loop._MONOTONIC_IS_COARSE_ON_WINDOWS", False),
            patch.object(
                asyncio,
                "WindowsSelectorEventLoopPolicy",
                create=True,
            ) as mock_policy_cls,
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            configure_event_loop_policy_for_platform()

        mock_policy_cls.assert_called_once_with()
        mock_set_policy.assert_called_once_with(mock_policy_cls.return_value)

    def test_sets_perf_counter_policy_when_windows_monotonic_is_coarse(
        self,
    ) -> None:
        with (
            patch("aiperf.common.event_loop.IS_WINDOWS", True),
            patch("aiperf.common.event_loop._MONOTONIC_IS_COARSE_ON_WINDOWS", True),
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            configure_event_loop_policy_for_platform()

        (policy,), _ = mock_set_policy.call_args
        assert isinstance(policy, PerfCounterSelectorEventLoopPolicy)

    def test_does_not_touch_event_loop_policy_when_not_windows(self) -> None:
        """On non-Windows platforms the helper is a no-op — leave the
        platform default in place (uvloop, or asyncio default)."""
        with (
            patch("aiperf.common.event_loop.IS_WINDOWS", False),
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            configure_event_loop_policy_for_platform()
            assert aiperf_event_loop_policy() is None

        mock_set_policy.assert_not_called()


class TestPerfCounterSelectorEventLoop:
    def test_policy_builds_a_perf_counter_selector_loop(self) -> None:
        loop = PerfCounterSelectorEventLoopPolicy().new_event_loop()
        try:
            assert isinstance(loop, PerfCounterSelectorEventLoop)
            assert isinstance(loop, asyncio.SelectorEventLoop)
        finally:
            loop.close()

    def test_loop_time_reads_perf_counter(self) -> None:
        # monotonic() and perf_counter() share a clock on Linux and macOS, so
        # skew monotonic() to tell them apart.
        real_monotonic = time.monotonic
        with patch("time.monotonic", lambda: real_monotonic() + 1000.0):
            loop = PerfCounterSelectorEventLoop()
            try:
                before = time.perf_counter()
                now = loop.time()
                after = time.perf_counter()
            finally:
                loop.close()
        assert before <= now <= after

    def test_loop_time_ignores_a_patched_global_perf_counter(self) -> None:
        """Tests patch time.perf_counter globally with short side_effect lists
        (e.g. via ``aiperf.dataset.dataset_manager.time.perf_counter``). The
        loop must keep reading the real clock rather than drain those values."""
        loop = PerfCounterSelectorEventLoop()
        try:
            with patch("time.perf_counter", side_effect=[0.0]):
                first = loop.time()
                second = loop.time()
        finally:
            loop.close()
        assert 0.0 < first <= second

    def test_timers_never_fire_early_when_monotonic_is_coarse(self) -> None:
        """Simulate Windows < 3.13: monotonic() advances in 15.625 ms ticks
        and reports that as its resolution. On a busy loop (aiperf's always
        is: ZMQ I/O wakes it constantly) the stock loop then fires timers up
        to one tick early; this loop must not fire any of them early."""
        real_monotonic, real_clock_info = time.monotonic, time.get_clock_info

        def coarse_monotonic() -> float:
            return math.floor(real_monotonic() / _WINDOWS_TICK) * _WINDOWS_TICK

        def coarse_clock_info(name: str) -> object:
            info = real_clock_info(name)
            if name != "monotonic":
                return info
            return SimpleNamespace(**{**vars(info), "resolution": _WINDOWS_TICK})

        delay = 0.03

        async def shortest_wait() -> float:
            loop = asyncio.get_running_loop()
            busy = True

            def keep_loop_busy() -> None:
                if busy:
                    loop.call_later(0.001, keep_loop_busy)

            keep_loop_busy()
            shortest = math.inf
            for _ in range(20):
                fired = loop.create_future()
                start = time.perf_counter()
                loop.call_later(delay, fired.set_result, None)
                await fired
                shortest = min(shortest, time.perf_counter() - start)
            busy = False
            return shortest

        with (
            patch("time.monotonic", coarse_monotonic),
            patch("time.get_clock_info", coarse_clock_info),
        ):
            loop = PerfCounterSelectorEventLoopPolicy().new_event_loop()
            try:
                shortest = loop.run_until_complete(shortest_wait())
            finally:
                loop.close()

        assert shortest >= delay


@pytest.mark.asyncio
async def test_pytest_asyncio_runs_tests_on_the_aiperf_policy_loop() -> None:
    """tests/conftest.py routes pytest-asyncio through aiperf's policy, so
    tests run on the loop aiperf itself uses on this platform."""
    expected = aiperf_event_loop_policy()
    loop = asyncio.get_running_loop()
    if expected is None:
        assert not isinstance(loop, PerfCounterSelectorEventLoop)
    else:
        expected_loop = expected.new_event_loop()
        try:
            # Plugins such as looptime wrap the loop class in a subclass.
            assert isinstance(loop, type(expected_loop))
        finally:
            expected_loop.close()

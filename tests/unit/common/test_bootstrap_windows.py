# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Windows-specific event loop policy fix in bootstrap.py.

Bug 5: pyzmq's async sockets call loop.add_reader() / add_writer(), which the
default Windows ProactorEventLoop does not implement. AIPerf forces the
SelectorEventLoopPolicy on Windows before the loop is created. These tests
mock IS_WINDOWS to exercise both branches from non-Windows CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from aiperf.common.bootstrap import (
    _configure_event_loop_policy_for_platform,
    _request_high_resolution_timer_on_windows,
)


class TestConfigureEventLoopPolicyForPlatform:
    """Verify the platform-conditional event-loop-policy setup."""

    def test_sets_event_loop_policy_when_windows(self) -> None:
        """On Windows, asyncio.set_event_loop_policy must be called once
        with a WindowsSelectorEventLoopPolicy instance."""
        # WindowsSelectorEventLoopPolicy doesn't exist on non-Windows Python,
        # so patch with create=True so the attribute lookup succeeds.
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", True),
            patch.object(
                asyncio,
                "WindowsSelectorEventLoopPolicy",
                create=True,
            ) as mock_policy_cls,
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            _configure_event_loop_policy_for_platform()

        mock_policy_cls.assert_called_once_with()
        mock_set_policy.assert_called_once_with(mock_policy_cls.return_value)

    def test_does_not_touch_event_loop_policy_when_not_windows(self) -> None:
        """On non-Windows platforms the helper is a no-op — leave the
        platform default in place (uvloop, or asyncio default)."""
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", False),
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            _configure_event_loop_policy_for_platform()

        mock_set_policy.assert_not_called()


class TestRequestHighResolutionTimerOnWindows:
    """Verify the Windows-only system-timer-resolution bump.

    Without this, asyncio.sleep is floored to ~15.6ms on Windows because
    of the default 15.625ms scheduling timer tick. The aiperf scheduler
    issues credits at sub-15ms intervals for >60 QPS, so the default tick
    causes credit issuance to clump and break constant-rate / Poisson
    pacing tests. ``timeBeginPeriod(1)`` requests 1ms resolution.
    """

    def test_calls_timeBeginPeriod_with_1ms_when_windows(self) -> None:
        """On Windows, winmm.timeBeginPeriod(1) must be called exactly once."""
        mock_winmm = MagicMock()
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", True),
            patch("ctypes.WinDLL", create=True, return_value=mock_winmm) as mock_dll,
        ):
            _request_high_resolution_timer_on_windows()

        mock_dll.assert_called_once_with("winmm")
        mock_winmm.timeBeginPeriod.assert_called_once_with(1)

    def test_does_not_call_timeBeginPeriod_when_not_windows(self) -> None:
        """No-op on POSIX — must not even attempt the winmm import."""
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", False),
            patch("ctypes.WinDLL", create=True) as mock_dll,
        ):
            _request_high_resolution_timer_on_windows()

        mock_dll.assert_not_called()

    def test_swallows_winmm_load_failure(self) -> None:
        """If winmm load fails (extremely unlikely — it's part of Windows),
        the function must not raise; aiperf should still run, just with
        coarser timing. Real users would hit this only on broken systems."""
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", True),
            patch("ctypes.WinDLL", create=True, side_effect=OSError("not found")),
        ):
            _request_high_resolution_timer_on_windows()  # must not raise

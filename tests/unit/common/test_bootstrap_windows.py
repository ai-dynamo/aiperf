# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Windows-specific high-resolution-timer fix in bootstrap.py.

Bug 5: pyzmq's async sockets call loop.add_reader() / add_writer(), which the
default Windows ProactorEventLoop does not implement. AIPerf forces the
SelectorEventLoopPolicy on Windows before the loop is created. That policy
switch now lives in ``aiperf.common.event_loop`` (see
``tests/unit/common/test_event_loop.py``) because it must be applied from
every real process entrypoint, not just ``bootstrap_and_run_service`` --
this file keeps only the Windows-timer-resolution tests, which are specific
to the service-bootstrap path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiperf.common.bootstrap import _request_high_resolution_timer_on_windows


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

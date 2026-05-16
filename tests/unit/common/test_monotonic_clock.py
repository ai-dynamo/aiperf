# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MonotonicClock.

Verifies the dual-clock bootstrap pattern: capture time.time_ns() once as a
wall-clock anchor, then derive all subsequent timestamps from perf_counter_ns()
deltas to produce monotonic wall-clock-domain values.
"""

from unittest.mock import MagicMock, patch

from aiperf.common.monotonic_clock import MonotonicClock

CLOCK_TIME = "aiperf.common.monotonic_clock.time"


def _make_mock_time(
    wall_anchor: int = 0, perf_anchor: int = 0
) -> tuple[MagicMock, list[int]]:
    """Create a mock time module with configurable anchors.

    Returns:
        (mock_time, perf_values) where perf_values is a mutable list that
        feeds perf_counter_ns. The first value is consumed at construction.
    """
    perf_values: list[int] = [perf_anchor]
    mock_time = MagicMock()
    mock_time.time_ns.return_value = wall_anchor
    mock_time.perf_counter_ns.side_effect = lambda: perf_values.pop(0)
    return mock_time, perf_values


class TestMonotonicClockInit:
    """Verify anchors are captured at construction."""

    def test_captures_perf_anchor(self) -> None:
        mock_time, _ = _make_mock_time(perf_anchor=42)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
        assert clock.perf_anchor_ns == 42

    def test_captures_wall_anchor(self) -> None:
        mock_time, _ = _make_mock_time(wall_anchor=1_000_000_000)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
        assert clock.wall_anchor_ns == 1_000_000_000


class TestMonotonicClockNowNs:
    """Verify now_ns derives wall clock from perf_counter deltas."""

    def test_at_construction_equals_wall_anchor(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=500)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(500)
            assert clock.now_ns() == 1_000_000

    def test_advances_with_perf_counter(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=500)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(700)
            assert clock.now_ns() == 1_000_200

    def test_immune_to_time_ns_changes(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=0)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            mock_time.time_ns.return_value = 9_999_999
            perf_values.append(100)
            assert clock.now_ns() == 1_000_100


class TestMonotonicClockElapsedNs:
    """Verify elapsed_ns returns perf_counter delta from anchor."""

    def test_zero_at_construction(self) -> None:
        mock_time, perf_values = _make_mock_time(perf_anchor=1000)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(1000)
            assert clock.elapsed_ns() == 0

    def test_returns_delta(self) -> None:
        mock_time, perf_values = _make_mock_time(perf_anchor=1000)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(1500)
            assert clock.elapsed_ns() == 500


class TestMonotonicClockElapsedSec:
    """Verify elapsed_sec returns seconds as float."""

    def test_converts_ns_to_seconds(self) -> None:
        mock_time, perf_values = _make_mock_time(perf_anchor=0)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(1_500_000_000)
            assert clock.elapsed_sec() == 1.5

    def test_zero_at_construction(self) -> None:
        mock_time, perf_values = _make_mock_time(perf_anchor=0)
        with patch(CLOCK_TIME, mock_time):
            clock = MonotonicClock()
            perf_values.append(0)
            assert clock.elapsed_sec() == 0.0

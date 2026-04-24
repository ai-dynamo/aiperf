# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ClockOffsetTracker.

Focuses on:
- Initial state before any measurements
- Minimum offset filtering over a sliding window
- Calibration readiness (is_calibrated)
- Jitter visibility (offset_range_ns)
- Timestamp correction (correct_timestamp)
- Window eviction behavior
- Pre-flight RTT measurement (measure_baseline_rtt)
- Estimated clock skew decomposition
- MonotonicClock integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.credit.messages import TimePong
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker

LOGGER_NAME = "test-worker"


# ============================================================
# Helpers
# ============================================================


def _make_mock_time(
    wall_anchor: int = 0, perf_anchor: int = 0
) -> tuple[MagicMock, list[int]]:
    """Create a mock time module with configurable anchors.

    The tracker's MonotonicClock calls perf_counter_ns() then time_ns() at init.
    now_ns() computes: wall_anchor + (perf_counter_ns() - perf_anchor)

    With both anchors at 0, now_ns() simply returns perf_counter_ns().

    Returns:
        (mock_time, perf_values) where perf_values is a mutable list that
        feeds perf_counter_ns via side_effect. Append values to control
        what now_ns() returns after construction.
    """
    perf_values: list[int] = [perf_anchor]
    mock_time = MagicMock()
    mock_time.time_ns.return_value = wall_anchor
    mock_time.perf_counter_ns.side_effect = lambda: perf_values.pop(0)
    return mock_time, perf_values


CLOCK_TIME = "aiperf.common.monotonic_clock.time"
TRACKER_TIME = "aiperf.workers.clock_offset_tracker.time"


def _make_tracker(
    mock_time: MagicMock,
    **kwargs,
) -> ClockOffsetTracker:
    """Construct a ClockOffsetTracker under a mocked time module."""
    with patch(CLOCK_TIME, mock_time):
        return ClockOffsetTracker(logger_name=LOGGER_NAME, **kwargs)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def tracker() -> ClockOffsetTracker:
    """Create a tracker with default settings (window=20, min_samples=5)."""
    return ClockOffsetTracker(logger_name=LOGGER_NAME)


# ============================================================
# Initial State
# ============================================================


class TestClockOffsetTrackerInitialState:
    """Verify state before any measurements."""

    def test_offset_ns_before_update_is_none(self, tracker: ClockOffsetTracker) -> None:
        assert tracker.offset_ns is None

    def test_sample_count_starts_at_zero(self, tracker: ClockOffsetTracker) -> None:
        assert tracker.sample_count == 0

    def test_is_calibrated_false_initially(self, tracker: ClockOffsetTracker) -> None:
        assert tracker.is_calibrated is False

    def test_offset_range_ns_none_initially(self, tracker: ClockOffsetTracker) -> None:
        assert tracker.offset_range_ns is None

    def test_baseline_rtt_none_initially(self, tracker: ClockOffsetTracker) -> None:
        assert tracker.baseline_rtt_ns is None
        assert tracker.estimated_one_way_ns is None

    def test_estimated_clock_skew_none_initially(
        self, tracker: ClockOffsetTracker
    ) -> None:
        assert tracker.estimated_clock_skew_ns is None


# ============================================================
# MonotonicClock Integration
# ============================================================


class TestClockOffsetTrackerMonotonicClock:
    """Verify tracker's clock derives wall time from perf_counter deltas."""

    def test_clock_at_construction_equals_wall_anchor(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=500)
        tracker = _make_tracker(mock_time)

        perf_values.append(500)
        with patch(CLOCK_TIME, mock_time):
            assert tracker._clock.now_ns() == 1_000_000

    def test_clock_advances_with_perf_counter(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=500)
        tracker = _make_tracker(mock_time)

        perf_values.append(700)
        with patch(CLOCK_TIME, mock_time):
            assert tracker._clock.now_ns() == 1_000_200

    def test_clock_immune_to_time_ns_changes(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000, perf_anchor=0)
        tracker = _make_tracker(mock_time)

        mock_time.time_ns.return_value = 9_999_999
        perf_values.append(100)
        with patch(CLOCK_TIME, mock_time):
            assert tracker._clock.now_ns() == 1_000_100


# ============================================================
# Minimum Offset Filtering
# ============================================================


class TestClockOffsetTrackerMinFilter:
    """Verify that offset_ns always equals the minimum sample in the window."""

    def test_update_first_sample_sets_offset(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # _now_ns returns 1_000_000_000, issued_at = 900_000_000
        perf_values.append(1_000_000_000)
        with patch(CLOCK_TIME, mock_time):
            result = tracker.update(900_000_000)

        assert result == 100_000_000
        assert tracker.offset_ns == 100_000_000

    def test_update_selects_minimum_across_samples(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # Three samples: offsets 100, 50, 200. Minimum = 50.
        for now_val in [1100, 1050, 1200]:
            perf_values.append(now_val)
        with patch(CLOCK_TIME, mock_time):
            for _ in range(3):
                tracker.update(1000)

        assert tracker.offset_ns == 50

    def test_update_new_minimum_replaces_previous(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.extend([1100, 1030])
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)
            assert tracker.offset_ns == 100

            tracker.update(1000)
            assert tracker.offset_ns == 30

    def test_update_higher_sample_does_not_change_offset(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.extend([1050, 1200])
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)
            assert tracker.offset_ns == 50

            tracker.update(1000)
            assert tracker.offset_ns == 50

    def test_update_returns_current_offset(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.append(1100)
        with patch(CLOCK_TIME, mock_time):
            result = tracker.update(1000)

        assert result == tracker.offset_ns


# ============================================================
# Window Eviction
# ============================================================


class TestClockOffsetTrackerWindowEviction:
    """Verify that old samples are evicted and offset updates accordingly."""

    def test_old_minimum_evicted_offset_rises(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time, window_size=3)

        # Fill window: [10, 200, 200]
        perf_values.extend([1010, 1200, 1200, 1150])
        with patch(CLOCK_TIME, mock_time):
            for _ in range(3):
                tracker.update(1000)
            assert tracker.offset_ns == 10

            # Push out the minimum: window becomes [200, 200, 150]
            tracker.update(1000)
            assert tracker.offset_ns == 150

    def test_sample_count_keeps_incrementing_past_window(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time, window_size=3)

        perf_values.extend([1100] * 10)
        with patch(CLOCK_TIME, mock_time):
            for _ in range(10):
                tracker.update(1000)

        assert tracker.sample_count == 10


# ============================================================
# Calibration
# ============================================================


class TestClockOffsetTrackerCalibration:
    """Verify is_calibrated threshold behavior."""

    def test_is_calibrated_false_below_min_samples(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time, min_samples=5)

        perf_values.extend([1100] * 4)
        with patch(CLOCK_TIME, mock_time):
            for i in range(4):
                tracker.update(1000)
                assert tracker.is_calibrated is False, (
                    f"Should not be calibrated after {i + 1} samples"
                )

    def test_is_calibrated_true_at_min_samples(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time, min_samples=5)

        perf_values.extend([1100] * 5)
        with patch(CLOCK_TIME, mock_time):
            for _ in range(5):
                tracker.update(1000)

        assert tracker.is_calibrated is True

    @pytest.mark.parametrize(
        "min_samples",
        [
            param(1, id="min-1"),
            param(3, id="min-3"),
            param(10, id="min-10"),
        ],
    )  # fmt: skip
    def test_is_calibrated_respects_custom_min_samples(self, min_samples: int) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time, min_samples=min_samples)

        perf_values.extend([1100] * (min_samples + 1))
        with patch(CLOCK_TIME, mock_time):
            for _ in range(min_samples - 1):
                tracker.update(1000)
            assert tracker.is_calibrated is False

            tracker.update(1000)
            assert tracker.is_calibrated is True


# ============================================================
# Offset Range (Jitter)
# ============================================================


class TestClockOffsetTrackerOffsetRange:
    """Verify offset_range_ns reflects jitter in the window."""

    def test_offset_range_single_sample_is_zero(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.append(1100)
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)

        assert tracker.offset_range_ns == 0

    def test_offset_range_reflects_spread(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # Samples: 50, 200, 100 -> range = 200 - 50 = 150
        perf_values.extend([1050, 1200, 1100])
        with patch(CLOCK_TIME, mock_time):
            for _ in range(3):
                tracker.update(1000)

        assert tracker.offset_range_ns == 150


# ============================================================
# Timestamp Correction
# ============================================================


class TestClockOffsetTrackerCorrectTimestamp:
    """Verify correct_timestamp converts worker time to controller time."""

    def test_correct_timestamp_no_offset_returns_input(
        self, tracker: ClockOffsetTracker
    ) -> None:
        ts = 5_000_000_000
        assert tracker.correct_timestamp(ts) == ts

    def test_correct_timestamp_subtracts_offset(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.append(1_100_000_000)
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1_000_000_000)

        worker_ts = 2_100_000_000
        assert tracker.correct_timestamp(worker_ts) == 2_000_000_000

    def test_correct_timestamp_uses_minimum_offset(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.extend([1_100_000_000, 1_050_000_000])
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1_000_000_000)
            tracker.update(1_000_000_000)

        worker_ts = 2_050_000_000
        assert tracker.correct_timestamp(worker_ts) == 2_000_000_000


# ============================================================
# Now With Offset
# ============================================================


class TestClockOffsetTrackerNowWithOffset:
    """Verify now_with_offset returns consistent timestamp and offset."""

    def test_returns_none_offset_before_update(self) -> None:
        mock_time, perf_values = _make_mock_time(wall_anchor=1_000_000)
        tracker = _make_tracker(mock_time)

        perf_values.append(500)
        with patch(CLOCK_TIME, mock_time):
            now_ns, offset_ns = tracker.now_with_offset()

        assert now_ns == 1_000_500
        assert offset_ns is None

    def test_returns_current_offset_after_update(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # Update: now=1100, issued=1000 -> offset=100
        perf_values.append(1100)
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)

        # now_with_offset: now=2000, offset should still be 100
        perf_values.append(2000)
        with patch(CLOCK_TIME, mock_time):
            now_ns, offset_ns = tracker.now_with_offset()

        assert now_ns == 2000
        assert offset_ns == 100

    def test_offset_matches_current_minimum(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # Two samples: offsets 200, 50 -> min = 50
        perf_values.extend([1200, 1050])
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)
            tracker.update(1000)

        perf_values.append(3000)
        with patch(CLOCK_TIME, mock_time):
            now_ns, offset_ns = tracker.now_with_offset()

        assert now_ns == 3000
        assert offset_ns == 50


# ============================================================
# Estimated Clock Skew
# ============================================================


class TestClockOffsetTrackerEstimatedClockSkew:
    """Verify estimated_clock_skew_ns decomposes offset into skew and transit."""

    def test_estimated_clock_skew_none_without_rtt(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        perf_values.append(1100)
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1000)

        assert tracker.estimated_clock_skew_ns is None

    def test_estimated_clock_skew_none_without_offset(self) -> None:
        tracker = ClockOffsetTracker(logger_name=LOGGER_NAME)
        tracker.baseline_rtt_ns = 2_000_000
        tracker.estimated_one_way_ns = 1_000_000

        assert tracker.estimated_clock_skew_ns is None

    def test_estimated_clock_skew_subtracts_one_way(self) -> None:
        mock_time, perf_values = _make_mock_time()
        tracker = _make_tracker(mock_time)

        # offset = 5ms, RTT = 2ms -> one-way = 1ms -> skew = 4ms
        perf_values.append(1_005_000_000)
        with patch(CLOCK_TIME, mock_time):
            tracker.update(1_000_000_000)
        tracker.baseline_rtt_ns = 2_000_000
        tracker.estimated_one_way_ns = 1_000_000

        assert tracker.estimated_clock_skew_ns == 4_000_000


# ============================================================
# Handle Pong
# ============================================================


class TestClockOffsetTrackerHandlePong:
    """Verify handle_pong resolves the pending future."""

    @pytest.mark.asyncio
    async def test_handle_pong_resolves_future(
        self, tracker: ClockOffsetTracker
    ) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        tracker._pending_pong_future = future

        pong = TimePong(sequence=0, sent_at_ns=1000)
        tracker.handle_pong(pong)

        assert future.done()
        assert future.result() is pong

    def test_handle_pong_no_pending_future_is_noop(
        self, tracker: ClockOffsetTracker
    ) -> None:
        pong = TimePong(sequence=0, sent_at_ns=1000)
        tracker.handle_pong(pong)  # Should not raise

    @pytest.mark.asyncio
    async def test_handle_pong_already_done_future_is_noop(
        self, tracker: ClockOffsetTracker
    ) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result(None)
        tracker._pending_pong_future = future

        pong = TimePong(sequence=0, sent_at_ns=1000)
        tracker.handle_pong(pong)  # Should not raise


# ============================================================
# Measure Baseline RTT
# ============================================================


class TestClockOffsetTrackerMeasureBaselineRtt:
    """Verify measure_baseline_rtt sends probes and records minimum RTT."""

    @patch(TRACKER_TIME)
    @patch(CLOCK_TIME)
    async def test_measure_baseline_rtt_records_minimum(
        self, mock_clock_time: MagicMock, mock_tracker_time: MagicMock
    ) -> None:
        # MonotonicClock init
        mock_clock_time.perf_counter_ns.return_value = 0
        mock_clock_time.time_ns.return_value = 0

        # RTT probes: 3 pairs (send, receive)
        mock_tracker_time.perf_counter_ns.side_effect = [
            1000,
            1200,  # probe 0: RTT 200
            1200,
            1350,  # probe 1: RTT 150
            1350,
            1600,  # probe 2: RTT 250
        ]

        tracker = ClockOffsetTracker(logger_name=LOGGER_NAME)

        async def mock_send_ping(ping: object) -> None:
            pong = TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
            tracker.handle_pong(pong)

        await tracker.measure_baseline_rtt(send_ping=mock_send_ping, probe_count=3)

        assert tracker.baseline_rtt_ns == 150
        assert tracker.estimated_one_way_ns == 75

    @pytest.mark.looptime
    @patch(TRACKER_TIME)
    @patch(CLOCK_TIME)
    async def test_measure_baseline_rtt_handles_timeout(
        self, mock_clock_time: MagicMock, mock_tracker_time: MagicMock
    ) -> None:
        mock_clock_time.perf_counter_ns.return_value = 0
        mock_clock_time.time_ns.return_value = 0
        mock_tracker_time.perf_counter_ns.return_value = 1000

        tracker = ClockOffsetTracker(logger_name=LOGGER_NAME)

        call_count = 0

        async def mock_send_ping_no_reply(ping: object) -> None:
            nonlocal call_count
            call_count += 1

        await tracker.measure_baseline_rtt(
            send_ping=mock_send_ping_no_reply, probe_count=2, timeout=1
        )

        assert call_count == 2
        assert tracker.baseline_rtt_ns is None
        assert tracker.estimated_one_way_ns is None

    @pytest.mark.looptime
    @patch(TRACKER_TIME)
    @patch(CLOCK_TIME)
    async def test_measure_baseline_rtt_partial_success(
        self, mock_clock_time: MagicMock, mock_tracker_time: MagicMock
    ) -> None:
        mock_clock_time.perf_counter_ns.return_value = 0
        mock_clock_time.time_ns.return_value = 0

        # 1 successful probe + 1 timeout probe
        mock_tracker_time.perf_counter_ns.side_effect = [
            1000,
            1100,  # probe 0: RTT 100
            1100,  # probe 1: send (no receive, times out)
        ]

        tracker = ClockOffsetTracker(logger_name=LOGGER_NAME)

        call_count = 0

        async def mock_send_ping(ping: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                pong = TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
                tracker.handle_pong(pong)

        await tracker.measure_baseline_rtt(
            send_ping=mock_send_ping, probe_count=2, timeout=1
        )

        assert tracker.baseline_rtt_ns == 100
        assert tracker.estimated_one_way_ns == 50

    async def test_measure_baseline_rtt_clears_pending_future(
        self, tracker: ClockOffsetTracker
    ) -> None:
        send_ping = AsyncMock()

        async def _send_and_resolve(ping: object) -> None:
            pong = TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
            tracker.handle_pong(pong)

        send_ping.side_effect = _send_and_resolve

        with patch(TRACKER_TIME) as mock_time:
            mock_time.perf_counter_ns.return_value = 1000
            await tracker.measure_baseline_rtt(send_ping=send_ping, probe_count=1)

        assert tracker._pending_pong_future is None

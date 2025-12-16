# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for windowed rate calculations for counters, gauges, and histograms."""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.server_metrics.export_stats import (
    _compute_counter_stats,
    _compute_counter_timeslices,
    _compute_gauge_stats,
    _compute_gauge_timeslices,
    _compute_histogram_stats,
    _compute_histogram_timeslices,
)
from aiperf.server_metrics.storage import ServerMetricsTimeSeries
from tests.unit.server_metrics.helpers import (
    add_counter_samples,
    add_counter_samples_with_timestamps,
    add_gauge_samples,
    add_gauge_samples_with_timestamps,
    add_histogram_snapshots,
    get_counter,
    get_gauge,
    get_histogram,
    hist,
    make_time_filter,
)


class TestComputeCounterTimeslices:
    """Test compute_counter_timeslices function."""

    def test_basic_windowed_rates(self):
        """Test basic windowed rate calculation with uniform data."""
        ts = ServerMetricsTimeSeries()
        # 5 samples at 1s intervals: 0, 100, 200, 300, 400
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # 4 complete 1s windows: [0-1), [1-2), [2-3), [3-4)
        assert len(rates) == 4
        for rp in rates:
            assert rp.rate == 100.0  # Uniform rate of 100/s

        # Verify from_ns and to_ns are correct
        assert rates[0].start_ns == 0
        assert rates[0].end_ns == NANOS_PER_SECOND
        assert rates[1].start_ns == NANOS_PER_SECOND
        assert rates[1].end_ns == 2 * NANOS_PER_SECOND

    def test_windowed_rates_variable_rate(self):
        """Test windowed rates with varying rates per window."""
        ts = ServerMetricsTimeSeries()
        # Increasing rates: 50/s, 100/s, 150/s, 200/s
        add_counter_samples(ts, "requests", [0.0, 50.0, 150.0, 300.0, 500.0])

        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        assert len(rates) == 4
        assert rates[0].rate == 50.0
        assert rates[1].rate == 100.0
        assert rates[2].rate == 150.0
        assert rates[3].rate == 200.0

    def test_windowed_rates_custom_window_size(self):
        """Test windowed rates with custom window size."""
        ts = ServerMetricsTimeSeries()
        # 10 samples at 1s intervals: 0, 100, 200, ..., 900
        add_counter_samples(ts, "requests", [float(i * 100) for i in range(10)])

        # 2 second windows
        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=2.0,
            time_filter=make_time_filter(start_ns=0, end_ns=8 * NANOS_PER_SECOND),
        )

        # 4 complete 2s windows: [0-2), [2-4), [4-6), [6-8)
        assert len(rates) == 4
        for rp in rates:
            assert rp.rate == 100.0  # 200 delta / 2s = 100/s

    def test_windowed_rates_with_time_filter(self):
        """Test windowed rates respect time filter (warmup exclusion)."""
        ts = ServerMetricsTimeSeries()
        # 10 samples: warmup (0-4s), actual run (5-9s)
        add_counter_samples(ts, "requests", [float(i * 100) for i in range(10)])

        # Filter excludes warmup (first 5 seconds)
        time_filter = TimeRangeFilter(
            start_ns=5 * NANOS_PER_SECOND, end_ns=9 * NANOS_PER_SECOND
        )
        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=time_filter,
        )

        # Reference point at 4s (value=400), data from 5-9s
        # Windows start from time_filter.start_ns (5s): [5-6), [6-7), [7-8), [8-9)
        # 4 complete windows (time_filter range is 5-9s)
        assert len(rates) == 4
        for rp in rates:
            assert rp.rate == 100.0
        # First window starts at time_filter.start_ns
        assert rates[0].start_ns == 5 * NANOS_PER_SECOND

    def test_windowed_rates_partial_windows_excluded(self):
        """Test partial windows at end are excluded."""
        ts = ServerMetricsTimeSeries()
        # 4.5 seconds of data (last sample at 4s)
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        # Only 4 complete 1s windows, no partial window at end
        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )
        assert len(rates) == 4

        # With 2s windows, only 2 complete windows
        rates_2s = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=2.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )
        assert len(rates_2s) == 2

    def test_windowed_rates_counter_reset(self):
        """Test counter reset results in zero rate for that window."""
        ts = ServerMetricsTimeSeries()
        # Counter resets at t=2s: 0, 100, 50 (reset!), 100, 200
        add_counter_samples(ts, "requests", [0.0, 100.0, 50.0, 100.0, 200.0])

        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        assert len(rates) == 4
        assert rates[0].rate == 100.0  # 0 -> 100
        assert rates[1].rate == 0.0  # Reset detected, clamped to 0
        assert rates[2].rate == 50.0  # 50 -> 100
        assert rates[3].rate == 100.0  # 100 -> 200

    def test_windowed_rates_empty_data(self):
        """Test empty data returns empty list."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [100.0])  # Single sample

        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )
        assert rates == []

    def test_windowed_rates_insufficient_duration(self):
        """Test duration less than window size returns empty list."""
        ts = ServerMetricsTimeSeries()
        # Half second of data
        add_counter_samples(
            ts, "requests", [0.0, 50.0], interval_ns=NANOS_PER_SECOND // 2
        )

        # time_filter also 0.5s - not enough for full 1s window, but returns partial
        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND // 2),
        )
        # Now returns partial slice instead of empty (best practice: preserve all data)
        assert len(rates) == 1
        assert not rates[0].is_complete
        assert rates[0].total == 50.0
        assert rates[0].rate == 100.0  # Normalized: 50 delta / 0.5s = 100/s

    def test_windowed_rates_invalid_window_size(self):
        """Test invalid window size raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_counter_timeslices(
                get_counter(ts, "requests"),
                slice_duration=0.0,
                time_filter=make_time_filter(),
            )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_counter_timeslices(
                get_counter(ts, "requests"),
                slice_duration=-1.0,
                time_filter=make_time_filter(),
            )

    def test_windowed_rates_sparse_data(self):
        """Test sparse data (gaps between samples) still computes correct rates."""
        ts = ServerMetricsTimeSeries()
        # Samples at 0s, 2s, 4s (sparse, 2s intervals)
        add_counter_samples(
            ts, "requests", [0.0, 200.0, 400.0], interval_ns=2 * NANOS_PER_SECOND
        )

        # 1s windows - uses interpolation from last known value
        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # Windows: [0-1), [1-2), [2-3), [3-4)
        # Value at 0s=0, at 2s=200, at 4s=400
        # [0-1): end value at 1s -> last known is 0 -> rate = 0
        # [1-2): end value at 2s -> 200 -> rate = 200/1s = 200? No, delta from start (0) = 200
        # Actually: searchsorted finds last value <= window boundary
        assert len(rates) == 4
        # First window [0-1): start=0, end=value at 1s (still 0) -> rate=0
        assert rates[0].rate == 0.0
        # Second window [1-2): start=0, end=value at 2s (200) -> rate=200
        assert rates[1].rate == 200.0
        # Third window [2-3): start=200, end=value at 3s (still 200) -> rate=0
        assert rates[2].rate == 0.0
        # Fourth window [3-4): start=200, end=value at 4s (400) -> rate=200
        assert rates[3].rate == 200.0


class TestComputeCounterStatsWithTimeslices:
    """Test compute_counter_stats integration with windowed rates."""

    def test_counter_stats_includes_windowed_rates(self):
        """Test counter stats includes rates list when slice_duration is specified."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        stats = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert stats.timeslices is not None
        assert len(stats.timeslices) == 4

    def test_counter_stats_rate_statistics_from_windows(self):
        """Test rate_min/max/avg/std are computed from windowed rates."""
        ts = ServerMetricsTimeSeries()
        # Variable rates: 50, 100, 150, 200 per second
        add_counter_samples(ts, "requests", [0.0, 50.0, 150.0, 300.0, 500.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert result.stats is not None
        assert result.stats.rate_min == 50.0
        assert result.stats.rate_max == 200.0
        assert result.stats.rate_avg == pytest.approx(125.0)  # (50+100+150+200)/4
        # std of [50, 100, 150, 200] with ddof=1
        assert result.stats.rate_std == pytest.approx(64.55, rel=0.01)

    def test_counter_stats_no_slice_duration(self):
        """Test slice_duration=None excludes windowed rates."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
            slice_duration=None,
        )

        assert result.timeslices is None
        assert result.stats is not None
        assert result.stats.rate_min is None
        assert result.stats.rate_max is None
        assert result.stats.rate_avg is None
        assert result.stats.rate_std is None
        # But still has overall rate
        assert result.stats.rate == 100.0

    def test_counter_stats_custom_window_size(self):
        """Test custom window size in counter stats."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [float(i * 100) for i in range(10)])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=8 * NANOS_PER_SECOND),
            slice_duration=2.0,
        )

        # 4 complete 2s windows
        assert result.timeslices is not None
        assert len(result.timeslices) == 4
        for rp in result.timeslices:
            assert rp.end_ns - rp.start_ns == 2 * NANOS_PER_SECOND

    def test_counter_stats_insufficient_data_for_windows(self):
        """Test insufficient data for fixed-size windows returns None timeslices."""
        ts = ServerMetricsTimeSeries()
        # Only 0.5s of data - not enough for 1s fixed windows
        add_counter_samples(
            ts, "requests", [0.0, 50.0], interval_ns=NANOS_PER_SECOND // 2
        )

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND // 2),
            slice_duration=1.0,
        )

        # Basic stats still computed
        assert result.stats is not None
        assert result.stats.total == 50.0
        assert result.stats.rate == 100.0  # 50 / 0.5s
        # Timeslices now include partial slice
        assert result.timeslices is not None
        assert len(result.timeslices) == 1
        assert not result.timeslices[0].is_complete
        # Rate stats should be None (no complete slices for aggregate calculation)
        assert result.stats.rate_min is None
        assert result.stats.rate_max is None
        assert result.stats.rate_avg is None

    def test_counter_stats_single_window(self):
        """Test stats with exactly one complete window."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=1 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert result.timeslices is not None
        assert len(result.timeslices) == 1
        assert result.stats is not None
        assert result.stats.rate_min == 100.0
        assert result.stats.rate_max == 100.0
        assert result.stats.rate_avg == 100.0
        assert result.stats.rate_std == 0.0  # Single sample, std=0


class TestComputeGaugeTimeslices:
    """Test compute_windowed_gauge_stats function."""

    def test_basic_windowed_gauge_stats(self):
        """Test basic windowed gauge stats with uniform data."""
        ts = ServerMetricsTimeSeries()
        # 5 samples at 1s intervals: 10, 20, 30, 40, 50
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # 4 complete 1s windows
        assert len(windows) == 4

        # Verify from_ns and to_ns
        assert windows[0].start_ns == 0
        assert windows[0].end_ns == NANOS_PER_SECOND

        # Each window has one sample, so avg=min=max
        assert windows[0].avg == 10.0
        assert windows[0].min == 10.0
        assert windows[0].max == 10.0

    def test_windowed_gauge_multiple_samples_per_window(self):
        """Test windowed gauge with multiple samples per window."""
        ts = ServerMetricsTimeSeries()
        # 4 samples per second, 2 seconds of data
        values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        add_gauge_samples(ts, "queue_depth", values, interval_ns=NANOS_PER_SECOND // 4)

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        # 2 complete 1s windows
        assert len(windows) == 2

        # First window: samples at 0, 0.25, 0.5, 0.75 (values 10, 15, 20, 25)
        assert windows[0].avg == pytest.approx(17.5)  # (10+15+20+25)/4
        assert windows[0].min == 10.0
        assert windows[0].max == 25.0

        # Second window: samples at 1.0, 1.25, 1.5, 1.75, 2.0 (values 30, 35, 40, 45, 50)
        # Note: the sample at t=2.0 is at the exact end boundary and is included
        assert windows[1].avg == pytest.approx(40.0)  # (30+35+40+45+50)/5
        assert windows[1].min == 30.0
        assert windows[1].max == 50.0

    def test_windowed_gauge_custom_window_size(self):
        """Test windowed gauge with custom window size."""
        ts = ServerMetricsTimeSeries()
        # 10 samples at 1s intervals
        add_gauge_samples(ts, "queue_depth", [float(i * 10) for i in range(10)])

        # 2 second windows
        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=2.0,
            time_filter=make_time_filter(start_ns=0, end_ns=8 * NANOS_PER_SECOND),
        )

        # 4 complete 2s windows: [0-2), [2-4), [4-6), [6-8)
        assert len(windows) == 4

        # First window has samples at 0s, 1s (values 0, 10)
        assert windows[0].avg == pytest.approx(5.0)
        assert windows[0].min == 0.0
        assert windows[0].max == 10.0

    def test_windowed_gauge_with_time_filter(self):
        """Test windowed gauge respects time filter."""
        ts = ServerMetricsTimeSeries()
        # 10 samples
        add_gauge_samples(ts, "queue_depth", [float(i * 10) for i in range(10)])

        # Filter excludes first 5 seconds
        time_filter = TimeRangeFilter(
            start_ns=5 * NANOS_PER_SECOND, end_ns=9 * NANOS_PER_SECOND
        )
        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=time_filter,
        )

        # Windows from 5s to 9s: [5-6), [6-7), [7-8), [8-9)
        assert len(windows) == 4
        assert windows[0].start_ns == 5 * NANOS_PER_SECOND
        assert windows[0].avg == 50.0  # value at 5s

    def test_windowed_gauge_partial_windows_excluded(self):
        """Test partial windows at end are excluded."""
        ts = ServerMetricsTimeSeries()
        # 4.5 seconds of data
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )
        # Only 4 complete 1s windows
        assert len(windows) == 4

    def test_windowed_gauge_empty_window_uses_last_value(self):
        """Test windows with no samples are skipped for gauges."""
        ts = ServerMetricsTimeSeries()
        # Sparse data: samples at 0s, 3s (gap from 1-2s)
        add_gauge_samples(
            ts, "queue_depth", [10.0, 40.0], interval_ns=3 * NANOS_PER_SECOND
        )

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        # Gauges only create timeslices for windows with actual samples
        # Windows: [0-1) has sample, [1-2) empty (skipped), [2-3) has sample at boundary
        assert len(windows) == 2
        assert windows[0].avg == 10.0  # Sample at 0s
        assert windows[1].avg == 40.0  # Sample at 3s (included at boundary)

    def test_windowed_gauge_empty_data(self):
        """Test empty/insufficient data returns None."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0])  # Single sample

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "queue_depth"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
        )
        assert windows is None

    def test_windowed_gauge_invalid_window_size(self):
        """Test invalid window size raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_gauge_timeslices(
                get_gauge(ts, "queue_depth"),
                slice_duration=0.0,
                time_filter=make_time_filter(),
            )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_gauge_timeslices(
                get_gauge(ts, "queue_depth"),
                slice_duration=-1.0,
                time_filter=make_time_filter(),
            )


class TestComputeGaugeStatsWithTimeslices:
    """Test compute_gauge_stats integration with windowed stats."""

    def test_gauge_stats_includes_windows_when_slice_duration_specified(self):
        """Test gauge stats includes windows list when slice_duration is specified."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert result.timeslices is not None
        assert len(result.timeslices) == 4

    def test_gauge_stats_no_slice_duration(self):
        """Test slice_duration=None excludes windowed stats."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
            slice_duration=None,
        )

        assert result.timeslices is None
        # But still has overall stats
        assert result.stats is not None
        assert result.stats.avg == pytest.approx(30.0)
        assert result.stats.min == 10.0
        assert result.stats.max == 50.0

    def test_gauge_stats_custom_window_size(self):
        """Test custom window size in gauge stats."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [float(i * 10) for i in range(10)])

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=8 * NANOS_PER_SECOND),
            slice_duration=2.0,
        )

        assert result.timeslices is not None
        # 4 complete 2s windows
        assert len(result.timeslices) == 4
        for wp in result.timeslices:
            assert wp.end_ns - wp.start_ns == 2 * NANOS_PER_SECOND

    def test_gauge_stats_insufficient_data_for_windows(self):
        """Test insufficient data for fixed-size windows returns None timeslices."""
        ts = ServerMetricsTimeSeries()
        # Only 0.5s of data - not enough for 1s fixed windows
        add_gauge_samples(
            ts, "queue_depth", [10.0, 20.0], interval_ns=NANOS_PER_SECOND // 2
        )

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=0.5 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # Basic stats still computed
        assert result.stats is not None
        assert result.stats.avg == pytest.approx(15.0)
        # Timeslices now include partial slice
        assert result.timeslices is not None
        assert len(result.timeslices) == 1
        assert not result.timeslices[0].is_complete

    def test_gauge_stats_constant_gauge_uses_value(self):
        """Test constant gauge (std=0) uses value instead of stats."""
        ts = ServerMetricsTimeSeries()
        # All same value
        add_gauge_samples(ts, "queue_depth", [42.0, 42.0, 42.0, 42.0, 42.0])

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # Constant gauge: stats with all values = 42.0, std = 0
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.std == 0.0
        assert result.stats.p50 == 42.0

    def test_gauge_stats_sparse_samples_uses_per_interval_mode(self):
        """Test that sparse samples create fixed-duration timeslices around each sample.

        With sparse samples (large gaps between samples), creates fixed-duration
        timeslices for each sample rather than filling all gaps.
        """
        ts = ServerMetricsTimeSeries()
        # 3 samples over 20 seconds with 2-second requested timeslices
        timestamps_ns = [
            1765612482013463899,  # t=0s
            1765612492013463899,  # t=10s
            1765612502013463899,  # t=20s (end)
        ]
        values = [30.0, 30.0, 31.0]
        samples = list(zip(timestamps_ns, values, strict=False))
        add_gauge_samples_with_timestamps(ts, "test_metric", samples)

        # Use actual timestamp range for time_filter
        result = _compute_gauge_stats(
            get_gauge(ts, "test_metric"),
            make_time_filter(start_ns=timestamps_ns[0], end_ns=timestamps_ns[-1]),
            slice_duration=2.0,
        )

        # Overall stats should include all 3 samples
        assert result.stats is not None
        assert result.stats.avg == pytest.approx(30.333333, rel=0.01)
        assert result.stats.min == 30.0
        assert result.stats.max == 31.0

        # Creates fixed-duration timeslices for each sample (3 timeslices, one per sample)
        assert result.timeslices is not None
        assert len(result.timeslices) == 3, (
            f"Expected 3 timeslices (one per sample), got {len(result.timeslices)}"
        )

        # First timeslice: interval from sample 0 to sample 1 (value at end = 30.0)
        assert result.timeslices[0].avg == 30.0
        assert result.timeslices[0].min == 30.0
        assert result.timeslices[0].max == 30.0

        # Second timeslice: interval from sample 1 to sample 2 (forward-filled from sample 1 = 30.0)
        assert result.timeslices[1].avg == 30.0
        assert result.timeslices[1].min == 30.0
        assert result.timeslices[1].max == 30.0


class TestComputeHistogramTimeslices:
    """Test compute_windowed_histogram_avg function."""

    def test_basic_windowed_histogram_avg(self) -> None:
        """Test basic windowed average computation for histogram."""
        ts = ServerMetricsTimeSeries()
        # Create histogram with consistent avg of 0.1 per window
        # Window 1: sum goes 0->10 (delta=10), count goes 0->100 (delta=100) -> avg=0.1
        # Window 2: sum goes 10->30 (delta=20), count goes 100->300 (delta=200) -> avg=0.1
        # Window 3: sum goes 30->70 (delta=40), count goes 300->700 (delta=400) -> avg=0.1
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 30.0, 300)),
                (3 * NANOS_PER_SECOND, hist(buckets, 70.0, 700)),
            ],
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
        )

        # 10 timeslices total (fills gaps with zeros)
        assert len(windows) == 10
        # First 3 windows have data with avg=0.1
        for i in range(3):
            assert windows[i].avg == pytest.approx(0.1, rel=0.01)
            assert windows[i].count > 0
        # Remaining 7 windows are zero-filled
        for i in range(3, 10):
            assert windows[i].count == 0
            assert windows[i].sum == 0.0

    def test_windowed_histogram_varying_avg(self) -> None:
        """Test histogram with varying average per window."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # Window 1: sum delta=10, count delta=100 -> avg=0.1
        # Window 2: sum delta=40, count delta=100 -> avg=0.4
        # Window 3: sum delta=20, count delta=100 -> avg=0.2
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 50.0, 200)),
                (3 * NANOS_PER_SECOND, hist(buckets, 70.0, 300)),
            ],
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
        )

        # 10 timeslices total (fills gaps with zeros)
        assert len(windows) == 10
        # First 3 windows have data with varying averages
        assert windows[0].avg == pytest.approx(0.1, rel=0.01)
        assert windows[1].avg == pytest.approx(0.4, rel=0.01)
        assert windows[2].avg == pytest.approx(0.2, rel=0.01)
        # Remaining 7 windows are zero-filled
        for i in range(3, 10):
            assert windows[i].count == 0

    def test_windowed_histogram_custom_window_size(self) -> None:
        """Test histogram with custom window size."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # 5 samples at 1s intervals covering 4 seconds
        # With 2s window: window 1 (0-2s), window 2 (2-4s)
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 20.0, 200)),
                (3 * NANOS_PER_SECOND, hist(buckets, 40.0, 400)),
                (4 * NANOS_PER_SECOND, hist(buckets, 50.0, 500)),
            ],
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=2.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        assert len(windows) == 2
        # Window 1: sum delta=20, count delta=200 -> avg=0.1
        # Window 2: sum delta=30, count delta=300 -> avg=0.1
        assert windows[0].avg == pytest.approx(0.1, rel=0.01)
        assert windows[1].avg == pytest.approx(0.1, rel=0.01)

    def test_windowed_histogram_with_time_filter(self) -> None:
        """Test histogram windowed avg respects time filter."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # 6 samples at 1s intervals
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),  # t=0
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),  # t=1
                (2 * NANOS_PER_SECOND, hist(buckets, 20.0, 200)),  # t=2
                (3 * NANOS_PER_SECOND, hist(buckets, 30.0, 300)),  # t=3
                (4 * NANOS_PER_SECOND, hist(buckets, 40.0, 400)),  # t=4
                (5 * NANOS_PER_SECOND, hist(buckets, 50.0, 500)),  # t=5
            ],
        )

        # Filter to only 2-5s (should use t=1 as reference)
        time_filter = TimeRangeFilter(
            start_ns=2 * NANOS_PER_SECOND,
            end_ns=5 * NANOS_PER_SECOND,
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=time_filter,
        )

        # Should get windows covering 1-2, 2-3, 3-4 (reference at t=1)
        assert len(windows) >= 2

    def test_windowed_histogram_no_observations_window_skipped(self) -> None:
        """Test that windows with no observations are included with zero values."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # Window 1: sum delta=10, count delta=100 -> avg=0.1
        # Window 2: sum delta=0, count delta=0 -> zero-filled
        # Window 3: sum delta=10, count delta=100 -> avg=0.1
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 10.0, 100)),  # No change
                (3 * NANOS_PER_SECOND, hist(buckets, 20.0, 200)),
            ],
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
        )

        # 10 timeslices total (fills gaps with zeros)
        assert len(windows) == 10
        assert windows[0].count == 100
        assert windows[1].count == 0  # Middle window with no observations
        assert windows[2].count == 100

    def test_windowed_histogram_empty_data(self) -> None:
        """Test histogram with insufficient data returns None."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(ts, "latency", [(0, hist(buckets, 0.0, 0))])

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=10 * NANOS_PER_SECOND),
        )

        assert windows is None

    def test_windowed_histogram_invalid_window_size(self) -> None:
        """Test that invalid window size raises ValueError."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(
            ts,
            "latency",
            [(0, hist(buckets, 0.0, 0)), (NANOS_PER_SECOND, hist(buckets, 10.0, 100))],
        )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_histogram_timeslices(
                get_histogram(ts, "latency"),
                slice_duration=0,
                time_filter=make_time_filter(),
            )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_histogram_timeslices(
                get_histogram(ts, "latency"),
                slice_duration=-1,
                time_filter=make_time_filter(),
            )


class TestComputeHistogramStatsWithTimeslices:
    """Test histogram stats integration with windowed averages."""

    def test_histogram_stats_includes_averages_by_default(self) -> None:
        """Test that _compute_histogram_stats includes averages by default."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 30.0, 300)),
                (3 * NANOS_PER_SECOND, hist(buckets, 60.0, 600)),
                (4 * NANOS_PER_SECOND, hist(buckets, 100.0, 1000)),
            ],
        )

        stats = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert stats.timeslices is not None
        assert len(stats.timeslices) == 4
        for wp in stats.timeslices:
            assert wp.avg == pytest.approx(0.1, rel=0.01)

    def test_histogram_stats_no_slice_duration(self) -> None:
        """Test that averages is None when slice_duration is None."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 20.0, 200)),
            ],
        )

        stats = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            slice_duration=None,
        )

        assert stats.timeslices is None

    def test_histogram_stats_custom_window_size(self) -> None:
        """Test histogram stats with custom window size."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (NANOS_PER_SECOND, hist(buckets, 10.0, 100)),
                (2 * NANOS_PER_SECOND, hist(buckets, 20.0, 200)),
                (3 * NANOS_PER_SECOND, hist(buckets, 30.0, 300)),
                (4 * NANOS_PER_SECOND, hist(buckets, 40.0, 400)),
                (5 * NANOS_PER_SECOND, hist(buckets, 50.0, 500)),
            ],
        )

        stats = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=2.0,
        )

        # 5 seconds of data with 2s windows = 2 complete windows
        assert stats.timeslices is not None
        assert len(stats.timeslices) == 2

    def test_histogram_stats_insufficient_data_for_windows(self) -> None:
        """Test histogram stats creates timeslices up to time filter, filling gaps with zeros."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # Only 0.5 seconds of data - creates timeslices up to time filter
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),
                (int(0.5 * NANOS_PER_SECOND), hist(buckets, 5.0, 50)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # Should still have regular stats
        assert result.stats is not None
        assert result.stats.avg == pytest.approx(0.1, rel=0.01)
        # Creates 3 timeslices (0-1s has data, 1-2s and 2-3s are zero-filled)
        assert result.timeslices is not None
        assert len(result.timeslices) == 3
        assert result.timeslices[0].count == 50
        assert result.timeslices[1].count == 0
        assert result.timeslices[2].count == 0

    def test_histogram_stats_varying_latency_over_time(self) -> None:
        """Test histogram that shows varying latency over windows."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 0.0, "+Inf": 0.0}
        # Latency starts low, increases, then decreases
        # Window 1: avg=0.05 (fast responses)
        # Window 2: avg=0.2 (slow responses)
        # Window 3: avg=0.1 (recovering)
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist(buckets, 0.0, 0)),  # t=0
                (
                    NANOS_PER_SECOND,
                    hist(buckets, 5.0, 100),
                ),  # t=1, delta: sum=5, count=100 -> avg=0.05
                (
                    2 * NANOS_PER_SECOND,
                    hist(buckets, 25.0, 200),
                ),  # t=2, delta: sum=20, count=100 -> avg=0.2
                (
                    3 * NANOS_PER_SECOND,
                    hist(buckets, 35.0, 300),
                ),  # t=3, delta: sum=10, count=100 -> avg=0.1
            ],
        )

        stats = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            make_time_filter(start_ns=0, end_ns=6 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert stats.timeslices is not None
        # 6 timeslices total (fills gaps with zeros)
        assert len(stats.timeslices) == 6
        # First 3 windows have data with varying latencies
        assert stats.timeslices[0].avg == pytest.approx(0.05, rel=0.01)
        assert stats.timeslices[1].avg == pytest.approx(0.2, rel=0.01)
        assert stats.timeslices[2].avg == pytest.approx(0.1, rel=0.01)
        # Remaining 3 windows are zero-filled
        for i in range(3, 6):
            assert stats.timeslices[i].count == 0


# =============================================================================
# Out-of-Order Data Handling Tests
# =============================================================================


class TestOutOfOrderDataHandling:
    """Test that timeslice functions handle out-of-order timestamps correctly.

    These tests verify that data arriving in non-chronological order
    (e.g., due to network delays or clock skew) is sorted before
    timeslice calculations are performed.
    """

    def test_counter_timeslices_out_of_order(self):
        """Test counter timeslices handle out-of-order data."""
        ts = ServerMetricsTimeSeries()
        # Add samples OUT OF ORDER: t=2, t=0, t=3, t=1, t=4
        add_counter_samples_with_timestamps(
            ts,
            "requests",
            [
                (2 * NANOS_PER_SECOND, 200.0),
                (0, 0.0),
                (3 * NANOS_PER_SECOND, 300.0),
                (1 * NANOS_PER_SECOND, 100.0),
                (4 * NANOS_PER_SECOND, 400.0),
            ],
        )

        rates = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # Should produce same result as sorted data: 4 windows, 100/s each
        assert len(rates) == 4
        for rp in rates:
            assert rp.rate == 100.0

    def test_gauge_timeslices_out_of_order(self):
        """Test gauge timeslices handle out-of-order data."""
        ts = ServerMetricsTimeSeries()
        # Add samples OUT OF ORDER with varying values
        # Sorted: t=0: 10, t=1: 20, t=2: 30, t=3: 40, t=4: 50
        add_gauge_samples_with_timestamps(
            ts,
            "cache_usage",
            [
                (2 * NANOS_PER_SECOND, 30.0),
                (0, 10.0),
                (4 * NANOS_PER_SECOND, 50.0),
                (1 * NANOS_PER_SECOND, 20.0),
                (3 * NANOS_PER_SECOND, 40.0),
            ],
        )

        windows = _compute_gauge_timeslices(
            get_gauge(ts, "cache_usage"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # Should produce same result as sorted data
        assert len(windows) == 4
        # Window [0-1): contains value 10
        assert windows[0].avg == 10.0
        # Window [1-2): contains value 20
        assert windows[1].avg == 20.0
        # Window [2-3): contains value 30
        assert windows[2].avg == 30.0
        # Window [3-4): contains values 40, 50 (sample at t=4 included at boundary)
        assert windows[3].avg == 45.0
        assert windows[3].min == 40.0
        assert windows[3].max == 50.0

    def test_histogram_timeslices_out_of_order(self):
        """Test histogram windowed avg handles out-of-order data."""
        ts = ServerMetricsTimeSeries()
        buckets = {"0.1": 100, "0.5": 200, "1.0": 300, "+Inf": 400}

        # Add snapshots OUT OF ORDER
        # Sorted: t=0: sum=0, count=0; t=1: sum=5, count=100; etc.
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (2 * NANOS_PER_SECOND, hist(buckets, 25.0, 200)),  # t=2
                (0, hist(buckets, 0.0, 0)),  # t=0
                (3 * NANOS_PER_SECOND, hist(buckets, 35.0, 300)),  # t=3
                (1 * NANOS_PER_SECOND, hist(buckets, 5.0, 100)),  # t=1
            ],
        )

        windows = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # 4 timeslices total (fills gaps with zeros)
        assert len(windows) == 4
        # Window [0-1): delta sum=5, delta count=100 -> avg=0.05
        assert windows[0].avg == pytest.approx(0.05, rel=0.01)
        # Window [1-2): delta sum=20, delta count=100 -> avg=0.2
        assert windows[1].avg == pytest.approx(0.2, rel=0.01)
        # Window [2-3): delta sum=10, delta count=100 -> avg=0.1
        assert windows[2].avg == pytest.approx(0.1, rel=0.01)
        # Window [3-4): no data
        assert windows[3].count == 0

    def test_counter_stats_with_timeslices_out_of_order_rate_statistics(self):
        """Test counter stats (rate_min/max/avg) work with out-of-order data."""
        ts = ServerMetricsTimeSeries()
        # Add samples OUT OF ORDER with variable rates
        # Sorted: 0->50, 50->150, 150->300, 300->500 (rates: 50, 100, 150, 200)
        add_counter_samples_with_timestamps(
            ts,
            "requests",
            [
                (3 * NANOS_PER_SECOND, 300.0),
                (0, 0.0),
                (4 * NANOS_PER_SECOND, 500.0),
                (2 * NANOS_PER_SECOND, 150.0),
                (1 * NANOS_PER_SECOND, 50.0),
            ],
        )

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert result.stats is not None
        assert result.stats.rate_min == 50.0
        assert result.stats.rate_max == 200.0
        assert result.stats.rate_avg == pytest.approx(125.0)  # (50+100+150+200)/4

    def test_gauge_stats_with_timeslices_out_of_order(self):
        """Test gauge stats compute correct percentiles with out-of-order data."""
        ts = ServerMetricsTimeSeries()
        # Add 10 samples out of order: values 1-10
        out_of_order_samples = [
            (5 * NANOS_PER_SECOND, 6.0),
            (0, 1.0),
            (9 * NANOS_PER_SECOND, 10.0),
            (2 * NANOS_PER_SECOND, 3.0),
            (7 * NANOS_PER_SECOND, 8.0),
            (1 * NANOS_PER_SECOND, 2.0),
            (4 * NANOS_PER_SECOND, 5.0),
            (8 * NANOS_PER_SECOND, 9.0),
            (3 * NANOS_PER_SECOND, 4.0),
            (6 * NANOS_PER_SECOND, 7.0),
        ]
        add_gauge_samples_with_timestamps(ts, "queue_depth", out_of_order_samples)

        result = _compute_gauge_stats(
            get_gauge(ts, "queue_depth"),
            make_time_filter(start_ns=0, end_ns=9 * NANOS_PER_SECOND),
        )

        # Stats should match sorted data: values [1,2,3,4,5,6,7,8,9,10]
        assert result.stats is not None
        assert result.stats.avg == pytest.approx(5.5)
        assert result.stats.min == 1.0
        assert result.stats.max == 10.0

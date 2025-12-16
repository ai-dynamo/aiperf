# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for basic export statistics: gauge, counter, histogram, and edge cases."""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.server_metrics.export_stats import (
    _compute_counter_stats,
    _compute_gauge_stats,
    _compute_histogram_stats,
    _compute_histogram_timeslices,
)
from aiperf.server_metrics.storage import (
    ServerMetricKey,
    ServerMetricsTimeSeries,
)
from tests.unit.server_metrics.helpers import (
    add_counter_samples,
    add_gauge_samples,
    add_histogram_snapshots,
    get_counter,
    get_gauge,
    get_histogram,
    hist,
    make_time_filter,
)

# =============================================================================
# Gauge Tests
# =============================================================================


class TestGaugeExportStats:
    """Test gauge export statistics computation."""

    def test_gauge_basic_statistics(self):
        """Test gauge computes correct avg, min, max, std."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue_depth", [10.0, 20.0, 30.0, 40.0, 50.0])

        result = _compute_gauge_stats(get_gauge(ts, "queue_depth"), make_time_filter())

        assert result.stats is not None
        assert result.stats.avg == 30.0  # (10+20+30+40+50)/5
        assert result.stats.min == 10.0
        assert result.stats.max == 50.0
        # Sample std (ddof=1): sqrt(500/4) = 15.811
        assert result.stats.std == pytest.approx(15.811, rel=0.01)

    def test_gauge_percentiles(self):
        """Test gauge computes all percentiles including p1, p5, p10, p25."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "kv_cache", [float(i) for i in range(100)])

        result = _compute_gauge_stats(get_gauge(ts, "kv_cache"), make_time_filter())

        assert result.stats is not None
        # Lower percentiles (p1, p5, p10, p25)
        assert result.stats.p1 == pytest.approx(0.99, rel=0.01)
        assert result.stats.p5 == pytest.approx(4.95, rel=0.01)
        assert result.stats.p10 == pytest.approx(9.9, rel=0.01)
        assert result.stats.p25 == pytest.approx(24.75, rel=0.01)
        # Standard percentiles
        assert result.stats.p50 == pytest.approx(49.5, rel=0.01)
        assert result.stats.p75 == pytest.approx(74.25, rel=0.01)
        assert result.stats.p90 == pytest.approx(89.1, rel=0.01)
        assert result.stats.p95 == pytest.approx(94.05, rel=0.01)
        assert result.stats.p99 == pytest.approx(98.01, rel=0.01)

    def test_gauge_with_time_filter(self):
        """Test gauge export stats respect time filtering (warmup exclusion)."""
        ts = ServerMetricsTimeSeries()
        # Warmup (0-9) + actual run (10-19)
        add_gauge_samples(ts, "queue", [float(i) for i in range(20)])

        # Filter excludes warmup (first 10 seconds)
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        # Should only include values 10-19
        assert result.stats is not None
        assert result.stats.avg == 14.5  # (10+11+...+19)/10
        assert result.stats.min == 10.0
        assert result.stats.max == 19.0


# =============================================================================
# Counter Tests
# =============================================================================


class TestCounterExportStats:
    """Test counter export statistics computation."""

    def test_counter_rate_calculation(self):
        """Test counter computes correct delta and rates."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 250.0, 450.0, 700.0])

        # Data spans 0-4s, use time_filter covering exactly that range
        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        assert result.stats is not None
        assert result.stats.total == 700.0
        assert result.stats.rate == 175.0  # 700 / 4s (overall throughput)
        # Point-to-point rates: [100, 150, 200, 250]
        assert result.stats.rate_avg == 175.0  # time-weighted: (100+150+200+250) / 4
        assert result.stats.rate_min == 100.0
        assert result.stats.rate_max == 250.0
        # Sample std (ddof=1): sqrt(12500/3) = 64.55
        assert result.stats.rate_std == pytest.approx(64.55, rel=0.01)

    def test_counter_with_warmup_filter(self):
        """Test counter delta calculation excludes warmup period."""
        ts = ServerMetricsTimeSeries()
        # Values: 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800
        add_counter_samples(ts, "requests", [float(i * 200) for i in range(10)])

        # Exclude warmup (first 5 seconds)
        time_filter = TimeRangeFilter(start_ns=5 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_counter_stats(get_counter(ts, "requests"), time_filter)

        # ref_idx = 4 (value = 800 at 4s), final = 1800 at 9s
        assert result.stats is not None
        assert result.stats.total == 1000.0  # 1800 - 800
        assert result.stats.rate == 200.0

    def test_counter_change_point_rates(self):
        """Test counter rates are computed between change points, not every sample.

        This tests the scenario where we sample faster than the server updates.
        Without change-point detection, we'd get misleading rates like:
        0/s, 0/s, 0/s, 1000/s (when the change is finally observed)

        With change-point detection, we correctly compute:
        100/s (the change attributed to the full time period)
        """
        ts = ServerMetricsTimeSeries()

        # Simulate sampling at 10Hz but server updating at ~1Hz
        # t=0.0s: 0, t=0.1s: 0, ..., t=1.0s: 100, t=1.1s: 100, ..., t=2.0s: 250
        interval_ns = NANOS_PER_SECOND // 10  # 0.1s intervals

        # First second: value stays at 0, then jumps to 100
        add_counter_samples(
            ts, "requests", [0.0] * 10, start_ns=0, interval_ns=interval_ns
        )
        # At t=1.0s, value becomes 100
        add_counter_samples(
            ts,
            "requests",
            [100.0] * 10,
            start_ns=10 * interval_ns,
            interval_ns=interval_ns,
        )
        # At t=2.0s, value becomes 250
        add_counter_samples(
            ts, "requests", [250.0], start_ns=20 * interval_ns, interval_ns=interval_ns
        )

        # Data spans 0-2s, use time_filter covering exactly that range
        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # Total: 250 over 2.0s
        assert result.stats is not None
        assert result.stats.total == 250.0
        assert result.stats.rate == 125.0  # 250 / 2.0s

        # Change points: t=0 (0), t=1.0 (100), t=2.0 (250)
        # Rates: 100/1.0s = 100/s, 150/1.0s = 150/s
        assert result.stats.rate_avg == 125.0  # (100 + 150) / 2
        assert result.stats.rate_min == 100.0
        assert result.stats.rate_max == 150.0

    def test_counter_no_changes_uses_value(self):
        """Test counter with no value changes uses value instead of stats."""
        ts = ServerMetricsTimeSeries()

        # Value never changes
        add_counter_samples(ts, "requests", [100.0] * 10)

        result = _compute_counter_stats(
            get_counter(ts, "requests"), make_time_filter(), slice_duration=1.0
        )

        # No change: stats with total=0, rate=0
        assert result.stats is not None
        assert result.stats.total == 0.0
        assert result.stats.rate == 0.0

    def test_counter_rate_avg_is_simple_mean_of_windows(self):
        """Test that rate_avg is the simple mean of windowed rates.

        With windowed rates, rate_avg is the average of all window rates.
        Each 1-second window contributes equally regardless of data point spacing.
        """
        ts = ServerMetricsTimeSeries()

        # 5 samples at 1s intervals: 0, 100, 200, 300, 400
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        # Data spans 0-4s, use time_filter covering exactly that range
        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # 4 windows, each with rate 100/s
        assert result.stats is not None
        assert result.stats.rate_avg == 100.0  # Simple mean of windowed rates
        assert result.stats.rate_min == 100.0
        assert result.stats.rate_max == 100.0

        # Variable rates
        ts2 = ServerMetricsTimeSeries()
        # Rates per window: 50, 100, 150, 200
        add_counter_samples(ts2, "requests", [0.0, 50.0, 150.0, 300.0, 500.0])

        # Data spans 0-4s, use time_filter covering exactly that range
        result2 = _compute_counter_stats(
            get_counter(ts2, "requests"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )

        # Simple mean: (50 + 100 + 150 + 200) / 4 = 125
        assert result2.stats is not None
        assert result2.stats.rate_avg == 125.0
        assert result2.stats.rate_min == 50.0
        assert result2.stats.rate_max == 200.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogramExportStats:
    """Test histogram export statistics computation."""

    def test_histogram_basic_stats(self):
        """Test histogram computes count, sum, avg, count_rate."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 30.0, "1.0": 120.0, "+Inf": 250.0}, 65.0, 250.0),
                ),
            ],
        )

        result = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        # Delta: count 250-100=150, sum 65-25=40
        assert result.stats is not None
        assert result.stats.count == 150
        assert result.stats.sum == 40.0
        assert result.stats.avg == pytest.approx(40.0 / 150, rel=0.01)
        assert result.stats.count_rate == 150.0  # 150 obs / 1s

    def test_histogram_bucket_delta_uses_correct_reference(self):
        """CRITICAL: Test histogram bucket deltas use ref_idx, not first snapshot."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                # Warmup snapshot
                (0, hist({"0.1": 100.0, "1.0": 500.0, "+Inf": 1000.0}, 250.0, 1000.0)),
                # Reference snapshot (last before filter starts)
                (
                    5 * NANOS_PER_SECOND,
                    hist({"0.1": 200.0, "1.0": 800.0, "+Inf": 1500.0}, 400.0, 1500.0),
                ),
                # Final snapshot
                (
                    10 * NANOS_PER_SECOND,
                    hist({"0.1": 300.0, "1.0": 1200.0, "+Inf": 2000.0}, 600.0, 2000.0),
                ),
            ],
        )

        # Filter: exclude warmup (first 5 seconds)
        time_filter = TimeRangeFilter(start_ns=6 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_histogram_stats(get_histogram(ts, "ttft"), time_filter)

        # Bucket deltas: final - ref_idx (NOT final - first!)
        assert result.buckets == {"0.1": 100, "1.0": 400, "+Inf": 500}
        assert result.stats is not None
        assert result.stats.count == 500  # 2000 - 1500
        assert result.stats.sum == 200.0  # 600 - 400

    def test_histogram_rate(self):
        """Test histogram rate calculation."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                (
                    5 * NANOS_PER_SECOND,
                    hist({"1.0": 300.0, "+Inf": 300.0}, 150.0, 300.0),
                ),
            ],
        )

        result = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        assert result.stats is not None
        assert isinstance(result.stats.count_rate, float)
        assert result.stats.count_rate == 40.0  # 200 obs / 5s

    def test_histogram_no_data_raises_key_error(self):
        """Test histogram raises KeyError when metric doesn't exist."""
        ts = ServerMetricsTimeSeries()

        with pytest.raises(KeyError):
            _ = ts.metrics[ServerMetricKey("nonexistent", ())]

    def test_histogram_bucket_counter_reset(self):
        """Test histogram returns None for buckets when counter reset detected.

        When a counter resets (server restart), the final value will be less than
        the reference. We return None to indicate invalid/incomplete data.
        """
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                # Before reset: high counts
                (
                    0,
                    hist(
                        {"0.1": 1000.0, "1.0": 5000.0, "+Inf": 10000.0}, 2500.0, 10000.0
                    ),
                ),
                # After reset: low counts (server restarted)
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 50.0, "1.0": 200.0, "+Inf": 500.0}, 125.0, 500.0),
                ),
            ],
        )

        stats = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        # Negative deltas indicate reset - return None for invalid data
        assert stats.buckets is None

    def test_histogram_percentile_estimates(self):
        """Test histogram computes estimated percentiles from buckets including p1, p5, p10, p25."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 0.0, "0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 1000 observations: 100 in [0,0.1], 400 in (0.1,0.5], 400 in (0.5,1.0], 100 in (1.0,+Inf]
                (
                    NANOS_PER_SECOND,
                    hist(
                        {"0.1": 100.0, "0.5": 500.0, "1.0": 900.0, "+Inf": 1000.0},
                        500.0,
                        1000.0,
                    ),
                ),
            ],
        )

        stats = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        # Percentile estimates should be flat fields on stats
        assert stats.stats is not None

        # Lower percentiles (p1, p5, p10) should be in the first bucket range [0, 0.1]
        assert stats.stats.p1_estimate is not None
        assert 0.0 <= stats.stats.p1_estimate <= 0.1

        assert stats.stats.p5_estimate is not None
        assert 0.0 <= stats.stats.p5_estimate <= 0.1

        assert stats.stats.p10_estimate is not None
        assert 0.0 <= stats.stats.p10_estimate <= 0.15

        # p25 should be in the (0.1, 0.5] bucket range
        assert stats.stats.p25_estimate is not None
        assert 0.1 <= stats.stats.p25_estimate <= 0.5

        # p50 should be around 0.3-0.5 (in the 0.1-0.5 bucket range)
        assert stats.stats.p50_estimate is not None
        assert 0.1 <= stats.stats.p50_estimate <= 0.6

        # p90 should be around 0.75-1.0 (in the 0.5-1.0 bucket range)
        assert stats.stats.p90_estimate is not None
        assert 0.5 <= stats.stats.p90_estimate <= 1.1

        # p95 and p99 - with +Inf bucket, these should be estimated above 1.0
        assert stats.stats.p95_estimate is not None
        assert stats.stats.p99_estimate is not None

    def test_histogram_percentile_estimates_none_on_counter_reset(self):
        """Test histogram percentile estimates are None when counter reset detected."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 1000.0, "+Inf": 2000.0}, 500.0, 2000.0)),
                # Counter reset - values decreased
                (NANOS_PER_SECOND, hist({"0.1": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
            ],
        )

        stats = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        assert stats.buckets is None
        # When counter reset detected, percentile estimates should be None
        assert stats.stats is None or stats.stats.p50_estimate is None


# =============================================================================
# Histogram Timeslice Tests
# =============================================================================


class TestHistogramTimeslices:
    """Test histogram timeslice computation including bucket deltas."""

    def test_histogram_timeslices_include_bucket_deltas(self):
        """Test that histogram timeslices include bucket deltas for each slice."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"0.1": 0.0, "0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # After 1s: 10 in [0,0.1], 20 in (0.1,0.5], 15 in (0.5,1.0], 5 in +Inf
                (
                    NANOS_PER_SECOND,
                    hist({"0.1": 10.0, "0.5": 30.0, "1.0": 45.0, "+Inf": 50.0}, 25.0, 50.0),
                ),
                # After 2s: additional 5 in [0,0.1], 10 in (0.1,0.5], 20 in (0.5,1.0], 15 in +Inf
                (
                    2 * NANOS_PER_SECOND,
                    hist({"0.1": 15.0, "0.5": 45.0, "1.0": 80.0, "+Inf": 100.0}, 75.0, 100.0),
                ),
            ],
        )  # fmt: skip

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        assert timeslices is not None
        assert len(timeslices) == 2

        # First timeslice (0s to 1s)
        ts1 = timeslices[0]
        assert ts1.count == 50
        assert ts1.sum == 25.0
        assert ts1.buckets is not None
        assert ts1.buckets == {"0.1": 10, "0.5": 30, "1.0": 45, "+Inf": 50}

        # Second timeslice (1s to 2s)
        ts2 = timeslices[1]
        assert ts2.count == 50
        assert ts2.sum == 50.0
        assert ts2.buckets is not None
        assert ts2.buckets == {"0.1": 5, "0.5": 15, "1.0": 35, "+Inf": 50}

    def test_histogram_timeslices_bucket_deltas_none_on_reset(self):
        """Test that bucket deltas are None when counter reset detected in timeslice."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"0.5": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # Normal data at 1s
                (NANOS_PER_SECOND, hist({"0.5": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                # Counter reset at 2s (values decreased)
                (2 * NANOS_PER_SECOND, hist({"0.5": 10.0, "+Inf": 20.0}, 10.0, 20.0)),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        # First timeslice should have buckets
        assert timeslices is not None
        assert len(timeslices) >= 1
        assert timeslices[0].buckets is not None

    def test_histogram_timeslices_basic_stats(self):
        """Test histogram timeslices compute count, sum, avg correctly."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                (
                    2 * NANOS_PER_SECOND,
                    hist({"1.0": 250.0, "+Inf": 250.0}, 125.0, 250.0),
                ),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "ttft"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        assert timeslices is not None
        assert len(timeslices) == 2

        # First timeslice
        assert timeslices[0].count == 100
        assert timeslices[0].sum == 50.0
        assert timeslices[0].avg == 0.5

        # Second timeslice
        assert timeslices[1].count == 150
        assert timeslices[1].sum == 75.0
        assert timeslices[1].avg == 0.5

    def test_histogram_timeslices_returns_none_when_insufficient_data(self):
        """Test returns None when not enough data for timeslices."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "ttft"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=0.5 * NANOS_PER_SECOND),
        )

        assert timeslices is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample_gauge(self):
        """Test gauge with single sample returns stats with constant value."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0])

        result = _compute_gauge_stats(get_gauge(ts, "queue"), make_time_filter())

        # Single sample: stats with all values = 42.0, std = 0
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.std == 0.0
        assert result.stats.p50 == 42.0

    def test_empty_time_range(self):
        """Test time filter that excludes all data returns None."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0], start_ns=NANOS_PER_SECOND)  # Data at 1s

        # Time filter starts after all data (2s), excludes the 1s data point
        time_filter = TimeRangeFilter(start_ns=2 * NANOS_PER_SECOND, end_ns=None)

        # When time filter excludes all data, returns None gracefully
        stats = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)
        assert stats is None

    def test_histogram_with_missing_buckets(self):
        """Test histogram handles missing buckets gracefully."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
                # Second snapshot missing "1.0" bucket
                (NANOS_PER_SECOND, hist({"0.1": 20.0, "+Inf": 150.0}, 35.0, 150.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "ttft"),
            make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )
        assert result.stats is not None
        assert result.stats.count == 50  # 150 - 100

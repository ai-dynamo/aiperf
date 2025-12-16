# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggressive edge case tests for export_stats.py functions.

These tests focus on boundary conditions, empty data, warmup exclusion,
type consistency, and other edge cases that could cause runtime errors.
"""

import numpy as np
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.server_metrics.export_stats import (
    _compute_counter_stats,
    _compute_gauge_stats,
    _compute_gauge_timeslices,
    _compute_histogram_stats,
    _compute_histogram_timeslices,
    compute_stats,
)
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricsTimeSeries,
)
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

# =============================================================================
# Empty Time Series Tests (Critical Bug Coverage)
# =============================================================================


class TestEmptyTimeSeriesHandling:
    """Test handling of completely empty time series - critical for IndexError prevention."""

    def test_empty_scalar_time_series_gauge_returns_none(self) -> None:
        """Test empty ScalarTimeSeries for gauge returns None without error."""
        series = ScalarTimeSeries()
        assert len(series) == 0

        result = _compute_gauge_stats(
            series,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is None

    def test_empty_scalar_time_series_counter_returns_none(self) -> None:
        """Test empty ScalarTimeSeries for counter returns None without error."""
        series = ScalarTimeSeries()
        assert len(series) == 0

        result = _compute_counter_stats(
            series,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is None

    def test_empty_histogram_time_series_returns_none(self) -> None:
        """Test empty HistogramTimeSeries returns None without IndexError."""
        series = HistogramTimeSeries()
        assert len(series) == 0

        result = _compute_histogram_stats(
            series,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is None

    def test_empty_histogram_timeslices_returns_none(self) -> None:
        """Test empty HistogramTimeSeries timeslice computation returns None."""
        series = HistogramTimeSeries()
        assert len(series) == 0

        result = _compute_histogram_timeslices(
            series,
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is None

    def test_compute_stats_with_empty_gauge(self) -> None:
        """Test compute_stats dispatcher handles empty gauge series."""
        series = ScalarTimeSeries()

        result = compute_stats(
            metric_type=PrometheusMetricType.GAUGE,
            time_series=series,
            time_filter=make_time_filter(),
        )

        assert result is None

    def test_compute_stats_with_empty_counter(self) -> None:
        """Test compute_stats dispatcher handles empty counter series."""
        series = ScalarTimeSeries()

        result = compute_stats(
            metric_type=PrometheusMetricType.COUNTER,
            time_series=series,
            time_filter=make_time_filter(),
        )

        assert result is None

    def test_compute_stats_with_empty_histogram(self) -> None:
        """Test compute_stats dispatcher handles empty histogram series."""
        series = HistogramTimeSeries()

        result = compute_stats(
            metric_type=PrometheusMetricType.HISTOGRAM,
            time_series=series,
            time_filter=make_time_filter(),
        )

        assert result is None


# =============================================================================
# Warmup Exclusion Edge Cases
# =============================================================================


class TestWarmupExclusionEdgeCases:
    """Test warmup exclusion with various boundary conditions."""

    def test_all_data_in_warmup_gauge(self) -> None:
        """Test gauge returns None when all data falls in warmup period."""
        ts = ServerMetricsTimeSeries()
        # All data at t=0 to t=5s (warmup)
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0, 40.0, 50.0])

        # Filter starts after all data (t=10s)
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is None

    def test_all_data_in_warmup_counter(self) -> None:
        """Test counter returns None when all data falls in warmup period."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0, 300.0, 400.0])

        # Filter starts after all data
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_counter_stats(get_counter(ts, "requests"), time_filter)

        assert result is None

    def test_all_data_in_warmup_histogram(self) -> None:
        """Test histogram returns None when all data falls in warmup period."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 10.0, "+Inf": 10.0}, 5.0, 10.0)),
                (NANOS_PER_SECOND, hist({"1.0": 20.0, "+Inf": 20.0}, 10.0, 20.0)),
            ],
        )

        # Filter starts after all data
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_histogram_stats(get_histogram(ts, "latency"), time_filter)

        # Should return stats with 0 count delta since final - ref = 0 (both point to last)
        assert result is not None
        assert result.stats.count == 0

    def test_single_point_at_warmup_boundary(self) -> None:
        """Test single data point exactly at warmup boundary."""
        ts = ServerMetricsTimeSeries()
        # Single point exactly at filter start
        add_gauge_samples(ts, "queue", [42.0], start_ns=5 * NANOS_PER_SECOND)

        time_filter = TimeRangeFilter(
            start_ns=5 * NANOS_PER_SECOND, end_ns=10 * NANOS_PER_SECOND
        )
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is not None
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.std == 0.0

    def test_data_spans_warmup_boundary(self) -> None:
        """Test data correctly uses warmup reference for delta calculation."""
        ts = ServerMetricsTimeSeries()
        # Warmup: t=0,1,2,3,4 with values 0,100,200,300,400
        # Profiling: t=5,6,7,8,9 with values 500,600,700,800,900
        add_counter_samples(ts, "requests", [float(i * 100) for i in range(10)])

        time_filter = TimeRangeFilter(
            start_ns=5 * NANOS_PER_SECOND, end_ns=10 * NANOS_PER_SECOND
        )
        result = _compute_counter_stats(get_counter(ts, "requests"), time_filter)

        # Delta should be 900 - 400 = 500 (final - last_warmup_reference)
        assert result is not None
        assert result.stats.total == 500.0

    def test_no_warmup_data_uses_first_profiling_as_reference(self) -> None:
        """Test that when no warmup data exists, first profiling point is used."""
        ts = ServerMetricsTimeSeries()
        # Data starts at t=5s (no warmup data)
        add_counter_samples(
            ts, "requests", [100.0, 200.0, 300.0], start_ns=5 * NANOS_PER_SECOND
        )

        # Filter starts at t=5s (where data starts)
        time_filter = TimeRangeFilter(
            start_ns=5 * NANOS_PER_SECOND, end_ns=10 * NANOS_PER_SECOND
        )
        result = _compute_counter_stats(get_counter(ts, "requests"), time_filter)

        # No reference point, so first profiling point (100) is used as baseline
        # Delta = 300 - 100 = 200
        assert result is not None
        assert result.stats.total == 200.0


# =============================================================================
# Type Consistency Tests
# =============================================================================


class TestTypeConsistency:
    """Test type consistency for model fields."""

    def test_histogram_timeslice_count_is_int(self) -> None:
        """Test that histogram timeslice count is always an integer."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 100.0}, 50.0, 100.0)),
                (
                    2 * NANOS_PER_SECOND,
                    hist({"1.0": 100.0, "+Inf": 200.0}, 100.0, 200.0),
                ),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        assert timeslices is not None
        for ts_item in timeslices:
            assert isinstance(ts_item.count, int), (
                f"count should be int, got {type(ts_item.count)}"
            )

    def test_histogram_stats_count_is_int(self) -> None:
        """Test that histogram stats count is always an integer."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        assert isinstance(result.stats.count, int)

    def test_bucket_deltas_are_ints(self) -> None:
        """Test that bucket deltas in timeslices are integers."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (
                    NANOS_PER_SECOND,
                    hist({"0.5": 25.0, "1.0": 75.0, "+Inf": 100.0}, 50.0, 100.0),
                ),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert timeslices is not None
        for ts_item in timeslices:
            if ts_item.buckets:
                for le, count in ts_item.buckets.items():
                    assert isinstance(count, int), f"bucket {le} count should be int"


# =============================================================================
# Time Filter Boundary Precision
# =============================================================================


class TestTimeFilterBoundaryPrecision:
    """Test exact boundary conditions with time filters."""

    def test_data_exactly_at_start_boundary_included(self) -> None:
        """Test data point exactly at start_ns is included."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0], start_ns=5 * NANOS_PER_SECOND)

        # Filter starts exactly at data point
        time_filter = TimeRangeFilter(start_ns=5 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is not None
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.std == 0.0

    def test_data_exactly_at_end_boundary_included(self) -> None:
        """Test data point exactly at end_ns is included."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0], start_ns=5 * NANOS_PER_SECOND)

        # Filter ends exactly at data point
        time_filter = TimeRangeFilter(start_ns=0, end_ns=5 * NANOS_PER_SECOND)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is not None
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.std == 0.0

    def test_data_one_nanosecond_before_start_excluded(self) -> None:
        """Test data point 1ns before start_ns is excluded from filtered data."""
        ts = ServerMetricsTimeSeries()
        # Data at t=4.999999999s
        add_gauge_samples(ts, "queue", [42.0], start_ns=5 * NANOS_PER_SECOND - 1)

        # Filter starts at t=5s
        time_filter = TimeRangeFilter(start_ns=5 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        # Data is before filter start, so no data in range
        assert result is None

    def test_data_one_nanosecond_after_end_excluded(self) -> None:
        """Test data point 1ns after end_ns is excluded."""
        ts = ServerMetricsTimeSeries()
        # Data at t=5.000000001s
        add_gauge_samples(ts, "queue", [42.0], start_ns=5 * NANOS_PER_SECOND + 1)

        # Filter ends at t=5s
        time_filter = TimeRangeFilter(start_ns=0, end_ns=5 * NANOS_PER_SECOND)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        # Data is after filter end, so no data in range
        assert result is None


# =============================================================================
# Out-of-Order Data Handling
# =============================================================================


class TestOutOfOrderDataHandling:
    """Test handling of out-of-order data arrivals."""

    def test_out_of_order_gauge_sorted_correctly(self) -> None:
        """Test out-of-order gauge data is sorted before statistics."""
        ts = ServerMetricsTimeSeries()
        # Data arrives out of order
        add_gauge_samples_with_timestamps(
            ts,
            "queue",
            [
                (3 * NANOS_PER_SECOND, 30.0),  # Third
                (1 * NANOS_PER_SECOND, 10.0),  # First
                (2 * NANOS_PER_SECOND, 20.0),  # Second
            ],
        )

        result = _compute_gauge_stats(
            get_gauge(ts, "queue"),
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        assert result is not None
        # Values should be sorted: [10, 20, 30]
        assert result.stats.min == 10.0
        assert result.stats.max == 30.0
        assert result.stats.avg == 20.0

    def test_out_of_order_counter_maintains_monotonicity(self) -> None:
        """Test out-of-order counter data is sorted for correct delta calculation."""
        ts = ServerMetricsTimeSeries()
        # Counters arrive out of order but values are monotonically increasing by time
        add_counter_samples_with_timestamps(
            ts,
            "requests",
            [
                (3 * NANOS_PER_SECOND, 300.0),
                (1 * NANOS_PER_SECOND, 100.0),
                (2 * NANOS_PER_SECOND, 200.0),
            ],
        )

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=4 * NANOS_PER_SECOND),
        )

        # After sorting: [100, 200, 300], total delta = 300 - 100 = 200
        assert result is not None
        assert result.stats.total == 200.0


# =============================================================================
# Division by Zero and Edge Numerics
# =============================================================================


class TestDivisionByZeroAndNumerics:
    """Test division by zero scenarios and numeric edge cases."""

    def test_histogram_zero_count_delta_no_division_error(self) -> None:
        """Test histogram with zero count delta doesn't cause division by zero."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                # No change in counts
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        assert result.stats.count == 0
        # avg should be None or 0 when count is 0 (no division)
        assert result.stats.avg is None

    def test_counter_zero_duration_no_rate_calculation(self) -> None:
        """Test counter with single point (zero duration) doesn't compute rate."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [100.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        # Single point, no change: stats with total=0, rate=0
        assert result is not None
        assert result.stats is not None
        assert result.stats.total == 0.0
        assert result.stats.rate == 0.0

    def test_histogram_timeslice_zero_count_avg_is_zero(self) -> None:
        """Test histogram timeslice with zero count sets avg to 0, not NaN."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 50.0, "+Inf": 50.0}, 25.0, 50.0)),
                # No change in first second
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 50.0}, 25.0, 50.0)),
                # Change in second second
                (
                    2 * NANOS_PER_SECOND,
                    hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0),
                ),
            ],
        )

        timeslices = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        assert timeslices is not None
        # First timeslice has count_delta=0, avg should be 0.0 not NaN
        assert timeslices[0].count == 0
        assert timeslices[0].avg == 0.0
        assert not np.isnan(timeslices[0].avg)


# =============================================================================
# Large Value and Precision Tests
# =============================================================================


class TestLargeValuesAndPrecision:
    """Test handling of very large values and floating point precision."""

    def test_very_large_counter_values(self) -> None:
        """Test counter with very large values doesn't lose precision."""
        ts = ServerMetricsTimeSeries()
        large_base = 1e15
        add_counter_samples(
            ts, "requests", [large_base, large_base + 1000, large_base + 2000]
        )

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        assert result is not None
        # Delta should be exactly 2000
        assert result.stats.total == 2000.0

    def test_very_small_gauge_values(self) -> None:
        """Test gauge with very small values maintains precision."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "ratio", [1e-10, 2e-10, 3e-10])

        result = _compute_gauge_stats(
            get_gauge(ts, "ratio"),
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        assert result is not None
        assert result.stats.min == pytest.approx(1e-10, rel=1e-6)
        assert result.stats.max == pytest.approx(3e-10, rel=1e-6)

    def test_histogram_very_large_bucket_counts(self) -> None:
        """Test histogram with very large bucket counts."""
        ts = ServerMetricsTimeSeries()
        large_count = 1e12
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (
                    NANOS_PER_SECOND,
                    hist(
                        {"1.0": large_count, "+Inf": large_count},
                        large_count / 2,
                        large_count,
                    ),
                ),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        assert result.stats.count == int(large_count)


# =============================================================================
# Multiple Data Points at Same Timestamp
# =============================================================================


class TestSameTimestampHandling:
    """Test handling of multiple data points at the same timestamp."""

    def test_duplicate_timestamps_gauge(self) -> None:
        """Test gauge handles duplicate timestamps (last value wins in storage)."""
        ts = ServerMetricsTimeSeries()
        # Add samples at same timestamp - sorted insertion handles this
        add_gauge_samples_with_timestamps(
            ts,
            "queue",
            [
                (NANOS_PER_SECOND, 10.0),
                (NANOS_PER_SECOND, 20.0),  # Same timestamp
                (NANOS_PER_SECOND, 30.0),  # Same timestamp
            ],
        )

        result = _compute_gauge_stats(
            get_gauge(ts, "queue"),
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        # All values are stored and contribute to statistics
        assert result is not None
        assert result.stats.avg == 20.0  # (10+20+30)/3


# =============================================================================
# Time Range Filter with None Values
# =============================================================================


class TestTimeFilterNoneValues:
    """Test time filter with None start or end values."""

    def test_none_start_includes_all_from_beginning(self) -> None:
        """Test None start_ns includes all data from beginning."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        time_filter = TimeRangeFilter(start_ns=None, end_ns=2 * NANOS_PER_SECOND)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is not None
        assert result.stats.avg == 20.0  # All 3 values

    def test_none_end_includes_all_to_end(self) -> None:
        """Test None end_ns includes all data to end."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        time_filter = TimeRangeFilter(start_ns=NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is not None
        # Should include values at t=1s and t=2s (20, 30)
        assert result.stats.avg == 25.0


# =============================================================================
# Timeslice Edge Cases
# =============================================================================


class TestTimesliceEdgeCases:
    """Test timeslice computation edge cases."""

    def test_timeslice_larger_than_data_range(self) -> None:
        """Test timeslice duration larger than data range returns None."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0], start_ns=0)

        # 1s of data, 10s timeslice - returns single partial slice
        result = _compute_gauge_timeslices(
            get_gauge(ts, "queue"),
            slice_duration=10.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        # Now returns a partial slice instead of None (best practice: preserve all data)
        assert result is not None
        assert len(result) == 1
        assert result[0].is_complete == False  # noqa: E712
        assert result[0].avg == 15.0  # Average of 10, 20

    def test_timeslice_exactly_fits_data(self) -> None:
        """Test timeslice that exactly fits data range."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        # 3 samples at 1s intervals, 1s timeslice, 2s range = 2 complete timeslices
        result = _compute_gauge_timeslices(
            get_gauge(ts, "queue"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        assert result is not None
        assert len(result) == 2

    def test_very_small_timeslice_duration(self) -> None:
        """Test very small timeslice duration (high granularity)."""
        ts = ServerMetricsTimeSeries()
        # Add 10 samples at 0.1s intervals
        interval_ns = NANOS_PER_SECOND // 10
        add_gauge_samples(
            ts, "queue", [float(i) for i in range(10)], interval_ns=interval_ns
        )

        # 0.2s timeslices over ~1s of data = ~4-5 complete timeslices
        result = _compute_gauge_timeslices(
            get_gauge(ts, "queue"),
            slice_duration=0.2,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        assert len(result) >= 4


# =============================================================================
# Counter Reset Detection
# =============================================================================


class TestCounterResetDetection:
    """Test counter reset detection and handling."""

    def test_counter_reset_clamps_to_zero(self) -> None:
        """Test counter reset results in clamped zero delta."""
        ts = ServerMetricsTimeSeries()
        # Value decreases (reset)
        add_counter_samples(ts, "requests", [1000.0, 500.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        # Negative delta clamped to 0
        assert result.stats is not None
        assert result.stats.total == 0.0
        assert result.stats.rate == 0.0

    def test_histogram_bucket_reset_nulls_buckets(self) -> None:
        """Test histogram bucket reset nullifies bucket data."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 1000.0, "+Inf": 1000.0}, 500.0, 1000.0)),
                # Reset - lower values
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        # Bucket deltas should be None due to reset
        assert result.buckets is None

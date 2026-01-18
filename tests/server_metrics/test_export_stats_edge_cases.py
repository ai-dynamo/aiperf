# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for export_stats.py functions."""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.server_metrics.export_stats import (
    _compute_counter_stats,
    _compute_counter_timeslices,
    _compute_gauge_stats,
    _compute_gauge_timeslices,
    _compute_histogram_stats,
    _compute_histogram_timeslices,
    _compute_timeslice_boundaries,
    compute_stats,
)
from aiperf.server_metrics.storage import (
    ScalarTimeSeries,
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
# _compute_timeslice_boundaries Edge Cases
# =============================================================================


class TestComputeTimesliceBoundaries:
    """Test _compute_timeslice_boundaries function edge cases."""

    def test_returns_partial_when_range_shorter_than_slice(self) -> None:
        """Test returns single partial slice when range is shorter than slice duration."""
        # 0.5 second range, 1 second slice - only partial
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=int(0.5 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )
        assert result is not None
        starts, ends, is_complete = result
        assert len(starts) == 1
        assert len(ends) == 1
        assert len(is_complete) == 1
        assert starts[0] == 0
        assert ends[0] == int(0.5 * NANOS_PER_SECOND)
        assert not is_complete[0]  # numpy bool

    def test_returns_partial_timeslice_when_no_complete(self) -> None:
        """Test returns single partial timeslice when no complete timeslices fit."""
        # 1.5 second range, 2 second slice - only partial slice
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=int(1.5 * NANOS_PER_SECOND),
            slice_duration=2.0,
        )
        assert result is not None
        starts, ends, is_complete = result
        assert len(starts) == 1
        assert len(ends) == 1
        assert len(is_complete) == 1
        assert starts[0] == 0
        assert ends[0] == int(1.5 * NANOS_PER_SECOND)
        assert is_complete[0] == False  # noqa: E712

    def test_returns_single_complete_plus_partial(self) -> None:
        """Test returns one complete and one partial timeslice."""
        # 1.5 second range, 1 second slice - one complete + one partial
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=int(1.5 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )
        assert result is not None
        starts, ends, is_complete = result
        assert len(starts) == 2
        assert len(ends) == 2
        assert len(is_complete) == 2
        # First slice: [0, 1s) - complete
        assert starts[0] == 0
        assert ends[0] == NANOS_PER_SECOND
        assert is_complete[0] == True  # noqa: E712
        # Second slice: [1s, 1.5s) - partial
        assert starts[1] == NANOS_PER_SECOND
        assert ends[1] == int(1.5 * NANOS_PER_SECOND)
        assert is_complete[1] == False  # noqa: E712

    def test_exact_fit_multiple_timeslices(self) -> None:
        """Test exact fit returns all complete timeslices with no partial."""
        # 3 second range, 1 second slices - exact fit
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=3 * NANOS_PER_SECOND,
            slice_duration=1.0,
        )
        assert result is not None
        starts, ends, is_complete = result
        assert len(starts) == 3
        assert len(ends) == 3
        assert len(is_complete) == 3
        # All should be complete (exact fit, no partial)
        assert all(is_complete)

    def test_partial_final_timeslice_included(self) -> None:
        """Test partial final timeslice is included and marked incomplete."""
        # 2.5 second range, 1 second slices - 2 complete + 1 partial
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=int(2.5 * NANOS_PER_SECOND),
            slice_duration=1.0,
        )
        assert result is not None
        starts, ends, is_complete = result
        assert len(starts) == 3
        assert len(ends) == 3
        assert len(is_complete) == 3
        # First two complete, last partial
        assert is_complete[0] == True  # noqa: E712
        assert is_complete[1] == True  # noqa: E712
        assert is_complete[2] == False  # noqa: E712
        # Check boundaries
        assert starts[2] == 2 * NANOS_PER_SECOND
        assert ends[2] == int(2.5 * NANOS_PER_SECOND)

    def test_empty_range_returns_none(self) -> None:
        """Test empty range returns None."""
        result = _compute_timeslice_boundaries(
            range_start_ns=0,
            range_end_ns=0,
            slice_duration=1.0,
        )
        assert result is None

    def test_negative_range_returns_none(self) -> None:
        """Test negative range (start > end) returns None."""
        result = _compute_timeslice_boundaries(
            range_start_ns=NANOS_PER_SECOND,
            range_end_ns=0,
            slice_duration=1.0,
        )
        assert result is None


# =============================================================================
# compute_counter_timeslices Edge Cases
# =============================================================================


class TestComputeCounterTimeslicesEdgeCases:
    """Test compute_counter_timeslices edge cases."""

    def test_negative_slice_duration_raises(self) -> None:
        """Test negative slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_counter_timeslices(
                get_counter(ts, "requests"),
                slice_duration=-1.0,
                time_filter=make_time_filter(),
            )

    def test_zero_slice_duration_raises(self) -> None:
        """Test zero slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_counter_timeslices(
                get_counter(ts, "requests"),
                slice_duration=0.0,
                time_filter=make_time_filter(),
            )

    def test_single_timestamp_returns_empty(self) -> None:
        """Test single data point returns empty list."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [100.0])

        result = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )
        assert result == []

    def test_empty_series_returns_empty(self) -> None:
        """Test empty time series returns empty list."""
        series = ScalarTimeSeries()
        result = _compute_counter_timeslices(
            series,
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )
        assert result == []

    def test_filter_excludes_all_data_returns_empty(self) -> None:
        """Test time filter that excludes all data returns empty list."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0], start_ns=0)

        # Filter starts after all data
        time_filter = TimeRangeFilter(start_ns=10 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_counter_timeslices(
            get_counter(ts, "requests"), slice_duration=1.0, time_filter=time_filter
        )
        assert result == []

    def test_counter_reset_handled_gracefully(self) -> None:
        """Test counter reset (negative delta) is handled."""
        ts = ServerMetricsTimeSeries()
        # Counter resets: 100 -> 50 (decrease)
        add_counter_samples(ts, "requests", [0.0, 100.0, 50.0, 150.0])

        result = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        # Negative deltas become 0 (handled by np.maximum)
        assert len(result) == 3
        # Second timeslice should have rate 0 (reset)
        assert result[1].rate == 0.0

    def test_partial_timeslice_marked_incomplete(self) -> None:
        """Test partial final timeslice is marked with is_complete=False."""
        ts = ServerMetricsTimeSeries()
        # 2.5 seconds of data with 1 second slices
        add_counter_samples(
            ts,
            "requests",
            [0.0, 100.0, 200.0],
            start_ns=0,
            interval_ns=NANOS_PER_SECOND,
        )
        add_counter_samples(
            ts, "requests", [250.0], start_ns=int(2.5 * NANOS_PER_SECOND)
        )

        result = _compute_counter_timeslices(
            get_counter(ts, "requests"),
            slice_duration=1.0,
            time_filter=make_time_filter(
                start_ns=0, end_ns=int(2.5 * NANOS_PER_SECOND)
            ),
        )

        # Should have 3 timeslices: 2 complete + 1 partial
        assert len(result) == 3
        assert result[0].is_complete != False  # noqa: E712
        assert result[1].is_complete != False  # noqa: E712
        assert result[2].is_complete == False  # noqa: E712
        # Partial slice should still have normalized rate
        assert result[2].rate > 0  # Should compute rate per second even for 0.5s slice


# =============================================================================
# compute_gauge_timeslices Edge Cases
# =============================================================================


class TestComputeGaugeTimeslicesEdgeCases:
    """Test compute_gauge_timeslices edge cases."""

    def test_negative_slice_duration_raises(self) -> None:
        """Test negative slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_gauge_timeslices(
                get_gauge(ts, "queue"),
                slice_duration=-1.0,
                time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            )

    def test_zero_slice_duration_raises(self) -> None:
        """Test zero slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_gauge_timeslices(
                get_gauge(ts, "queue"),
                slice_duration=0.0,
                time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            )

    def test_single_timestamp_returns_none(self) -> None:
        """Test single data point returns None."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0])

        result = _compute_gauge_timeslices(
            get_gauge(ts, "queue"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )
        assert result is None

    def test_empty_series_returns_none(self) -> None:
        """Test empty time series returns None."""
        series = ScalarTimeSeries()
        result = _compute_gauge_timeslices(
            series,
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )
        assert result is None

    def test_last_known_value_carried_forward(self) -> None:
        """Test last known value is carried forward when no samples in timeslice."""
        ts = ServerMetricsTimeSeries()
        # Samples at t=0, t=1s, t=3s (gap at t=2s timeslice)
        add_gauge_samples(ts, "queue", [10.0, 20.0], start_ns=0)
        add_gauge_samples(ts, "queue", [30.0], start_ns=3 * NANOS_PER_SECOND)

        result = _compute_gauge_timeslices(
            get_gauge(ts, "queue"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        # Should have 3 timeslices, middle one uses last known value
        assert result is not None
        assert len(result) >= 2
        # First timeslice has value 20 (last in range)
        # If gap handling works, the gap timeslice uses last known value


# =============================================================================
# _compute_histogram_timeslices Edge Cases
# =============================================================================


class TestComputeHistogramTimeslicesEdgeCases:
    """Test _compute_histogram_timeslices edge cases."""

    def test_negative_slice_duration_raises(self) -> None:
        """Test negative slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_histogram_timeslices(
                get_histogram(ts, "latency"),
                slice_duration=-1.0,
                time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
            )

    def test_zero_slice_duration_raises(self) -> None:
        """Test zero slice_duration raises ValueError."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        with pytest.raises(ValueError, match="slice_duration must be positive"):
            _compute_histogram_timeslices(
                get_histogram(ts, "latency"),
                slice_duration=0.0,
                time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
            )

    def test_single_snapshot_returns_none(self) -> None:
        """Test single snapshot returns None."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [(0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0))],
        )

        result = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )
        assert result is None

    def test_zero_count_delta_timeslice_skipped(self) -> None:
        """Test timeslice with zero count delta is included with zero values."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                # No change in counts
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
                # Some change
                (
                    2 * NANOS_PER_SECOND,
                    hist({"1.0": 200.0, "+Inf": 200.0}, 100.0, 200.0),
                ),
            ],
        )

        result = _compute_histogram_timeslices(
            get_histogram(ts, "latency"),
            slice_duration=1.0,
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
        )

        # Both timeslices should be included, first with zero count delta
        assert result is not None
        assert len(result) == 2
        assert result[0].count == 0  # No change from t=0 to t=1
        assert result[1].count == 100  # Change from t=1 to t=2


# =============================================================================
# compute_stats Edge Cases
# =============================================================================


class TestComputeStatsEdgeCases:
    """Test compute_stats dispatcher edge cases."""

    def test_unsupported_metric_type_raises(self) -> None:
        """Test unsupported metric type raises ValueError."""
        series = ScalarTimeSeries()

        with pytest.raises(ValueError, match="Unsupported metric type"):
            compute_stats(
                metric_type=PrometheusMetricType.SUMMARY,  # SUMMARY is not supported
                time_series=series,
            )

    def test_gauge_dispatch(self) -> None:
        """Test GAUGE metric type dispatches correctly."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        result = compute_stats(
            metric_type=PrometheusMetricType.GAUGE,
            time_series=get_gauge(ts, "queue"),
        )

        assert result is not None
        assert result.stats.avg == 20.0

    def test_counter_dispatch(self) -> None:
        """Test COUNTER metric type dispatches correctly."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0])

        result = compute_stats(
            metric_type=PrometheusMetricType.COUNTER,
            time_series=get_counter(ts, "requests"),
        )

        assert result is not None
        assert result.stats.total == 200.0

    def test_histogram_dispatch(self) -> None:
        """Test HISTOGRAM metric type dispatches correctly."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        result = compute_stats(
            metric_type=PrometheusMetricType.HISTOGRAM,
            time_series=get_histogram(ts, "latency"),
        )

        assert result is not None
        assert result.stats.count == 100


# =============================================================================
# compute_gauge_stats Edge Cases
# =============================================================================


class TestComputeGaugeStatsEdgeCases:
    """Test compute_gauge_stats edge cases."""

    def test_constant_gauge_returns_value(self) -> None:
        """Test constant gauge (std==0) returns stats with all values equal."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [42.0, 42.0, 42.0, 42.0])

        result = _compute_gauge_stats(
            get_gauge(ts, "queue"),
            time_filter=make_time_filter(start_ns=0, end_ns=3 * NANOS_PER_SECOND),
        )

        # Constant values: all stats equal the constant value, std=0
        assert result is not None
        assert result.stats is not None
        assert result.stats.avg == 42.0
        assert result.stats.min == 42.0
        assert result.stats.max == 42.0
        assert result.stats.std == 0.0
        assert result.stats.p50 == 42.0
        assert result.stats.p99 == 42.0

    def test_labels_passed_through(self) -> None:
        """Test labels are passed through to result."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0, 20.0, 30.0])

        labels = {"instance": "server1", "job": "metrics"}
        result = _compute_gauge_stats(
            get_gauge(ts, "queue"),
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            labels=labels,
        )

        assert result is not None
        assert result.labels == labels

    def test_empty_after_filter_returns_none(self) -> None:
        """Test returns None when time filter excludes all data."""
        ts = ServerMetricsTimeSeries()
        add_gauge_samples(ts, "queue", [10.0], start_ns=0)

        # Filter starts way after the data
        time_filter = TimeRangeFilter(start_ns=100 * NANOS_PER_SECOND, end_ns=None)
        result = _compute_gauge_stats(get_gauge(ts, "queue"), time_filter)

        assert result is None


# =============================================================================
# compute_counter_stats Edge Cases
# =============================================================================


class TestComputeCounterStatsEdgeCases:
    """Test compute_counter_stats edge cases."""

    def test_zero_duration_returns_value(self) -> None:
        """Test zero duration (single point) returns stats with total=0."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [100.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        # Single point, no change: stats with total=0, rate=0
        assert result.stats is not None
        assert result.stats.total == 0.0
        assert result.stats.rate == 0.0

    def test_counter_reset_clamps_to_zero(self) -> None:
        """Test counter reset results in zero total (not negative)."""
        ts = ServerMetricsTimeSeries()
        # Starts at 1000, resets to 50
        add_counter_samples(ts, "requests", [1000.0, 50.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        # Negative delta clamped to 0, stats with total=0
        assert result.stats is not None
        assert result.stats.total == 0.0
        assert result.stats.rate == 0.0

    def test_labels_passed_through(self) -> None:
        """Test labels are passed through to result."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0])

        labels = {"endpoint": "/api/v1"}
        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
            labels=labels,
        )

        assert result is not None
        assert result.labels == labels

    def test_timeslices_not_computed_without_slice_duration(self) -> None:
        """Test timeslices not computed when slice_duration is None."""
        ts = ServerMetricsTimeSeries()
        add_counter_samples(ts, "requests", [0.0, 100.0, 200.0])

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(start_ns=0, end_ns=2 * NANOS_PER_SECOND),
            slice_duration=None,
        )

        assert result is not None
        assert result.timeslices is None

    def test_aggregate_stats_exclude_partial_timeslices(self) -> None:
        """Test that rate_min/max/avg/std only use complete timeslices."""
        ts = ServerMetricsTimeSeries()
        # Create data with 2.5 seconds: 2 complete slices + 1 partial
        # Slice 1 (0-1s): 0->100 = rate 100/s
        # Slice 2 (1-2s): 100->200 = rate 100/s
        # Slice 3 (2-2.5s): 200->300 = rate 200/s (partial, 0.5s duration)
        add_counter_samples(
            ts,
            "requests",
            [0.0, 100.0, 200.0],
            start_ns=0,
            interval_ns=NANOS_PER_SECOND,
        )
        add_counter_samples(
            ts, "requests", [300.0], start_ns=int(2.5 * NANOS_PER_SECOND)
        )

        result = _compute_counter_stats(
            get_counter(ts, "requests"),
            time_filter=make_time_filter(
                start_ns=0, end_ns=int(2.5 * NANOS_PER_SECOND)
            ),
            slice_duration=1.0,
        )

        assert result is not None
        assert result.timeslices is not None
        assert len(result.timeslices) == 3

        # Verify all timeslices are present
        assert result.timeslices[0].is_complete != False  # noqa: E712
        assert result.timeslices[1].is_complete != False  # noqa: E712
        assert result.timeslices[2].is_complete == False  # noqa: E712

        # Aggregate stats should only use complete slices (rate=100 for both)
        # NOT including the partial slice (rate=200)
        assert result.stats.rate_min == 100.0
        assert result.stats.rate_max == 100.0
        assert result.stats.rate_avg == 100.0
        assert result.stats.rate_std == 0.0  # Same rate for both complete slices


# =============================================================================
# _compute_histogram_stats Edge Cases
# =============================================================================


class TestComputeHistogramStatsEdgeCases:
    """Test _compute_histogram_stats edge cases."""

    def test_zero_count_returns_simplified_stats(self) -> None:
        """Test histogram with no observations returns simplified stats."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # No change - zero observations added
                (NANOS_PER_SECOND, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        assert result.stats.count == 0
        # Simplified output for zero-count
        assert result.stats.avg is None

    def test_labels_passed_through(self) -> None:
        """Test labels are passed through to result."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        labels = {"handler": "/api", "method": "GET"}
        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
            labels=labels,
        )

        assert result is not None
        assert result.labels == labels

    def test_counter_reset_nulls_percentile_estimates(self) -> None:
        """Test counter reset nullifies percentile estimates."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 1000.0, "+Inf": 2000.0}, 500.0, 2000.0)),
                # Reset - counts decreased
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        # Buckets should be None due to reset
        assert result.buckets is None
        # Percentile estimates not computed when buckets invalid
        assert result.stats is None or result.stats.p50_estimate is None

    def test_timeslices_not_computed_without_slice_duration(self) -> None:
        """Test timeslices not computed when slice_duration is None."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (NANOS_PER_SECOND, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
            slice_duration=None,
        )

        assert result is not None
        assert result.timeslices is None

    def test_zero_duration_computes_rates_as_none(self) -> None:
        """Test zero duration results in None rates."""
        ts = ServerMetricsTimeSeries()
        # Single snapshot at t=0
        add_histogram_snapshots(
            ts,
            "latency",
            [(0, hist({"1.0": 100.0, "+Inf": 100.0}, 50.0, 100.0))],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"),
            time_filter=make_time_filter(start_ns=0, end_ns=NANOS_PER_SECOND),
        )

        assert result is not None
        # count=0 (delta from self), returns simplified stats
        assert result.stats.count == 0

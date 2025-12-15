# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Edge case tests for storage.py time series classes.

Tests boundary conditions, capacity handling, out-of-order data,
and numeric edge cases for ScalarTimeSeries, HistogramTimeSeries,
ServerMetricsTimeSeries, and related components.
"""

import numpy as np
import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    TimeRangeFilter,
)
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
    ServerMetricKey,
    ServerMetricsHierarchy,
    ServerMetricsTimeSeries,
    _bucket_sort_key,
)

# =============================================================================
# Test Helpers
# =============================================================================

_NS = 1_000_000_000  # 1 second in nanoseconds


def make_scalar_ts(values: list[float], interval_ns: int = _NS) -> ScalarTimeSeries:
    """Create a ScalarTimeSeries with sequential timestamps."""
    ts = ScalarTimeSeries()
    for i, val in enumerate(values):
        ts.append(i * interval_ns, MetricSample(value=val))
    return ts


def make_histogram_ts(
    snapshots: list[tuple[dict[str, float], float, float]],
    interval_ns: int = _NS,
) -> HistogramTimeSeries:
    """Create a HistogramTimeSeries from (buckets, sum, count) tuples."""
    ts = HistogramTimeSeries()
    for i, (buckets, sum_, count) in enumerate(snapshots):
        ts.append(i * interval_ns, MetricSample(buckets=buckets, sum=sum_, count=count))
    return ts


def make_gauge_family(value: float = 42.0) -> MetricFamily:
    """Create a simple gauge MetricFamily."""
    return MetricFamily(
        type=PrometheusMetricType.GAUGE,
        description="Test gauge",
        samples=[MetricSample(value=value)],
    )


def make_record(
    timestamp_ns: int,
    metrics: dict[str, MetricFamily],
    is_duplicate: bool = False,
) -> ServerMetricsRecord:
    """Create a ServerMetricsRecord for testing."""
    return ServerMetricsRecord(
        endpoint_url="http://test:8081/metrics",
        timestamp_ns=timestamp_ns,
        endpoint_latency_ns=5_000_000,
        metrics=metrics,
        is_duplicate=is_duplicate,
    )


# =============================================================================
# ScalarTimeSeries Tests
# =============================================================================


class TestScalarTimeSeriesCapacity:
    """Test capacity expansion with many data points."""

    @pytest.mark.parametrize("count", [300, 2000])
    def test_capacity_expansion(self, count: int) -> None:
        """Test that capacity expands correctly for various sizes."""
        ts = make_scalar_ts([float(i) for i in range(count)])

        assert len(ts) == count
        np.testing.assert_array_equal(ts.values, np.arange(count, dtype=np.float64))


class TestScalarTimeSeriesOutOfOrder:
    """Test out-of-order data insertion."""

    @pytest.mark.parametrize(
        "timestamps,values,expected_order",
        [
            ([1, 3, 2], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]),  # Single out-of-order
            ([1, 2, 0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]),  # Insert at beginning
            ([3, 2, 1, 0], [3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]),  # Fully reversed
        ],
    )  # fmt: skip
    def test_out_of_order_sorting(
        self, timestamps: list[int], values: list[float], expected_order: list[float]
    ) -> None:
        """Test that out-of-order data is sorted correctly."""
        ts = ScalarTimeSeries()
        for t, v in zip(timestamps, values, strict=True):
            ts.append(t * _NS, MetricSample(value=v))

        np.testing.assert_array_equal(ts.values, expected_order)

    def test_scrambled_order(self) -> None:
        """Test multiple out-of-order insertions."""
        order = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0]
        ts = ScalarTimeSeries()
        for idx in order:
            ts.append(idx * _NS, MetricSample(value=float(idx)))

        np.testing.assert_array_equal(ts.values, np.arange(10, dtype=np.float64))


class TestScalarTimeSeriesFiltering:
    """Test time filtering operations."""

    @pytest.fixture
    def ts_10_points(self) -> ScalarTimeSeries:
        """ScalarTimeSeries with 10 points at 0-9 seconds."""
        return make_scalar_ts([float(i) for i in range(10)])

    @pytest.mark.parametrize(
        "start_ns,end_ns,expected_mask",
        [
            (None, None, [True] * 10),  # No filter
            (5 * _NS, None, [False] * 5 + [True] * 5),  # Start only
            (None, 5 * _NS, [True] * 6 + [False] * 4),  # End only
            (3 * _NS, 7 * _NS, [False] * 3 + [True] * 5 + [False] * 2),  # Both bounds
            (100 * _NS, None, [False] * 10),  # No matching data
        ],
    )  # fmt: skip
    def test_get_time_mask(
        self,
        ts_10_points: ScalarTimeSeries,
        start_ns: int | None,
        end_ns: int | None,
        expected_mask: list[bool],
    ) -> None:
        """Test time mask generation with various filters."""
        time_filter = (
            TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)
            if start_ns or end_ns
            else None
        )
        mask = ts_10_points.get_time_mask(time_filter)
        np.testing.assert_array_equal(mask, expected_mask)

    @pytest.mark.parametrize(
        "start_ns,expected_ref_idx",
        [
            (5 * _NS, 4),  # Mid-range
            (0, None),  # At first point (no point before 0)
        ],
    )  # fmt: skip
    def test_get_reference_idx(
        self,
        ts_10_points: ScalarTimeSeries,
        start_ns: int,
        expected_ref_idx: int | None,
    ) -> None:
        """Test reference index for delta calculations."""
        time_filter = TimeRangeFilter(start_ns=start_ns)
        assert ts_10_points.get_reference_idx(time_filter) == expected_ref_idx


class TestScalarTimeSeriesEmpty:
    """Test empty time series behavior."""

    def test_empty_properties(self) -> None:
        """Test that empty series returns valid empty arrays."""
        ts = ScalarTimeSeries()

        assert len(ts) == 0
        assert len(ts.timestamps) == 0
        assert len(ts.values) == 0
        assert len(ts.get_time_mask(None)) == 0
        assert ts.get_reference_idx(TimeRangeFilter(start_ns=_NS)) is None


class TestScalarTimeSeriesNumericEdgeCases:
    """Test numeric edge cases."""

    @pytest.mark.parametrize(
        "value", [1e308, 1e-308, -100.5, 0.0, float("inf"), float("-inf"), float("nan")]
    )
    def test_extreme_values(self, value: float) -> None:
        """Test that extreme numeric values are stored correctly."""
        ts = ScalarTimeSeries()
        ts.append(_NS, MetricSample(value=value))

        if np.isnan(value):
            assert np.isnan(ts.values[0])
        else:
            assert ts.values[0] == value or np.isinf(ts.values[0])

    def test_none_value_raises(self) -> None:
        """Test that None value raises ValueError."""
        ts = ScalarTimeSeries()
        with pytest.raises(ValueError, match="Value is required"):
            ts.append(
                _NS,
                MetricSample(buckets={"0.1": 1.0, "+Inf": 10.0}, sum=5.0, count=10.0),
            )


# =============================================================================
# HistogramTimeSeries Tests
# =============================================================================


class TestHistogramTimeSeriesBasics:
    """Test histogram time series basics."""

    def test_capacity_expansion(self) -> None:
        """Test capacity expansion with many histogram samples."""
        buckets = {"0.1": 0.0, "1.0": 0.0, "+Inf": 0.0}
        snapshots = [(dict(buckets), float(i * 10), float(i * 3)) for i in range(300)]
        ts = make_histogram_ts(snapshots)

        assert len(ts) == 300
        assert ts.bucket_counts.shape == (300, 3)

    def test_bucket_schema_sorted_numerically(self) -> None:
        """Test that bucket les are sorted numerically, not alphabetically."""
        ts = HistogramTimeSeries()
        ts.append(
            _NS,
            MetricSample(
                buckets={"10": 50.0, "5": 30.0, "1": 10.0, "+Inf": 100.0},
                sum=100.0,
                count=100.0,
            ),
        )

        assert ts.bucket_les == ("1", "5", "10", "+Inf")

    def test_missing_bucket_defaults_to_zero(self) -> None:
        """Test that missing buckets in later samples get 0.0."""
        ts = HistogramTimeSeries()
        ts.append(
            _NS,
            MetricSample(
                buckets={"0.1": 1.0, "1.0": 5.0, "+Inf": 10.0}, sum=10.0, count=10.0
            ),
        )
        ts.append(
            2 * _NS,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 20.0}, sum=20.0, count=20.0),
        )

        assert ts.bucket_counts[1, 0] == 0.0  # Missing "0.1" bucket


class TestHistogramTimeSeriesIndices:
    """Test get_indices_for_filter method."""

    @pytest.fixture
    def ts_10_histograms(self) -> HistogramTimeSeries:
        """HistogramTimeSeries with 10 samples."""
        buckets = {"1.0": 0.0, "+Inf": 0.0}
        return make_histogram_ts(
            [(dict(buckets), float(i * 100), float(i * 20)) for i in range(10)]
        )

    @pytest.mark.parametrize(
        "start_ns,end_ns,expected_ref,expected_final",
        [
            (None, None, None, 9),  # No filter
            (5 * _NS, None, 4, 9),  # Start only
            (None, 5 * _NS, None, 5),  # End only
            (3 * _NS, 7 * _NS, 2, 7),  # Both bounds
        ],
    )  # fmt: skip
    def test_indices_for_filter(
        self,
        ts_10_histograms: HistogramTimeSeries,
        start_ns: int | None,
        end_ns: int | None,
        expected_ref: int | None,
        expected_final: int,
    ) -> None:
        """Test index computation for various filters."""
        time_filter = (
            TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)
            if start_ns or end_ns
            else None
        )
        ref_idx, final_idx = ts_10_histograms.get_indices_for_filter(time_filter)

        assert ref_idx == expected_ref
        assert final_idx == expected_final

    def test_empty_histogram_indices(self) -> None:
        """Test indices on empty series returns -1 for final."""
        ts = HistogramTimeSeries()
        ref_idx, final_idx = ts.get_indices_for_filter(None)

        assert ref_idx is None
        assert final_idx == -1


class TestHistogramTimeSeriesRates:
    """Test observation rate calculation."""

    def test_basic_rates(self) -> None:
        """Test basic observation rate calculation."""
        buckets = {"1.0": 0.0, "+Inf": 0.0}
        # Cumulative counts: 0, 100, 200, 300, 400 -> 100/s each interval
        snapshots = [(dict(buckets), float(i * 1000), float(i * 100)) for i in range(5)]
        ts = make_histogram_ts(snapshots)
        rates = ts.get_observation_rates()

        assert len(rates) == 4
        np.testing.assert_array_almost_equal(rates, [100.0, 100.0, 100.0, 100.0])

    @pytest.mark.parametrize("scenario", ["empty", "single", "zero_delta"])
    def test_empty_rates(self, scenario: str) -> None:
        """Test that various edge cases return empty rates."""
        if scenario == "empty":
            ts = HistogramTimeSeries()
        elif scenario == "single":
            ts = make_histogram_ts([({"1.0": 100.0, "+Inf": 100.0}, 1000.0, 100.0)])
        else:  # zero_delta - same timestamp
            ts = HistogramTimeSeries()
            for i in range(3):
                ts.append(
                    _NS,
                    MetricSample(
                        buckets={"1.0": float(i * 100), "+Inf": float(i * 100)},
                        sum=float(i * 1000),
                        count=float(i * 100),
                    ),
                )

        assert len(ts.get_observation_rates()) == 0


class TestHistogramTimeSeriesEmpty:
    """Test empty histogram series properties."""

    def test_empty_properties(self) -> None:
        """Test that empty histogram series returns valid defaults."""
        ts = HistogramTimeSeries()

        assert len(ts) == 0
        assert len(ts.timestamps) == 0
        assert len(ts.sums) == 0
        assert len(ts.counts) == 0
        assert ts.bucket_les == ()
        assert ts.bucket_counts.shape == (0, 0)
        assert ts.get_bucket_dict(0) == {}


# =============================================================================
# ServerMetricsTimeSeries Tests
# =============================================================================


class TestServerMetricsTimeSeriesDuplicates:
    """Test duplicate record handling."""

    def test_duplicate_tracking(self) -> None:
        """Test that duplicates affect fetch counts but not unique counts."""
        ts = ServerMetricsTimeSeries()
        gauge = make_gauge_family()

        ts.append_snapshot(make_record(_NS, {"gauge": gauge}, is_duplicate=False))
        ts.append_snapshot(make_record(2 * _NS, {"gauge": gauge}, is_duplicate=True))

        assert ts._unique_update_count == 1
        assert ts._total_fetch_count == 2
        assert len(ts._fetch_latencies_ns) == 2

    def test_empty_metrics_skipped(self) -> None:
        """Test that records with empty metrics dict are skipped."""
        ts = ServerMetricsTimeSeries()
        ts.append_snapshot(make_record(_NS, {}))

        assert len(ts) == 0
        assert len(ts.metrics) == 0


class TestServerMetricsTimeSeriesTimestamps:
    """Test timestamp handling."""

    def test_out_of_order_updates_bounds(self) -> None:
        """Test that out-of-order records update first/last correctly."""
        ts = ServerMetricsTimeSeries()
        gauge = make_gauge_family()

        ts.append_snapshot(make_record(5 * _NS, {"gauge": gauge}))
        ts.append_snapshot(make_record(2 * _NS, {"gauge": gauge}))

        assert ts.first_update_ns == 2 * _NS
        assert ts.last_update_ns == 5 * _NS

    def test_interval_calculation(self) -> None:
        """Test that intervals are computed from sorted timestamps."""
        ts = ServerMetricsTimeSeries()
        gauge = make_gauge_family()

        for t in [3 * _NS, _NS, 2 * _NS]:  # Out of order
            ts.append_snapshot(make_record(t, {"gauge": gauge}))

        assert ts._update_intervals_ns == [_NS, _NS]


# =============================================================================
# ServerMetricKey Tests
# =============================================================================


class TestServerMetricKey:
    """Test ServerMetricKey creation and properties."""

    @pytest.mark.parametrize(
        "labels,expected_labels,expected_dict",
        [
            (None, (), None),  # No labels
            ({}, (), None),  # Empty dict
            ({"a": "1"}, (("a", "1"),), {"a": "1"}),  # Single label
            ({"z": "2", "a": "1"}, (("a", "1"), ("z", "2")), {"a": "1", "z": "2"}),  # Sorted
        ],
    )  # fmt: skip
    def test_key_creation(
        self,
        labels: dict[str, str] | None,
        expected_labels: tuple,
        expected_dict: dict | None,
    ) -> None:
        """Test key creation with various label configurations."""
        key = ServerMetricKey.from_name_and_labels("metric", labels)

        assert key.labels == expected_labels
        assert key.labels_dict == expected_dict

    def test_key_hashable(self) -> None:
        """Test that keys are hashable and work in dicts."""
        key1 = ServerMetricKey.from_name_and_labels("m", {"a": "1"})
        key2 = ServerMetricKey.from_name_and_labels("m", {"a": "1"})

        assert key1 == key2
        assert {key1: "v"}[key2] == "v"


# =============================================================================
# ServerMetricEntry Tests
# =============================================================================


class TestServerMetricEntry:
    """Test ServerMetricEntry creation from metric families."""

    @pytest.mark.parametrize(
        "metric_type,expected_data_type",
        [
            (PrometheusMetricType.GAUGE, ScalarTimeSeries),
            (PrometheusMetricType.COUNTER, ScalarTimeSeries),
            (PrometheusMetricType.HISTOGRAM, HistogramTimeSeries),
        ],
    )
    def test_from_metric_family(
        self, metric_type: PrometheusMetricType, expected_data_type: type
    ) -> None:
        """Test entry creation for each metric type."""
        if metric_type == PrometheusMetricType.HISTOGRAM:
            samples = [
                MetricSample(buckets={"1.0": 5.0, "+Inf": 10.0}, sum=50.0, count=10.0)
            ]
        else:
            samples = [MetricSample(value=42.0)]

        family = MetricFamily(type=metric_type, description="test", samples=samples)
        entry = ServerMetricEntry.from_metric_family(family)

        assert entry.metric_type == metric_type
        assert isinstance(entry.data, expected_data_type)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestBucketSortKey:
    """Test _bucket_sort_key function."""

    @pytest.mark.parametrize(
        "buckets,expected",
        [
            (["10", "2", "1", "0.5", "100"], ["0.5", "1", "2", "10", "100"]),  # Numeric
            (["+Inf", "1", "10", "0.1"], ["0.1", "1", "10", "+Inf"]),  # With +Inf
            (["-1", "0", "1", "-10", "+Inf"], ["-10", "-1", "0", "1", "+Inf"]),  # Negative
        ],
    )  # fmt: skip
    def test_bucket_sorting(self, buckets: list[str], expected: list[str]) -> None:
        """Test bucket boundary sorting."""
        assert sorted(buckets, key=_bucket_sort_key) == expected


# =============================================================================
# ServerMetricsHierarchy Tests
# =============================================================================


class TestServerMetricsHierarchy:
    """Test ServerMetricsHierarchy storage."""

    def test_multiple_metrics_same_endpoint(self) -> None:
        """Test multiple metrics from same endpoint."""
        hierarchy = ServerMetricsHierarchy()
        gauge = make_gauge_family(42.0)
        counter = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            description="counter",
            samples=[MetricSample(value=100.0)],
        )

        hierarchy.add_record(make_record(_NS, {"gauge": gauge, "counter": counter}))

        ts = hierarchy.endpoints["http://test:8081/metrics"]
        assert len(ts.metrics) == 2

    def test_same_metric_different_labels(self) -> None:
        """Test same metric with different labels creates separate entries."""
        hierarchy = ServerMetricsHierarchy()
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="gauge",
            samples=[
                MetricSample(labels={"model": "a"}, value=10.0),
                MetricSample(labels={"model": "b"}, value=20.0),
            ],
        )

        hierarchy.add_record(make_record(_NS, {"gauge": gauge}))

        ts = hierarchy.endpoints["http://test:8081/metrics"]
        assert len(ts.metrics) == 2
        assert ServerMetricKey("gauge", (("model", "a"),)) in ts.metrics
        assert ServerMetricKey("gauge", (("model", "b"),)) in ts.metrics

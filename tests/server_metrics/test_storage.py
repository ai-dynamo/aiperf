# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for server metrics storage classes."""

from __future__ import annotations

import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models import MetricFamily, MetricSample, TimeRangeFilter
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
    ServerMetricKey,
    ServerMetricsHierarchy,
    ServerMetricsTimeSeries,
)


class TestScalarTimeSeries:
    """Tests for ScalarTimeSeries class."""

    def test_empty_series(self) -> None:
        """Test empty series properties."""
        series = ScalarTimeSeries()

        assert len(series) == 0
        assert len(series.timestamps) == 0
        assert len(series.values) == 0

    def test_append_single_value(self) -> None:
        """Test appending a single value."""
        series = ScalarTimeSeries()
        sample = MetricSample(value=42.0)

        series.append(1000, sample)

        assert len(series) == 1
        assert series.timestamps[0] == 1000
        assert series.values[0] == 42.0

    def test_append_multiple_values_in_order(self) -> None:
        """Test appending multiple values in timestamp order."""
        series = ScalarTimeSeries()

        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))

        assert len(series) == 3
        assert list(series.timestamps) == [1000, 2000, 3000]
        assert list(series.values) == [1.0, 2.0, 3.0]

    def test_append_out_of_order_inserts_correctly(self) -> None:
        """Test that out-of-order appends maintain sorted order."""
        series = ScalarTimeSeries()

        series.append(3000, MetricSample(value=3.0))
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))

        assert len(series) == 3
        assert list(series.timestamps) == [1000, 2000, 3000]
        assert list(series.values) == [1.0, 2.0, 3.0]

    def test_append_duplicate_timestamps(self) -> None:
        """Test appending values with duplicate timestamps."""
        series = ScalarTimeSeries()

        series.append(1000, MetricSample(value=1.0))
        series.append(1000, MetricSample(value=2.0))

        assert len(series) == 2
        # Both should be present, order maintained
        assert series.timestamps[0] == 1000
        assert series.timestamps[1] == 1000

    def test_append_none_value_raises(self) -> None:
        """Test that appending None value raises ValueError."""
        series = ScalarTimeSeries()
        # Create sample with buckets to bypass MetricSample validation
        sample = MetricSample(buckets={"1.0": 5.0})
        # Manually set value to None to test ScalarTimeSeries validation
        sample.value = None

        with pytest.raises(ValueError, match="Value is required"):
            series.append(1000, sample)

    def test_capacity_growth(self) -> None:
        """Test that capacity grows automatically."""
        series = ScalarTimeSeries()

        # Append more than initial capacity (256)
        for i in range(300):
            series.append(i * 1000, MetricSample(value=float(i)))

        assert len(series) == 300
        assert series.values[-1] == 299.0

    def test_get_time_mask_no_filter(self) -> None:
        """Test get_time_mask with no filter returns all True."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))

        mask = series.get_time_mask(None)

        assert all(mask)
        assert len(mask) == 3

    def test_get_time_mask_with_start_filter(self) -> None:
        """Test get_time_mask with start_ns filter."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))

        time_filter = TimeRangeFilter(start_ns=2000)
        mask = series.get_time_mask(time_filter)

        assert list(mask) == [False, True, True]

    def test_get_time_mask_with_end_filter(self) -> None:
        """Test get_time_mask with end_ns filter."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))

        time_filter = TimeRangeFilter(end_ns=2000)
        mask = series.get_time_mask(time_filter)

        assert list(mask) == [True, True, False]

    def test_get_time_mask_with_both_filters(self) -> None:
        """Test get_time_mask with both start and end filters."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))
        series.append(4000, MetricSample(value=4.0))

        time_filter = TimeRangeFilter(start_ns=2000, end_ns=3000)
        mask = series.get_time_mask(time_filter)

        assert list(mask) == [False, True, True, False]

    def test_get_time_mask_empty_series(self) -> None:
        """Test get_time_mask on empty series."""
        series = ScalarTimeSeries()
        mask = series.get_time_mask(None)

        assert len(mask) == 0

    def test_get_reference_idx_no_filter(self) -> None:
        """Test get_reference_idx with no filter returns None."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))

        assert series.get_reference_idx(None) is None

    def test_get_reference_idx_with_start_filter(self) -> None:
        """Test get_reference_idx returns last point before start_ns."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))
        series.append(3000, MetricSample(value=3.0))

        time_filter = TimeRangeFilter(start_ns=2500)
        ref_idx = series.get_reference_idx(time_filter)

        assert ref_idx == 1  # Index of timestamp 2000

    def test_get_reference_idx_no_point_before_start(self) -> None:
        """Test get_reference_idx when no point exists before start_ns."""
        series = ScalarTimeSeries()
        series.append(1000, MetricSample(value=1.0))
        series.append(2000, MetricSample(value=2.0))

        time_filter = TimeRangeFilter(start_ns=500)
        ref_idx = series.get_reference_idx(time_filter)

        assert ref_idx is None


class TestHistogramTimeSeries:
    """Tests for HistogramTimeSeries class."""

    def test_empty_series(self) -> None:
        """Test empty series properties."""
        series = HistogramTimeSeries()

        assert len(series) == 0
        assert len(series.timestamps) == 0
        assert series.bucket_les == ()

    def test_append_single_histogram(self) -> None:
        """Test appending a single histogram sample."""
        series = HistogramTimeSeries()
        sample = MetricSample(
            buckets={"0.1": 5.0, "1.0": 10.0, "+Inf": 15.0},
            sum=50.0,
            count=15.0,
        )

        series.append(1000, sample)

        assert len(series) == 1
        assert series.timestamps[0] == 1000
        assert series.sums[0] == 50.0
        assert series.counts[0] == 15.0
        assert series.bucket_les == ("0.1", "1.0", "+Inf")

    def test_append_without_buckets_raises(self) -> None:
        """Test that appending without buckets raises ValueError."""
        series = HistogramTimeSeries()
        # Create sample with buckets to bypass MetricSample validation
        sample = MetricSample(buckets={"1.0": 5.0}, sum=10.0, count=5.0)
        # Manually set buckets to None to test HistogramTimeSeries validation
        sample.buckets = None

        with pytest.raises(ValueError, match="Buckets are required"):
            series.append(1000, sample)

    def test_bucket_sorting(self) -> None:
        """Test that bucket boundaries are sorted correctly."""
        series = HistogramTimeSeries()
        # Provide buckets in unsorted order
        sample = MetricSample(
            buckets={"+Inf": 10.0, "1.0": 5.0, "0.1": 2.0},
            sum=10.0,
            count=10.0,
        )

        series.append(1000, sample)

        # Should be sorted numerically with +Inf last
        assert series.bucket_les == ("0.1", "1.0", "+Inf")

    def test_append_multiple_in_order(self) -> None:
        """Test appending multiple histograms in order."""
        series = HistogramTimeSeries()

        series.append(
            1000, MetricSample(buckets={"1.0": 5.0, "+Inf": 10.0}, sum=10.0, count=10.0)
        )
        series.append(
            2000,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 20.0}, sum=25.0, count=20.0),
        )

        assert len(series) == 2
        assert list(series.timestamps) == [1000, 2000]
        assert list(series.sums) == [10.0, 25.0]
        assert list(series.counts) == [10.0, 20.0]

    def test_append_out_of_order_maintains_sorted(self) -> None:
        """Test that out-of-order appends maintain sorted order."""
        series = HistogramTimeSeries()

        series.append(
            3000,
            MetricSample(buckets={"1.0": 30.0, "+Inf": 30.0}, sum=30.0, count=30.0),
        )
        series.append(
            1000,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 10.0}, sum=10.0, count=10.0),
        )
        series.append(
            2000,
            MetricSample(buckets={"1.0": 20.0, "+Inf": 20.0}, sum=20.0, count=20.0),
        )

        assert len(series) == 3
        assert list(series.timestamps) == [1000, 2000, 3000]
        assert list(series.sums) == [10.0, 20.0, 30.0]

    def test_get_bucket_dict(self) -> None:
        """Test get_bucket_dict returns correct dict."""
        series = HistogramTimeSeries()
        series.append(
            1000,
            MetricSample(
                buckets={"0.1": 5.0, "1.0": 10.0, "+Inf": 15.0}, sum=10.0, count=15.0
            ),
        )

        bucket_dict = series.get_bucket_dict(0)

        assert bucket_dict == {"0.1": 5.0, "1.0": 10.0, "+Inf": 15.0}

    def test_get_bucket_dict_empty_series(self) -> None:
        """Test get_bucket_dict on empty series returns empty dict."""
        series = HistogramTimeSeries()
        assert series.get_bucket_dict(0) == {}

    def test_missing_bucket_defaults_to_zero(self) -> None:
        """Test that missing buckets default to 0.0."""
        series = HistogramTimeSeries()
        # First sample establishes bucket schema
        series.append(
            1000,
            MetricSample(
                buckets={"0.1": 5.0, "1.0": 10.0, "+Inf": 15.0}, sum=10.0, count=15.0
            ),
        )
        # Second sample is missing "0.1" bucket
        series.append(
            2000,
            MetricSample(buckets={"1.0": 20.0, "+Inf": 30.0}, sum=20.0, count=30.0),
        )

        bucket_dict = series.get_bucket_dict(1)
        assert bucket_dict["0.1"] == 0.0
        assert bucket_dict["1.0"] == 20.0

    def test_get_indices_for_filter_no_filter(self) -> None:
        """Test get_indices_for_filter with no filter."""
        series = HistogramTimeSeries()
        series.append(
            1000, MetricSample(buckets={"1.0": 5.0, "+Inf": 5.0}, sum=5.0, count=5.0)
        )
        series.append(
            2000,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 10.0}, sum=10.0, count=10.0),
        )

        ref_idx, final_idx = series.get_indices_for_filter(None)

        assert ref_idx is None
        assert final_idx == 1

    def test_get_indices_for_filter_with_start(self) -> None:
        """Test get_indices_for_filter with start_ns."""
        series = HistogramTimeSeries()
        series.append(
            1000, MetricSample(buckets={"1.0": 5.0, "+Inf": 5.0}, sum=5.0, count=5.0)
        )
        series.append(
            2000,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 10.0}, sum=10.0, count=10.0),
        )
        series.append(
            3000,
            MetricSample(buckets={"1.0": 15.0, "+Inf": 15.0}, sum=15.0, count=15.0),
        )

        time_filter = TimeRangeFilter(start_ns=1500)
        ref_idx, final_idx = series.get_indices_for_filter(time_filter)

        assert ref_idx == 0  # Last point before 1500
        assert final_idx == 2

    def test_get_observation_rates_empty_series(self) -> None:
        """Test get_observation_rates on empty series."""
        series = HistogramTimeSeries()
        rates = series.get_observation_rates()

        assert len(rates) == 0

    def test_get_observation_rates_single_point(self) -> None:
        """Test get_observation_rates with single point."""
        series = HistogramTimeSeries()
        series.append(
            1000, MetricSample(buckets={"1.0": 5.0, "+Inf": 5.0}, sum=5.0, count=5.0)
        )

        rates = series.get_observation_rates()

        assert len(rates) == 0  # Need at least 2 points for rates

    def test_get_observation_rates_computation(self) -> None:
        """Test get_observation_rates computes rates correctly."""
        series = HistogramTimeSeries()
        # 1 second apart, count increases by 10 -> rate = 10/s
        series.append(
            0, MetricSample(buckets={"1.0": 0.0, "+Inf": 0.0}, sum=0.0, count=0.0)
        )
        series.append(
            1_000_000_000,
            MetricSample(buckets={"1.0": 10.0, "+Inf": 10.0}, sum=50.0, count=10.0),
        )

        rates = series.get_observation_rates()

        assert len(rates) == 1
        assert abs(rates[0] - 10.0) < 0.01

    def test_capacity_growth(self) -> None:
        """Test that capacity grows automatically."""
        series = HistogramTimeSeries()

        # Append more than initial capacity (256)
        for i in range(300):
            series.append(
                i * 1000,
                MetricSample(
                    buckets={"1.0": float(i), "+Inf": float(i)},
                    sum=float(i),
                    count=float(i),
                ),
            )

        assert len(series) == 300


class TestServerMetricKey:
    """Tests for ServerMetricKey class."""

    def test_no_labels(self) -> None:
        """Test key with no labels."""
        key = ServerMetricKey.from_name_and_labels("my_metric", None)

        assert key.name == "my_metric"
        assert key.labels == ()
        assert key.labels_dict is None

    def test_empty_labels_dict(self) -> None:
        """Test key with empty labels dict."""
        key = ServerMetricKey.from_name_and_labels("my_metric", {})

        assert key.name == "my_metric"
        assert key.labels == ()
        assert key.labels_dict is None

    def test_with_labels(self) -> None:
        """Test key with labels."""
        key = ServerMetricKey.from_name_and_labels(
            "http_requests", {"method": "GET", "status": "200"}
        )

        assert key.name == "http_requests"
        assert key.labels == (("method", "GET"), ("status", "200"))
        assert key.labels_dict == {"method": "GET", "status": "200"}

    def test_labels_sorted_by_key(self) -> None:
        """Test that labels are sorted by key for consistent ordering."""
        key1 = ServerMetricKey.from_name_and_labels(
            "metric", {"z": "1", "a": "2", "m": "3"}
        )
        key2 = ServerMetricKey.from_name_and_labels(
            "metric", {"a": "2", "z": "1", "m": "3"}
        )

        # Both should have same tuple representation
        assert key1 == key2
        assert key1.labels == (("a", "2"), ("m", "3"), ("z", "1"))

    def test_hashable(self) -> None:
        """Test that keys are hashable and can be used as dict keys."""
        key1 = ServerMetricKey.from_name_and_labels("metric", {"a": "1"})
        key2 = ServerMetricKey.from_name_and_labels("metric", {"a": "1"})

        d = {key1: "value"}
        assert d[key2] == "value"


class TestServerMetricEntry:
    """Tests for ServerMetricEntry class."""

    def test_from_gauge_metric_family(self) -> None:
        """Test creating entry from gauge metric family."""
        family = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Test gauge",
            samples=[],
        )

        entry = ServerMetricEntry.from_metric_family(family)

        assert entry.metric_type == PrometheusMetricType.GAUGE
        assert entry.description == "Test gauge"
        assert isinstance(entry.data, ScalarTimeSeries)

    def test_from_counter_metric_family(self) -> None:
        """Test creating entry from counter metric family."""
        family = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            description="Test counter",
            samples=[],
        )

        entry = ServerMetricEntry.from_metric_family(family)

        assert entry.metric_type == PrometheusMetricType.COUNTER
        assert isinstance(entry.data, ScalarTimeSeries)

    def test_from_histogram_metric_family(self) -> None:
        """Test creating entry from histogram metric family."""
        family = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            description="Test histogram",
            samples=[],
        )

        entry = ServerMetricEntry.from_metric_family(family)

        assert entry.metric_type == PrometheusMetricType.HISTOGRAM
        assert isinstance(entry.data, HistogramTimeSeries)


class TestServerMetricsTimeSeries:
    """Tests for ServerMetricsTimeSeries class."""

    def test_empty_time_series(self) -> None:
        """Test empty time series properties."""
        ts = ServerMetricsTimeSeries()

        assert len(ts) == 0
        assert ts.first_update_ns == 0
        assert ts.last_update_ns == 0
        assert len(ts.metrics) == 0

    def test_append_snapshot_updates_timestamps(self) -> None:
        """Test that append_snapshot updates timestamp tracking."""
        ts = ServerMetricsTimeSeries()

        record = ServerMetricsRecord(
            timestamp_ns=1000,
            endpoint_url="http://localhost/metrics",
            endpoint_latency_ns=1000,
            metrics={
                "gauge_metric": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Test",
                    samples=[MetricSample(value=42.0)],
                ),
            },
        )

        ts.append_snapshot(record)

        assert len(ts) == 1
        assert ts.first_update_ns == 1000
        assert ts.last_update_ns == 1000

    def test_append_multiple_snapshots(self) -> None:
        """Test appending multiple snapshots."""
        ts = ServerMetricsTimeSeries()

        for i in range(3):
            record = ServerMetricsRecord(
                timestamp_ns=(i + 1) * 1000,
                endpoint_url="http://localhost/metrics",
                endpoint_latency_ns=1000,
                metrics={
                    "gauge_metric": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Test",
                        samples=[MetricSample(value=float(i))],
                    ),
                },
            )
            ts.append_snapshot(record)

        assert len(ts) == 3
        assert ts.first_update_ns == 1000
        assert ts.last_update_ns == 3000

    def test_append_empty_record(self) -> None:
        """Test that empty record doesn't update state."""
        ts = ServerMetricsTimeSeries()

        record = ServerMetricsRecord(
            timestamp_ns=1000,
            endpoint_url="http://localhost/metrics",
            endpoint_latency_ns=1000,
            metrics={},
        )

        ts.append_snapshot(record)

        assert len(ts) == 0

    def test_scrape_latencies_tracked(self) -> None:
        """Test that scrape latencies are tracked."""
        ts = ServerMetricsTimeSeries()

        record = ServerMetricsRecord(
            timestamp_ns=1000,
            endpoint_url="http://localhost/metrics",
            endpoint_latency_ns=50000,
            metrics={
                "gauge_metric": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Test",
                    samples=[MetricSample(value=1.0)],
                ),
            },
        )

        ts.append_snapshot(record)

        assert ts._fetch_latencies_ns == [50000]


class TestServerMetricsHierarchy:
    """Tests for ServerMetricsHierarchy class."""

    def test_empty_hierarchy(self) -> None:
        """Test empty hierarchy."""
        hierarchy = ServerMetricsHierarchy()

        assert len(hierarchy.endpoints) == 0

    def test_add_record_creates_endpoint(self) -> None:
        """Test that add_record creates new endpoint entry."""
        hierarchy = ServerMetricsHierarchy()

        record = ServerMetricsRecord(
            timestamp_ns=1000,
            endpoint_url="http://localhost:8080/metrics",
            endpoint_latency_ns=1000,
            metrics={
                "test_metric": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Test",
                    samples=[MetricSample(value=1.0)],
                ),
            },
        )

        hierarchy.add_record(record)

        assert "http://localhost:8080/metrics" in hierarchy.endpoints

    def test_add_records_multiple_endpoints(self) -> None:
        """Test adding records from multiple endpoints."""
        hierarchy = ServerMetricsHierarchy()

        for port in [8080, 8081]:
            record = ServerMetricsRecord(
                timestamp_ns=1000,
                endpoint_url=f"http://localhost:{port}/metrics",
                endpoint_latency_ns=1000,
                metrics={
                    "test_metric": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Test",
                        samples=[MetricSample(value=1.0)],
                    ),
                },
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 2
        assert "http://localhost:8080/metrics" in hierarchy.endpoints
        assert "http://localhost:8081/metrics" in hierarchy.endpoints

    def test_add_records_same_endpoint_accumulates(self) -> None:
        """Test that records to same endpoint accumulate."""
        hierarchy = ServerMetricsHierarchy()

        for i in range(5):
            record = ServerMetricsRecord(
                timestamp_ns=i * 1000,
                endpoint_url="http://localhost:8080/metrics",
                endpoint_latency_ns=1000,
                metrics={
                    "test_metric": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Test",
                        samples=[MetricSample(value=float(i))],
                    ),
                },
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 1
        ts = hierarchy.endpoints["http://localhost:8080/metrics"]
        assert len(ts) == 5

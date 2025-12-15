# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    ServerMetricsResults,
)
from aiperf.server_metrics.storage import (
    ServerMetricKey,
    ServerMetricsHierarchy,
    ServerMetricsTimeSeries,
)


class TestMetricSampleValidation:
    """Test MetricSample mutual exclusivity validation."""

    def test_value_only_is_valid(self):
        """Test that value-only sample is valid."""
        sample = MetricSample(value=42.0)
        assert sample.value == 42.0
        assert sample.buckets is None

    def test_value_with_labels_is_valid(self):
        """Test that value with labels is valid."""
        sample = MetricSample(
            labels={"model": "test-model", "status": "success"},
            value=100.0,
        )
        assert sample.labels == {"model": "test-model", "status": "success"}
        assert sample.value == 100.0
        assert sample.buckets is None

    def test_histogram_only_is_valid(self):
        """Test that histogram-only sample (buckets, sum, count) is valid."""
        sample = MetricSample(
            buckets={"0.1": 10, "1.0": 50, "+Inf": 100},
            sum=100.0,
            count=100,
        )
        assert sample.buckets == {"0.1": 10, "1.0": 50, "+Inf": 100}
        assert sample.sum == 100.0
        assert sample.count == 100

    def test_histogram_with_labels_is_valid(self):
        """Test that histogram with labels is valid."""
        sample = MetricSample(
            labels={"model": "test"},
            buckets={"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0},
            sum=125.5,
            count=100.0,
        )
        assert sample.labels == {"model": "test"}
        assert sample.buckets == {"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0}
        assert sample.sum == 125.5
        assert sample.count == 100.0

    def test_neither_value_nor_buckets_raises(self):
        """Test that setting neither value nor buckets raises ValidationError."""
        with pytest.raises(ValueError, match="One of value or buckets must be set"):
            MetricSample(labels={"key": "value"})

    def test_value_and_buckets_raises(self):
        """Test that setting both value and buckets raises ValidationError."""
        with pytest.raises(ValueError, match="Only one of value or buckets can be set"):
            MetricSample(value=42.0, buckets={"0.1": 10})

    def test_value_with_sum_raises(self):
        """Test that setting value with sum raises ValidationError."""
        with pytest.raises(
            ValueError, match="If value is set, sum and count must not be set"
        ):
            MetricSample(value=42.0, sum=100.0)

    def test_value_with_count_raises(self):
        """Test that setting value with count raises ValidationError."""
        with pytest.raises(
            ValueError, match="If value is set, sum and count must not be set"
        ):
            MetricSample(value=42.0, count=100)


class TestServerMetricsRecordConversion:
    """Test ServerMetricsRecord to slim format conversion."""

    def test_full_record_to_slim(
        self,
        sample_counter_metric: MetricFamily,
        sample_histogram_metric: MetricFamily,
    ):
        """Test converting complete record with multiple metric types."""
        record = ServerMetricsRecord(
            endpoint_url="http://node1:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": sample_counter_metric,
                "ttft": sample_histogram_metric,
            },
        )

        slim = record.to_slim()

        assert slim.endpoint_url == "http://node1:8081/metrics"
        assert slim.timestamp_ns == 1_000_000_000
        assert slim.endpoint_latency_ns == 5_000_000
        assert len(slim.metrics) == 2
        assert "requests_total" in slim.metrics
        assert "ttft" in slim.metrics

        # Samples are already MetricSample (slim format)
        assert isinstance(slim.metrics["requests_total"][0], MetricSample)
        assert slim.metrics["requests_total"][0].value == 150.0

        assert isinstance(slim.metrics["ttft"][0], MetricSample)
        assert slim.metrics["ttft"][0].buckets is not None

    def test_slim_record_preserves_endpoint_url(self, sample_server_metrics_record):
        """Test that endpoint_url is preserved in slim format."""
        slim = sample_server_metrics_record.to_slim()
        assert slim.endpoint_url == sample_server_metrics_record.endpoint_url


class TestServerMetricsTimeSeries:
    """Test ServerMetricsTimeSeries storage."""

    def test_append_snapshot_stores_metrics(self, sample_server_metrics_record):
        """Test appending a record stores metrics in unified dict."""
        ts = ServerMetricsTimeSeries()

        ts.append_snapshot(sample_server_metrics_record)

        assert len(ts) == 1
        # Should have extracted gauge metrics using MetricKey
        gauge_key = ServerMetricKey(
            "vllm:gpu_cache_usage_perc",
            (("model_name", "meta-llama/Llama-3.1-8B-Instruct"),),
        )
        assert gauge_key in ts.metrics
        # Should have extracted counter metrics
        counter_key = ServerMetricKey(
            "vllm:request_success_total",
            (("model_name", "meta-llama/Llama-3.1-8B-Instruct"),),
        )
        assert counter_key in ts.metrics

    def test_append_multiple_snapshots(
        self, sample_gauge_metric, sample_counter_metric
    ):
        """Test appending multiple records accumulates data."""
        ts = ServerMetricsTimeSeries()

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "gauge_metric": sample_gauge_metric,
                    "counter_metric": sample_counter_metric,
                },
            )
            ts.append_snapshot(record)

        assert len(ts) == 3


class TestServerMetricsHierarchy:
    """Test ServerMetricsHierarchy storage model."""

    def test_add_record_creates_endpoint(self, sample_server_metrics_record):
        """Test adding a record creates endpoint entry."""
        hierarchy = ServerMetricsHierarchy()

        hierarchy.add_record(sample_server_metrics_record)

        assert "http://localhost:8081/metrics" in hierarchy.endpoints
        time_series = hierarchy.endpoints["http://localhost:8081/metrics"]
        assert isinstance(time_series, ServerMetricsTimeSeries)
        assert len(time_series) == 1

    def test_add_record_multiple_endpoints(self, sample_gauge_metric):
        """Test adding records from multiple endpoints."""
        hierarchy = ServerMetricsHierarchy()

        for i, endpoint in enumerate(
            ["http://node1:8081/metrics", "http://node2:8081/metrics"]
        ):
            record = ServerMetricsRecord(
                endpoint_url=endpoint,
                timestamp_ns=1_000_000_000 + i,
                endpoint_latency_ns=5_000_000,
                metrics={"gauge_metric": sample_gauge_metric},
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 2
        assert "http://node1:8081/metrics" in hierarchy.endpoints
        assert "http://node2:8081/metrics" in hierarchy.endpoints

    def test_add_record_updates_existing_endpoint(self, sample_gauge_metric):
        """Test adding multiple records to same endpoint accumulates data."""
        hierarchy = ServerMetricsHierarchy()

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"gauge_metric": sample_gauge_metric},
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 1
        time_series = hierarchy.endpoints["http://localhost:8081/metrics"]
        assert len(time_series) == 3


class TestServerMetricsResults:
    """Test ServerMetricsResults model."""

    def test_results_creation(self):
        """Test creating ServerMetricsResults with all fields."""
        results = ServerMetricsResults(
            benchmark_id="test-benchmark-id",
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            endpoints_configured=["http://node1:8081/metrics"],
            endpoints_successful=["http://node1:8081/metrics"],
            error_summary=[],
        )

        assert results.start_ns == 1_000_000_000
        assert results.end_ns == 2_000_000_000
        assert results.endpoints_configured == ["http://node1:8081/metrics"]
        assert results.endpoints_successful == ["http://node1:8081/metrics"]
        assert len(results.error_summary) == 0

    def test_results_with_errors(self):
        """Test creating ServerMetricsResults with error summary."""
        error = ErrorDetails(message="Connection failed")
        results = ServerMetricsResults(
            benchmark_id="test-benchmark-id",
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            error_summary=[ErrorDetailsCount(error_details=error, count=5)],
        )

        assert len(results.error_summary) == 1
        assert results.error_summary[0].count == 5

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    MetricSchema,
    ServerMetricsData,
    ServerMetricsHierarchy,
    ServerMetricsMetadata,
    ServerMetricsRecord,
    ServerMetricsTimeSeries,
    SlimMetricSample,
    SummaryData,
)


class TestMetricSampleConversion:
    """Test MetricSample to SlimMetricSample conversion."""

    def test_counter_to_slim(self):
        """Test converting simple counter metric to slim format."""
        sample = MetricSample(
            labels={"model": "test-model", "status": "success"},
            value=100.0,
        )
        slim = sample.to_slim()

        assert slim.labels == {"model": "test-model", "status": "success"}
        assert slim.value == 100.0
        assert slim.histogram is None
        assert slim.summary is None

    def test_gauge_to_slim(self):
        """Test converting gauge metric to slim format."""
        sample = MetricSample(labels=None, value=0.75)
        slim = sample.to_slim()

        assert slim.labels is None
        assert slim.value == 0.75
        assert slim.histogram is None

    def test_histogram_to_slim(self):
        """Test converting histogram metric to slim dict format."""
        sample = MetricSample(
            labels={"model": "test"},
            histogram=HistogramData(
                buckets={"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0},
                sum=125.5,
                count=100.0,
            ),
        )
        slim = sample.to_slim()

        assert slim.labels == {"model": "test"}
        assert slim.value is None
        assert slim.histogram == {"0.01": 5.0, "0.1": 15.0, "1.0": 50.0, "+Inf": 100.0}
        assert slim.sum == 125.5
        assert slim.count == 100.0

    def test_summary_to_slim(self):
        """Test converting summary metric to slim dict format."""
        sample = MetricSample(
            labels={"endpoint": "/v1/chat"},
            summary=SummaryData(
                quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
                sum=50.0,
                count=100.0,
            ),
        )
        slim = sample.to_slim()

        assert slim.labels == {"endpoint": "/v1/chat"}
        assert slim.summary == {"0.5": 0.1, "0.9": 0.5, "0.99": 1.0}
        assert slim.sum == 50.0
        assert slim.count == 100.0


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

        assert isinstance(slim.metrics["requests_total"][0], SlimMetricSample)
        assert slim.metrics["requests_total"][0].value == 150.0

        assert isinstance(slim.metrics["ttft"][0], SlimMetricSample)
        assert slim.metrics["ttft"][0].histogram is not None

    def test_slim_record_preserves_endpoint_url(self, sample_server_metrics_record):
        """Test that endpoint_url is preserved in slim format."""
        slim = sample_server_metrics_record.to_slim()
        assert slim.endpoint_url == sample_server_metrics_record.endpoint_url


class TestServerMetricsHierarchy:
    """Test hierarchical storage of server metrics data."""

    def test_add_record_creates_endpoint(self):
        """Test adding record to new endpoint creates hierarchy automatically."""
        hierarchy = ServerMetricsHierarchy()
        record = ServerMetricsRecord(
            endpoint_url="http://node1:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        hierarchy.add_record(record)

        assert "http://node1:8081/metrics" in hierarchy.endpoints
        assert (
            len(hierarchy.endpoints["http://node1:8081/metrics"].time_series.records)
            == 1
        )

    def test_add_multiple_records_same_endpoint(self):
        """Test adding multiple records to same endpoint accumulates time series."""
        hierarchy = ServerMetricsHierarchy()

        for i in range(3):
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
            hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 1
        assert (
            len(hierarchy.endpoints["http://node1:8081/metrics"].time_series.records)
            == 3
        )

    def test_add_records_multiple_endpoints(self):
        """Test adding records from different endpoints creates separate hierarchies."""
        hierarchy = ServerMetricsHierarchy()

        for i in range(2):
            for endpoint in ["http://node1:8081/metrics", "http://node2:8081/metrics"]:
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                hierarchy.add_record(record)

        assert len(hierarchy.endpoints) == 2
        assert (
            len(hierarchy.endpoints["http://node1:8081/metrics"].time_series.records)
            == 2
        )
        assert (
            len(hierarchy.endpoints["http://node2:8081/metrics"].time_series.records)
            == 2
        )

    def test_metadata_initialized_correctly(self):
        """Test that metadata is properly initialized when adding first record."""
        hierarchy = ServerMetricsHierarchy()
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        hierarchy.add_record(record)

        metadata = hierarchy.endpoints["http://localhost:8081/metrics"].metadata
        assert metadata.endpoint_url == "http://localhost:8081/metrics"
        assert metadata.endpoint_display == "http://localhost:8081/metrics"


class TestServerMetricsTimeSeries:
    """Test time series accumulation functionality."""

    def test_empty_time_series(self):
        """Test newly created time series is empty."""
        ts = ServerMetricsTimeSeries()
        assert len(ts.records) == 0

    def test_add_single_record(self):
        """Test adding single record to time series."""
        ts = ServerMetricsTimeSeries()
        record = ServerMetricsRecord(
            endpoint_url="http://test:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        ts.add_record(record)
        assert len(ts.records) == 1
        assert ts.records[0].timestamp_ns == 1_000_000_000

    def test_chronological_ordering(self):
        """Test that records maintain insertion order (chronological)."""
        ts = ServerMetricsTimeSeries()

        for i in range(5):
            record = ServerMetricsRecord(
                endpoint_url="http://test:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
            ts.add_record(record)

        assert len(ts.records) == 5
        for i in range(5):
            assert ts.records[i].timestamp_ns == 1_000_000_000 + i * 1_000_000


class TestServerMetricsData:
    """Test combined metadata and time series data structure."""

    def test_initialization_with_metadata(self):
        """Test ServerMetricsData initialization with metadata."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://node1:8081/metrics",
            endpoint_display="Node 1 vLLM",
        )
        data = ServerMetricsData(metadata=metadata)

        assert data.metadata.endpoint_url == "http://node1:8081/metrics"
        assert data.metadata.endpoint_display == "Node 1 vLLM"
        assert len(data.time_series.records) == 0

    def test_add_record_to_data(self):
        """Test adding record to ServerMetricsData updates time series."""
        metadata = ServerMetricsMetadata(
            endpoint_url="http://test:8081/metrics",
            endpoint_display="Test",
        )
        data = ServerMetricsData(metadata=metadata)

        record = ServerMetricsRecord(
            endpoint_url="http://test:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        data.add_record(record)
        assert len(data.time_series.records) == 1


class TestMetricSchema:
    """Test MetricSchema model for metadata."""

    def test_counter_schema(self):
        """Test schema for counter metric."""
        schema = MetricSchema(
            type=PrometheusMetricType.COUNTER,
            help="Total number of requests",
        )

        assert schema.type == PrometheusMetricType.COUNTER
        assert schema.help == "Total number of requests"
        assert schema.bucket_labels is None
        assert schema.quantile_labels is None

    def test_histogram_schema_with_buckets(self):
        """Test schema for histogram includes bucket labels."""
        schema = MetricSchema(
            type=PrometheusMetricType.HISTOGRAM,
            help="Request duration histogram",
            bucket_labels=["0.01", "0.1", "1.0", "+Inf"],
        )

        assert schema.type == PrometheusMetricType.HISTOGRAM
        assert schema.bucket_labels == ["0.01", "0.1", "1.0", "+Inf"]
        assert schema.quantile_labels is None

    def test_summary_schema_with_quantiles(self):
        """Test schema for summary includes quantile labels."""
        schema = MetricSchema(
            type=PrometheusMetricType.SUMMARY,
            help="Request latency quantiles",
            quantile_labels=["0.5", "0.9", "0.99"],
        )

        assert schema.type == PrometheusMetricType.SUMMARY
        assert schema.quantile_labels == ["0.5", "0.9", "0.99"]
        assert schema.bucket_labels is None


class TestHistogramAndSummaryData:
    """Test HistogramData and SummaryData models."""

    def test_histogram_data_structure(self):
        """Test HistogramData contains buckets, sum, and count."""
        hist = HistogramData(
            buckets={"0.01": 10.0, "0.1": 25.0, "+Inf": 50.0},
            sum=5.5,
            count=50.0,
        )

        assert len(hist.buckets) == 3
        assert hist.buckets["0.01"] == 10.0
        assert hist.sum == 5.5
        assert hist.count == 50.0

    def test_summary_data_structure(self):
        """Test SummaryData contains quantiles, sum, and count."""
        summary = SummaryData(
            quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
            sum=50.0,
            count=100.0,
        )

        assert len(summary.quantiles) == 3
        assert summary.quantiles["0.5"] == 0.1
        assert summary.sum == 50.0
        assert summary.count == 100.0

    def test_histogram_optional_fields(self):
        """Test histogram with optional fields as None."""
        hist = HistogramData(
            buckets={"0.01": 10.0},
            sum=None,
            count=None,
        )

        assert hist.buckets == {"0.01": 10.0}
        assert hist.sum is None
        assert hist.count is None

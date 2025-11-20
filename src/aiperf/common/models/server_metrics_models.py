# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount


class HistogramData(AIPerfBaseModel):
    """Structured histogram data with buckets, sum, and count."""

    buckets: dict[str, float] = Field(
        description="Bucket upper bounds to counts {le: value}"
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SummaryData(AIPerfBaseModel):
    """Structured summary data with quantiles, sum, and count."""

    quantiles: dict[str, float] = Field(
        description="Quantile to value {quantile: value}"
    )
    sum: float | None = Field(default=None, description="Sum of all observed values")
    count: float | None = Field(
        default=None, description="Total number of observations"
    )


class SlimMetricSample(AIPerfBaseModel):
    """Slim metric sample with minimal data using dictionary-based format.

    Optimized for JSONL export. Uses dictionary format for
    histogram/summary data for clarity:
    - Type and help text are in schema
    - Histogram bucket labels (le values) map to their counts
    - Summary quantile labels map to their values
    - sum/count are optional fields at sample level (used for histogram/summary)

    Format examples:
    - Counter/Gauge: {"value": 42.0} or {"labels": {...}, "value": 42.0}
    - Histogram: {"histogram": {"0.01": 10, "0.1": 25, "1.0": 50, ...}, "sum": 100.0, "count": 50}
    - Summary: {"summary": {"0.5": 0.1, "0.9": 0.5, "0.99": 1.0, ...}, "sum": 100.0, "count": 50}
    """

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: dict[str, float] | None = Field(
        default=None,
        description="Histogram bucket upper bounds (le) to counts mapping",
    )
    summary: dict[str, float] | None = Field(
        default=None,
        description="Summary quantile to value mapping",
    )
    sum: float | None = Field(
        default=None,
        description="Sum of all observed values (for histogram/summary)",
    )
    count: float | None = Field(
        default=None,
        description="Total number of observations (for histogram/summary)",
    )


class MetricSample(AIPerfBaseModel):
    """Single metric sample with labels and value."""

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram/summary special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    histogram: HistogramData | None = Field(
        default=None, description="Histogram data if metric is histogram type"
    )
    summary: SummaryData | None = Field(
        default=None, description="Summary data if metric is summary type"
    )

    def to_slim(self) -> SlimMetricSample:
        """Convert to slim metric sample format.

        For histograms and summaries, converts to dictionary format where
        bucket/quantile labels map to their counts/values.

        Returns:
            SlimMetricSample with dictionary-based histogram/summary data
        """
        if self.histogram:
            return SlimMetricSample(
                labels=self.labels,
                value=self.value,
                histogram=self.histogram.buckets,
                sum=self.histogram.sum,
                count=self.histogram.count,
            )

        if self.summary:
            return SlimMetricSample(
                labels=self.labels,
                value=self.value,
                summary=self.summary.quantiles,
                sum=self.summary.sum,
                count=self.summary.count,
            )

        return SlimMetricSample(
            labels=self.labels,
            value=self.value,
        )


class MetricFamily(AIPerfBaseModel):
    """Group of related metrics with same name and type."""

    type: PrometheusMetricType = Field(description="Metric type as enum")
    help: str = Field(description="Metric description from HELP text")
    samples: list[MetricSample] = Field(
        description="Metric samples grouped by base labels"
    )


class MetricSchema(AIPerfBaseModel):
    """Schema information for a metric (type, help text, and bucket/quantile labels).

    Sent once per metric in ServerMetricsMetadata to avoid repeating in every record.
    For histograms and summaries, includes static bucket/quantile labels so only
    counts/values need to be sent in each record.
    """

    type: PrometheusMetricType = Field(description="Metric type as enum")
    help: str = Field(description="Metric description from HELP text")
    bucket_labels: list[str] | None = Field(
        default=None,
        description="Histogram bucket upper bounds (le values) in order. Only for histogram metrics.",
    )
    quantile_labels: list[str] | None = Field(
        default=None,
        description="Summary quantile labels in order. Only for summary metrics.",
    )
    unique_label_values: dict[str, list[str]] | None = Field(
        default=None,
        description="Unique values seen for each label key during collection (e.g., {'status': ['success', 'error']}). Limited by AIPERF_SERVER_METRICS_MAX_UNIQUE_LABEL_VALUES per key.",
    )


class ServerMetricsSlimRecord(AIPerfBaseModel):
    """Slim server metrics record containing only time-varying data.

    This record excludes static metadata (endpoint_url, metric types, help text)
    to reduce ZMQ message size and JSONL file size. The metadata and schemas are sent
    once separately via ServerMetricsMetadataMessage.

    Format is optimized with flat structure and slim samples:
    - Metrics map directly to sample lists (no 'samples' key nesting)
    - Histogram/summary samples only include counts/values (bucket/quantile labels in schema)
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    endpoint_latency_ns: int = Field(
        description="Nanoseconds it took to collect the metrics from the endpoint"
    )
    metrics: dict[str, list[SlimMetricSample]] = Field(
        description="Metrics grouped by family name, mapping directly to slim sample list"
    )


class ServerMetricsRecord(AIPerfBaseModel):
    """Single server metrics data point from Prometheus endpoint.

    This record contains all metrics scraped from one Prometheus endpoint at one point in time.
    Used for hierarchical storage: endpoint_url -> time series data.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when metrics were collected (time_ns)"
    )
    endpoint_latency_ns: int = Field(
        description="Nanoseconds it took to collect the metrics from the endpoint"
    )
    metrics: dict[str, MetricFamily] = Field(
        description="Metrics grouped by family name"
    )

    def to_slim(self) -> ServerMetricsSlimRecord:
        """Convert to slim record using array-based format for histograms/summaries.

        Creates flat structure where metrics map directly to slim sample lists.
        For histograms and summaries, uses array format with bucket counts/quantile values
        and sum/count at the sample level.

        Returns:
            ServerMetricsSlimRecord with only timestamp and slim samples (flat structure)
        """
        slim_metrics = {
            name: [sample.to_slim() for sample in family.samples]
            for name, family in self.metrics.items()
        }

        return ServerMetricsSlimRecord(
            timestamp_ns=self.timestamp_ns,
            endpoint_latency_ns=self.endpoint_latency_ns,
            endpoint_url=self.endpoint_url,
            metrics=slim_metrics,
        )


class ServerMetricsMetadata(AIPerfBaseModel):
    """Metadata for a server metrics endpoint that doesn't change over time.

    Includes metric schemas (type and help text) to avoid sending them in every record.
    """

    endpoint_url: str = Field(description="Prometheus metrics endpoint URL")
    endpoint_display: str = Field(description="Human-readable endpoint display name")
    metric_schemas: dict[str, MetricSchema] = Field(
        default_factory=dict,
        description="Metric schemas (type and help) sent once to avoid repetition",
    )


class ServerMetricsMetadataFile(AIPerfBaseModel):
    """Container for all server metrics endpoint metadata.

    This model represents the complete server_metrics_metadata.json file structure,
    mapping endpoint URLs to their metadata.
    """

    endpoints: dict[str, ServerMetricsMetadata] = Field(
        default_factory=dict,
        description="Dict mapping endpoint_url to ServerMetricsMetadata",
    )


class ServerMetricsTimeSeries(AIPerfBaseModel):
    """Time series data for server metrics from a single endpoint.

    Stores chronological records of all metrics collected from one Prometheus endpoint.
    """

    records: list[ServerMetricsRecord] = Field(
        default_factory=list, description="Chronological server metrics records"
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add new server metrics record to time series.

        Args:
            record: New server metrics data point from Prometheus endpoint
        """
        self.records.append(record)


class ServerMetricsData(AIPerfBaseModel):
    """Complete server metrics data for one endpoint: metadata + time series.

    This combines static endpoint information with dynamic time-series data,
    providing the complete picture for one Prometheus endpoint's metrics.
    Parallel to GpuTelemetryData in the telemetry system.
    """

    metadata: ServerMetricsMetadata = Field(description="Static endpoint information")
    time_series: ServerMetricsTimeSeries = Field(
        default_factory=ServerMetricsTimeSeries,
        description="Time series for all metrics from this endpoint",
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to time series.

        Args:
            record: New server metrics data from Prometheus endpoint
        """
        self.time_series.add_record(record)


class ServerMetricsHierarchy(AIPerfBaseModel):
    """Hierarchical storage: endpoint_url -> complete endpoint server metrics data.

    This provides hierarchical structure for efficient access to server metrics
    data organized by Prometheus endpoint.

    Structure:
    {
        "http://localhost:8081/metrics": ServerMetricsData(metadata + time series),
        "http://node2:8081/metrics": ServerMetricsData(metadata + time series)
    }
    """

    endpoints: dict[str, ServerMetricsData] = Field(
        default_factory=dict,
        description="Dict: endpoint_url -> server metrics endpoint data",
    )

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to hierarchical storage.

        Args:
            record: New server metrics data from Prometheus endpoint

        Note: Automatically creates hierarchy levels as needed:
        - New endpoints get initialized with metadata and empty time series
        """
        if record.endpoint_url not in self.endpoints:
            metadata = ServerMetricsMetadata(
                endpoint_url=record.endpoint_url,
                endpoint_display=record.endpoint_url,  # Can be enhanced with display name
            )
            self.endpoints[record.endpoint_url] = ServerMetricsData(metadata=metadata)

        self.endpoints[record.endpoint_url].add_record(record)


class ServerMetricsResults(AIPerfBaseModel):
    """Results from server metrics collection during a profile run.

    This class contains all server metrics data and metadata collected during
    a benchmarking session, separate from inference performance results.
    """

    server_metrics_data: ServerMetricsHierarchy = Field(
        description="Hierarchical server metrics data organized by Prometheus endpoint"
    )
    start_ns: int = Field(
        description="Start time of server metrics collection in nanoseconds"
    )
    end_ns: int = Field(
        description="End time of server metrics collection in nanoseconds"
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs configured",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs that successfully provided data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )


class ProcessServerMetricsResult(AIPerfBaseModel):
    """Result of server metrics processing - mirrors ProcessRecordsResult pattern.

    This provides a parallel structure to ProcessRecordsResult for the server metrics pipeline,
    maintaining complete separation while following the same architectural patterns.
    """

    results: ServerMetricsResults = Field(
        description="The processed server metrics results"
    )
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any errors that occurred while processing server metrics data",
    )

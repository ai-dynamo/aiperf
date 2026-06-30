# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pydantic import Field, SerializeAsAny, model_validator
from typing_extensions import Self

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount


@dataclass(frozen=True, slots=True)
class TimeRangeFilter:
    """Filter for selecting metrics within a specific time range.

    Immutable time range specification used throughout the metrics processing
    pipeline to exclude warmup and cooldown periods from statistics computation.

    Supports partial ranges (None on either end) for flexibility. Automatically
    validates that start < end to catch configuration errors early.

    Args:
        start_ns: Start of valid time range in nanoseconds (inclusive).
                 None means include from beginning of data collection.
        end_ns: End of valid time range in nanoseconds (inclusive).
               None means include to end of data collection.

    Raises:
        ValueError: If both bounds specified and start_ns >= end_ns

    Example:
        >>> # Filter for profiling phase only (exclude 5s warmup)
        >>> filter = TimeRangeFilter(
        ...     start_ns=5_000_000_000,  # 5 seconds in nanoseconds
        ...     end_ns=65_000_000_000    # 65 seconds (60s profiling)
        ... )
        >>> filter.includes(3_000_000_000)  # 3s timestamp
        False  # Before start_ns
        >>> filter.includes(30_000_000_000)  # 30s timestamp
        True  # Within range
    """

    start_ns: int | None = None
    end_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate that start_ns < end_ns if both are specified.

        Called automatically after dataclass initialization to ensure valid time range.

        Raises:
            ValueError: If start_ns >= end_ns (empty or reversed range)
        """
        if (
            self.start_ns is not None
            and self.end_ns is not None
            and self.start_ns >= self.end_ns
        ):
            raise ValueError(
                f"start_ns ({self.start_ns}) must be less than end_ns ({self.end_ns})"
            )

    def includes(self, timestamp_ns: int) -> bool:
        """Check if a timestamp falls within this time range (inclusive bounds).

        Args:
            timestamp_ns: Timestamp to check in nanoseconds

        Returns:
            True if timestamp is within [start_ns, end_ns] inclusive range,
            False if outside. None bounds are treated as unbounded (always include).
        """
        return not (
            (self.start_ns is not None and timestamp_ns < self.start_ns)
            or (self.end_ns is not None and timestamp_ns > self.end_ns)
        )


# =============================================================================
# Data Models (Prometheus metrics records and metadata)
# =============================================================================


class MetricSample(AIPerfBaseModel):
    """Single metric sample from Prometheus exposition format.

    Represents one data point from a Prometheus metric scrape. Format depends
    on metric type:
    - Counter/Gauge: Uses `value` field only
    - Histogram: Uses `buckets`, `sum`, and `count` fields

    Labels provide dimensional data for grouping and filtering (e.g., HTTP method,
    status code, instance ID). Histogram labels exclude the special "le" label
    which is used for bucket boundaries instead.

    Validation ensures mutual exclusivity between value and histogram fields
    to prevent malformed samples.
    """

    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels (excluding histogram special labels). None if no labels.",
    )
    value: float | None = Field(
        default=None, description="Simple metric value (counter/gauge)"
    )
    buckets: dict[str, float] | None = Field(
        default=None,
        description='Histogram bucket upper bounds (le="less than or equal") to counts. Keys are strings like "0.01", "0.1", "1.0"',
    )
    sum: float | None = Field(
        default=None,
        description="Sum of all observed values (for histogram only)",
    )
    count: float | None = Field(
        default=None,
        description="Total number of observations (for histogram only)",
    )

    @model_validator(mode="after")
    def _validate_mutual_exclusivity(self) -> Self:
        """Ensure metric sample has valid field combination.

        Validates that:
        1. Exactly one of {value, buckets} is set (not both, not neither)
        2. If value set (counter/gauge), then sum/count must be None
        3. If buckets set (histogram), value must be None

        Raises:
            ValueError: If field combination is invalid

        Returns:
            Self for method chaining
        """
        if self.value is not None and self.buckets is not None:
            raise ValueError("Only one of value or buckets can be set")
        if self.value is None and self.buckets is None:
            raise ValueError("One of value or buckets must be set")
        if self.value is not None and (self.sum is not None or self.count is not None):
            raise ValueError("If value is set, sum and count must not be set")
        return self


class MetricFamily(AIPerfBaseModel):
    """Group of related metrics with same name and type from Prometheus.

    Represents a complete metric family from Prometheus exposition format
    (all samples under one TYPE and HELP declaration). Contains metadata
    (type, description) and all samples with their label dimensions.

    For multi-dimensional metrics, samples list contains one entry per unique
    label combination. For histograms, each sample contains all buckets for
    that label set.

    Args:
        type: Prometheus metric type (COUNTER, GAUGE, HISTOGRAM, etc.)
        description: Human-readable description from HELP text
        samples: List of MetricSample objects, one per unique label combination
    """

    type: PrometheusMetricType = Field(description="Metric type as enum")
    description: str = Field(description="Metric description from HELP text")
    samples: list[MetricSample] = Field(
        description="Metric samples grouped by base labels"
    )


class SlimRecord(AIPerfBaseModel):
    """Slim server metrics record containing only time-varying data.

    This record excludes static metadata (metric types, help text)
    to reduce JSONL file size. Includes HTTP trace timing fields for
    precise correlation with client request timestamps.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp representing when server generated metrics"
    )
    endpoint_latency_ns: int = Field(
        description="Nanoseconds for total HTTP round-trip (request start to completion)"
    )
    metrics: dict[str, list[MetricSample]] = Field(
        description="Metrics grouped by family name, mapping directly to metric sample list"
    )
    request_sent_ns: int | None = Field(
        default=None,
        description="Wall-clock timestamp in nanoseconds when HTTP request was initiated",
    )
    first_byte_ns: int | None = Field(
        default=None,
        description="Wall-clock timestamp in nanoseconds when first response byte received from server",
    )


class ServerMetricsRecord(AIPerfBaseModel):
    """Single server metrics data point from Prometheus endpoint.

    This record contains all metrics scraped from one Prometheus endpoint at one point in time.
    Used for hierarchical storage: endpoint_url -> time series data.

    The trace timing fields provide precise correlation between server metrics and client
    requests by capturing when the server actually generated the metrics, not just when
    the client received the full response.
    """

    endpoint_url: str = Field(
        description="Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp representing when server generated metrics. "
        "Uses first_byte_ns if available (most accurate), otherwise falls back to time after request completes."
    )
    endpoint_latency_ns: int | None = Field(
        default=None,
        description="Nanoseconds for total HTTP round-trip (request start to completion)",
    )
    metrics: dict[str, MetricFamily] = Field(
        description="Metrics grouped by family name"
    )
    request_sent_ns: int | None = Field(
        default=None,
        description="Wall-clock timestamp in nanoseconds when HTTP request was initiated (from aiohttp trace)",
    )
    first_byte_ns: int | None = Field(
        default=None,
        description="Wall-clock timestamp in nanoseconds when first response byte received from server. "
        "Best approximation of when server generated the metrics.",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True if this record's metrics are identical to the previous fetch from this endpoint",
    )

    def to_slim(self) -> SlimRecord:
        """Convert to slim record.

        Excludes metrics ending in _info as they are typically used for metadata and not metrics,
        so they will be include in the final export, but not in the JSONL records.

        Returns:
            ServerMetricsSlimRecord with only timestamp and slim samples (flat structure)
        """
        slim_metrics = {
            name: family.samples
            for name, family in self.metrics.items()
            if not name.endswith("_info")
        }

        return SlimRecord(
            timestamp_ns=self.timestamp_ns,
            endpoint_latency_ns=self.endpoint_latency_ns,
            endpoint_url=self.endpoint_url,
            metrics=slim_metrics,
            request_sent_ns=self.request_sent_ns,
            first_byte_ns=self.first_byte_ns,
        )


# =============================================================================
# Server Metrics Export Data
# =============================================================================


class BaseTimeslice(AIPerfBaseModel):
    """Base timeslice for server metrics.

    Timeslices represent fixed-duration windows of time for analyzing metrics.
    The `is_complete` flag indicates whether the timeslice covers a full duration
    or is a partial slice (typically the final slice when data ends mid-window).

    Partial timeslices should be included in exports for data completeness but
    excluded from aggregate statistics to avoid skewing rate calculations.

    For space efficiency, `is_complete` is omitted from JSON exports when True
    (most timeslices are complete). Missing field is treated as True on deserialization.
    """

    start_ns: int = Field(description="Timeslice start timestamp in nanoseconds")
    end_ns: int = Field(description="Timeslice end timestamp in nanoseconds")
    is_complete: bool | None = Field(
        default=None,
        description="False for partial timeslices (typically the final slice). "
        "None or True for complete timeslices covering the full configured duration. "
        "Partial slices should be excluded from aggregate statistics. "
        "None by default to save space in JSON exports (treated as complete).",
    )


class CounterTimeslice(BaseTimeslice):
    """Single counter timeslice in a windowed time series."""

    total: float = Field(
        description="Total increase in counter value during this timeslice"
    )
    rate: float = Field(
        description="Rate of counter value increase per second during this timeslice"
    )


class GaugeTimeslice(BaseTimeslice):
    """Single gauge timeslice in a windowed time series."""

    avg: float = Field(description="Average value during this timeslice")
    min: float = Field(description="Minimum value during this timeslice")
    max: float = Field(description="Maximum value during this timeslice")


class HistogramTimeslice(BaseTimeslice):
    """Single histogram timeslice in a windowed time series."""

    count: int = Field(
        description="Change in count (count_delta) during this timeslice"
    )
    sum: float = Field(description="Change in sum (sum_delta) during this timeslice")
    avg: float = Field(
        description="Average value during this timeslice (sum_delta / count_delta)"
    )
    buckets: dict[str, int] | None = Field(
        default=None,
        description="Histogram bucket upper bounds to delta counts during this timeslice",
    )


class ServerMetricsEndpointInfo(AIPerfBaseModel):
    """Metadata about a single endpoint's collection statistics."""

    # Fetch statistics (all HTTP requests, including duplicates)
    total_fetches: int = Field(
        description="Total number of HTTP fetches from this endpoint"
    )
    first_fetch_ns: int = Field(description="Timestamp of first fetch in nanoseconds")
    last_fetch_ns: int = Field(description="Timestamp of last fetch in nanoseconds")
    avg_fetch_latency_ms: float = Field(
        description="Average time to fetch metrics from this endpoint in milliseconds"
    )
    # Unique update statistics (only when metrics changed)
    unique_updates: int = Field(
        description="Number of fetches that returned changed metrics"
    )
    first_update_ns: int = Field(
        description="Timestamp of first unique update in nanoseconds"
    )
    last_update_ns: int = Field(
        description="Timestamp of last unique update in nanoseconds"
    )
    duration_seconds: float = Field(
        description="Time span from first to last unique update in seconds"
    )
    avg_update_interval_ms: float = Field(
        description="Average time between unique metric updates in milliseconds"
    )
    median_update_interval_ms: float | None = Field(
        default=None,
        description="Median time between unique metric updates in milliseconds. "
        "More robust to outliers than average. None if fewer than 2 intervals.",
    )


class ServerMetricsSummary(AIPerfBaseModel):
    """Summary information for server metrics collection."""

    endpoints_configured: list[str] = Field(
        description="List of configured endpoint identifiers (normalized)"
    )
    endpoints_successful: list[str] = Field(
        description="List of successful endpoint identifiers (normalized)"
    )
    start_time: datetime
    end_time: datetime
    endpoint_info: dict[str, ServerMetricsEndpointInfo] | None = Field(
        default=None,
        description="Per-endpoint collection metadata keyed by normalized endpoint identifier",
    )


# =============================================================================
# Server Metrics Export Data (keyed metrics + flat stats)
# =============================================================================


class BaseSeries(AIPerfBaseModel):
    """Base series."""

    # Note: Optional during computation, filled in for export
    endpoint_url: str | None = Field(
        default=None,
        description="Full endpoint URL (e.g., 'http://localhost:8081/metrics')",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Metric labels. None/missing if the metric has no labels.",
    )


class GaugeStats(AIPerfBaseModel):
    """Server gauge statistics."""

    avg: float | None = Field(default=None, description="Average value")
    min: float | None = Field(default=None, description="Minimum value")
    max: float | None = Field(default=None, description="Maximum value")
    std: float | None = Field(default=None, description="Standard deviation")
    p1: float | None = Field(default=None, description="1st percentile")
    p5: float | None = Field(default=None, description="5th percentile")
    p10: float | None = Field(default=None, description="10th percentile")
    p25: float | None = Field(default=None, description="25th percentile")
    p50: float | None = Field(default=None, description="50th percentile (median)")
    p75: float | None = Field(default=None, description="75th percentile")
    p90: float | None = Field(default=None, description="90th percentile")
    p95: float | None = Field(default=None, description="95th percentile")
    p99: float | None = Field(default=None, description="99th percentile")


class GaugeSeries(BaseSeries):
    """Server gauge series."""

    stats: GaugeStats | None = Field(default=None, description="Gauge statistics")
    timeslices: list[GaugeTimeslice] | None = Field(
        default=None,
        description="Statistics per timeslice",
    )


class CounterStats(AIPerfBaseModel):
    """Server counter statistics."""

    total: float | None = Field(
        default=None,
        description="Total increase in counter value over collection period.",
    )
    rate: float | None = Field(
        default=None,
        description="Overall rate of counter value increase per second.",
    )
    rate_avg: float | None = Field(
        default=None,
        description="Time-weighted average rate between change points (counter)",
    )
    rate_min: float | None = Field(
        default=None, description="Minimum point-to-point rate per second (counter)"
    )
    rate_max: float | None = Field(
        default=None, description="Maximum point-to-point rate per second (counter)"
    )
    rate_std: float | None = Field(
        default=None, description="Standard deviation of point-to-point rates (counter)"
    )


class CounterSeries(BaseSeries):
    """Server counter series."""

    stats: CounterStats | None = Field(
        default=None,
        description="Counter statistics",
    )
    timeslices: list[CounterTimeslice] | None = Field(
        default=None,
        description="Statistics per timeslice",
    )


class HistogramStats(AIPerfBaseModel):
    """Server histogram statistics."""

    count: int | None = Field(
        default=None,
        description="Total count change over collection period.",
    )
    sum: float | None = Field(
        default=None,
        description="Total sum change over collection period.",
    )
    avg: float | None = Field(
        default=None,
        description="Overall average value over collection period (sum / count)",
    )
    count_rate: float | None = Field(
        default=None,
        description="Average count change per second.",
    )
    sum_rate: float | None = Field(
        default=None,
        description="Average sum change per second.",
    )
    p1_estimate: float | None = Field(
        default=None, description="Estimated 1st percentile"
    )
    p5_estimate: float | None = Field(
        default=None, description="Estimated 5th percentile"
    )
    p10_estimate: float | None = Field(
        default=None, description="Estimated 10th percentile"
    )
    p25_estimate: float | None = Field(
        default=None, description="Estimated 25th percentile"
    )
    p50_estimate: float | None = Field(
        default=None, description="Estimated 50th percentile (median)"
    )
    p75_estimate: float | None = Field(
        default=None, description="Estimated 75th percentile"
    )
    p90_estimate: float | None = Field(
        default=None, description="Estimated 90th percentile"
    )
    p95_estimate: float | None = Field(
        default=None, description="Estimated 95th percentile"
    )
    p99_estimate: float | None = Field(
        default=None, description="Estimated 99th percentile"
    )


class HistogramSeries(BaseSeries):
    """Server histogram series."""

    stats: HistogramStats | None = Field(
        default=None,
        description="Histogram statistics",
    )
    buckets: dict[str, int] | None = Field(
        default=None,
        description="Histogram bucket upper bounds to delta counts during collection period (e.g., {'0.1': 2000, '+Inf': 5000})",
    )
    timeslices: list[HistogramTimeslice] | None = Field(
        default=None,
        description="Statistics per timeslice",
    )


class BaseServerMetricData(AIPerfBaseModel):
    """Base metric data with type, description, unit, and base series stats.

    Used in hybrid export format where metrics are keyed by name for O(1) lookup,
    but stats within each series are flattened for easy access.
    """

    discriminator_field: ClassVar[str] = "type"

    type: PrometheusMetricType = Field(description="Metric type")

    description: str = Field(description="Metric description from HELP text")
    unit: str | None = Field(
        default=None,
        description="Unit inferred from metric name suffix (_seconds, _bytes, etc.)",
    )


class GaugeMetricData(BaseServerMetricData):
    """Server gauge metric data."""

    type: PrometheusMetricType = PrometheusMetricType.GAUGE

    series: list[GaugeSeries] = Field(
        default_factory=list,
        description="Statistics for each unique endpoint + label combination",
    )


class CounterMetricData(BaseServerMetricData):
    """Server counter metric data."""

    type: PrometheusMetricType = PrometheusMetricType.COUNTER

    series: list[CounterSeries] = Field(
        default_factory=list,
        description="Statistics for each unique endpoint + label combination",
    )


class HistogramMetricData(BaseServerMetricData):
    """Server histogram metric data."""

    type: PrometheusMetricType = PrometheusMetricType.HISTOGRAM

    series: list[HistogramSeries] = Field(
        default_factory=list,
        description="Statistics for each unique endpoint + label combination",
    )


class ServerMetricsExportData(AIPerfBaseModel):
    """Server metrics in hybrid format: keyed metrics with flat stats.

    Provides O(1) metric lookup by name while keeping stats flat within each series.
    Best of both worlds: easy to find specific metrics AND easy to access their stats.

    Example access:
        data["metrics"]["vllm:kv_cache_usage_perc"]["series"][0]["stats"]["p99"]
    """

    # Increment on breaking changes to the export structure
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for this export format.",
    )
    aiperf_version: str | None = Field(
        default=None,
        description="AIPerf version that generated this export. None for legacy exports.",
    )
    benchmark_id: str | None = Field(
        default=None,
        description="Unique identifier for this benchmark run (UUID), shared across all export formats. "
        "None for legacy exports.",
    )
    summary: ServerMetricsSummary
    metrics: SerializeAsAny[
        dict[
            str,
            GaugeMetricData | CounterMetricData | HistogramMetricData,
        ]
    ] = Field(
        default_factory=dict,
        description="Metrics keyed by name, each with type-specific series stats",
    )
    input_config: dict = Field(
        default_factory=dict,
        description="User configuration that was used for this profiling run (exclude_unset=True)",
    )


class ServerMetricsEndpointSummary(AIPerfBaseModel):
    """Summary of server metrics data for a single endpoint.

    Unified structure combining metadata and type-specific aggregated statistics:
    - Each metric uses stats matching its semantic type (gauge, counter, histogram)
    - Mirrors JSONL structure with labels as proper objects
    - Includes metric description from metadata
    """

    endpoint_url: str
    info: ServerMetricsEndpointInfo = Field(
        description="Collection statistics for this endpoint"
    )
    metrics: SerializeAsAny[
        dict[
            str,
            GaugeMetricData | CounterMetricData | HistogramMetricData,
        ]
    ] = Field(
        default_factory=dict,
        description="All metrics keyed by metric name, with type-specific series stats",
    )


class ServerMetricsResults(AIPerfBaseModel):
    """Results from server metrics collection during a profile run.

    Pre-computed summaries (endpoint_summaries) are computed in the subprocess
    and sent as JSON-serializable Pydantic models.
    """

    benchmark_id: str | None = Field(
        default=None,
        description="Unique identifier for this benchmark run (UUID), shared across all export formats. "
        "None for legacy exports created before this field was added.",
    )
    endpoint_summaries: dict[str, ServerMetricsEndpointSummary] | None = Field(
        default=None,
        description="Pre-computed endpoint summaries ready for export (sent over ZMQ)",
    )
    start_ns: int = Field(
        description="Start time of server metrics collection in nanoseconds"
    )
    end_ns: int = Field(
        description="End time of server metrics collection in nanoseconds"
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of server metrics endpoint URLs in configured scope for display",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of server metrics endpoint URLs that successfully provided data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
    # Time filter for aggregation (excludes warmup)
    aggregation_time_filter: TimeRangeFilter | None = Field(
        default=None,
        description="Time filter for aggregation, excluding warmup periods",
    )


class ProcessServerMetricsResult(AIPerfBaseModel):
    """Result of server metrics processing - mirrors ProcessTelemetryResult pattern."""

    results: ServerMetricsResults | None = Field(
        default=None, description="The processed server metrics results"
    )

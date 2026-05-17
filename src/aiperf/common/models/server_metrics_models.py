# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from pydantic import Field

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.finite import FiniteFloat
from aiperf.common.models._server_metrics_export import (
    ProcessServerMetricsResult as ProcessServerMetricsResult,
)
from aiperf.common.models._server_metrics_export import (
    ServerMetricsEndpointInfo as ServerMetricsEndpointInfo,
)
from aiperf.common.models._server_metrics_export import (
    ServerMetricsEndpointSummary as ServerMetricsEndpointSummary,
)
from aiperf.common.models._server_metrics_export import (
    ServerMetricsExportData as ServerMetricsExportData,
)
from aiperf.common.models._server_metrics_export import (
    ServerMetricsResults as ServerMetricsResults,
)
from aiperf.common.models._server_metrics_export import (
    ServerMetricsSummary as ServerMetricsSummary,
)
from aiperf.common.models._server_metrics_records import MetricFamily as MetricFamily
from aiperf.common.models._server_metrics_records import MetricSample as MetricSample
from aiperf.common.models._server_metrics_records import (
    ServerMetricsRecord as ServerMetricsRecord,
)
from aiperf.common.models._server_metrics_records import SlimRecord as SlimRecord
from aiperf.common.models._server_metrics_records import (
    TimeRangeFilter as TimeRangeFilter,
)
from aiperf.common.models.base_models import AIPerfBaseModel

# =============================================================================
# Data Models (Prometheus metrics records and metadata)
# =============================================================================


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
    rate_avg: FiniteFloat | None = Field(
        default=None,
        description="Time-weighted average rate between change points (counter)",
    )
    rate_min: FiniteFloat | None = Field(
        default=None, description="Minimum point-to-point rate per second (counter)"
    )
    rate_max: FiniteFloat | None = Field(
        default=None, description="Maximum point-to-point rate per second (counter)"
    )
    rate_std: FiniteFloat | None = Field(
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


class UnknownMetricData(BaseServerMetricData):
    """Server metric data for Prometheus `untyped` / UNKNOWN families.

    Prometheus `# TYPE foo untyped` declarations carry scalar samples with the
    same wire shape as a gauge (single `value`, no buckets/sum/count), but the
    exporter is explicitly declining to commit to gauge or counter semantics.
    AIPerf treats them as gauge-equivalent for storage and statistics, but
    preserves the original `unknown` tag in the export so downstream consumers
    can tell a real `gauge` apart from an exporter-untyped scalar.
    """

    type: PrometheusMetricType = PrometheusMetricType.UNKNOWN

    series: list[GaugeSeries] = Field(
        default_factory=list,
        description="Statistics for each unique endpoint + label combination",
    )

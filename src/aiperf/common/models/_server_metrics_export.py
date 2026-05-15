# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from pydantic import ConfigDict, Field

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models._server_metrics_records import TimeRangeFilter
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount

# =============================================================================
# Server Metrics Export Data
# =============================================================================


@dataclass(slots=True, kw_only=True)
class ServerTimeslice:
    """Single timeslice in a windowed time series.

    Unified dataclass replacing ``CounterTimeslice``/``GaugeTimeslice``/
    ``HistogramTimeslice``. msgspec rejects unions of multiple dataclasses
    when decoding, so the three legacy shapes collapse into one with
    type-specific fields as optional. Callers select fields based on the
    parent ``ServerMetricData.type``:

    - COUNTER: ``total``, ``rate``
    - GAUGE:   ``avg``, ``min``, ``max``
    - HISTOGRAM: ``count``, ``sum``, ``avg``, ``buckets``
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    start_ns: int
    end_ns: int
    is_complete: bool | None = None
    # Counter fields
    total: float | None = None
    rate: float | None = None
    # Gauge + Histogram fields (avg is shared)
    avg: float | None = None
    min: float | None = None
    max: float | None = None
    # Histogram fields
    count: int | None = None
    sum: float | None = None
    buckets: dict[str, int] | None = None


def CounterTimeslice(
    *,
    start_ns: int,
    end_ns: int,
    total: float,
    rate: float,
    is_complete: bool | None = None,
) -> ServerTimeslice:
    """Factory — builds a COUNTER-typed ``ServerTimeslice``."""
    return ServerTimeslice(
        start_ns=start_ns,
        end_ns=end_ns,
        is_complete=is_complete,
        total=total,
        rate=rate,
    )


def GaugeTimeslice(
    *,
    start_ns: int,
    end_ns: int,
    avg: float,
    min: float,
    max: float,
    is_complete: bool | None = None,
) -> ServerTimeslice:
    """Factory — builds a GAUGE-typed ``ServerTimeslice``."""
    return ServerTimeslice(
        start_ns=start_ns,
        end_ns=end_ns,
        is_complete=is_complete,
        avg=avg,
        min=min,
        max=max,
    )


def HistogramTimeslice(
    *,
    start_ns: int,
    end_ns: int,
    count: int,
    sum: float,
    avg: float,
    buckets: dict[str, int] | None = None,
    is_complete: bool | None = None,
) -> ServerTimeslice:
    """Factory — builds a HISTOGRAM-typed ``ServerTimeslice``."""
    return ServerTimeslice(
        start_ns=start_ns,
        end_ns=end_ns,
        is_complete=is_complete,
        count=count,
        sum=sum,
        avg=avg,
        buckets=buckets,
    )


# Aliases kept as class-style identities for backward compat with isinstance checks
# and type hints. All three resolve to the same underlying dataclass.
BaseTimeslice = ServerTimeslice


@dataclass(slots=True, kw_only=True)
class ServerMetricsEndpointInfo:
    """Metadata about a single endpoint's collection statistics."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    total_fetches: int
    first_fetch_ns: int
    last_fetch_ns: int
    avg_fetch_latency_ms: float
    unique_updates: int
    first_update_ns: int
    last_update_ns: int
    duration_seconds: float
    avg_update_interval_ms: float
    median_update_interval_ms: float | None = None


@dataclass(slots=True, kw_only=True)
class ServerMetricsSummary:
    """Summary information for server metrics collection."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoints_configured: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime
    endpoint_info: dict[str, ServerMetricsEndpointInfo] | None = None


# =============================================================================
# Server Metrics Export Data (keyed metrics + flat stats)
# =============================================================================


@dataclass(slots=True, kw_only=True)
class ServerSeriesStats:
    """Unified server metric series statistics.

    Replaces ``GaugeStats``/``CounterStats``/``HistogramStats`` — msgspec
    rejects unions of multiple dataclasses when decoding. Callers select
    fields based on the parent ``ServerMetricData.type``:

    - GAUGE: ``avg``, ``min``, ``max``, ``std``, ``p1``..``p99``
    - COUNTER: ``total``, ``rate``, ``rate_avg``, ``rate_min``, ``rate_max``,
               ``rate_std``
    - HISTOGRAM: ``count``, ``sum``, ``avg``, ``count_rate``, ``sum_rate``,
                 ``p1_estimate``..``p99_estimate``
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    # Gauge
    avg: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    # Counter
    total: float | None = None
    rate: float | None = None
    rate_avg: float | None = None
    rate_min: float | None = None
    rate_max: float | None = None
    rate_std: float | None = None
    # Histogram
    count: int | None = None
    sum: float | None = None
    count_rate: float | None = None
    sum_rate: float | None = None
    p1_estimate: float | None = None
    p5_estimate: float | None = None
    p10_estimate: float | None = None
    p25_estimate: float | None = None
    p50_estimate: float | None = None
    p75_estimate: float | None = None
    p90_estimate: float | None = None
    p95_estimate: float | None = None
    p99_estimate: float | None = None


# Legacy aliases for backward compatibility — all three stats types collapse
# into one dataclass. isinstance() checks against these aliases still work
# because they're the same class.
GaugeStats = ServerSeriesStats
CounterStats = ServerSeriesStats
HistogramStats = ServerSeriesStats


@dataclass(slots=True, kw_only=True)
class ServerSeries:
    """Unified server metric series.

    Replaces ``GaugeSeries``/``CounterSeries``/``HistogramSeries`` — single
    dataclass with type-specific optional fields. Disambiguated by the
    parent ``ServerMetricData.type``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoint_url: str | None = None
    labels: dict[str, str] | None = None
    stats: ServerSeriesStats | None = None
    timeslices: list[ServerTimeslice] | None = None
    # Histogram-only: per-series bucket counts
    buckets: dict[str, int] | None = None


# Legacy aliases
BaseSeries = ServerSeries
GaugeSeries = ServerSeries
CounterSeries = ServerSeries
HistogramSeries = ServerSeries


@dataclass(slots=True, kw_only=True)
class ServerMetricData:
    """Unified server metric data for all three Prometheus types.

    Collapses ``GaugeMetricData``/``CounterMetricData``/``HistogramMetricData``
    into a single dataclass. msgspec rejects unions of multiple dataclasses
    when decoding, so we carry a single dataclass and disambiguate via the
    ``type`` field. Callers inspect ``type`` to know which series/stats
    fields are populated.

    The three legacy names (``GaugeMetricData``, ``CounterMetricData``,
    ``HistogramMetricData``) are kept as factory functions for ergonomic
    construction and backward compatibility at call sites.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: PrometheusMetricType
    description: str
    unit: str | None = None
    series: list[ServerSeries] = field(default_factory=list)


# Legacy alias — all *MetricData types are ServerMetricData.
BaseServerMetricData = ServerMetricData


def GaugeMetricData(
    *,
    description: str,
    unit: str | None = None,
    series: list[ServerSeries] | None = None,
) -> ServerMetricData:
    """Factory — builds a GAUGE-typed ``ServerMetricData``."""
    return ServerMetricData(
        type=PrometheusMetricType.GAUGE,
        description=description,
        unit=unit,
        series=list(series) if series else [],
    )


def CounterMetricData(
    *,
    description: str,
    unit: str | None = None,
    series: list[ServerSeries] | None = None,
) -> ServerMetricData:
    """Factory — builds a COUNTER-typed ``ServerMetricData``."""
    return ServerMetricData(
        type=PrometheusMetricType.COUNTER,
        description=description,
        unit=unit,
        series=list(series) if series else [],
    )


def HistogramMetricData(
    *,
    description: str,
    unit: str | None = None,
    series: list[ServerSeries] | None = None,
) -> ServerMetricData:
    """Factory — builds a HISTOGRAM-typed ``ServerMetricData``."""
    return ServerMetricData(
        type=PrometheusMetricType.HISTOGRAM,
        description=description,
        unit=unit,
        series=list(series) if series else [],
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
    metrics: dict[str, ServerMetricData] = Field(
        default_factory=dict,
        description="Metrics keyed by name, each with type-specific series stats",
    )
    input_config: dict = Field(
        default_factory=dict,
        description="User configuration that was used for this profiling run (exclude_unset=True)",
    )


@dataclass(slots=True, kw_only=True)
class ServerMetricsEndpointSummary:
    """Summary of server metrics data for a single endpoint.

    Unified structure combining metadata and type-specific aggregated statistics:
    - Each metric uses stats matching its semantic type (gauge, counter, histogram)
    - Mirrors JSONL structure with labels as proper objects
    - Includes metric description from metadata
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoint_url: str
    info: ServerMetricsEndpointInfo
    metrics: dict[str, ServerMetricData] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class ServerMetricsResults:
    """Results from server metrics collection during a profile run.

    Slotted dataclass — shared between msgspec envelopes
    (``ProcessServerMetricsResultMessage.server_metrics_result.results``) and
    Pydantic parents.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    start_ns: int
    end_ns: int
    benchmark_id: str | None = None
    endpoint_summaries: dict[str, ServerMetricsEndpointSummary] | None = None
    endpoints_configured: list[str] = field(default_factory=list)
    endpoints_successful: list[str] = field(default_factory=list)
    error_summary: list[ErrorDetailsCount] = field(default_factory=list)
    aggregation_time_filter: TimeRangeFilter | None = None


@dataclass(slots=True, kw_only=True)
class ProcessServerMetricsResult:
    """Result of server metrics processing - mirrors ProcessTelemetryResult pattern.

    Slotted dataclass — shared between msgspec envelopes
    (``ProcessServerMetricsResultMessage.server_metrics_result``) and Pydantic
    parents via ``__pydantic_config__``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    results: ServerMetricsResults | None = None

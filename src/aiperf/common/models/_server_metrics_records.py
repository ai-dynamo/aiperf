# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import msgspec
from pydantic_core import InitErrorDetails, ValidationError

from aiperf.common.enums import PrometheusMetricType


def _raise_metric_sample_error(message: str) -> None:
    """Raise a pydantic ``ValidationError`` for an invalid ``MetricSample``.

    ``MetricSample`` is a ``msgspec.Struct`` for zero-copy JSONL serialization,
    so it cannot use a pydantic ``@model_validator``. The producer hot path
    (``data_collector._process_*_family``) catches ``pydantic.ValidationError``
    to drop only the offending sample, and the project's NaN/Inf Discipline
    tests assert ``pydantic.ValidationError`` carrying a "finite" message. A
    real ``pydantic_core.ValidationError`` satisfies both: it subclasses
    ``ValueError`` (so the mutual-exclusivity ``pytest.raises(ValueError)``
    assertions still match) while remaining the exact pydantic type the
    producer and finite-contract tests expect.

    Raises:
        ValidationError: always, wrapping ``message``.
    """
    raise ValidationError.from_exception_data(
        "MetricSample",
        [
            InitErrorDetails(
                type="value_error",
                loc=(),
                input={},
                ctx={"error": ValueError(message)},
            )
        ],
    )


@dataclass(frozen=True, slots=True)
class TimeRangeFilter:
    """Filter for selecting metrics within a specific time range.

    Immutable time range specification used throughout the metrics processing
    pipeline to exclude warmup and cooldown periods from statistics computation.

    Supports partial ranges (None on either end) for flexibility. Automatically
    validates that start < end to catch configuration errors early.

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
    """Start of valid time range in nanoseconds (inclusive). None means unbounded."""

    end_ns: int | None = None
    """End of valid time range in nanoseconds (inclusive). None means unbounded."""

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


class MetricSample(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Single metric sample from Prometheus exposition format.

    Represents one data point from a Prometheus metric scrape. Format depends
    on metric type:
    - Counter/Gauge: Uses `value` field only
    - Histogram: Uses `buckets`, `sum`, and `count` fields

    Labels provide dimensional data for grouping and filtering (e.g., HTTP method,
    status code, instance ID). Histogram labels exclude the special "le" label
    which is used for bucket boundaries instead.
    """

    labels: dict[str, str] | None = None
    """Metric labels (excluding histogram special labels). None if no labels."""

    value: float | None = None
    """Simple metric value (counter/gauge)."""

    buckets: dict[str, float] | None = None
    """Histogram bucket upper bounds (le="less than or equal") to counts.
    Keys are strings like "0.01", "0.1", "1.0"."""

    sum: float | None = None
    """Sum of all observed values (for histogram only)."""

    count: float | None = None
    """Total number of observations (for histogram only)."""

    def __post_init__(self) -> None:
        """Validate finiteness and value/histogram mutual exclusivity.

        Enforces the same contract main's pydantic ``MetricSample`` did, but
        from a ``msgspec.Struct`` (see :func:`_raise_metric_sample_error`):

        1. ``value``, ``sum``, ``count``, and every ``buckets`` count must be
           finite when present (NaN/Inf would orjson-encode to ``null`` on the
           ZMQ/JSONL hop and poison downstream histogram series).
        2. Exactly one of ``{value, buckets}`` is set (not both, not neither).
        3. If ``value`` is set (counter/gauge), ``sum`` and ``count`` must be
           unset (those belong only to histograms).

        Raises:
            ValidationError: If any rule above is violated.
        """
        if self.value is not None and not math.isfinite(self.value):
            _raise_metric_sample_error(f"value must be finite, got {self.value!r}")
        if self.sum is not None and not math.isfinite(self.sum):
            _raise_metric_sample_error(f"sum must be finite, got {self.sum!r}")
        if self.count is not None and not math.isfinite(self.count):
            _raise_metric_sample_error(f"count must be finite, got {self.count!r}")
        if self.buckets is not None:
            for bound, bucket_count in self.buckets.items():
                if not math.isfinite(bucket_count):
                    _raise_metric_sample_error(
                        f"bucket {bound!r} count must be finite, got {bucket_count!r}"
                    )

        if self.value is not None and self.buckets is not None:
            _raise_metric_sample_error("Only one of value or buckets can be set")
        if self.value is None and self.buckets is None:
            _raise_metric_sample_error("One of value or buckets must be set")
        if self.value is not None and (self.sum is not None or self.count is not None):
            _raise_metric_sample_error("If value is set, sum and count must not be set")


class MetricFamily(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Group of related metrics with same name and type from Prometheus.

    Represents a complete metric family from Prometheus exposition format
    (all samples under one TYPE and HELP declaration). Contains metadata
    (type, description) and all samples with their label dimensions.

    For multi-dimensional metrics, samples list contains one entry per unique
    label combination. For histograms, each sample contains all buckets for
    that label set.
    """

    type: PrometheusMetricType
    """Metric type as enum."""

    description: str
    """Metric description from HELP text."""

    samples: list[MetricSample]
    """Metric samples grouped by base labels."""


class SlimRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Slim server metrics record containing only time-varying data.

    This record excludes static metadata (metric types, help text)
    to reduce JSONL file size. Includes HTTP trace timing fields for
    precise correlation with client request timestamps.
    """

    endpoint_url: str
    """Source Prometheus metrics endpoint URL (e.g., 'http://localhost:8081/metrics')."""

    timestamp_ns: int
    """Nanosecond wall-clock timestamp representing when server generated metrics."""

    metrics: dict[str, list[MetricSample]]
    """Metrics grouped by family name, mapping directly to metric sample list."""

    endpoint_latency_ns: int | None = None
    """Nanoseconds for total HTTP round-trip (request start to completion)."""

    request_sent_ns: int | None = None
    """Wall-clock timestamp in nanoseconds when HTTP request was initiated."""

    first_byte_ns: int | None = None
    """Wall-clock timestamp in nanoseconds when first response byte received from server."""


class ServerMetricsRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Single server metrics data point from Prometheus endpoint.

    This record contains all metrics scraped from one Prometheus endpoint at one point in time.
    Used for hierarchical storage: endpoint_url -> time series data.

    The trace timing fields provide precise correlation between server metrics and client
    requests by capturing when the server actually generated the metrics, not just when
    the client received the full response.
    """

    endpoint_url: str
    """Source Prometheus metrics endpoint URL."""

    timestamp_ns: int
    """Nanosecond wall-clock timestamp representing when server generated metrics.
    Uses first_byte_ns if available (most accurate), otherwise falls back to
    time after request completes."""

    metrics: dict[str, MetricFamily]
    """Metrics grouped by family name."""

    endpoint_latency_ns: int | None = None
    """Nanoseconds for total HTTP round-trip (request start to completion)."""

    request_sent_ns: int | None = None
    """Wall-clock timestamp in nanoseconds when HTTP request was initiated
    (from aiohttp trace)."""

    first_byte_ns: int | None = None
    """Wall-clock timestamp in nanoseconds when first response byte received from server.
    Best approximation of when server generated the metrics."""

    is_duplicate: bool = False
    """True if this record's metrics are identical to the previous fetch from this endpoint."""

    def to_slim(self) -> SlimRecord:
        """Convert to slim record.

        Excludes metrics ending in _info as they are typically used for metadata and not metrics,
        so they will be include in the final export, but not in the JSONL records.
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

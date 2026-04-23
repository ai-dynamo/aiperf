# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

from typing_extensions import Self

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models import MetricFamily
from aiperf.server_metrics._storage_histogram import HistogramTimeSeries
from aiperf.server_metrics._storage_scalar import ScalarTimeSeries


class ServerMetricKey(NamedTuple):
    """Structured key for metric identification with labels.

    Immutable, hashable key for uniquely identifying a metric time series.
    Combines metric name with label dimensions to distinguish between series
    (e.g., http_requests_total with method=GET vs method=POST are different series).

    Uses immutable tuple of tuples for labels to be hashable as dict key.
    Labels are stored as sorted (key, value) pairs for consistent ordering,
    ensuring that the same labels in different order produce identical keys.

    This enables efficient dict-based storage and O(1) metric lookup by name+labels.

    Args:
        name: Prometheus metric name (e.g., "http_requests_total", "cache_hit_ratio")
        labels: Sorted tuple of (key, value) label pairs for metric dimensions

    Examples:
        >>> # Metric with no labels
        >>> key1 = ServerMetricKey("vllm:kv_cache_usage_perc", ())

        >>> # Metric with labels
        >>> key2 = ServerMetricKey("http_requests_total", (("method", "GET"), ("status", "200")))

        >>> # Create from dict (labels get sorted automatically)
        >>> key3 = ServerMetricKey.from_name_and_labels(
        ...     "http_requests_total",
        ...     {"status": "200", "method": "GET"}  # Order doesn't matter
        ... )
        >>> key3.labels
        (('method', 'GET'), ('status', '200'))  # Sorted by key
    """

    name: str
    labels: tuple[tuple[str, str], ...] = ()

    @property
    def labels_dict(self) -> dict[str, str] | None:
        """Convert labels tuple to dict for easy access.

        Returns:
            Dict mapping label keys to values, or None if no labels.
            Useful for passing to statistics models and export functions.
        """
        return dict(self.labels) if self.labels else None

    @classmethod
    def from_name_and_labels(cls, name: str, labels: dict[str, str] | None) -> Self:
        """Create ServerMetricKey from metric name and optional labels dict.

        Convenience constructor that handles dict-to-tuple conversion and sorting.
        Ensures consistent key generation regardless of dict iteration order.

        Args:
            name: Prometheus metric name
            labels: Optional dict of label key-value pairs

        Returns:
            ServerMetricKey with labels sorted by key for consistent hashing
        """
        if not labels:
            return cls(name, ())
        sorted_labels = tuple(sorted(labels.items()))
        return cls(name, sorted_labels)


@dataclass(slots=True)
class ServerMetricEntry:
    """Unified container for server metric type, description, and time series data.

    Self-describing storage for a single metric time series. Combines metadata
    (type, description) with the actual time series data in one structure,
    eliminating the need for separate metadata lookups.

    This design enables:
    - Type-appropriate statistics computation without external type info
    - Description propagation through the processing pipeline
    - Polymorphic storage (ScalarTimeSeries or HistogramTimeSeries based on type)
    """

    metric_type: PrometheusMetricType
    """Prometheus metric type (GAUGE, COUNTER, or HISTOGRAM)."""

    description: str
    """Human-readable description from Prometheus HELP text."""

    data: ScalarTimeSeries | HistogramTimeSeries
    """Type-appropriate time series storage."""

    @classmethod
    def from_metric_family(cls, metric_family: MetricFamily) -> Self:
        """Create a ServerMetricEntry from a MetricFamily.

        Factory method that automatically selects the appropriate time series
        storage type based on the metric type. Gauges and counters use
        ScalarTimeSeries, histograms use HistogramTimeSeries.

        Args:
            metric_family: MetricFamily from parsed Prometheus metrics containing
                          type, description, and initial samples

        Returns:
            ServerMetricEntry with appropriate storage initialized
        """
        return cls(
            metric_type=metric_family.type,
            description=metric_family.description,
            data=ScalarTimeSeries()
            if metric_family.type
            in (PrometheusMetricType.GAUGE, PrometheusMetricType.COUNTER)
            else HistogramTimeSeries(),
        )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for server metrics export stats tests."""

import numpy as np

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.server_metrics.storage import (
    ServerMetricKey,
    ServerMetricsTimeSeries,
)


def add_gauge_samples(
    ts: ServerMetricsTimeSeries,
    name: str,
    values: list[float],
    start_ns: int = 0,
    interval_ns: int = NANOS_PER_SECOND,
) -> None:
    """Add gauge samples at regular intervals."""
    for i, value in enumerate(values):
        record = ServerMetricsRecord(
            endpoint_url="test://",
            timestamp_ns=start_ns + i * interval_ns,
            endpoint_latency_ns=0,
            metrics={
                name: MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="",
                    samples=[MetricSample(value=value)],
                )
            },
        )
        ts.append_snapshot(record)


def add_counter_samples(
    ts: ServerMetricsTimeSeries,
    name: str,
    values: list[float],
    start_ns: int = 0,
    interval_ns: int = NANOS_PER_SECOND,
) -> None:
    """Add counter samples at regular intervals."""
    for i, value in enumerate(values):
        record = ServerMetricsRecord(
            endpoint_url="test://",
            timestamp_ns=start_ns + i * interval_ns,
            endpoint_latency_ns=0,
            metrics={
                name: MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="",
                    samples=[MetricSample(value=value)],
                )
            },
        )
        ts.append_snapshot(record)


class HistogramSnapshot:
    """Simple container for histogram snapshot data used in tests."""

    __slots__ = ("buckets", "sum", "count")

    def __init__(
        self,
        buckets: dict[str, float],
        sum: float,
        count: float,  # noqa: A002
    ) -> None:
        self.buckets = buckets
        self.sum = sum
        self.count = count


def add_histogram_snapshots(
    ts: ServerMetricsTimeSeries,
    name: str,
    snapshots: list[tuple[int, HistogramSnapshot]],
) -> None:
    """Add histogram snapshots at specified timestamps."""
    for timestamp_ns, histogram in snapshots:
        record = ServerMetricsRecord(
            endpoint_url="test://",
            timestamp_ns=timestamp_ns,
            endpoint_latency_ns=0,
            metrics={
                name: MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    description="",
                    samples=[
                        MetricSample(
                            buckets=histogram.buckets,
                            sum=histogram.sum,
                            count=histogram.count,
                        )
                    ],
                )
            },
        )
        ts.append_snapshot(record)


def hist(buckets: dict[str, float], sum_: float, count: float) -> HistogramSnapshot:
    """Shorthand for creating histogram snapshot data for tests."""
    return HistogramSnapshot(buckets=buckets, sum=sum_, count=count)


def add_gauge_samples_with_timestamps(
    ts: ServerMetricsTimeSeries,
    name: str,
    samples: list[tuple[int, float]],  # (timestamp_ns, value)
) -> None:
    """Add gauge samples at explicit timestamps (useful for out-of-order tests)."""
    for timestamp_ns, value in samples:
        record = ServerMetricsRecord(
            endpoint_url="test://",
            timestamp_ns=timestamp_ns,
            endpoint_latency_ns=0,
            metrics={
                name: MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="",
                    samples=[MetricSample(value=value)],
                )
            },
        )
        ts.append_snapshot(record)


def add_counter_samples_with_timestamps(
    ts: ServerMetricsTimeSeries,
    name: str,
    samples: list[tuple[int, float]],  # (timestamp_ns, value)
) -> None:
    """Add counter samples at explicit timestamps (useful for out-of-order tests)."""
    for timestamp_ns, value in samples:
        record = ServerMetricsRecord(
            endpoint_url="test://",
            timestamp_ns=timestamp_ns,
            endpoint_latency_ns=0,
            metrics={
                name: MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="",
                    samples=[MetricSample(value=value)],
                )
            },
        )
        ts.append_snapshot(record)


def get_gauge(ts: ServerMetricsTimeSeries, name: str):
    """Get gauge data from unified metrics dict."""
    key = ServerMetricKey(name, ())
    return ts.metrics[key].data


def get_counter(ts: ServerMetricsTimeSeries, name: str):
    """Get counter data from unified metrics dict."""
    key = ServerMetricKey(name, ())
    return ts.metrics[key].data


def get_histogram(ts: ServerMetricsTimeSeries, name: str):
    """Get histogram data from unified metrics dict."""
    key = ServerMetricKey(name, ())
    return ts.metrics[key].data


def bucket_sort_key(le: str) -> float:
    """Sort key for bucket boundaries: '+Inf' sorts last."""
    return float("inf") if le == "+Inf" else float(le)


def convert_bucket_snapshots(
    bucket_snapshots: list[dict[str, float]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Convert list of bucket dicts to (bucket_les, bucket_counts) format.

    Args:
        bucket_snapshots: List of dicts mapping bucket boundaries to counts

    Returns:
        Tuple of (bucket_les, bucket_counts) where:
        - bucket_les: Sorted tuple of bucket boundary strings
        - bucket_counts: 2D array of shape (n_snapshots, n_buckets)
    """
    if not bucket_snapshots:
        return (), np.empty((0, 0), dtype=np.float64)

    # Get all bucket boundaries from first snapshot
    bucket_les = tuple(sorted(bucket_snapshots[0].keys(), key=bucket_sort_key))
    n_buckets = len(bucket_les)
    n_snapshots = len(bucket_snapshots)

    # Build 2D array
    bucket_counts = np.zeros((n_snapshots, n_buckets), dtype=np.float64)
    for i, snapshot in enumerate(bucket_snapshots):
        for j, le in enumerate(bucket_les):
            bucket_counts[i, j] = snapshot.get(le, 0.0)

    return bucket_les, bucket_counts


def make_time_filter(
    start_ns: int = 0,
    end_ns: int = 1000 * NANOS_PER_SECOND,
) -> "TimeRangeFilter":  # noqa: F821
    """Create a TimeRangeFilter for tests.

    Args:
        start_ns: Start time in nanoseconds (default: 0)
        end_ns: End time in nanoseconds (default: 1000s)

    Returns:
        TimeRangeFilter covering the specified range
    """
    from aiperf.common.models.server_metrics_models import TimeRangeFilter

    return TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)

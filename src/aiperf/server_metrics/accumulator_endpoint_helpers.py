# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for ServerMetricsAccumulator endpoint summary computation.

Extracted from ``accumulator.py`` so that module stays under the file-size
ergonomics threshold; kept private (leading underscore) since they're only
called via the static-method delegators on the accumulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    CounterMetricData,
    GaugeMetricData,
    HistogramMetricData,
    ServerMetricsEndpointInfo,
    TimeRangeFilter,
)
from aiperf.server_metrics.export_stats import compute_stats


def _build_endpoint_metrics(
    time_series: Any,
    time_filter: TimeRangeFilter,
    slice_duration: float | None,
    fast_histogram_percentiles: bool,
) -> dict[str, GaugeMetricData | CounterMetricData | HistogramMetricData]:
    """Compute per-metric stats for a single endpoint's time series."""
    metrics: dict[
        str,
        GaugeMetricData | CounterMetricData | HistogramMetricData,
    ] = {}

    for metric_key, metric_entry in time_series.metrics.items():
        base_name = metric_key.name
        series_stats = compute_stats(
            metric_entry.metric_type,
            metric_entry.data,
            time_filter=time_filter,
            labels=metric_key.labels_dict,
            slice_duration=slice_duration,
            fast_histogram_percentiles=fast_histogram_percentiles,
        )
        if series_stats is None:
            continue
        if base_name not in metrics:
            match metric_entry.metric_type:
                case PrometheusMetricType.GAUGE:
                    metrics[base_name] = GaugeMetricData(
                        description=metric_entry.description,
                        series=[series_stats],
                    )
                case PrometheusMetricType.COUNTER:
                    metrics[base_name] = CounterMetricData(
                        description=metric_entry.description,
                        series=[series_stats],
                    )
                case PrometheusMetricType.HISTOGRAM:
                    metrics[base_name] = HistogramMetricData(
                        description=metric_entry.description,
                        series=[series_stats],
                    )
        else:
            metrics[base_name].series.append(series_stats)
    return metrics


def _build_endpoint_info(time_series: Any) -> ServerMetricsEndpointInfo:
    """Compute fetch/update collection metadata for a single endpoint."""
    unique_count = time_series._unique_update_count
    duration_seconds = (
        (time_series.last_update_ns - time_series.first_update_ns) / NANOS_PER_SECOND
        if unique_count > 0
        else 0.0
    )
    avg_update_interval_ms = (
        (duration_seconds * MILLIS_PER_SECOND) / (unique_count - 1)
        if unique_count > 1
        else 0.0
    )
    median_update_interval_ms: float | None = None
    if time_series._update_intervals_ns:
        intervals_ns = np.array(time_series._update_intervals_ns, dtype=np.int64)
        median_update_interval_ms = float(np.median(intervals_ns)) / NANOS_PER_MILLIS

    avg_fetch_latency_ms = (
        sum(time_series._fetch_latencies_ns)
        / len(time_series._fetch_latencies_ns)
        / NANOS_PER_MILLIS
        if time_series._fetch_latencies_ns
        else 0.0
    )

    return ServerMetricsEndpointInfo(
        total_fetches=time_series._total_fetch_count,
        first_fetch_ns=time_series.first_fetch_ns,
        last_fetch_ns=time_series.last_fetch_ns,
        avg_fetch_latency_ms=avg_fetch_latency_ms,
        unique_updates=unique_count,
        first_update_ns=time_series.first_update_ns,
        last_update_ns=time_series.last_update_ns,
        duration_seconds=duration_seconds,
        avg_update_interval_ms=avg_update_interval_ms,
        median_update_interval_ms=median_update_interval_ms,
    )

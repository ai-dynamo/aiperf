# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export statistics computation for server metrics.

This module provides functions to compute statistics from time series data
into type-specific series models (GaugeSeries, CounterSeries, HistogramSeries).
"""

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    CounterSeries,
    GaugeSeries,
    HistogramSeries,
    TimeRangeFilter,
)
from aiperf.server_metrics._counter_stats import (
    _compute_counter_stats,
    _compute_counter_timeslices,
)
from aiperf.server_metrics._gauge_stats import (
    _compute_gauge_stats,
    _compute_gauge_timeslices,
)
from aiperf.server_metrics._histogram_stats import (
    _compute_histogram_stats,
    _compute_histogram_timeslices,
)
from aiperf.server_metrics._timeslice_boundaries import _compute_timeslice_boundaries
from aiperf.server_metrics.storage import HistogramTimeSeries, ScalarTimeSeries

__all__ = [
    "_compute_counter_stats",
    "_compute_counter_timeslices",
    "_compute_gauge_stats",
    "_compute_gauge_timeslices",
    "_compute_histogram_stats",
    "_compute_histogram_timeslices",
    "_compute_timeslice_boundaries",
    "compute_stats",
]


def compute_stats(
    metric_type: PrometheusMetricType,
    time_series: ScalarTimeSeries | HistogramTimeSeries,
    *,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
    fast_histogram_percentiles: bool = False,
) -> GaugeSeries | CounterSeries | HistogramSeries | None:
    """Compute statistics from a time series based on metric type.

    Routes to type-specific computation functions (gauge, counter, histogram)
    and returns appropriate statistics model. Supports time filtering to exclude
    warmup periods and optional timeslice-based analysis.

    Args:
        metric_type: The type of metric to compute statistics for (GAUGE, COUNTER, or HISTOGRAM)
        time_series: The time series to compute statistics from (ScalarTimeSeries or HistogramTimeSeries)
        time_filter: Optional time range filter to exclude warmup/cooldown periods.
                     Uses reference point (last sample before start_ns) for counter/histogram deltas.
        labels: Optional labels to attach to the output statistics (e.g., {"method": "GET", "status": "200"})
        slice_duration: Duration of each timeslice in seconds. If None, timeslices are not computed.
                        Timeslices provide time-series analysis of how metrics vary over the profiling period.
        fast_histogram_percentiles: Algorithm selection for histogram percentile estimation.
                                    True = Prometheus linear interpolation (~15-40% error, instant).
                                    False = Polynomial histogram with learned means (~5% error, slower).

    Returns:
        Type-specific series statistics (GaugeSeries, CounterSeries, or HistogramSeries) with:
        - Gauge: avg, min, max, std, percentiles
        - Counter: total delta, rate, rate statistics from timeslices
        - Histogram: count, sum, rates, estimated percentiles from buckets
        Returns None if no data in time range.

    Example:
        >>> from aiperf.server_metrics.storage import ScalarTimeSeries
        >>> from aiperf.common.models import MetricSample
        >>> # Create gauge time series
        >>> ts = ScalarTimeSeries()
        >>> ts.append(1000000000, MetricSample(value=42.5))
        >>> ts.append(2000000000, MetricSample(value=43.1))
        >>> ts.append(3000000000, MetricSample(value=41.8))
        >>> # Compute statistics
        >>> stats = compute_stats(
        ...     PrometheusMetricType.GAUGE,
        ...     ts,
        ...     labels={"instance": "server-1"}
        ... )
        >>> print(stats.stats.avg)  # Average across all samples
        42.47
    """
    match metric_type:
        case PrometheusMetricType.GAUGE:
            return _compute_gauge_stats(
                time_series,
                time_filter,
                labels,
                slice_duration,
            )
        case PrometheusMetricType.COUNTER:
            return _compute_counter_stats(
                time_series,
                time_filter,
                labels,
                slice_duration,
            )
        case PrometheusMetricType.HISTOGRAM:
            return _compute_histogram_stats(
                time_series,
                time_filter,
                labels,
                slice_duration,
                fast_histogram_percentiles=fast_histogram_percentiles,
            )
        case _:
            raise ValueError(f"Unsupported metric type: {metric_type}")

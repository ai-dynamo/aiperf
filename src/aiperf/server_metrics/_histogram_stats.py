# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram statistics computation for server metrics export."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.server_metrics_models import (
    HistogramSeries,
    HistogramStats,
    HistogramTimeslice,
    TimeRangeFilter,
)
from aiperf.server_metrics._timeslice_boundaries import _compute_timeslice_boundaries
from aiperf.server_metrics.histogram_percentiles import (
    accumulate_bucket_statistics,
    compute_estimated_percentiles,
    compute_prometheus_percentiles,
)
from aiperf.server_metrics.storage import HistogramTimeSeries

_logger = AIPerfLogger(__name__)


def _histogram_timeslice_bucket_deltas(
    bucket_les: list[str],
    bucket_counts: np.ndarray,
    boundary_start_idx: int,
    boundary_end_idx: int,
) -> dict[str, int] | None:
    """Compute bucket deltas for a single histogram timeslice.

    Returns None if a counter reset is detected (any negative bucket delta).
    """
    if len(bucket_les) == 0 or len(bucket_counts) == 0:
        return None

    start_buckets = bucket_counts[boundary_start_idx]
    end_buckets = bucket_counts[boundary_end_idx]
    bucket_deltas: dict[str, int] = {}
    for i, le in enumerate(bucket_les):
        delta = end_buckets[i] - start_buckets[i]
        if delta < 0:
            return None
        bucket_deltas[le] = int(delta)
    return bucket_deltas


def _build_histogram_timeslice(
    *,
    timestamps: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    bucket_les: list[str],
    bucket_counts: np.ndarray,
    timeslice_start: int,
    timeslice_end: int,
    complete: bool,
) -> HistogramTimeslice | None:
    """Build one HistogramTimeslice. Returns None on counter reset or empty slice."""
    boundary_start_idx = np.searchsorted(timestamps, timeslice_start, side="right") - 1
    boundary_end_idx = np.searchsorted(timestamps, timeslice_end, side="right") - 1

    boundary_start_idx = max(0, min(boundary_start_idx, len(timestamps) - 1))
    boundary_end_idx = max(0, min(boundary_end_idx, len(timestamps) - 1))

    sum_delta = sums[boundary_end_idx] - sums[boundary_start_idx]
    count_delta = counts[boundary_end_idx] - counts[boundary_start_idx]

    if sum_delta < 0 or count_delta < 0:
        return None

    avg_value = sum_delta / count_delta if count_delta > 0 else 0.0

    bucket_deltas = _histogram_timeslice_bucket_deltas(
        bucket_les, bucket_counts, boundary_start_idx, boundary_end_idx
    )

    return HistogramTimeslice(
        start_ns=int(timeslice_start),
        end_ns=int(timeslice_end),
        count=int(count_delta),
        sum=float(sum_delta),
        avg=float(avg_value),
        buckets=bucket_deltas,
        is_complete=None if complete else False,
    )


def _compute_histogram_timeslices(
    time_series: HistogramTimeSeries,
    slice_duration: float,
    time_filter: TimeRangeFilter,
) -> list[HistogramTimeslice] | None:
    """Compute time-sliced average values for a histogram metric.

    Divides the time range into fixed-size timeslices and computes the
    average value (sum_delta / count_delta) for each timeslice. Includes
    a final partial timeslice if the range doesn't align with slice boundaries.

    Partial slices are marked with is_complete=False. They contain valid data
    but should be excluded from comparative analysis to avoid skewing results.

    Args:
        time_series: Histogram time series data
        slice_duration: Duration of each timeslice in seconds
        time_filter: Time filter defining benchmark time range (excludes warmup)

    Returns:
        List of HistogramTimeslice, one per timeslice (complete + optional partial).
        None if insufficient data.

    Raises:
        ValueError: If slice_duration <= 0
    """
    if slice_duration <= 0:
        raise ValueError("slice_duration must be positive")

    reference_idx, final_idx = time_series.get_indices_for_filter(time_filter)
    start_idx = reference_idx if reference_idx is not None else 0
    if final_idx <= start_idx:
        return None

    boundaries = _compute_timeslice_boundaries(
        time_filter.start_ns, time_filter.end_ns, slice_duration
    )
    if boundaries is None:
        return None
    timeslice_starts, timeslice_ends, is_complete = boundaries

    results: list[HistogramTimeslice] = []
    for timeslice_start, timeslice_end, complete in zip(
        timeslice_starts, timeslice_ends, is_complete, strict=True
    ):
        ts = _build_histogram_timeslice(
            timestamps=time_series.timestamps,
            sums=time_series.sums,
            counts=time_series.counts,
            bucket_les=time_series.bucket_les,
            bucket_counts=time_series.bucket_counts,
            timeslice_start=timeslice_start,
            timeslice_end=timeslice_end,
            complete=complete,
        )
        if ts is not None:
            results.append(ts)

    return results if results else None


def _histogram_reference_and_final(
    time_series: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None,
) -> tuple[int | None, int, float, float, int, float, float, int, dict[str, float]]:
    """Resolve reference and final points for histogram delta computation.

    Returns: (reference_idx, start_idx, reference_sum, reference_count, reference_timestamp,
              final_sum, final_count, final_timestamp, final_buckets)
    """
    reference_idx, final_idx = time_series.get_indices_for_filter(time_filter)

    if reference_idx is not None:
        reference_sum = float(time_series.sums[reference_idx])
        reference_count = float(time_series.counts[reference_idx])
        reference_timestamp = time_series.timestamps[reference_idx]
    else:
        reference_sum = float(time_series.sums[0])
        reference_count = float(time_series.counts[0])
        reference_timestamp = time_series.timestamps[0]

    final_sum = float(time_series.sums[final_idx])
    final_count = float(time_series.counts[final_idx])
    final_timestamp = time_series.timestamps[final_idx]
    final_buckets = (
        time_series.get_bucket_dict(final_idx) if len(time_series) > 0 else {}
    )

    start_idx = reference_idx if reference_idx is not None else 0
    return (
        reference_idx,
        start_idx,
        reference_sum,
        reference_count,
        reference_timestamp,
        final_sum,
        final_count,
        final_timestamp,
        final_buckets,
    )


def _compute_bucket_deltas(
    reference_buckets: dict[str, float],
    final_buckets: dict[str, float],
) -> dict[str, int] | None:
    """Compute bucket deltas; return None on counter reset."""
    bucket_deltas: dict[str, int] = {}
    for le_bound, final_bucket_count in final_buckets.items():
        reference_bucket_count = reference_buckets.get(le_bound, 0.0)
        bucket_delta = final_bucket_count - reference_bucket_count
        if bucket_delta < 0:
            return None
        bucket_deltas[le_bound] = int(bucket_delta)
    return bucket_deltas


def _empty_histogram_series(
    time_series: HistogramTimeSeries,
    reference_idx: int | None,
    final_buckets: dict[str, float],
    labels: dict[str, str] | None,
) -> HistogramSeries:
    """Build empty-count HistogramSeries (count=0) with bucket deltas for API consistency."""
    reference_bucket_idx = reference_idx if reference_idx is not None else 0
    reference_buckets = (
        time_series.get_bucket_dict(reference_bucket_idx)
        if len(time_series) > 0
        else {}
    )
    bucket_deltas = _compute_bucket_deltas(reference_buckets, final_buckets)
    return HistogramSeries(
        labels=labels,
        stats=HistogramStats(count=0),
        buckets=bucket_deltas,
    )


def _estimate_percentiles(
    *,
    time_series: HistogramTimeSeries,
    bucket_deltas: dict[str, int],
    count_delta: int,
    sum_delta: float,
    start_idx: int,
    fast_histogram_percentiles: bool,
):
    """Compute percentile estimates (P1-P99) from bucket distribution."""
    if fast_histogram_percentiles:
        # Standard Prometheus linear interpolation within buckets.
        # Assumes uniform distribution - good for realtime display where speed matters.
        return compute_prometheus_percentiles(
            bucket_cumulative=bucket_deltas,
            total_count=count_delta,
        )
    # Polynomial histogram algorithm with learned bucket means.
    # Learns per-bucket distributions from scrape sequences for better estimates.
    bucket_stats = accumulate_bucket_statistics(
        time_series.sums,
        time_series.counts,
        time_series.bucket_les,
        time_series.bucket_counts,
        start_idx=start_idx,
    )
    return compute_estimated_percentiles(
        bucket_deltas=bucket_deltas,
        bucket_stats=bucket_stats,
        total_sum=sum_delta,
        total_count=count_delta,
    )


def _histogram_bucket_deltas_with_reset_logging(
    time_series: HistogramTimeSeries,
    reference_idx: int | None,
    final_buckets: dict[str, float],
    labels: dict[str, str] | None,
) -> dict[str, int] | None:
    """Compute histogram bucket deltas; log warning on counter reset."""
    reference_bucket_idx = reference_idx if reference_idx is not None else 0
    reference_buckets = (
        time_series.get_bucket_dict(reference_bucket_idx)
        if len(time_series) > 0
        else {}
    )
    bucket_deltas = _compute_bucket_deltas(reference_buckets, final_buckets)
    if bucket_deltas is None:
        metric_label = "histogram metric" + (f" with labels {labels}" if labels else "")
        _logger.warning(
            f"Detected bucket counter reset in {metric_label}. "
            f"Histogram bucket data will be omitted from export. "
            f"Percentile estimates may be inaccurate."
        )
    return bucket_deltas


def _log_histogram_counter_reset_if_negative(
    sum_delta: float,
    count_delta: int,
    labels: dict[str, str] | None,
) -> None:
    """Log warning if histogram sum/count deltas are negative (counter reset)."""
    if sum_delta >= 0 and count_delta >= 0:
        return
    metric_label = "histogram metric" + (f" with labels {labels}" if labels else "")
    _logger.warning(
        f"Detected histogram counter reset in {metric_label}. "
        f"This typically indicates that server is behind a load balancer. "
        f"Sum delta: {sum_delta:.2f}, Count delta: {count_delta}. "
        f"Statistics may be inaccurate."
    )


def _build_histogram_rates(
    sum_delta: float,
    count_delta: int,
    duration_ns: int,
) -> tuple[float, float | None, float | None]:
    """Return (avg_value, count_rate, sum_rate) for a non-empty histogram window."""
    avg_value = sum_delta / count_delta
    duration_seconds = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
    count_rate = count_delta / duration_seconds if duration_seconds > 0 else None
    sum_rate = sum_delta / duration_seconds if duration_seconds > 0 else None
    return avg_value, count_rate, sum_rate


def _compute_histogram_deltas(
    time_series: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None,
) -> tuple[int | None, int, float, int, int, dict[str, float]]:
    """Compute core histogram deltas.

    Returns: (reference_idx, start_idx, sum_delta, count_delta, duration_ns, final_buckets)
    """
    (
        reference_idx,
        start_idx,
        reference_sum,
        reference_count,
        reference_timestamp,
        final_sum,
        final_count,
        final_timestamp,
        final_buckets,
    ) = _histogram_reference_and_final(time_series, time_filter)

    sum_delta = final_sum - reference_sum
    count_delta = int(final_count - reference_count)
    duration_ns = final_timestamp - reference_timestamp
    return reference_idx, start_idx, sum_delta, count_delta, duration_ns, final_buckets


def _compute_histogram_stats(
    time_series: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
    *,
    fast_histogram_percentiles: bool = False,
) -> HistogramSeries | None:
    """Compute histogram statistics (count/rates/percentiles/buckets/timeslices).

    Args:
        time_series: Histogram time series data
        time_filter: Time range filter (excludes warmup/cooldown)
        labels: Optional labels to attach to output statistics
        slice_duration: Timeslice duration in seconds (None = no timeslices)
        fast_histogram_percentiles: True = Prometheus linear interpolation (fast,
            ~15-40% error). False = polynomial histogram with learned bucket
            means (accurate, ~5% error).

    Returns:
        HistogramSeries, or None if the series is empty.
    """
    if len(time_series) == 0:
        return None

    (
        reference_idx,
        start_idx,
        sum_delta,
        count_delta,
        duration_ns,
        final_buckets,
    ) = _compute_histogram_deltas(time_series, time_filter)

    _log_histogram_counter_reset_if_negative(sum_delta, count_delta, labels)

    if count_delta == 0:
        return _empty_histogram_series(
            time_series, reference_idx, final_buckets, labels
        )

    avg_value, count_rate, sum_rate = _build_histogram_rates(
        sum_delta, count_delta, duration_ns
    )

    bucket_deltas = _histogram_bucket_deltas_with_reset_logging(
        time_series, reference_idx, final_buckets, labels
    )

    estimated = None
    if bucket_deltas:
        estimated = _estimate_percentiles(
            time_series=time_series,
            bucket_deltas=bucket_deltas,
            count_delta=count_delta,
            sum_delta=sum_delta,
            start_idx=start_idx,
            fast_histogram_percentiles=fast_histogram_percentiles,
        )

    timeslices: list[HistogramTimeslice] | None = None
    if slice_duration is not None:
        timeslices = _compute_histogram_timeslices(
            time_series, slice_duration, time_filter
        )

    return HistogramSeries(
        labels=labels,
        stats=HistogramStats(
            count=count_delta,
            count_rate=count_rate,
            sum=sum_delta,
            sum_rate=sum_rate,
            avg=avg_value,
            **(asdict(estimated) if estimated else {}),
        ),
        buckets=bucket_deltas,
        timeslices=timeslices,
    )

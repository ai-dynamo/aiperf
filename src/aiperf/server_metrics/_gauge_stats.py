# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gauge statistics computation for server metrics export."""

import numpy as np

from aiperf.common.models.server_metrics_models import (
    GaugeSeries,
    GaugeStats,
    GaugeTimeslice,
    TimeRangeFilter,
)
from aiperf.server_metrics._timeslice_boundaries import _compute_timeslice_boundaries
from aiperf.server_metrics.storage import ScalarTimeSeries


def _build_gauge_timeslice(
    timeslice_start: int,
    timeslice_end: int,
    timeslice_values: np.ndarray,
    complete: bool,
) -> GaugeTimeslice:
    """Build a single GaugeTimeslice from values in a timeslice."""
    return GaugeTimeslice(
        start_ns=int(timeslice_start),
        end_ns=int(timeslice_end),
        avg=float(np.mean(timeslice_values)),
        min=float(np.min(timeslice_values)),
        max=float(np.max(timeslice_values)),
        is_complete=None if complete else False,
    )


def _compute_gauge_timeslices(
    time_series: ScalarTimeSeries,
    slice_duration: float,
    time_filter: TimeRangeFilter,
) -> list[GaugeTimeslice] | None:
    """Compute time-sliced statistics for a gauge metric.

    Divides the time range into fixed-size timeslices and computes the
    average, min, and max values for each timeslice. Includes a final partial
    timeslice if the range doesn't align with slice boundaries.

    Partial slices are marked with is_complete=False. They contain valid data
    but should be excluded from comparative analysis to avoid skewing results.

    Uses np.searchsorted for O(log n) timeslice boundary lookups instead of
    O(n) boolean masks per timeslice.

    Args:
        time_series: Gauge time series data
        slice_duration: Duration of each timeslice in seconds
        time_filter: Time filter defining benchmark time range (excludes warmup)

    Returns:
        List of GaugeTimeslice, one per timeslice (complete + optional partial).
        None if insufficient data.

    Raises:
        ValueError: If slice_duration <= 0
    """
    if slice_duration <= 0:
        raise ValueError("slice_duration must be positive")

    time_mask = time_series.get_time_mask(time_filter)
    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]

    if len(filtered_timestamps) < 2:
        return None

    boundaries = _compute_timeslice_boundaries(
        time_filter.start_ns, time_filter.end_ns, slice_duration
    )
    if boundaries is None:
        return None
    timeslice_starts, timeslice_ends, is_complete = boundaries

    # Vectorized: find indices at all timeslice boundaries using searchsorted O(log n)
    sample_starts = np.searchsorted(filtered_timestamps, timeslice_starts, side="left")
    sample_ends = np.searchsorted(filtered_timestamps, timeslice_ends, side="left")

    # Special case: if the last timeslice ends exactly at the last sample,
    # include that sample (closed interval on the right) so it isn't dropped.
    if len(timeslice_ends) > 0 and timeslice_ends[-1] == filtered_timestamps[-1]:
        sample_ends[-1] = len(filtered_timestamps)

    results: list[GaugeTimeslice] = []
    for slice_idx, (timeslice_start, timeslice_end, complete) in enumerate(
        zip(timeslice_starts, timeslice_ends, is_complete, strict=True)
    ):
        sample_start_idx = int(sample_starts[slice_idx])
        sample_end_idx = int(sample_ends[slice_idx])
        if sample_start_idx >= sample_end_idx:
            continue

        timeslice_values = filtered_values[sample_start_idx:sample_end_idx]
        results.append(
            _build_gauge_timeslice(
                timeslice_start, timeslice_end, timeslice_values, complete
            )
        )

    return results if results else None


def _compute_gauge_stats(
    time_series: ScalarTimeSeries,
    time_filter: TimeRangeFilter | None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
) -> GaugeSeries | None:
    """Compute gauge statistics from a ScalarTimeSeries.

    Gauges represent instantaneous values (e.g., current queue depth, cache usage %).
    Statistics are computed over all samples in the aggregation period.

    Always returns full stats (avg, min, max, std, percentiles) for consistent API,
    even for constant gauges where std=0 and all percentiles equal the constant value.

    Args:
        time_series: The scalar time series to compute stats from
        time_filter: Time range filter defining benchmark period (excludes warmup)
        labels: Optional labels to attach to the output statistics
        slice_duration: Duration of each timeslice in seconds. If None, timeslices
                        are not computed.

    Returns:
        GaugeSeriesStats with gauge statistics, or None if no data in range
    """
    time_mask = time_series.get_time_mask(time_filter)
    filtered_values = time_series.values[time_mask]

    if len(filtered_values) == 0:
        return None

    # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
    std_dev = (
        float(np.std(filtered_values, ddof=1)) if len(filtered_values) > 1 else 0.0
    )

    timeslices: list[GaugeTimeslice] | None = None
    if slice_duration is not None:
        timeslices = _compute_gauge_timeslices(time_series, slice_duration, time_filter)

    # For constant gauges (std=0), all percentiles equal the constant value
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        filtered_values, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )

    return GaugeSeries(
        labels=labels,
        stats=GaugeStats(
            avg=float(np.mean(filtered_values)),
            min=float(np.min(filtered_values)),
            max=float(np.max(filtered_values)),
            std=std_dev,
            p1=float(p1),
            p5=float(p5),
            p10=float(p10),
            p25=float(p25),
            p50=float(p50),
            p75=float(p75),
            p90=float(p90),
            p95=float(p95),
            p99=float(p99),
        ),
        timeslices=timeslices,
    )

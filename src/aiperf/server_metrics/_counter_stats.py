# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counter statistics computation for server metrics export."""

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models.server_metrics_models import (
    CounterSeries,
    CounterStats,
    CounterTimeslice,
    TimeRangeFilter,
)
from aiperf.server_metrics._timeslice_boundaries import _compute_timeslice_boundaries
from aiperf.server_metrics.storage import ScalarTimeSeries

_logger = AIPerfLogger(__name__)


def _build_counter_series_with_reference(
    time_series: ScalarTimeSeries,
    time_filter: TimeRangeFilter,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (timestamps, values) arrays including reference point, or None if <2 samples."""
    reference_idx = time_series.get_reference_idx(time_filter)
    time_mask = time_series.get_time_mask(time_filter)

    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]

    if len(filtered_timestamps) == 0:
        return None

    if reference_idx is not None:
        reference_timestamp = time_series.timestamps[reference_idx]
        reference_value = time_series.values[reference_idx]
        timestamps = np.concatenate([[reference_timestamp], filtered_timestamps])
        values = np.concatenate([[reference_value], filtered_values])
    else:
        timestamps = filtered_timestamps
        values = filtered_values

    if len(timestamps) < 2:
        return None

    return timestamps, values


def _compute_counter_timeslices(
    time_series: ScalarTimeSeries,
    slice_duration: float,
    time_filter: TimeRangeFilter,
) -> list[CounterTimeslice]:
    """Compute time-sliced rates for a counter metric.

    Divides the time range into fixed-size timeslices and computes the rate
    (value delta / time) for each timeslice. Includes a final partial timeslice
    if the range doesn't align with slice boundaries.

    Partial slices are marked with is_complete=False and should be excluded from
    aggregate statistics (rate_min/max/avg/std) to avoid skewing results.

    Args:
        time_series: Counter time series data
        slice_duration: Duration of each timeslice in seconds
        time_filter: Time filter defining benchmark time range (excludes warmup)

    Returns:
        List of CounterTimeslice, one per timeslice (complete + optional partial).
        Empty list if insufficient data.

    Raises:
        ValueError: If slice_duration <= 0
    """
    if slice_duration <= 0:
        raise ValueError("slice_duration must be positive")

    series = _build_counter_series_with_reference(time_series, time_filter)
    if series is None:
        return []
    timestamps, values = series

    boundaries = _compute_timeslice_boundaries(
        time_filter.start_ns, time_filter.end_ns, slice_duration
    )
    if boundaries is None:
        return []
    timeslice_starts, timeslice_ends, is_complete = boundaries

    # Find values at timeslice boundaries using searchsorted (last value <= boundary)
    start_indices = np.searchsorted(timestamps, timeslice_starts, side="right") - 1
    end_indices = np.searchsorted(timestamps, timeslice_ends, side="right") - 1
    start_indices = np.clip(start_indices, 0, len(values) - 1)
    end_indices = np.clip(end_indices, 0, len(values) - 1)

    # Compute deltas and rates vectorized; handle counter resets (negative deltas -> 0)
    deltas = np.maximum(values[end_indices] - values[start_indices], 0)
    durations_s = (timeslice_ends - timeslice_starts) / NANOS_PER_SECOND
    rates = np.where(durations_s > 0, deltas / durations_s, 0)

    return [
        CounterTimeslice(
            start_ns=int(timeslice_start),
            end_ns=int(timeslice_end),
            total=float(delta),
            rate=float(rate),
            is_complete=None if complete else False,
        )
        for timeslice_start, timeslice_end, delta, rate, complete in zip(
            timeslice_starts, timeslice_ends, deltas, rates, is_complete, strict=True
        )
    ]


def _counter_rate_statistics(
    timeslices: list[CounterTimeslice],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute rate (avg, min, max, std) from complete timeslices only."""
    complete = [ts for ts in timeslices if ts.is_complete is not False]
    if not complete:
        return None, None, None, None

    slice_rates = np.array([ts.rate for ts in complete], dtype=np.float64)
    rate_std = float(np.std(slice_rates, ddof=1)) if len(slice_rates) > 1 else 0.0
    return (
        float(np.mean(slice_rates)),
        float(np.min(slice_rates)),
        float(np.max(slice_rates)),
        rate_std,
    )


def _log_counter_resets(
    reset_count: int,
    raw_delta: float,
    labels: dict[str, str] | None,
) -> None:
    metric_label = "counter metric" + (f" with labels {labels}" if labels else "")
    _logger.warning(
        f"Detected {reset_count} counter reset(s) in {metric_label}. "
        f"This typically indicates server restart(s) during profiling. "
        f"Statistics may be inaccurate. Raw delta: {raw_delta:.2f}"
    )


def _counter_reference_point(
    time_series: ScalarTimeSeries,
    reference_idx: int | None,
    filtered_timestamps: np.ndarray,
    filtered_values: np.ndarray,
) -> tuple[float, int]:
    """Return (reference_value, reference_timestamp) for counter delta computation."""
    if reference_idx is not None:
        return (
            float(time_series.values[reference_idx]),
            time_series.timestamps[reference_idx],
        )
    return float(filtered_values[0]), filtered_timestamps[0]


def _compute_counter_stats(
    time_series: ScalarTimeSeries,
    time_filter: TimeRangeFilter | None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
) -> CounterSeries | None:
    """Compute counter statistics from a ScalarTimeSeries.

    Counters represent cumulative totals (e.g., total requests, total bytes).
    We report the delta and rate statistics over the aggregation period.

    Always returns full stats (total, rate, rate_avg, rate_min, rate_max, rate_std)
    for consistent API, even for zero-change counters where all rates are 0.

    Rate statistics (rate_min, rate_max, rate_avg, rate_std) are computed from
    timeslices - fixed-duration time slices that provide consistent,
    comparable rate measurements across the collection period.

    Args:
        time_series: The scalar time series to compute stats from
        time_filter: Time range filter defining benchmark period (excludes warmup)
        labels: Optional labels to attach to the output statistics
        slice_duration: Duration of each timeslice in seconds. If None, timeslices
                        are not computed.

    Returns:
        CounterSeriesStats with counter statistics, or None if no data in range
    """
    reference_idx = time_series.get_reference_idx(time_filter)
    time_mask = time_series.get_time_mask(time_filter)

    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]

    if len(filtered_values) == 0:
        return None

    reference_value, reference_timestamp = _counter_reference_point(
        time_series, reference_idx, filtered_timestamps, filtered_values
    )

    raw_delta = float(filtered_values[-1]) - reference_value
    duration_ns = filtered_timestamps[-1] - reference_timestamp

    # Detect counter resets (Prometheus counters should be monotonically increasing)
    reset_count = int(np.sum(np.diff(filtered_values) < 0))
    if reset_count > 0:
        _log_counter_resets(reset_count, raw_delta, labels)

    total_delta = max(raw_delta, 0.0)
    duration_seconds = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0.0
    rate_per_second = total_delta / duration_seconds if duration_seconds > 0 else 0.0

    timeslices: list[CounterTimeslice] | None = None
    rate_avg = rate_min = rate_max = rate_std = None

    if slice_duration is not None:
        timeslices = _compute_counter_timeslices(
            time_series, slice_duration, time_filter
        )
        if timeslices:
            rate_avg, rate_min, rate_max, rate_std = _counter_rate_statistics(
                timeslices
            )

    return CounterSeries(
        labels=labels,
        stats=CounterStats(
            total=total_delta,
            rate=rate_per_second,
            rate_avg=rate_avg,
            rate_min=rate_min,
            rate_max=rate_max,
            rate_std=rate_std,
        ),
        timeslices=timeslices,
    )

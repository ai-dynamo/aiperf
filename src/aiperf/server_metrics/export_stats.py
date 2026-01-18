# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export statistics computation for server metrics.

This module provides functions to compute statistics from time series data
into type-specific series models (GaugeSeries, CounterSeries, HistogramSeries).
"""

from dataclasses import asdict

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    CounterSeries,
    CounterStats,
    CounterTimeslice,
    GaugeSeries,
    GaugeStats,
    GaugeTimeslice,
    HistogramSeries,
    HistogramStats,
    HistogramTimeslice,
    TimeRangeFilter,
)
from aiperf.server_metrics.histogram_percentiles import (
    accumulate_bucket_statistics,
    compute_estimated_percentiles,
)
from aiperf.server_metrics.storage import HistogramTimeSeries, ScalarTimeSeries

_logger = AIPerfLogger(__name__)

# =============================================================================
# Public API: Statistics Computation
# =============================================================================


def compute_stats(
    metric_type: PrometheusMetricType,
    time_series: ScalarTimeSeries | HistogramTimeSeries,
    time_filter: TimeRangeFilter | None = None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
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
            )
        case _:
            raise ValueError(f"Unsupported metric type: {metric_type}")


# =============================================================================
# Timeslice Boundary Computation
# =============================================================================


def _compute_timeslice_boundaries(
    range_start_ns: int,
    range_end_ns: int,
    slice_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute timeslice start/end boundaries and completeness flags.

    Generates evenly-spaced timeslice boundaries covering the time range, including
    a partial final timeslice if the range doesn't align with slice boundaries.

    Best practice: Include all data (even partial slices) for completeness, but mark
    which slices are complete vs partial so aggregate statistics can filter appropriately.

    Args:
        range_start_ns: Start of the time range in nanoseconds (inclusive)
        range_end_ns: End of the time range in nanoseconds (inclusive)
        slice_duration: Duration of each timeslice in seconds

    Returns:
        Tuple of (starts, ends, is_complete) numpy arrays where:
        - starts[i] and ends[i] define the i-th timeslice boundaries
        - is_complete[i] is True if the slice covers the full duration, False for partial
        Returns None if slice_duration > range duration (no slices fit).

    Example:
        >>> # 10 second range with 3 second slices
        >>> starts, ends, complete = _compute_timeslice_boundaries(0, 10_000_000_000, 3.0)
        >>> # Returns: [0, 3s, 6s, 9s], [3s, 6s, 9s, 10s], [True, True, True, False]
        >>> #          ^-- 3 complete slices + 1 partial (1s duration)
    """
    timeslice_size_ns = int(slice_duration * NANOS_PER_SECOND)

    # Generate all complete timeslice starts
    timeslice_starts = np.arange(
        range_start_ns, range_end_ns, timeslice_size_ns, dtype=np.int64
    )

    if len(timeslice_starts) == 0:
        return None

    # Compute corresponding ends
    timeslice_ends = timeslice_starts + timeslice_size_ns

    # Mark which timeslices are complete (end <= range_end_ns)
    is_complete = timeslice_ends <= range_end_ns

    # Clip ends to range_end_ns (converts incomplete slices to partial)
    timeslice_ends = np.minimum(timeslice_ends, range_end_ns)

    # Add partial final slice if there's remaining time after last complete slice
    if not is_complete[-1]:
        # Last slice is already partial, no need to add another
        pass
    elif timeslice_ends[-1] < range_end_ns:
        # Add partial final slice from last complete slice end to range end
        final_start = timeslice_ends[-1]
        timeslice_starts = np.append(timeslice_starts, final_start)
        timeslice_ends = np.append(timeslice_ends, range_end_ns)
        is_complete = np.append(is_complete, False)

    return timeslice_starts, timeslice_ends, is_complete


# =============================================================================
# Counter Calculations
# =============================================================================


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

    # Get reference point and filtered data
    reference_idx = time_series.get_reference_idx(time_filter)
    time_mask = time_series.get_time_mask(time_filter)

    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]

    if len(filtered_timestamps) == 0:
        return []

    # Build complete series including reference point
    if reference_idx is not None:
        reference_timestamp = time_series.timestamps[reference_idx]
        reference_value = time_series.values[reference_idx]
        timestamps = np.concatenate([[reference_timestamp], filtered_timestamps])
        values = np.concatenate([[reference_value], filtered_values])
    else:
        timestamps = filtered_timestamps
        values = filtered_values

    if len(timestamps) < 2:
        return []

    # Use time_filter bounds for timeslice generation (not data bounds)
    boundaries = _compute_timeslice_boundaries(
        time_filter.start_ns, time_filter.end_ns, slice_duration
    )
    if boundaries is None:
        return []
    timeslice_starts, timeslice_ends, is_complete = boundaries

    # Find values at timeslice boundaries using searchsorted
    # Use 'right' to find the last value <= boundary timestamp
    start_indices = np.searchsorted(timestamps, timeslice_starts, side="right") - 1
    end_indices = np.searchsorted(timestamps, timeslice_ends, side="right") - 1

    # Clip to valid range
    start_indices = np.clip(start_indices, 0, len(values) - 1)
    end_indices = np.clip(end_indices, 0, len(values) - 1)

    # Compute deltas and rates vectorized
    start_values = values[start_indices]
    end_values = values[end_indices]
    deltas = end_values - start_values

    # Handle counter resets (negative deltas become 0)
    deltas = np.maximum(deltas, 0)

    # Compute normalized rates (per second) even for partial slices
    durations_ns = timeslice_ends - timeslice_starts
    durations_s = durations_ns / NANOS_PER_SECOND
    rates = np.where(durations_s > 0, deltas / durations_s, 0)

    return [
        CounterTimeslice(
            start_ns=int(timeslice_start),
            end_ns=int(timeslice_end),
            total=float(delta),
            rate=float(rate),
            is_complete=None
            if complete
            else False,  # None for complete (space-efficient)
        )
        for timeslice_start, timeslice_end, delta, rate, complete in zip(
            timeslice_starts, timeslice_ends, deltas, rates, is_complete, strict=True
        )
    ]


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

    # Return None if time filter excludes all data
    if len(filtered_values) == 0:
        return None

    # Reference for delta calculation
    reference_value = (
        float(time_series.values[reference_idx])
        if reference_idx is not None
        else float(filtered_values[0])
    )
    reference_timestamp = (
        time_series.timestamps[reference_idx]
        if reference_idx is not None
        else filtered_timestamps[0]
    )

    # Total delta and duration
    raw_delta = float(filtered_values[-1]) - reference_value
    duration_ns = filtered_timestamps[-1] - reference_timestamp

    # Detect counter resets (common during server restarts)
    # Prometheus counters should be monotonically increasing, but resets cause negative deltas
    # Check for multiple resets by examining all deltas
    deltas = np.diff(filtered_values)
    reset_count = np.sum(deltas < 0)

    if reset_count > 0:
        metric_label = "counter metric" + (f" with labels {labels}" if labels else "")
        _logger.warning(
            f"Detected {reset_count} counter reset(s) in {metric_label}. "
            f"This typically indicates server restart(s) during profiling. "
            f"Statistics may be inaccurate. Raw delta: {raw_delta:.2f}"
        )

    # Handle counter reset - use max(0) for negative deltas
    total_delta = max(raw_delta, 0.0)

    # Always populate stats (even for zero-change counters) for consistent API
    duration_seconds = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0.0
    rate_per_second = total_delta / duration_seconds if duration_seconds > 0 else 0.0

    # Compute timeslices if slice_duration is specified
    timeslices: list[CounterTimeslice] | None = None
    rate_avg = None
    rate_min = None
    rate_max = None
    rate_std = None

    if slice_duration is not None:
        timeslices = _compute_counter_timeslices(
            time_series, slice_duration, time_filter
        )
        if timeslices:
            # Compute rate statistics from COMPLETE timeslices only
            # Partial slices are included in output but excluded from aggregate stats
            # is_complete=None or True means complete, False means partial
            complete_timeslices = [
                ts for ts in timeslices if ts.is_complete is not False
            ]
            if complete_timeslices:
                slice_rates = np.array(
                    [ts.rate for ts in complete_timeslices], dtype=np.float64
                )
                rate_avg = float(np.mean(slice_rates))
                rate_min = float(np.min(slice_rates))
                rate_max = float(np.max(slice_rates))
                rate_std = (
                    float(np.std(slice_rates, ddof=1)) if len(slice_rates) > 1 else 0.0
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


# =============================================================================
# Gauge Calculations
# =============================================================================


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

    # Get filtered data
    time_mask = time_series.get_time_mask(time_filter)
    filtered_timestamps = time_series.timestamps[time_mask]
    filtered_values = time_series.values[time_mask]

    if len(filtered_timestamps) < 2:
        return None

    # Use time_filter bounds for timeslice generation (not data bounds)
    boundaries = _compute_timeslice_boundaries(
        time_filter.start_ns, time_filter.end_ns, slice_duration
    )
    if boundaries is None:
        return None
    timeslice_starts, timeslice_ends, is_complete = boundaries

    # Vectorized: find indices at all timeslice boundaries using searchsorted O(log n)
    # sample_starts[i] = first index where timestamp >= timeslice_starts[i]
    # sample_ends[i] = first index where timestamp >= timeslice_ends[i] (exclusive)
    sample_starts = np.searchsorted(filtered_timestamps, timeslice_starts, side="left")
    sample_ends = np.searchsorted(filtered_timestamps, timeslice_ends, side="left")

    # Special case: if the last timeslice ends exactly at the last sample,
    # include that sample in the timeslice (making it a closed interval on the right).
    # This ensures that when there are fewer samples than timeslices, the last sample
    # is not excluded from all timeslices.
    if len(timeslice_ends) > 0 and timeslice_ends[-1] == filtered_timestamps[-1]:
        sample_ends[-1] = len(filtered_timestamps)

    results: list[GaugeTimeslice] = []

    for slice_idx, (timeslice_start, timeslice_end, complete) in enumerate(
        zip(timeslice_starts, timeslice_ends, is_complete, strict=True)
    ):
        sample_start_idx = int(sample_starts[slice_idx])
        sample_end_idx = int(sample_ends[slice_idx])

        # Skip timeslices with no samples (shouldn't happen with deduplication disabled)
        if sample_start_idx >= sample_end_idx:
            continue

        # Get slice of values (O(1) view, not copy)
        timeslice_values = filtered_values[sample_start_idx:sample_end_idx]

        # Compute timeslice statistics
        results.append(
            GaugeTimeslice(
                start_ns=int(timeslice_start),
                end_ns=int(timeslice_end),
                avg=float(np.mean(timeslice_values)),
                min=float(np.min(timeslice_values)),
                max=float(np.max(timeslice_values)),
                is_complete=None
                if complete
                else False,  # None for complete (space-efficient)
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

    # Return None if time filter excludes all data
    if len(filtered_values) == 0:
        return None

    # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
    std_dev = (
        float(np.std(filtered_values, ddof=1)) if len(filtered_values) > 1 else 0.0
    )

    # Always populate stats (even for constant gauges) for consistent API
    # Compute timeslices if slice_duration is specified
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


# =============================================================================
# Histogram Calculations
# =============================================================================


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

    timestamps = time_series.timestamps
    sums = time_series.sums
    counts = time_series.counts
    bucket_les = time_series.bucket_les
    bucket_counts = time_series.bucket_counts

    start_idx = reference_idx if reference_idx is not None else 0

    if final_idx <= start_idx:
        return None

    # Use time_filter bounds for timeslice generation (not data bounds)
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
        # Find last sample <= boundary
        boundary_start_idx = (
            np.searchsorted(timestamps, timeslice_start, side="right") - 1
        )
        boundary_end_idx = np.searchsorted(timestamps, timeslice_end, side="right") - 1

        # Clip to valid range
        boundary_start_idx = max(0, min(boundary_start_idx, len(timestamps) - 1))
        boundary_end_idx = max(0, min(boundary_end_idx, len(timestamps) - 1))

        # Compute deltas
        sum_delta = sums[boundary_end_idx] - sums[boundary_start_idx]
        count_delta = counts[boundary_end_idx] - counts[boundary_start_idx]

        # Skip only on counter resets (negative deltas)
        if sum_delta < 0 or count_delta < 0:
            continue

        avg_value = sum_delta / count_delta if count_delta > 0 else 0.0

        # Compute bucket deltas for this timeslice
        bucket_deltas: dict[str, int] | None = None
        if len(bucket_les) > 0 and len(bucket_counts) > 0:
            start_buckets = bucket_counts[boundary_start_idx]
            end_buckets = bucket_counts[boundary_end_idx]
            bucket_deltas: dict[str, int] = {}
            has_reset = False
            for i, le in enumerate(bucket_les):
                delta = end_buckets[i] - start_buckets[i]
                if delta < 0:
                    has_reset = True
                    break
                bucket_deltas[le] = int(delta)
            if has_reset:
                bucket_deltas = None

        results.append(
            HistogramTimeslice(
                start_ns=int(timeslice_start),
                end_ns=int(timeslice_end),
                count=int(count_delta),
                sum=float(sum_delta),
                avg=float(avg_value),
                buckets=bucket_deltas,
                is_complete=None
                if complete
                else False,  # None for complete (space-efficient)
            )
        )

    return results if results else None


def _compute_histogram_stats(
    time_series: HistogramTimeSeries,
    time_filter: TimeRangeFilter | None,
    labels: dict[str, str] | None = None,
    slice_duration: float | None = None,
) -> HistogramSeries | None:
    """Compute histogram statistics from a HistogramTimeSeries.

    Histograms track distributions (e.g., request latencies). We report:
    - Count and count rate (observations per second)
    - Sum and sum rate (total value per second)
    - Average value per observation
    - Estimated percentiles using polynomial histogram algorithm
    - Raw bucket data for downstream analysis
    - Timeslices over time (when slice_duration is specified)

    Args:
        time_series: The histogram time series to compute stats from
        time_filter: Time range filter defining benchmark period (excludes warmup)
        labels: Optional labels to attach to the output statistics
        slice_duration: Duration of each timeslice in seconds. If None, timeslices
                        are not computed.

    Returns:
        HistogramSeriesStats with histogram statistics, or None if no data in range
    """
    # Return None if time series is empty
    if len(time_series) == 0:
        return None

    reference_idx, final_idx = time_series.get_indices_for_filter(time_filter)

    # Reference values
    if reference_idx is not None:
        reference_sum = float(time_series.sums[reference_idx])
        reference_count = float(time_series.counts[reference_idx])
        reference_timestamp = time_series.timestamps[reference_idx]
    else:
        reference_sum = float(time_series.sums[0])
        reference_count = float(time_series.counts[0])
        reference_timestamp = time_series.timestamps[0]

    # Final values
    final_sum = float(time_series.sums[final_idx])
    final_count = float(time_series.counts[final_idx])
    final_timestamp = time_series.timestamps[final_idx]
    final_buckets = (
        time_series.get_bucket_dict(final_idx) if len(time_series) > 0 else {}
    )

    # Compute deltas
    sum_delta = final_sum - reference_sum
    count_delta = int(final_count - reference_count)
    duration_ns = final_timestamp - reference_timestamp

    # Detect histogram counter resets (sum/count are cumulative)
    if sum_delta < 0 or count_delta < 0:
        metric_label = "histogram metric" + (f" with labels {labels}" if labels else "")
        _logger.warning(
            f"Detected histogram counter reset in {metric_label}. "
            f"This typically indicates that server is behind a load balancer. "
            f"Sum delta: {sum_delta:.2f}, Count delta: {count_delta}. "
            f"Statistics may be inaccurate."
        )

    # For empty histograms (count=0), still include buckets for API consistency
    if count_delta == 0:
        # Compute bucket deltas (all zeros for empty histogram)
        reference_bucket_idx = reference_idx if reference_idx is not None else 0
        reference_buckets = (
            time_series.get_bucket_dict(reference_bucket_idx)
            if len(time_series) > 0
            else {}
        )

        bucket_deltas: dict[str, int] = {}
        for le_bound, final_bucket_count in final_buckets.items():
            reference_bucket_count = reference_buckets.get(le_bound, 0.0)
            bucket_delta = final_bucket_count - reference_bucket_count
            if bucket_delta < 0:
                # Counter reset - omit buckets
                bucket_deltas = None
                break
            bucket_deltas[le_bound] = int(bucket_delta)

        return HistogramSeries(
            labels=labels,
            stats=HistogramStats(count=0),
            buckets=bucket_deltas,
        )

    avg_value = sum_delta / count_delta
    duration_seconds = duration_ns / NANOS_PER_SECOND if duration_ns > 0 else 0
    count_rate = count_delta / duration_seconds if duration_seconds > 0 else None
    sum_rate = sum_delta / duration_seconds if duration_seconds > 0 else None

    # Bucket delta calculation
    reference_bucket_idx = reference_idx if reference_idx is not None else 0
    reference_buckets = (
        time_series.get_bucket_dict(reference_bucket_idx)
        if len(time_series) > 0
        else {}
    )

    bucket_deltas: dict[str, int] | None = {}
    bucket_reset_detected = False
    for le_bound, final_bucket_count in final_buckets.items():
        reference_bucket_count = reference_buckets.get(le_bound, 0.0)
        bucket_delta = final_bucket_count - reference_bucket_count
        if bucket_delta < 0:
            bucket_reset_detected = True
            bucket_deltas = None
            break
        bucket_deltas[le_bound] = int(bucket_delta)

    if bucket_reset_detected:
        metric_label = "histogram metric" + (f" with labels {labels}" if labels else "")
        _logger.warning(
            f"Detected bucket counter reset in {metric_label}. "
            f"Histogram bucket data will be omitted from export. "
            f"Percentile estimates may be inaccurate."
        )

    # Compute estimated percentiles
    estimated = None
    if bucket_deltas:
        start_idx = reference_idx if reference_idx is not None else 0
        bucket_stats = accumulate_bucket_statistics(
            time_series.sums,
            time_series.counts,
            time_series.bucket_les,
            time_series.bucket_counts,
            start_idx=start_idx,
        )
        estimated = compute_estimated_percentiles(
            bucket_deltas=bucket_deltas,
            bucket_stats=bucket_stats,
            total_sum=sum_delta,
            total_count=count_delta,
        )

    # Compute timeslices if slice_duration is specified
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

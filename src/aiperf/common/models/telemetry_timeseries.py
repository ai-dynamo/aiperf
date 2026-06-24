# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter


class GpuMetricTimeSeries:
    """NumPy-backed columnar storage for GPU telemetry.

    Stores timestamps once with separate value arrays per metric.

    The metric schema is *dynamic*: a DCGM/AMDSMI scrape may transiently omit
    a field when a sensor read fails, or only start reporting a field partway
    through a run. To keep every value array index-aligned with the shared
    ``_timestamps`` array, missing samples are stored as ``NaN``:

    - A metric absent from a scrape gets ``NaN`` written at that index rather
      than uninitialized ``np.empty`` garbage.
    - A metric that first appears in a later scrape allocates its array
      ``NaN``-backfilled for every prior index.

    All stat computations are therefore NaN-aware: gauges aggregate only the
    real (non-NaN) samples, counters walk back past NaN reference/final
    samples, and an all-NaN window raises ``NoMetricValue`` instead of emitting
    NaN stats.

    Data is kept sorted by timestamp using insert-sorted approach:
    O(1) for in-order appends (99.9% of cases), O(k) for out-of-order.
    """

    __slots__ = ("_timestamps", "_metrics", "_size", "_capacity")

    _INITIAL_CAPACITY = 128

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(self._INITIAL_CAPACITY, dtype=np.int64)
        self._metrics: dict[str, np.ndarray] = {}
        self._size: int = 0
        self._capacity: int = self._INITIAL_CAPACITY

    def append_snapshot(self, metrics: dict[str, float], timestamp_ns: int) -> None:
        """Append all metrics from a single DCGM scrape (insert-sorted).

        Args:
            metrics: Dict of metric_name -> value (only present metrics)
            timestamp_ns: Timestamp for this scrape

        Note:
            - The metric schema is dynamic. Metrics missing from this scrape are
              stored as ``NaN`` at the new index; metrics appearing for the
              first time allocate a new ``NaN``-backfilled array.
            - Data kept sorted by timestamp (O(1) in-order, O(k) out-of-order).
        """
        if self._size >= self._capacity:
            self._grow()

        # Register any newly-seen metrics, NaN-backfilled for all prior indices.
        for name in metrics:
            if name not in self._metrics:
                self._metrics[name] = np.full(self._capacity, np.nan, dtype=np.float64)

        # Fast path: in-order append (99.9% of cases)
        if self._size == 0 or timestamp_ns >= self._timestamps[self._size - 1]:
            insert_pos = self._size
        else:
            # Slow path: find insert position from end (reverse linear search)
            insert_pos = self._size - 1
            while insert_pos > 0 and self._timestamps[insert_pos - 1] > timestamp_ns:
                insert_pos -= 1

            # Shift timestamps right
            self._timestamps[insert_pos + 1 : self._size + 1] = self._timestamps[
                insert_pos : self._size
            ]

            # Shift all metric arrays right
            for arr in self._metrics.values():
                arr[insert_pos + 1 : self._size + 1] = arr[insert_pos : self._size]

        # Insert timestamp at position
        self._timestamps[insert_pos] = timestamp_ns

        # Write every known metric at this index: NaN for any absent from this
        # scrape (disappearing metric), real value for those present.
        for name, arr in self._metrics.items():
            arr[insert_pos] = metrics.get(name, np.nan)

        self._size += 1

    def _grow(self) -> None:
        """Double capacity of all arrays.

        New metric arrays are ``NaN``-initialized so any index written without
        an explicit value reads back as missing rather than garbage.
        """
        new_capacity = self._capacity * 2

        # Grow timestamps
        new_ts = np.empty(new_capacity, dtype=np.int64)
        new_ts[: self._size] = self._timestamps[: self._size]
        self._timestamps = new_ts

        # Grow each metric array, NaN-filling the freshly-allocated tail.
        for name, old_arr in self._metrics.items():
            new_arr = np.full(new_capacity, np.nan, dtype=np.float64)
            new_arr[: self._size] = old_arr[: self._size]
            self._metrics[name] = new_arr

        self._capacity = new_capacity

    @property
    def timestamps(self) -> np.ndarray:
        """View of timestamps array (no copy)."""
        return self._timestamps[: self._size]

    def get_metric_array(self, metric_name: str) -> np.ndarray | None:
        """Get values array for a metric (no copy). Returns None if metric unknown."""
        if metric_name not in self._metrics:
            return None
        return self._metrics[metric_name][: self._size]

    def to_metric_result(
        self, metric_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Compute stats for a metric using vectorized NumPy operations.

        NaN-aware: only the real (non-NaN) samples feed the min/max/avg/std/
        percentile computation. ``count`` is the number of scrapes (array
        length), not the non-NaN sample count, so it stays aligned with the
        timestamp axis.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement

        Returns:
            MetricResult with min/max/avg/percentiles computed from all values

        Raises:
            NoMetricValue: If no data for this metric, or every sample is NaN
        """
        arr = self.get_metric_array(metric_name)
        if arr is None or len(arr) == 0:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )

        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            raise NoMetricValue(
                f"All samples for metric '{metric_name}' are NaN (no valid telemetry)"
            )

        return self._stats_result(
            valid, tag, header, unit, count=len(arr), current=_last_valid(arr)
        )

    def get_time_mask(self, time_filter: TimeRangeFilter | None) -> NDArray[np.bool_]:
        """Get boolean mask for points within time range.

        Uses np.searchsorted for O(log n) binary search on sorted timestamps,
        then slice assignment for mask creation (10-100x faster than element-wise
        boolean comparisons for large arrays).

        Args:
            time_filter: Time range filter specifying start_ns and/or end_ns bounds.
                        None returns all-True mask.

        Returns:
            Boolean mask array where True indicates timestamp within range
        """
        if time_filter is None:
            return np.ones(self._size, dtype=bool)

        timestamps = self.timestamps
        first_idx = 0
        last_idx = self._size

        # O(log n) binary search for range boundaries
        if time_filter.start_ns is not None:
            first_idx = int(
                np.searchsorted(timestamps, time_filter.start_ns, side="left")
            )
        if time_filter.end_ns is not None:
            last_idx = int(
                np.searchsorted(timestamps, time_filter.end_ns, side="right")
            )

        # Single allocation + slice assignment
        mask = np.zeros(self._size, dtype=bool)
        mask[first_idx:last_idx] = True
        return mask

    def get_reference_idx(self, time_filter: TimeRangeFilter | None) -> int | None:
        """Get index of last point BEFORE time filter start (for delta calculation).

        Uses np.searchsorted for O(log n) lookup. Returns None if no baseline exists
        (i.e., time_filter is None, start_ns is None, or no data before start_ns).

        Args:
            time_filter: Time range filter. Reference point is found before start_ns.

        Returns:
            Index of last timestamp before start_ns, or None if no baseline exists
        """
        if time_filter is None or time_filter.start_ns is None:
            return None
        insert_pos = int(
            np.searchsorted(self.timestamps, time_filter.start_ns, side="left")
        )
        return insert_pos - 1 if insert_pos > 0 else None

    def _counter_delta_result(
        self,
        arr: np.ndarray,
        filtered: np.ndarray,
        *,
        metric_name: str,
        time_filter: TimeRangeFilter | None,
        tag: str,
        header: str,
        unit: str,
    ) -> MetricResult:
        """Compute counter delta from baseline, clamped to 0 on resets.

        NaN-aware: the in-window final sample skips trailing NaNs; the baseline
        reference walks back past NaN samples before start_ns, falling back to
        the first valid in-window sample when no valid pre-window baseline
        exists. An all-NaN window raises ``NoMetricValue``.
        """
        # Last valid in-window sample.
        filtered_last = _last_valid(filtered)
        if filtered_last is None:
            raise NoMetricValue(
                f"No valid (non-NaN) samples in time range for counter "
                f"metric '{metric_name}'"
            )

        # Walk back from the chosen reference index past NaN samples.
        reference_value: float | None = None
        reference_idx = self.get_reference_idx(time_filter)
        if reference_idx is not None:
            for idx in range(reference_idx, -1, -1):
                if not np.isnan(arr[idx]):
                    reference_value = float(arr[idx])
                    break

        # No valid pre-window baseline: anchor on the first valid in-window value.
        if reference_value is None:
            reference_value = _first_valid(filtered)

        raw_delta = filtered_last - reference_value

        # Handle counter resets (e.g., DCGM restart) by clamping to 0
        delta = max(raw_delta, 0.0)

        # Counters report a single delta value, not a distribution
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            avg=delta,
        )

    @staticmethod
    def _stats_result(
        valid: np.ndarray,
        tag: str,
        header: str,
        unit: str,
        *,
        count: int,
        current: float | None,
    ) -> MetricResult:
        """Compute vectorized gauge stats from the non-NaN sample array.

        ``valid`` must already be filtered to non-NaN values. ``count`` is the
        number of scrapes (kept aligned with the timestamp axis) while the
        std ddof guard uses the non-NaN sample count.
        """
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            valid, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        # Use sample std (ddof=1) for unbiased estimate; 0 for a single valid
        # sample (ddof=1 with one sample is degrees-of-freedom 0 -> NaN+warning).
        std_dev = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.min(valid)),
            max=float(np.max(valid)),
            avg=float(np.mean(valid)),
            sum=float(np.sum(valid)),
            std=std_dev,
            count=count,
            current=current,
            p1=p1,
            p5=p5,
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
        )

    def to_metric_result_filtered(
        self,
        metric_name: str,
        tag: str,
        header: str,
        unit: str,
        time_filter: TimeRangeFilter | None = None,
        is_counter: bool = False,
    ) -> MetricResult:
        """Compute stats with time filtering and optional delta for counters.

        For gauges: Uses vectorized NumPy on the non-NaN filtered samples
        (np.mean, np.std, np.percentile).
        For counters: Computes delta from reference point before profiling start.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement
            time_filter: Optional time range filter to exclude warmup/cooldown periods
            is_counter: If True, compute delta from baseline instead of statistics

        Returns:
            MetricResult with min/max/avg/percentiles for gauges, or delta for counters

        Raises:
            NoMetricValue: If no data for this metric, no data in the filtered
                range, or every in-range sample is NaN
        """
        arr = self.get_metric_array(metric_name)
        if arr is None or len(arr) == 0:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )

        time_mask = self.get_time_mask(time_filter)
        filtered = arr[time_mask]
        if len(filtered) == 0:
            raise NoMetricValue(f"No data in time range for metric '{metric_name}'")

        if is_counter:
            return self._counter_delta_result(
                arr,
                filtered,
                metric_name=metric_name,
                time_filter=time_filter,
                tag=tag,
                header=header,
                unit=unit,
            )

        valid = filtered[~np.isnan(filtered)]
        if len(valid) == 0:
            raise NoMetricValue(
                f"All in-range samples for metric '{metric_name}' are NaN "
                f"(no valid telemetry in filtered window)"
            )
        return self._stats_result(
            valid,
            tag,
            header,
            unit,
            count=len(filtered),
            current=_last_valid(filtered),
        )

    def __len__(self) -> int:
        """Return the number of snapshots in the time series."""
        return self._size


def _last_valid(arr: np.ndarray) -> float | None:
    """Return the last non-NaN value in ``arr``, or ``None`` if all NaN."""
    mask = ~np.isnan(arr)
    return float(arr[mask][-1]) if mask.any() else None


def _first_valid(arr: np.ndarray) -> float | None:
    """Return the first non-NaN value in ``arr``, or ``None`` if all NaN."""
    mask = ~np.isnan(arr)
    return float(arr[mask][0]) if mask.any() else None

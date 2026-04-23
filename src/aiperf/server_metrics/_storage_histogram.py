# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricSample, TimeRangeFilter
from aiperf.server_metrics._storage_scalar import _INITIAL_CAPACITY


def _bucket_sort_key(le: str) -> float:
    """Sort key for histogram bucket boundaries.

    Prometheus histograms use string keys for bucket upper bounds (le values).
    The special '+Inf' bucket must sort after all numeric buckets to maintain
    proper bucket ordering for cumulative histogram semantics.

    Args:
        le: Bucket upper bound as string (e.g., "0.01", "1.0", "+Inf")

    Returns:
        float("inf") for "+Inf" bucket, otherwise the numeric value
    """
    return float("inf") if le == "+Inf" else float(le)


class HistogramTimeSeries:
    """Storage for histogram metrics with fully vectorized bucket storage.

    Efficient storage for Prometheus histogram metrics using NumPy arrays.
    Maintains sorted order by timestamp while supporting efficient bucket-based
    percentile estimation and rate calculations.

    Storage strategy:
    - Bucket schema initialized on first append (tuple of sorted le values)
    - Parallel 1D arrays for timestamps, sums, counts
    - Single 2D array for all bucket counts (shape: n_snapshots × n_buckets)
    - Fully vectorized operations for statistics and delta computation

    Enables:
    - Observation rate (count/sec) - e.g., requests/second
    - Value rate (sum/sec) - e.g., total latency/second
    - Average value (sum/count) - e.g., average latency per request
    - Vectorized bucket delta computation for percentiles
    - Time-filtered analysis with O(log n) queries

    Data is always maintained in sorted order by timestamp. Out-of-order
    appends are handled efficiently by inserting at the correct position,
    optimized for the common case where data arrives nearly in order (99.9%).

    Example:
        >>> from aiperf.common.models import MetricSample
        >>> ts = HistogramTimeSeries()
        >>> # Add histogram snapshot
        >>> sample = MetricSample(
        ...     buckets={"0.01": 10, "0.1": 45, "1.0": 98, "+Inf": 100},
        ...     sum=32.5,
        ...     count=100
        ... )
        >>> ts.append(1_000_000_000, sample)
        >>> len(ts)
        1
        >>> ts.bucket_les
        ('0.01', '0.1', '1.0', '+Inf')
        >>> ts.counts[0]
        100.0
    """

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.int64)
        self._sums: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._counts: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._size: int = 0
        self._bucket_les: tuple[str, ...] | None = None
        self._bucket_counts: np.ndarray | None = None
        self._logger = logging.getLogger(__name__)

    def append(self, timestamp_ns: int, sample: MetricSample) -> None:
        """Append a histogram sample, maintaining sorted order by timestamp.

        Optimized for chronological data arrival (99.9% of cases) with O(1)
        fast path append. Falls back to O(k) insertion for out-of-order data.
        All array operations use fully vectorized NumPy for performance.

        Histogram storage maintains:
        - Sorted timestamps for efficient time-based queries
        - Sum and count arrays for rate calculations
        - 2D bucket counts array for percentile estimation

        Automatically grows capacity by doubling when full. On first append,
        initializes bucket schema from the sample's bucket keys (sorted order).
        Subsequent samples must have compatible bucket boundaries.

        Args:
            timestamp_ns: Nanosecond timestamp for this histogram snapshot
            sample: MetricSample containing buckets, sum, and count

        Raises:
            ValueError: If sample.buckets is None (required for histogram series)
        """
        if sample.buckets is None:
            raise ValueError("Buckets are required for histogram time series")

        if self._bucket_les is None:
            self._init_bucket_schema(sample.buckets)

        self._warn_on_schema_mismatch(sample.buckets)

        # Convert dict to row (order matches _bucket_les, 0.0 for missing buckets)
        bucket_row = np.array([sample.buckets.get(le, 0.0) for le in self._bucket_les])

        if self._size >= len(self._timestamps):
            self._grow_capacity()

        sum_val = sample.sum or 0.0
        count_val = sample.count or 0.0

        # Fast path: in-order append (99.9% of cases)
        if self._size == 0 or timestamp_ns >= self._timestamps[self._size - 1]:
            self._insert_at(
                self._size,
                timestamp_ns=timestamp_ns,
                sum_val=sum_val,
                count_val=count_val,
                bucket_row=bucket_row,
            )
            self._size += 1
            return

        # Slow path: out-of-order insert (fully vectorized)
        idx = self._size - 1
        while idx > 0 and self._timestamps[idx - 1] > timestamp_ns:
            idx -= 1

        self._shift_right(idx)
        self._insert_at(
            idx,
            timestamp_ns=timestamp_ns,
            sum_val=sum_val,
            count_val=count_val,
            bucket_row=bucket_row,
        )
        self._size += 1

    def _init_bucket_schema(self, buckets: dict[str, float]) -> None:
        """Initialize bucket schema from the first sample's bucket keys."""
        self._bucket_les = tuple(sorted(buckets.keys(), key=_bucket_sort_key))
        self._bucket_counts = np.empty(
            (_INITIAL_CAPACITY, len(self._bucket_les)), dtype=np.float64
        )

    def _warn_on_schema_mismatch(self, buckets: dict[str, float]) -> None:
        """Log warnings if sample buckets don't match the initialized schema."""
        sample_bucket_keys = set(buckets.keys())
        expected_bucket_keys = set(self._bucket_les or ())

        if sample_bucket_keys == expected_bucket_keys:
            return

        missing_in_sample = expected_bucket_keys - sample_bucket_keys
        extra_in_sample = sample_bucket_keys - expected_bucket_keys

        if missing_in_sample:
            self._logger.warning(
                f"Histogram bucket schema mismatch: sample is missing buckets {sorted(missing_in_sample)}. "
                f"Missing buckets will be filled with 0.0. Expected schema: {self._bucket_les}"
            )

        if extra_in_sample:
            self._logger.warning(
                f"Histogram bucket schema mismatch: sample has unexpected buckets {sorted(extra_in_sample)}. "
                f"Extra buckets will be ignored. Expected schema: {self._bucket_les}"
            )

    def _grow_capacity(self) -> None:
        """Double the capacity of all parallel storage arrays."""
        new_cap = len(self._timestamps) * 2
        new_ts = np.empty(new_cap, dtype=np.int64)
        new_sums = np.empty(new_cap, dtype=np.float64)
        new_counts = np.empty(new_cap, dtype=np.float64)
        new_buckets = np.empty((new_cap, len(self._bucket_les)), dtype=np.float64)
        new_ts[: self._size] = self._timestamps[: self._size]
        new_sums[: self._size] = self._sums[: self._size]
        new_counts[: self._size] = self._counts[: self._size]
        new_buckets[: self._size] = self._bucket_counts[: self._size]
        self._timestamps = new_ts
        self._sums = new_sums
        self._counts = new_counts
        self._bucket_counts = new_buckets

    def _shift_right(self, idx: int) -> None:
        """Shift all parallel arrays right by 1 starting at idx."""
        self._timestamps[idx + 1 : self._size + 1] = self._timestamps[idx : self._size]
        self._sums[idx + 1 : self._size + 1] = self._sums[idx : self._size]
        self._counts[idx + 1 : self._size + 1] = self._counts[idx : self._size]
        self._bucket_counts[idx + 1 : self._size + 1] = self._bucket_counts[
            idx : self._size
        ]

    def _insert_at(
        self,
        idx: int,
        *,
        timestamp_ns: int,
        sum_val: float,
        count_val: float,
        bucket_row: np.ndarray,
    ) -> None:
        """Write a snapshot tuple into all parallel arrays at idx."""
        self._timestamps[idx] = timestamp_ns
        self._sums[idx] = sum_val
        self._counts[idx] = count_val
        self._bucket_counts[idx] = bucket_row

    def get_bucket_dict(self, idx: int) -> dict[str, float]:
        """Get bucket snapshot at index as dict for percentile estimation.

        Retrieves the histogram bucket counts at a specific time index, formatted
        as a dict for use in percentile computation algorithms. The bucket counts
        are cumulative (Prometheus le="less than or equal" semantics).

        Args:
            idx: Index of the snapshot to retrieve (0 to len-1)

        Returns:
            Dict mapping bucket upper bounds (le strings) to cumulative counts.
            Empty dict if no buckets initialized yet.

        Example:
            >>> # After appending histogram samples
            >>> bucket_dict = histogram_ts.get_bucket_dict(0)
            >>> bucket_dict
            {"0.01": 10, "0.1": 45, "1.0": 98, "+Inf": 100}
        """
        if self._bucket_les is None or self._bucket_counts is None:
            return {}
        return dict(zip(self._bucket_les, self._bucket_counts[idx], strict=True))

    @property
    def timestamps(self) -> NDArray[np.int64]:
        """Nanosecond timestamps for each histogram snapshot, in sorted order.

        Returns:
            1D array of shape (size,) with monotonically increasing timestamps.
            View of underlying storage (no copy).
        """
        return self._timestamps[: self._size]

    @property
    def sums(self) -> NDArray[np.float64]:
        """Cumulative sum of observed values at each timestamp.

        For Prometheus histograms, this is the total sum of all observations
        seen since the metric was created (or last server restart). Use deltas
        between snapshots to get sum for a specific time period.

        Returns:
            1D array of shape (size,) with cumulative sums. View of underlying storage.
        """
        return self._sums[: self._size]

    @property
    def counts(self) -> NDArray[np.float64]:
        """Cumulative count of observations at each timestamp.

        For Prometheus histograms, this is the total number of observations
        recorded since the metric was created (or last server restart). Use deltas
        between snapshots to get observation count for a specific time period.

        Returns:
            1D array of shape (size,) with cumulative counts. View of underlying storage.
        """
        return self._counts[: self._size]

    @property
    def bucket_les(self) -> tuple[str, ...]:
        """Sorted bucket boundary strings (e.g., ('0.01', '0.1', '+Inf')).

        Bucket boundaries are initialized on first append and remain fixed.
        Sorted in ascending numeric order with '+Inf' last.

        Returns:
            Tuple of bucket upper bound strings. Empty tuple if no data appended yet.
        """
        return self._bucket_les or ()

    @property
    def bucket_counts(self) -> NDArray[np.float64]:
        """2D array of cumulative bucket counts, shape (n_snapshots, n_buckets).

        Each row represents one histogram snapshot with cumulative counts for
        all buckets (Prometheus le="less than or equal" semantics).

        Returns:
            2D array where bucket_counts[i, j] is the cumulative count for
            bucket j at snapshot i. Empty array if no data appended yet.
            View of underlying storage (no copy).
        """
        if self._bucket_counts is None:
            return np.empty((0, 0), dtype=np.float64)
        return self._bucket_counts[: self._size]

    def __len__(self) -> int:
        return self._size

    def get_indices_for_filter(
        self, time_filter: TimeRangeFilter | None
    ) -> tuple[int | None, int]:
        """Get (reference_idx, final_idx) indices for time-filtered histogram processing.

        For histogram metrics (cumulative counters), we need:
        - reference_idx: Last sample BEFORE profiling period (baseline for deltas)
        - final_idx: Last sample WITHIN profiling period (end point for deltas)

        This enables delta calculation: final_value - reference_value gives the
        change during the profiling period, excluding warmup and end buffer.

        Uses np.searchsorted for O(log n) binary search on sorted timestamps.

        Args:
            time_filter: Optional time range filter for profiling period bounds

        Returns:
            Tuple of (reference_idx, final_idx) where:
            - reference_idx: Index of last sample < start_ns, or None if none exists
            - final_idx: Index of last sample <= end_ns, or last index if no end bound
        """
        reference_idx = None
        final_idx = self._size - 1

        if time_filter is not None:
            timestamps = self.timestamps
            if time_filter.start_ns is not None:
                # Find last point < start_ns (reference point for delta calculation)
                insert_pos = int(
                    np.searchsorted(timestamps, time_filter.start_ns, side="left")
                )
                reference_idx = insert_pos - 1 if insert_pos > 0 else None
            if time_filter.end_ns is not None:
                # Find last point <= end_ns
                insert_pos = int(
                    np.searchsorted(timestamps, time_filter.end_ns, side="right")
                )
                final_idx = insert_pos - 1 if insert_pos > 0 else self._size - 1

        return reference_idx, final_idx

    def get_observation_rates(
        self, time_filter: TimeRangeFilter | None = None
    ) -> NDArray[np.float64]:
        """Get point-to-point observation rates (observations per second).

        Computes instantaneous observation rates between consecutive histogram
        snapshots by dividing count deltas by time deltas. This provides insight
        into how request arrival rate varies over time.

        Zero-duration intervals (consecutive samples with same timestamp) are
        automatically filtered out to avoid division by zero. This can occur
        when metrics are scraped faster than the server updates them.

        Uses fully vectorized NumPy operations for efficiency on large time series.

        Args:
            time_filter: Optional time range to compute rates within

        Returns:
            Array of observation rates in observations/second, one per valid interval.
            Empty array if fewer than 2 samples or all intervals have zero duration.
        """
        ref_idx, final_idx = self.get_indices_for_filter(time_filter)
        start_idx = ref_idx if ref_idx is not None else 0

        ts = self.timestamps[start_idx : final_idx + 1]
        counts = self.counts[start_idx : final_idx + 1]

        if len(ts) < 2:
            return np.array([], dtype=np.float64)

        count_deltas = np.diff(counts)
        time_deltas_ns = np.diff(ts)

        # Filter out zero-duration intervals
        valid_mask = time_deltas_ns > 0
        if not np.any(valid_mask):
            return np.array([], dtype=np.float64)

        time_deltas_s = time_deltas_ns[valid_mask] / NANOS_PER_SECOND
        return count_deltas[valid_mask] / time_deltas_s

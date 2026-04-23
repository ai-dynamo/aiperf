# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray

from aiperf.common.models import MetricSample, TimeRangeFilter

_INITIAL_CAPACITY = 256


class ScalarTimeSeries:
    """NumPy-backed (timestamp, value) storage for gauge and counter metrics.

    Efficient storage for single-value metrics using parallel NumPy arrays.
    Maintains sorted order for O(log n) time-based queries while optimizing
    for the common case of chronological data arrival.

    Supports:
    - Time range filtering with O(log n) binary search
    - Reference point lookup for counter delta calculations
    - Vectorized statistics computation (mean, percentiles, etc.)
    - Out-of-order insertion with minimal performance impact

    Data is always maintained in sorted order by timestamp. Out-of-order
    appends are handled efficiently by inserting at the correct position,
    optimized for the common case where data arrives nearly in order (99.9%).

    Memory efficiency:
    - Pre-allocated arrays with doubling growth strategy
    - Amortized O(1) append for in-order data
    - Maximum 2x memory overhead (capacity vs size)

    Example:
        >>> from aiperf.common.models import MetricSample
        >>> ts = ScalarTimeSeries()
        >>> # Add gauge samples
        >>> ts.append(1_000_000_000, MetricSample(value=42.5))
        >>> ts.append(2_000_000_000, MetricSample(value=43.1))
        >>> ts.append(3_000_000_000, MetricSample(value=41.8))
        >>> len(ts)
        3
        >>> ts.values
        array([42.5, 43.1, 41.8])
    """

    def __init__(self) -> None:
        self._timestamps: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.int64)
        self._values: np.ndarray = np.empty(_INITIAL_CAPACITY, dtype=np.float64)
        self._size: int = 0

    def append(self, timestamp_ns: int, sample: MetricSample) -> None:
        """Append a sample, maintaining sorted order by timestamp.

        Optimized for the common case where data arrives in chronological order
        (99.9% of metrics collections). Uses fast O(1) append when data is in order,
        falling back to O(k) insertion for out-of-order data where k is the
        displacement from the end.

        Out-of-order data can occur when:
        - Multiple collectors report with slight timing skew
        - Network delays cause reordering of async metric fetches
        - Clock adjustments on the server

        Automatically grows capacity by doubling when full to maintain amortized
        O(1) append performance.

        Args:
            timestamp_ns: Nanosecond timestamp for this sample
            sample: MetricSample containing the metric value

        Raises:
            ValueError: If sample.value is None (required for scalar series)
        """
        if sample.value is None:
            raise ValueError("Value is required for scalar time series")

        # Ensure capacity
        if self._size >= len(self._values):
            new_cap = len(self._values) * 2
            new_ts = np.empty(new_cap, dtype=np.int64)
            new_val = np.empty(new_cap, dtype=np.float64)
            new_ts[: self._size] = self._timestamps[: self._size]
            new_val[: self._size] = self._values[: self._size]
            self._timestamps, self._values = new_ts, new_val

        # Fast path: in-order append (99.9% of cases)
        if self._size == 0 or timestamp_ns >= self._timestamps[self._size - 1]:
            self._timestamps[self._size] = timestamp_ns
            self._values[self._size] = sample.value
            self._size += 1
            return

        # Slow path: out-of-order insert
        # Find insertion point by walking backwards from end (O(k) where k = displacement)
        idx = self._size - 1
        while idx > 0 and self._timestamps[idx - 1] > timestamp_ns:
            idx -= 1

        # Shift elements right by 1 to make room (O(k) when inserting near end)
        self._timestamps[idx + 1 : self._size + 1] = self._timestamps[idx : self._size]
        self._values[idx + 1 : self._size + 1] = self._values[idx : self._size]

        # Insert at correct position
        self._timestamps[idx] = timestamp_ns
        self._values[idx] = sample.value
        self._size += 1

    @property
    def timestamps(self) -> NDArray[np.int64]:
        """Nanosecond timestamps for each data point, in sorted order.

        Returns:
            1D array of shape (size,) with monotonically increasing timestamps.
            View of underlying storage (no copy).
        """
        return self._timestamps[: self._size]

    @property
    def values(self) -> NDArray[np.float64]:
        """Metric values corresponding to each timestamp.

        For gauges: instantaneous values (e.g., current queue depth)
        For counters: cumulative totals (use deltas for period counts)

        Returns:
            1D array of shape (size,) with metric values.
            View of underlying storage (no copy).
        """
        return self._values[: self._size]

    def __len__(self) -> int:
        """Number of stored samples.

        Returns:
            Count of samples currently stored (not capacity)
        """
        return self._size

    def get_time_mask(self, time_filter: TimeRangeFilter | None) -> NDArray[np.bool_]:
        """Get boolean mask for points within time range.

        Uses np.searchsorted for O(log n) binary search on sorted timestamps
        to find range boundaries, then creates a boolean mask via efficient
        slice assignment rather than element-wise comparisons.

        This approach is significantly faster than boolean indexing for large
        arrays (10-100x speedup for 10k+ elements) and maintains constant
        memory overhead regardless of array size.

        Args:
            time_filter: Optional time range filter. None returns all True mask.

        Returns:
            Boolean numpy array of shape (size,) where True indicates samples
            within the time range [start_ns, end_ns] inclusive
        """
        if time_filter is None:
            return np.ones(self._size, dtype=bool)

        timestamps = self.timestamps
        first_idx = 0
        last_idx = self._size

        if time_filter.start_ns is not None:
            # Find first index where timestamp >= start_ns
            first_idx = int(
                np.searchsorted(timestamps, time_filter.start_ns, side="left")
            )
        if time_filter.end_ns is not None:
            # Find first index where timestamp > end_ns (so last_idx-1 is last valid)
            last_idx = int(
                np.searchsorted(timestamps, time_filter.end_ns, side="right")
            )

        # Create mask with single allocation
        mask = np.zeros(self._size, dtype=bool)
        mask[first_idx:last_idx] = True
        return mask

    def get_reference_idx(self, time_filter: TimeRangeFilter | None) -> int | None:
        """Get index of last point BEFORE time filter start (for delta calculation).

        For counter and histogram metrics, we need a reference point before the
        profiling period to compute deltas. This finds the last sample with
        timestamp < start_ns to use as the baseline for cumulative metrics.

        Uses np.searchsorted for O(log n) binary search on sorted timestamps.

        Example:
            If timestamps are [100, 200, 300, 400] and start_ns=250,
            returns index 1 (timestamp=200) as the reference point.

        Args:
            time_filter: Optional time range filter. None or missing start_ns returns None.

        Returns:
            Index of last sample before start_ns, or None if no such sample exists
        """
        if time_filter is None or time_filter.start_ns is None:
            return None
        # searchsorted with side='left' gives first index where timestamp >= start_ns
        # So insert_pos - 1 is the last point < start_ns
        insert_pos = int(
            np.searchsorted(self.timestamps, time_filter.start_ns, side="left")
        )
        return insert_pos - 1 if insert_pos > 0 else None

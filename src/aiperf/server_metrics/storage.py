# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    TimeRangeFilter,
)

# =============================================================================
# Aggregation (Multi-metric time series container)
# =============================================================================


class ServerMetricsTimeSeries:
    """Unified per-metric storage for server metrics from a single endpoint.

    Container for all metrics scraped from one Prometheus endpoint over time.
    Each metric (identified by name + labels) gets its own time series with
    type-appropriate storage.

    Design:
    - Single dict mapping ServerMetricKey -> ServerMetricEntry
    - Each MetricEntry is self-describing (type, description, data)
    - No global alignment, no NaN padding for sparse data
    - NumPy arrays for memory efficiency and vectorized operations
    - Time filtering supported via O(log n) index lookups

    Storage by type:
    - Gauges/Counters: ScalarTimeSeries (timestamp, value) pairs
    - Histograms: HistogramTimeSeries (timestamp, sum, count, buckets)

    Tracks dual timelines:
    - Fetch timeline: All HTTP requests (including duplicates where metrics unchanged)
    - Update timeline: Only unique updates where metric values changed

    This separation enables:
    - Accurate fetch latency statistics (endpoint reliability monitoring)
    - Accurate update interval statistics (server metric update frequency)
    - Storage optimization (don't duplicate unchanged metric values)

    Example:
        >>> ts = ServerMetricsTimeSeries()
        >>> # Add first record
        >>> record1 = ServerMetricsRecord(timestamp_ns=1000, metrics={...})
        >>> ts.append_snapshot(record1)
        >>> # Add duplicate (same metrics)
        >>> record2 = ServerMetricsRecord(timestamp_ns=2000, is_duplicate=True)
        >>> ts.append_snapshot(record2)
        >>> ts._total_fetch_count  # 2 fetches
        2
        >>> ts._unique_update_count  # 1 unique update
        1
    """

    def __init__(self) -> None:
        self.metrics: dict[ServerMetricKey, ServerMetricEntry] = {}
        # Timestamps for unique updates only (when metrics changed)
        self.first_update_ns: int = 0
        self.last_update_ns: int = 0
        self._unique_update_count: int = 0
        self._unique_update_timestamps: list[
            int
        ] = []  # All unique update timestamps (for interval calc)
        # Timestamps for all fetches (including duplicates)
        self.first_fetch_ns: int = 0
        self.last_fetch_ns: int = 0
        self._total_fetch_count: int = 0
        self._fetch_latencies_ns: list[int] = []

    @property
    def _update_intervals_ns(self) -> list[int]:
        """Compute intervals between unique updates from sorted timestamps.

        Calculated on-demand to handle out-of-order data correctly by sorting
        timestamps before computing intervals. This ensures accurate median
        interval calculation even when records arrive out of chronological order.

        Used for computing median update interval statistics to assess how
        frequently the server updates its metrics (independent of how often
        we scrape them).

        Returns:
            List of intervals in nanoseconds between consecutive unique updates.
            Empty list if fewer than 2 unique updates recorded.
        """
        if len(self._unique_update_timestamps) < 2:
            return []

        # Sort timestamps and compute intervals
        sorted_timestamps = sorted(self._unique_update_timestamps)
        return [
            int(sorted_timestamps[i] - sorted_timestamps[i - 1])
            for i in range(1, len(sorted_timestamps))
        ]

    def append_snapshot(self, record: ServerMetricsRecord) -> None:
        """Append all metrics from a ServerMetricsRecord.

        Extracts gauge, counter, and histogram metrics from the record and
        stores them in the appropriate time series. Handles both unique updates
        (metrics changed) and duplicate records (same metric values as previous).

        For duplicate records (where metrics haven't changed), only fetch
        timestamps and latencies are tracked - metric data is not duplicated.
        This optimizes storage while maintaining accurate fetch statistics for
        monitoring endpoint reliability.

        Duplicate detection is performed by the data collector via response hash
        comparison before parsing, making this a lightweight operation.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics and metadata
        """
        timestamp_ns = record.timestamp_ns

        if not record.metrics:
            return

        # Always track fetch timestamps and latencies
        if self._total_fetch_count == 0 or timestamp_ns < self.first_fetch_ns:
            self.first_fetch_ns = timestamp_ns
        if timestamp_ns > self.last_fetch_ns:
            self.last_fetch_ns = timestamp_ns
        self._total_fetch_count += 1
        if record.endpoint_latency_ns is not None:
            self._fetch_latencies_ns.append(record.endpoint_latency_ns)

        # Track unique updates (only for non-duplicates) for metadata/statistics
        # But store ALL samples (including duplicates) for consistent timeslice boundaries
        if not record.is_duplicate:
            # Track unique update timestamps (handles out-of-order data with min/max)
            if self._unique_update_count == 0:
                self.first_update_ns = timestamp_ns
                self.last_update_ns = timestamp_ns
            else:
                # Use min/max to handle out-of-order arrivals
                self.first_update_ns = min(self.first_update_ns, timestamp_ns)
                self.last_update_ns = max(self.last_update_ns, timestamp_ns)

            self._unique_update_count += 1
            # Track this unique update timestamp for interval calculation later
            self._unique_update_timestamps.append(timestamp_ns)

        # Append to time series (for all records, including duplicates)
        for metric_name, metric_family in record.metrics.items():
            for sample in metric_family.samples:
                key = ServerMetricKey.from_name_and_labels(metric_name, sample.labels)

                if key not in self.metrics:
                    self.metrics[key] = ServerMetricEntry.from_metric_family(
                        metric_family
                    )
                self.metrics[key].data.append(timestamp_ns, sample)

    def __len__(self) -> int:
        """Number of unique metric updates (excluding duplicates).

        Returns:
            Count of times metrics actually changed, not total fetch count.
            Used for computing update interval statistics.
        """
        return self._unique_update_count


# =============================================================================
# Hierarchy & Results (Hierarchy and processing results)
# =============================================================================


class ServerMetricsHierarchy:
    """Hierarchical storage container for multi-endpoint server metrics.

    Top-level storage structure organizing metrics by endpoint URL. Enables
    collecting from multiple Prometheus endpoints simultaneously (e.g., multiple
    inference servers in a distributed deployment).

    Structure:
    {
        "http://localhost:8081/metrics": ServerMetricsTimeSeries,
        "http://localhost:8082/metrics": ServerMetricsTimeSeries
    }

    Each endpoint gets its own ServerMetricsTimeSeries which contains all metrics
    scraped from that endpoint over time. Endpoints are automatically created
    on first record arrival.

    Example:
        >>> hierarchy = ServerMetricsHierarchy()
        >>> # Add record from first endpoint
        >>> record1 = ServerMetricsRecord(endpoint_url="http://server1/metrics", ...)
        >>> hierarchy.add_record(record1)
        >>> # Add record from second endpoint
        >>> record2 = ServerMetricsRecord(endpoint_url="http://server2/metrics", ...)
        >>> hierarchy.add_record(record2)
        >>> len(hierarchy.endpoints)
        2
    """

    def __init__(self) -> None:
        self.endpoints: dict[str, ServerMetricsTimeSeries] = {}

    def add_record(self, record: ServerMetricsRecord) -> None:
        """Add server metrics record to hierarchical storage.

        Automatically creates new endpoints as needed. Descriptions are stored
        in the ServerMetricEntry alongside the time series data.
        """
        url = record.endpoint_url

        if url not in self.endpoints:
            self.endpoints[url] = ServerMetricsTimeSeries()

        self.endpoints[url].append_snapshot(record)


# =============================================================================
# Time Series Storage Classes
# =============================================================================

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

        # Initialize bucket schema on first append
        if self._bucket_les is None:
            self._bucket_les = tuple(
                sorted(sample.buckets.keys(), key=_bucket_sort_key)
            )
            n_buckets = len(self._bucket_les)
            self._bucket_counts = np.empty(
                (_INITIAL_CAPACITY, n_buckets), dtype=np.float64
            )

        # Validate bucket schema consistency
        sample_bucket_keys = set(sample.buckets.keys())
        expected_bucket_keys = set(self._bucket_les)

        if sample_bucket_keys != expected_bucket_keys:
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

        # Convert dict to row (order matches _bucket_les, 0.0 for missing buckets)
        bucket_row = np.array([sample.buckets.get(le, 0.0) for le in self._bucket_les])

        # Ensure capacity for all arrays
        if self._size >= len(self._timestamps):
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

        # Fast path: in-order append (99.9% of cases)
        if self._size == 0 or timestamp_ns >= self._timestamps[self._size - 1]:
            self._timestamps[self._size] = timestamp_ns
            self._sums[self._size] = sample.sum or 0.0
            self._counts[self._size] = sample.count or 0.0
            self._bucket_counts[self._size] = bucket_row
            self._size += 1
            return

        # Slow path: out-of-order insert (fully vectorized)
        idx = self._size - 1
        while idx > 0 and self._timestamps[idx - 1] > timestamp_ns:
            idx -= 1

        # Shift all arrays right by 1 (single vectorized op per array)
        self._timestamps[idx + 1 : self._size + 1] = self._timestamps[idx : self._size]
        self._sums[idx + 1 : self._size + 1] = self._sums[idx : self._size]
        self._counts[idx + 1 : self._size + 1] = self._counts[idx : self._size]
        self._bucket_counts[idx + 1 : self._size + 1] = self._bucket_counts[
            idx : self._size
        ]

        # Insert at correct position
        self._timestamps[idx] = timestamp_ns
        self._sums[idx] = sample.sum or 0.0
        self._counts[idx] = sample.count or 0.0
        self._bucket_counts[idx] = bucket_row
        self._size += 1

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


# =============================================================================
# Metric Key and Entry (Unified storage structure)
# =============================================================================


class ServerMetricKey(NamedTuple):
    """Structured key for metric identification with labels.

    Immutable, hashable key for uniquely identifying a metric time series.
    Combines metric name with label dimensions to distinguish between series
    (e.g., http_requests_total with method=GET vs method=POST are different series).

    Uses immutable tuple of tuples for labels to be hashable as dict key.
    Labels are stored as sorted (key, value) pairs for consistent ordering,
    ensuring that the same labels in different order produce identical keys.

    This enables efficient dict-based storage and O(1) metric lookup by name+labels.

    Args:
        name: Prometheus metric name (e.g., "http_requests_total", "cache_hit_ratio")
        labels: Sorted tuple of (key, value) label pairs for metric dimensions

    Examples:
        >>> # Metric with no labels
        >>> key1 = ServerMetricKey("vllm:kv_cache_usage_perc", ())

        >>> # Metric with labels
        >>> key2 = ServerMetricKey("http_requests_total", (("method", "GET"), ("status", "200")))

        >>> # Create from dict (labels get sorted automatically)
        >>> key3 = ServerMetricKey.from_name_and_labels(
        ...     "http_requests_total",
        ...     {"status": "200", "method": "GET"}  # Order doesn't matter
        ... )
        >>> key3.labels
        (('method', 'GET'), ('status', '200'))  # Sorted by key
    """

    name: str
    labels: tuple[tuple[str, str], ...] = ()

    @property
    def labels_dict(self) -> dict[str, str] | None:
        """Convert labels tuple to dict for easy access.

        Returns:
            Dict mapping label keys to values, or None if no labels.
            Useful for passing to statistics models and export functions.
        """
        return dict(self.labels) if self.labels else None

    @classmethod
    def from_name_and_labels(cls, name: str, labels: dict[str, str] | None) -> Self:
        """Create ServerMetricKey from metric name and optional labels dict.

        Convenience constructor that handles dict-to-tuple conversion and sorting.
        Ensures consistent key generation regardless of dict iteration order.

        Args:
            name: Prometheus metric name
            labels: Optional dict of label key-value pairs

        Returns:
            ServerMetricKey with labels sorted by key for consistent hashing
        """
        if not labels:
            return cls(name, ())
        sorted_labels = tuple(sorted(labels.items()))
        return cls(name, sorted_labels)


@dataclass(slots=True)
class ServerMetricEntry:
    """Unified container for server metric type, description, and time series data.

    Self-describing storage for a single metric time series. Combines metadata
    (type, description) with the actual time series data in one structure,
    eliminating the need for separate metadata lookups.

    This design enables:
    - Type-appropriate statistics computation without external type info
    - Description propagation through the processing pipeline
    - Polymorphic storage (ScalarTimeSeries or HistogramTimeSeries based on type)

    Args:
        metric_type: Prometheus metric type (GAUGE, COUNTER, or HISTOGRAM)
        description: Human-readable description from Prometheus HELP text
        data: Type-appropriate time series storage (ScalarTimeSeries for
              gauge/counter, HistogramTimeSeries for histogram)
    """

    metric_type: PrometheusMetricType
    description: str
    data: ScalarTimeSeries | HistogramTimeSeries

    @classmethod
    def from_metric_family(cls, metric_family: MetricFamily) -> Self:
        """Create a ServerMetricEntry from a MetricFamily.

        Factory method that automatically selects the appropriate time series
        storage type based on the metric type. Gauges and counters use
        ScalarTimeSeries, histograms use HistogramTimeSeries.

        Args:
            metric_family: MetricFamily from parsed Prometheus metrics containing
                          type, description, and initial samples

        Returns:
            ServerMetricEntry with appropriate storage initialized
        """
        return cls(
            metric_type=metric_family.type,
            description=metric_family.description,
            data=ScalarTimeSeries()
            if metric_family.type
            in (PrometheusMetricType.GAUGE, PrometheusMetricType.COUNTER)
            else HistogramTimeSeries(),
        )

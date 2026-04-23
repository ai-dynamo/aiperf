# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.models import ServerMetricsRecord
from aiperf.server_metrics._storage_entry import ServerMetricEntry, ServerMetricKey
from aiperf.server_metrics._storage_histogram import (
    HistogramTimeSeries,
    _bucket_sort_key,
)
from aiperf.server_metrics._storage_scalar import (
    _INITIAL_CAPACITY,
    ScalarTimeSeries,
)

__all__ = [
    "HistogramTimeSeries",
    "ScalarTimeSeries",
    "ServerMetricEntry",
    "ServerMetricKey",
    "ServerMetricsHierarchy",
    "ServerMetricsTimeSeries",
    "_INITIAL_CAPACITY",
    "_bucket_sort_key",
]


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

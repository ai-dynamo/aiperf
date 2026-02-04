# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter


class TelemetryMetrics(AIPerfBaseModel):
    """GPU metrics collected at a single point in time.

    All fields are optional to handle cases where specific metrics are not available
    from the DCGM exporter or are filtered out due to invalid values.

    Custom metrics from user-provided CSV files are supported via extra='allow'.
    """

    model_config = ConfigDict(extra="allow")

    gpu_power_usage: float | None = Field(
        default=None, description="Current GPU power usage in W"
    )
    energy_consumption: float | None = Field(
        default=None, description="Cumulative energy consumption in MJ"
    )
    gpu_utilization: float | None = Field(
        default=None, description="GPU utilization percentage (0-100)"
    )
    gpu_memory_used: float | None = Field(
        default=None, description="GPU memory used in GB"
    )
    gpu_temperature: float | None = Field(
        default=None, description="GPU temperature in °C"
    )
    xid_errors: float | None = Field(
        default=None, description="Value of the last XID error encountered"
    )
    power_violation: float | None = Field(
        default=None,
        description="Throttling duration due to power constraints in microseconds",
    )


class GpuMetadata(AIPerfBaseModel):
    """Static metadata for a GPU that doesn't change over time.

    This is stored once per GPU and referenced by all telemetry data points
    to avoid duplicating metadata in every time-series entry.
    """

    gpu_index: int = Field(
        description="GPU index on this node (0, 1, 2, etc.) - used for display ordering"
    )
    gpu_uuid: str = Field(
        description="Unique GPU identifier (e.g., 'GPU-ef6ef310-...') - primary key for data"
    )
    gpu_model_name: str = Field(
        description="GPU model name (e.g., 'NVIDIA RTX 6000 Ada Generation')"
    )
    pci_bus_id: str | None = Field(
        default=None, description="PCI Bus ID (e.g., '00000000:02:00.0')"
    )
    device: str | None = Field(
        default=None, description="Device identifier (e.g., 'nvidia0')"
    )
    hostname: str | None = Field(
        default=None, description="Hostname where GPU is located"
    )
    namespace: str | None = Field(
        default=None, description="Namespace where the GPU is located (kubernetes only)"
    )
    pod_name: str | None = Field(
        default=None, description="Pod name where the GPU is located (kubernetes only)"
    )


class TelemetryRecord(GpuMetadata):
    """Single telemetry data point from GPU monitoring.

    This record contains all telemetry data for one GPU at one point in time,
    along with metadata to identify the source DCGM endpoint and specific GPU.
    Used for hierarchical storage: dcgm_url -> gpu_uuid -> time series data.

    Inherits from GpuMetadata to avoid duplicating metadata fields.
    """

    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when telemetry was collected (time_ns)"
    )
    dcgm_url: str = Field(
        description="Source DCGM endpoint URL (e.g., 'http://node1:9401/metrics')"
    )
    telemetry_data: TelemetryMetrics = Field(
        description="GPU metrics snapshot collected at this timestamp"
    )


class GpuTelemetrySnapshot(AIPerfBaseModel):
    """All metrics for a single GPU at one point in time.

    Groups all metric values collected during a single collection cycle,
    eliminating timestamp duplication across individual metrics.
    """

    timestamp_ns: int = Field(description="Collection timestamp for all metrics")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="All metric values at this timestamp"
    )


class GpuMetricTimeSeries:
    """NumPy-backed columnar storage for GPU telemetry.

    Stores timestamps once with separate value arrays per metric.
    Metric schema is determined on first snapshot - all subsequent snapshots
    must contain the same metrics (DCGM metrics are static per run).

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
            - Metric schema is determined on first snapshot. All subsequent snapshots
              must contain the same metrics (DCGM metrics are static per run).
            - Data kept sorted by timestamp (O(1) in-order, O(k) out-of-order).
        """
        if self._size >= self._capacity:
            self._grow()

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

        # Initialize metric arrays on first snapshot (schema determined here)
        if not self._metrics:
            for name in metrics:
                self._metrics[name] = np.empty(self._capacity, dtype=np.float64)

        # Set values for all metrics at insert position
        for name, value in metrics.items():
            self._metrics[name][insert_pos] = value

        self._size += 1

    def _grow(self) -> None:
        """Double capacity of all arrays."""
        new_capacity = self._capacity * 2

        # Grow timestamps
        new_ts = np.empty(new_capacity, dtype=np.int64)
        new_ts[: self._size] = self._timestamps[: self._size]
        self._timestamps = new_ts

        # Grow each metric array
        for name, old_arr in self._metrics.items():
            new_arr = np.empty(new_capacity, dtype=np.float64)
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

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement

        Returns:
            MetricResult with min/max/avg/percentiles computed from all values

        Raises:
            NoMetricValue: If no data for this metric
        """
        arr = self.get_metric_array(metric_name)
        if arr is None or len(arr) == 0:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )

        # Vectorized stats computation
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
        std_dev = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            avg=float(np.mean(arr)),
            std=std_dev,
            count=len(arr),
            current=float(arr[-1]),
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

        For gauges: Uses vectorized NumPy on filtered array (np.mean, np.std, np.percentile)
        For counters: Computes delta from reference point before profiling start

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
            NoMetricValue: If no data for this metric or no data in filtered range
        """
        arr = self.get_metric_array(metric_name)
        if arr is None or len(arr) == 0:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )

        # Common: apply time filter
        time_mask = self.get_time_mask(time_filter)
        filtered = arr[time_mask]
        if len(filtered) == 0:
            raise NoMetricValue(f"No data in time range for metric '{metric_name}'")

        if is_counter:
            # Counter: compute delta from baseline
            reference_idx = self.get_reference_idx(time_filter)
            reference_value = (
                arr[reference_idx] if reference_idx is not None else filtered[0]
            )
            raw_delta = float(filtered[-1] - reference_value)

            # Handle counter resets (e.g., DCGM restart) by clamping to 0
            delta = max(raw_delta, 0.0)

            # Counters report a single delta value, not a distribution
            return MetricResult(
                tag=tag,
                header=header,
                unit=unit,
                avg=delta,
            )

        # Gauge: vectorized stats on filtered data
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            filtered, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        # Use sample std (ddof=1) for unbiased estimate; 0 for single sample
        std_dev = float(np.std(filtered, ddof=1)) if len(filtered) > 1 else 0.0

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.min(filtered)),
            max=float(np.max(filtered)),
            avg=float(np.mean(filtered)),
            std=std_dev,
            count=len(filtered),
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

    def __len__(self) -> int:
        """Return the number of snapshots in the time series."""
        return self._size


class GpuTelemetryData(AIPerfBaseModel):
    """Complete telemetry data for one GPU: metadata + grouped metric time series.

    This combines static GPU information with dynamic time-series data,
    providing the complete picture for one GPU's telemetry using efficient columnar storage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: GpuMetadata = Field(description="Static GPU information")
    time_series: GpuMetricTimeSeries = Field(
        default_factory=GpuMetricTimeSeries,
        description="Columnar time series for all metrics",
        exclude=True,  # Numpy arrays are not serializable by default
    )

    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record as a grouped snapshot.

        Args:
            record: New telemetry data point from DCGM collector

        Note: Groups all metric values from the record into a single snapshot
        """
        metric_mapping = record.telemetry_data.model_dump()
        valid_metrics = {k: v for k, v in metric_mapping.items() if v is not None}
        if valid_metrics:
            self.time_series.append_snapshot(valid_metrics, record.timestamp_ns)

    def get_metric_result(
        self,
        metric_name: str,
        tag: str,
        header: str,
        unit: str,
        time_filter: TimeRangeFilter | None = None,
        is_counter: bool = False,
    ) -> MetricResult:
        """Get MetricResult for a specific metric with optional time filtering.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement
            time_filter: Optional time range filter to exclude warmup/cooldown periods
            is_counter: If True, compute delta from baseline instead of statistics

        Returns:
            MetricResult with statistical summary for the specified metric
        """
        if time_filter is not None or is_counter:
            return self.time_series.to_metric_result_filtered(
                metric_name, tag, header, unit, time_filter, is_counter
            )
        return self.time_series.to_metric_result(metric_name, tag, header, unit)


class TelemetryHierarchy(AIPerfBaseModel):
    """Hierarchical storage: dcgm_url -> gpu_uuid -> complete GPU telemetry data.

    This provides the requested hierarchical structure while maintaining efficient
    access patterns for both real-time display and final aggregation.

    Structure:
    {
        "http://node1:9401/metrics": {
            "GPU-ef6ef310-...": GpuTelemetryData(metadata + time series),
            "GPU-a1b2c3d4-...": GpuTelemetryData(metadata + time series)
        },
        "http://node2:9401/metrics": {
            "GPU-f5e6d7c8-...": GpuTelemetryData(metadata + time series)
        }
    }
    """

    dcgm_endpoints: dict[str, dict[str, GpuTelemetryData]] = Field(
        default_factory=dict,
        description="Nested dict: dcgm_url -> gpu_uuid -> telemetry data",
    )

    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record to hierarchical storage.

        Args:
            record: New telemetry data from GPU monitoring

        Note: Automatically creates hierarchy levels as needed:
        - New DCGM endpoints get empty GPU dict
        - New GPUs get initialized with metadata and empty metrics
        """

        if record.dcgm_url not in self.dcgm_endpoints:
            self.dcgm_endpoints[record.dcgm_url] = {}

        dcgm_data = self.dcgm_endpoints[record.dcgm_url]

        if record.gpu_uuid not in dcgm_data:
            dcgm_data[record.gpu_uuid] = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=record.gpu_index,
                    gpu_uuid=record.gpu_uuid,
                    gpu_model_name=record.gpu_model_name,
                    hostname=record.hostname,
                    namespace=record.namespace,
                    pod_name=record.pod_name,
                ),
            )

        dcgm_data[record.gpu_uuid].add_record(record)


class ProcessTelemetryResult(AIPerfBaseModel):
    """Result of telemetry processing - mirrors ProcessRecordsResult pattern.

    This provides a parallel structure to ProcessRecordsResult for the telemetry pipeline,
    maintaining complete separation while following the same architectural patterns.

    Note: Uses TelemetryExportData (wire-safe, pre-computed stats) rather than
    TelemetryResults (internal, contains non-serializable GpuMetricTimeSeries).
    """

    results: TelemetryExportData | None = Field(
        default=None, description="Pre-computed telemetry export data (wire-safe)"
    )

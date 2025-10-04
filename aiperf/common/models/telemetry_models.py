# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pydantic import Field

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.record_models import MetricResult


class TelemetryMetrics(AIPerfBaseModel):
    """GPU metrics collected at a single point in time.

    All fields are optional to handle cases where specific metrics are not available
    from the DCGM exporter or are filtered out due to invalid values.
    """

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
    sm_clock_frequency: float | None = Field(
        default=None, description="SM clock frequency in MHz"
    )
    memory_clock_frequency: float | None = Field(
        default=None, description="Memory clock frequency in MHz"
    )
    memory_temperature: float | None = Field(
        default=None, description="Memory temperature in °C"
    )
    gpu_temperature: float | None = Field(
        default=None, description="GPU temperature in °C"
    )
    memory_copy_utilization: float | None = Field(
        default=None, description="Memory copy utilization percentage (0-100)"
    )
    xid_errors: float | None = Field(
        default=None, description="Value of the last XID error encountered"
    )
    power_violation: float | None = Field(
        default=None,
        description="Throttling duration due to power constraints in microseconds",
    )
    thermal_violation: float | None = Field(
        default=None,
        description="Throttling duration due to thermal constraints in microseconds",
    )


class TelemetryRecord(AIPerfBaseModel):
    """Single telemetry data point from GPU monitoring.

    This record contains all telemetry data for one GPU at one point in time,
    along with metadata to identify the source DCGM endpoint and specific GPU.
    Used for hierarchical storage: dcgm_url -> gpu_uuid -> time series data.
    """

    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when telemetry was collected (time_ns)"
    )

    dcgm_url: str = Field(
        description="Source DCGM endpoint URL (e.g., 'http://node1:9401/metrics')"
    )

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

    telemetry_data: TelemetryMetrics = Field(
        description="GPU metrics snapshot collected at this timestamp"
    )


class GpuMetadata(AIPerfBaseModel):
    """Static metadata for a GPU that doesn't change over time.

    This is stored once per GPU and referenced by all telemetry data points
    to avoid duplicating metadata in every time-series entry.
    """

    gpu_index: int = Field(description="GPU index for display ordering (0, 1, 2, etc.)")
    gpu_uuid: str = Field(description="Unique GPU identifier - primary key")
    model_name: str = Field(description="GPU hardware model name")
    pci_bus_id: str | None = Field(default=None, description="PCI Bus location")
    device: str | None = Field(default=None, description="System device identifier")
    hostname: str | None = Field(default=None, description="Host machine name")


class GpuTelemetrySnapshot(AIPerfBaseModel):
    """All metrics for a single GPU at one point in time.

    Groups all metric values collected during a single collection cycle,
    eliminating timestamp duplication across individual metrics.
    """

    timestamp_ns: int = Field(description="Collection timestamp for all metrics")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="All metric values at this timestamp"
    )


class GpuMetricTimeSeries(AIPerfBaseModel):
    """Time series data for all metrics on a single GPU.

    Uses grouped snapshots instead of individual metric time series to eliminate
    timestamp duplication and improve storage efficiency.
    """

    snapshots: list[GpuTelemetrySnapshot] = Field(
        default_factory=list, description="Chronological snapshots of all metrics"
    )

    def append_snapshot(self, metrics: dict[str, float], timestamp_ns: int) -> None:
        """Add new snapshot with all metrics at once.

        Args:
            metrics: Dictionary of metric_name -> value for this timestamp
            timestamp_ns: Timestamp when measurements were taken
        """
        snapshot = GpuTelemetrySnapshot(
            timestamp_ns=timestamp_ns,
            metrics={k: v for k, v in metrics.items() if v is not None},
        )
        self.snapshots.append(snapshot)

    def get_metric_values(self, metric_name: str) -> list[tuple[float, int]]:
        """Extract time series data for a specific metric.

        Args:
            metric_name: Name of the metric to extract

        Returns:
            List of (value, timestamp_ns) tuples for the specified metric
        """
        return [
            (snapshot.metrics[metric_name], snapshot.timestamp_ns)
            for snapshot in self.snapshots
            if metric_name in snapshot.metrics
        ]

    def to_metric_result(
        self, metric_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Convert metric time series to MetricResult with statistical summary.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric (used by dashboard, exports, API)
            header: Human-readable name for display
            unit: Unit of measurement (e.g., "W" for Watts, "%" for percentage)

        Returns:
            MetricResult with min/max/avg/percentiles computed from time series

        Raises:
            NoMetricValue: If no data points are available for the specified metric
        """
        data_points = self.get_metric_values(metric_name)

        if not data_points:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )

        values = np.array([point[0] for point in data_points])
        p1, p5, p25, p50, p75, p90, p95, p99 = np.percentile(
            values, [1, 5, 25, 50, 75, 90, 95, 99]
        )

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=np.min(values),
            max=np.max(values),
            avg=float(np.mean(values)),
            std=float(np.std(values)),
            count=len(values),
            p1=p1,
            p5=p5,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
        )


class GpuTelemetryData(AIPerfBaseModel):
    """Complete telemetry data for one GPU: metadata + grouped metric time series.

    This combines static GPU information with dynamic time-series data,
    providing the complete picture for one GPU's telemetry using efficient grouped snapshots.
    """

    metadata: GpuMetadata = Field(description="Static GPU information")
    time_series: GpuMetricTimeSeries = Field(
        default_factory=GpuMetricTimeSeries,
        description="Grouped time series for all metrics",
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
        self, metric_name: str, tag: str, header: str, unit: str
    ) -> MetricResult:
        """Get MetricResult for a specific metric.

        Args:
            metric_name: Name of the metric to analyze
            tag: Unique identifier for this metric
            header: Human-readable name for display
            unit: Unit of measurement

        Returns:
            MetricResult with statistical summary for the specified metric
        """
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
            metadata = GpuMetadata(
                gpu_index=record.gpu_index,
                gpu_uuid=record.gpu_uuid,
                model_name=record.gpu_model_name,
                pci_bus_id=record.pci_bus_id,
                device=record.device,
                hostname=record.hostname,
            )
            dcgm_data[record.gpu_uuid] = GpuTelemetryData(metadata=metadata)

        dcgm_data[record.gpu_uuid].add_record(record)


class TelemetryResults(AIPerfBaseModel):
    """Results from GPU telemetry collection during a profile run.

    This class contains all telemetry data and metadata collected during
    a benchmarking session, separate from inference performance results.
    """

    telemetry_data: TelemetryHierarchy = Field(
        description="Hierarchical telemetry data organized by DCGM endpoint and GPU"
    )
    start_ns: int = Field(
        description="Start time of telemetry collection in nanoseconds"
    )
    end_ns: int = Field(description="End time of telemetry collection in nanoseconds")
    endpoints_tested: list[str] = Field(
        default_factory=list,
        description="List of DCGM endpoint URLs that were tested for reachability",
    )
    endpoints_successful: list[str] = Field(
        default_factory=list,
        description="List of DCGM endpoint URLs that successfully provided telemetry data",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )


class ProcessTelemetryResult(AIPerfBaseModel):
    """Result of telemetry processing - mirrors ProcessRecordsResult pattern.

    This provides a parallel structure to ProcessRecordsResult for the telemetry pipeline,
    maintaining complete separation while following the same architectural patterns.
    """

    results: TelemetryResults = Field(description="The processed telemetry results")
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any errors that occurred while processing telemetry data",
    )

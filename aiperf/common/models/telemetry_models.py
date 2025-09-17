# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pydantic import Field

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.record_models import MetricResult


class TelemetryRecord(AIPerfBaseModel):
    """Single telemetry data point from GPU monitoring.
    
    This record contains all telemetry data for one GPU at one point in time,
    along with metadata to identify the source DCGM endpoint and specific GPU.
    Used for hierarchical storage: dcgm_url -> gpu_uuid -> time series data.
    """
    
    timestamp_ns: int = Field(description="Nanosecond timestamp when telemetry was collected (perf_counter_ns)")
    
    # Source identification - enables multi-DCGM deployment support
    dcgm_url: str = Field(description="Source DCGM endpoint URL (e.g., 'http://node1:9401/metrics')")
    
    # GPU identification - both index (for ordering) and UUID (for uniqueness)
    gpu_index: int = Field(description="GPU index on this node (0, 1, 2, etc.) - used for display ordering")
    gpu_uuid: str = Field(description="Unique GPU identifier (e.g., 'GPU-ef6ef310-...') - primary key for data")
    
    # GPU metadata - static information about the hardware
    gpu_model_name: str = Field(description="GPU model name (e.g., 'NVIDIA RTX 6000 Ada Generation')")
    pci_bus_id: str | None = Field(default=None, description="PCI Bus ID (e.g., '00000000:02:00.0')")
    device: str | None = Field(default=None, description="Device identifier (e.g., 'nvidia0')")
    hostname: str | None = Field(default=None, description="Hostname where GPU is located")
    
    # Actual telemetry metrics - the time-series data we want to track
    gpu_power_usage: float | None = Field(default=None, description="Current GPU power usage in Watts")
    gpu_power_limit: float | None = Field(default=None, description="GPU power limit in Watts")
    energy_consumption: float | None = Field(default=None, description="Cumulative energy consumption in Millijoules")
    gpu_utilization: float | None = Field(default=None, description="GPU utilization percentage (0-100)")
    gpu_memory_used: float | None = Field(default=None, description="GPU memory used in GB")
    total_gpu_memory: float | None = Field(default=None, description="Total GPU memory in GB")


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


class GpuMetricTimeSeries(AIPerfBaseModel):
    """Time series data for a single metric on a single GPU.
    
    Stores arrays of values and timestamps for efficient statistical computation.
    """
    
    # Parallel arrays - values[i] corresponds to timestamps_ns[i]
    values: list[float] = Field(default_factory=list, description="Metric values over time")
    timestamps_ns: list[int] = Field(default_factory=list, description="Corresponding timestamps in nanoseconds")
    
    def append(self, value: float, timestamp_ns: int) -> None:
        """Add new data point to time series.
        
        Args:
            value: Metric value (e.g., power usage in Watts)
            timestamp_ns: Timestamp when measurement was taken
        """
        self.values.append(value)
        self.timestamps_ns.append(timestamp_ns)
    
    def to_metric_result(self, tag: str, header: str, unit: str) -> MetricResult:
        """Convert time series to MetricResult with statistical summary.
        
        Args:
            tag: Unique identifier for this metric (used by dashboard, exports, API)
            header: Human-readable name for display
            unit: Unit of measurement (e.g., "W" for Watts, "%" for percentage)
            
        Returns:
            MetricResult with min/max/avg/percentiles computed from time series
            
        Raises:
            NoMetricValue: If no data points are available
        """
        if not self.values:
            raise NoMetricValue("No telemetry data available for statistical computation")
        
        # Use numpy for efficient statistical computation (same as existing MetricArray)
        arr = np.array(self.values)
        p1, p5, p25, p50, p75, p90, p95, p99 = np.percentile(arr, [1, 5, 25, 50, 75, 90, 95, 99])
        
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=np.min(arr),
            max=np.max(arr),
            avg=float(np.mean(arr)),
            std=float(np.std(arr)),
            count=len(arr),
            p1=p1, p5=p5, p25=p25, p50=p50, p75=p75, p90=p90, p95=p95, p99=p99
        )


class GpuTelemetryData(AIPerfBaseModel):
    """Complete telemetry data for one GPU: metadata + all metric time series.
    
    This combines static GPU information with dynamic time-series data,
    providing the complete picture for one GPU's telemetry.
    """
    
    metadata: GpuMetadata = Field(description="Static GPU information")
    metrics: dict[str, GpuMetricTimeSeries] = Field(
        default_factory=dict, 
        description="Time series for each metric type (power_usage, utilization, etc.)"
    )
    
    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record to appropriate metric time series.
        
        Args:
            record: New telemetry data point from DCGM collector
            
        Note: Automatically creates new time series for metrics that don't exist yet
        """
        # Map TelemetryRecord fields to metric names for time series storage
        metric_mapping = {
            "gpu_power_usage": record.gpu_power_usage,
            "gpu_power_limit": record.gpu_power_limit, 
            "energy_consumption": record.energy_consumption,
            "gpu_utilization": record.gpu_utilization,
            "gpu_memory_used": record.gpu_memory_used,
            "total_gpu_memory": record.total_gpu_memory,
        }
        
        # Add each non-null metric value to its time series
        for metric_name, value in metric_mapping.items():
            if value is not None:  # Skip null values from DCGM
                if metric_name not in self.metrics:
                    # Lazy initialization - create time series when first data arrives
                    self.metrics[metric_name] = GpuMetricTimeSeries()
                self.metrics[metric_name].append(value, record.timestamp_ns)


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
        description="Nested dict: dcgm_url -> gpu_uuid -> telemetry data"
    )
    
    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record to hierarchical storage.
        
        Args:
            record: New telemetry data from GPU monitoring
            
        Note: Automatically creates hierarchy levels as needed:
        - New DCGM endpoints get empty GPU dict
        - New GPUs get initialized with metadata and empty metrics
        """
        # Create DCGM endpoint level if it doesn't exist
        if record.dcgm_url not in self.dcgm_endpoints:
            self.dcgm_endpoints[record.dcgm_url] = {}
        
        dcgm_data = self.dcgm_endpoints[record.dcgm_url]
        
        # Create GPU level if this is the first time we see this GPU
        if record.gpu_uuid not in dcgm_data:
            # Initialize with metadata from the first record
            metadata = GpuMetadata(
                gpu_index=record.gpu_index,
                gpu_uuid=record.gpu_uuid,
                model_name=record.gpu_model_name,
                pci_bus_id=record.pci_bus_id,
                device=record.device,
                hostname=record.hostname
            )
            dcgm_data[record.gpu_uuid] = GpuTelemetryData(metadata=metadata)
        
        # Add the record's data to the appropriate GPU's time series
        dcgm_data[record.gpu_uuid].add_record(record)
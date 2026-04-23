# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar

import msgspec
from pydantic import ConfigDict

from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.common.models.telemetry_timeseries import GpuMetricTimeSeries

__all__ = [
    "GpuMetadata",
    "GpuMetricTimeSeries",
    "GpuTelemetryData",
    "GpuTelemetrySnapshot",
    "ProcessTelemetryResult",
    "TelemetryHierarchy",
    "TelemetryRecord",
]


class GpuMetadata(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Static metadata for a GPU that doesn't change over time.

    This is stored once per GPU and referenced by all telemetry data points
    to avoid duplicating metadata in every time-series entry.
    """

    gpu_index: int
    """GPU index on this node (0, 1, 2, etc.) - used for display ordering."""

    gpu_uuid: str
    """Unique GPU identifier (e.g., 'GPU-ef6ef310-...') - primary key for data."""

    gpu_model_name: str
    """GPU model name (e.g., 'NVIDIA RTX 6000 Ada Generation')."""

    pci_bus_id: str | None = None
    """PCI Bus ID (e.g., '00000000:02:00.0')."""

    device: str | None = None
    """Device identifier (e.g., 'nvidia0')."""

    hostname: str | None = None
    """Hostname where GPU is located."""

    namespace: str | None = None
    """Namespace where the GPU is located (kubernetes only)."""

    pod_name: str | None = None
    """Pod name where the GPU is located (kubernetes only)."""


class TelemetryRecord(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Single telemetry data point from GPU monitoring.

    This record contains all telemetry data for one GPU at one point in time,
    along with metadata to identify the source DCGM endpoint and specific GPU.
    Used for hierarchical storage: dcgm_url -> gpu_uuid -> time series data.

    `telemetry_data` is a plain ``dict[str, float]`` mapping canonical metric
    names (``gpu_power_usage``, ``gpu_utilization``, ...) to values. Custom
    metrics loaded from user CSV files go into the same dict keyed by their
    ``DCGM_TO_FIELD_MAPPING`` entry.
    """

    gpu_index: int
    """GPU index on this node (0, 1, 2, etc.) - used for display ordering."""

    gpu_uuid: str
    """Unique GPU identifier (e.g., 'GPU-ef6ef310-...') - primary key for data."""

    gpu_model_name: str
    """GPU model name (e.g., 'NVIDIA RTX 6000 Ada Generation')."""

    timestamp_ns: int
    """Nanosecond wall-clock timestamp when telemetry was collected (time_ns)."""

    dcgm_url: str
    """Source identifier (DCGM URL e.g., 'http://node1:9401/metrics' or
    'pynvml://localhost')."""

    telemetry_data: dict[str, float]
    """GPU metrics snapshot collected at this timestamp."""

    pci_bus_id: str | None = None
    """PCI Bus ID (e.g., '00000000:02:00.0')."""

    device: str | None = None
    """Device identifier (e.g., 'nvidia0')."""

    hostname: str | None = None
    """Hostname where GPU is located."""

    namespace: str | None = None
    """Namespace where the GPU is located (kubernetes only)."""

    pod_name: str | None = None
    """Pod name where the GPU is located (kubernetes only)."""


class GpuTelemetrySnapshot(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """All metrics for a single GPU at one point in time.

    Groups all metric values collected during a single collection cycle,
    eliminating timestamp duplication across individual metrics.
    """

    timestamp_ns: int
    """Collection timestamp for all metrics."""

    metrics: dict[str, float] = msgspec.field(default_factory=dict)
    """All metric values at this timestamp."""


class GpuTelemetryData(msgspec.Struct, kw_only=True):
    """Complete telemetry data for one GPU: metadata + grouped metric time series.

    This combines static GPU information with dynamic time-series data,
    providing the complete picture for one GPU's telemetry using efficient
    columnar storage. Not frozen because ``time_series`` is mutated in place
    as records arrive.
    """

    metadata: GpuMetadata
    """Static GPU information."""

    time_series: GpuMetricTimeSeries = msgspec.field(
        default_factory=GpuMetricTimeSeries
    )
    """Columnar time series for all metrics. Numpy-backed, not serialized."""

    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record as a grouped snapshot.

        Args:
            record: New telemetry data point from DCGM collector

        Note: Groups all metric values from the record into a single snapshot.
        Producers (DCGM + pynvml collectors, test factories) always build
        dense dicts — missing metrics are absent, not None — so no filtering
        is needed here.
        """
        if record.telemetry_data:
            self.time_series.append_snapshot(record.telemetry_data, record.timestamp_ns)

    def get_metric_result(
        self,
        metric_name: str,
        tag: str,
        header: str,
        unit: str,
        *,
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
                metric_name,
                tag,
                header,
                unit,
                time_filter=time_filter,
                is_counter=is_counter,
            )
        return self.time_series.to_metric_result(metric_name, tag, header, unit)


class TelemetryHierarchy(msgspec.Struct, kw_only=True):
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

    dcgm_endpoints: dict[str, dict[str, GpuTelemetryData]] = msgspec.field(
        default_factory=dict
    )
    """Nested dict: dcgm_url -> gpu_uuid -> telemetry data."""

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
                    pci_bus_id=record.pci_bus_id,
                    device=record.device,
                    hostname=record.hostname,
                    namespace=record.namespace,
                    pod_name=record.pod_name,
                ),
            )

        dcgm_data[record.gpu_uuid].add_record(record)


@dataclass(slots=True, kw_only=True)
class ProcessTelemetryResult:
    """Result of telemetry processing - mirrors ProcessRecordsResult pattern.

    Slotted dataclass — shared between msgspec envelopes
    (``ProcessTelemetryResultMessage.telemetry_result``) and Pydantic parents
    via ``__pydantic_config__``.

    Note: Uses TelemetryExportData (wire-safe, pre-computed stats) rather than
    TelemetryResults (internal, contains non-serializable GpuMetricTimeSeries).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    results: TelemetryExportData | None = None

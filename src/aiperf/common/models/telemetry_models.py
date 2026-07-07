# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import msgspec
from pydantic import ConfigDict

from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.common.models.telemetry_timeseries import (
    GpuMetricTimeSeries,
    _last_valid,
)

__all__ = [
    "GpuMetadata",
    "GpuMetricTimeSeries",
    "GpuTelemetryData",
    "GpuTelemetrySnapshot",
    "ProcessTelemetryResult",
    "TelemetryHierarchy",
    "TelemetryMetrics",
    "TelemetryRecord",
    "_last_valid",
]


class TelemetryMetrics(msgspec.Struct, kw_only=True, omit_defaults=True):
    """GPU metrics collected at a single point in time.

    All fields are optional to handle cases where specific metrics are not
    available from the DCGM exporter or are filtered out due to invalid values.

    Custom metrics from user-provided CSV files arrive via DCGM with field
    names not declared here; they are routed into ``custom_metrics`` by the
    DCGM collector. Collectors that set declared fields directly
    (pynvml, amdsmi) bypass that dict.
    """

    # perf: validation moved to msgspec/protocol boundary
    gpu_power_usage: float | None = None
    """Current GPU power usage in W."""
    energy_consumption: float | None = None
    """Cumulative energy consumption in MJ."""
    gpu_utilization: float | None = None
    """GPU utilization percentage (0-100)."""
    gpu_memory_used: float | None = None
    """GPU memory used in GB."""
    gpu_temperature: float | None = None
    """GPU temperature in C."""
    mem_utilization: float | None = None
    """Memory bandwidth utilization percentage (0-100)."""
    sm_utilization: float | None = None
    """Streaming multiprocessor utilization percentage (0-100)."""
    decoder_utilization: float | None = None
    """Video decoder (NVDEC) utilization percentage (0-100)."""
    encoder_utilization: float | None = None
    """Video encoder (NVENC) utilization percentage (0-100)."""
    jpg_utilization: float | None = None
    """JPEG decoder utilization percentage (0-100)."""
    xid_errors: float | None = None
    """Value of the last XID error encountered."""
    power_violation: float | None = None
    """Throttling duration due to power constraints in microseconds."""

    # AMD ROCm telemetry (collected by AMDSMITelemetryCollector). These mirror
    # the amdsmi field names rather than being aliased onto NVML-shaped fields,
    # because the underlying signals do not always measure the same physical
    # quantity (e.g. gfx_activity vs sm_utilization sample differently).
    amd_power: float | None = None
    """AMD GPU current socket power in W."""
    amd_energy_consumption: float | None = None
    """AMD GPU cumulative energy consumption in MJ."""
    amd_gfx_activity: float | None = None
    """AMD GPU graphics engine activity percentage (0-100)."""
    amd_umc_activity: float | None = None
    """AMD GPU memory controller activity percentage (0-100)."""
    amd_mm_activity: float | None = None
    """AMD GPU multimedia engine activity percentage (0-100)."""
    amd_memory_used: float | None = None
    """AMD GPU VRAM used in GB."""
    amd_temperature: float | None = None
    """AMD GPU temperature in C."""
    amd_ecc_uncorrectable: float | None = None
    """AMD GPU cumulative uncorrectable ECC error count."""
    amd_throttle_status: float | None = None
    """AMD GPU throttle status snapshot."""

    custom_metrics: dict[str, float] = msgspec.field(default_factory=dict)
    """User-CSV-defined GPU metrics not in the canonical fields. Populated by
    the DCGM collector when a metric's mapped field name is not a declared
    attribute on this struct."""

    @classmethod
    def from_mapping(cls, metrics: dict[str, float]) -> TelemetryMetrics:
        """Construct from a flat ``{field_name: value}`` mapping.

        Field names matching declared attributes set them directly; unknown
        keys are routed into ``custom_metrics``. This replaces the prior
        Pydantic ``extra="allow"`` setattr semantics used by the DCGM
        collector for custom-CSV metrics.
        """
        known = _TELEMETRY_METRICS_FIELDS
        declared = {k: v for k, v in metrics.items() if k in known}
        custom = {k: v for k, v in metrics.items() if k not in known}
        return cls(**declared, custom_metrics=custom)

    def to_flat_dict(self) -> dict[str, float]:
        """Return all known + custom metrics as a single flat mapping.

        None values are dropped. Used by ``GpuTelemetryData.add_record`` to
        feed the numpy time series.
        """
        out: dict[str, float] = {}
        for name in _TELEMETRY_METRICS_FIELDS:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.custom_metrics)
        return out

    def any_field_set(self) -> bool:
        """True if any declared field is non-None or any custom metric is set.

        Replaces the prior Pydantic ``model_fields_set`` check used in
        pynvml_collector to skip emitting records with no metrics.
        """
        if self.custom_metrics:
            return True
        for name in _TELEMETRY_METRICS_FIELDS:
            if getattr(self, name) is not None:
                return True
        return False


# Names of declared TelemetryMetrics fields; excludes ``custom_metrics`` since
# that is the catch-all bucket, not a declared metric.
_TELEMETRY_METRICS_FIELDS: frozenset[str] = frozenset(
    f for f in TelemetryMetrics.__struct_fields__ if f != "custom_metrics"
)


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
    GpuMetadata,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Single telemetry data point from GPU monitoring.

    Inherits from GpuMetadata to avoid duplicating metadata fields.
    """

    timestamp_ns: int
    """Nanosecond wall-clock timestamp when telemetry was collected."""

    dcgm_url: str
    """Source identifier (DCGM URL e.g., 'http://node1:9401/metrics' or
    'pynvml://localhost')."""

    telemetry_data: TelemetryMetrics
    """GPU metrics snapshot collected at this timestamp."""

    @classmethod
    def model_validate(cls, value: Any) -> TelemetryRecord:
        """Pydantic-compat constructor from a dict-like value."""
        return msgspec.convert(value, type=cls)

    @classmethod
    def model_validate_json(cls, value: str | bytes) -> TelemetryRecord:
        """Pydantic-compat constructor from a JSON string / bytes."""
        return msgspec.json.decode(value, type=cls)

    def model_dump_json(self) -> str:
        """Pydantic-compat JSON serializer."""
        return msgspec.json.encode(self).decode("utf-8")


class GpuTelemetrySnapshot(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """All metrics for a single GPU at one point in time."""

    timestamp_ns: int
    """Collection timestamp for all metrics."""

    metrics: dict[str, float] = msgspec.field(default_factory=dict)
    """All metric values at this timestamp."""


class GpuTelemetryData(msgspec.Struct, kw_only=True):
    """Complete telemetry data for one GPU: metadata + grouped metric time series.

    Not frozen because ``time_series`` is mutated in place as records arrive.
    ``time_series`` is a numpy-backed columnar store; it is excluded from
    serialization via ``_omit_fields_`` to keep wire payloads small.
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
        """
        flat = record.telemetry_data.to_flat_dict()
        if flat:
            self.time_series.append_snapshot(flat, record.timestamp_ns)

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
        """Get MetricResult for a specific metric with optional time filtering."""
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
    """Hierarchical storage: dcgm_url -> gpu_uuid -> complete GPU telemetry data."""

    dcgm_endpoints: dict[str, dict[str, GpuTelemetryData]] = msgspec.field(
        default_factory=dict
    )
    """Nested dict: dcgm_url -> gpu_uuid -> telemetry data."""

    def add_record(self, record: TelemetryRecord) -> None:
        """Add telemetry record to hierarchical storage."""
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

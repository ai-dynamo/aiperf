# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import msgspec
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.record_models import MetricResult
from aiperf.common.models.server_metrics_models import TimeRangeFilter

__all__ = [
    "GpuMetadata",
    "GpuMetricTimeSeries",
    "GpuTelemetryData",
    "GpuTelemetrySnapshot",
    "ProcessTelemetryResult",
    "TelemetryHierarchy",
    "TelemetryMetrics",
    "TelemetryRecord",
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


def _last_valid(arr: np.ndarray) -> float | None:
    """Return the last non-NaN value in ``arr``, or ``None`` if all NaN."""
    mask = ~np.isnan(arr)
    return float(arr[mask][-1]) if mask.any() else None


class GpuMetricTimeSeries:
    """NumPy-backed columnar storage for GPU telemetry.

    Stores timestamps once with separate value arrays per metric. The metric
    schema is the union of all keys ever seen — late-arriving keys allocate
    a new array NaN-backfilled for prior positions, and known keys absent
    from a given snapshot are written as NaN at that index. Stat methods
    use ``np.nan*`` variants so NaN-padded slots don't poison results.

    This dynamic-schema behavior accommodates collectors like AMDSMI whose
    sensors can fail transiently (a missing baseline field is not the same
    as a reading of zero). Static-schema collectors (DCGM, PyNVML) emit the
    same keys every scrape, so the NaN handling is a no-op for them.

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
        """Append all metrics from a single scrape (insert-sorted)."""
        if self._size >= self._capacity:
            self._grow()

        if self._size == 0 or timestamp_ns >= self._timestamps[self._size - 1]:
            insert_pos = self._size
        else:
            insert_pos = self._size - 1
            while insert_pos > 0 and self._timestamps[insert_pos - 1] > timestamp_ns:
                insert_pos -= 1

            self._timestamps[insert_pos + 1 : self._size + 1] = self._timestamps[
                insert_pos : self._size
            ]
            for arr in self._metrics.values():
                arr[insert_pos + 1 : self._size + 1] = arr[insert_pos : self._size]

        self._timestamps[insert_pos] = timestamp_ns

        for name in metrics:
            if name not in self._metrics:
                self._metrics[name] = np.full(self._capacity, np.nan, dtype=np.float64)

        for name, arr in self._metrics.items():
            arr[insert_pos] = metrics.get(name, np.nan)

        self._size += 1

    def _grow(self) -> None:
        """Double capacity of all arrays."""
        new_capacity = self._capacity * 2

        new_ts = np.empty(new_capacity, dtype=np.int64)
        new_ts[: self._size] = self._timestamps[: self._size]
        self._timestamps = new_ts

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
        """Compute stats for a metric using vectorized NumPy operations."""
        arr = self.get_metric_array(metric_name)
        if arr is None or len(arr) == 0:
            raise NoMetricValue(
                f"No telemetry data available for metric '{metric_name}'"
            )
        if np.all(np.isnan(arr)):
            raise NoMetricValue(
                f"All samples for metric '{metric_name}' are NaN "
                f"(sensor never returned a successful read)"
            )

        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.nanpercentile(
            arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        non_nan = int(np.count_nonzero(~np.isnan(arr)))
        std_dev = float(np.nanstd(arr, ddof=1)) if non_nan > 1 else 0.0

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.nanmin(arr)),
            max=float(np.nanmax(arr)),
            avg=float(np.nanmean(arr)),
            sum=float(np.nansum(arr)),
            std=std_dev,
            count=len(arr),
            current=_last_valid(arr),
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
        """Get boolean mask for points within time range."""
        if time_filter is None:
            return np.ones(self._size, dtype=bool)

        timestamps = self.timestamps
        first_idx = 0
        last_idx = self._size

        if time_filter.start_ns is not None:
            first_idx = int(
                np.searchsorted(timestamps, time_filter.start_ns, side="left")
            )
        if time_filter.end_ns is not None:
            last_idx = int(
                np.searchsorted(timestamps, time_filter.end_ns, side="right")
            )

        mask = np.zeros(self._size, dtype=bool)
        mask[first_idx:last_idx] = True
        return mask

    def get_reference_idx(self, time_filter: TimeRangeFilter | None) -> int | None:
        """Get index of last point BEFORE time filter start (for delta calculation)."""
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
        """Compute stats with time filtering and optional delta for counters."""
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
            filtered_last = _last_valid(filtered)
            if filtered_last is None:
                raise NoMetricValue(
                    f"No valid (non-NaN) samples in filtered range for "
                    f"metric '{metric_name}'"
                )

            reference_idx = self.get_reference_idx(time_filter)
            reference_value: float | None
            if reference_idx is not None:
                reference_value = _last_valid(arr[: reference_idx + 1])
            else:
                reference_value = None

            if reference_value is None:
                mask = ~np.isnan(filtered)
                reference_value = float(filtered[mask][0])

            delta = max(filtered_last - reference_value, 0.0)

            return MetricResult(
                tag=tag,
                header=header,
                unit=unit,
                avg=delta,
            )

        if np.all(np.isnan(filtered)):
            raise NoMetricValue(
                f"All in-range samples for metric '{metric_name}' are NaN"
            )
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.nanpercentile(
            filtered, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )

        non_nan = int(np.count_nonzero(~np.isnan(filtered)))
        std_dev = float(np.nanstd(filtered, ddof=1)) if non_nan > 1 else 0.0

        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=float(np.nanmin(filtered)),
            max=float(np.nanmax(filtered)),
            avg=float(np.nanmean(filtered)),
            sum=float(np.nansum(filtered)),
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
                metric_name, tag, header, unit, time_filter, is_counter
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

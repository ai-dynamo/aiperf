# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client.parser import text_string_to_metric_families
from pydantic import ValidationError

from aiperf.common.environment import Environment
from aiperf.common.finite import is_finite_value
from aiperf.common.mixins import (
    BaseMetricsCollectorMixin,
    TErrorCallback,
    TRecordCallback,
)
from aiperf.common.models import GpuMetadata, TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.constants import AMD_GPU_TELEMETRY_PLATFORM

if TYPE_CHECKING:
    from prometheus_client.samples import Sample

__all__ = ["AMDDMETelemetryCollector"]

# DME labels its VRAM gauges "MB", but the values are mebibytes: gpu.proto
# reports them straight from the SMI, which counts 1024*1024-byte units. The
# telemetry fields are documented in GB, so the factor is 2**20 / 1e9 rather
# than 1/1024. The two differ by 7.4%, which is the whole reason this is not
# left as a tidy-up.
_MIB_TO_GB = (1024.0 * 1024.0) / 1e9

SCALING_FACTORS = {
    "amd_energy_consumption": 1e-12,
    "amd_memory_used": _MIB_TO_GB,
    "amd_memory_free": _MIB_TO_GB,
    "amd_memory_total": _MIB_TO_GB,
}

# DME gives every partition of a partitioned card the parent card's serial
# number, so the serial alone is not unique per reported GPU. gpu_partition_id
# is a default DME label and reads "NA" on an unpartitioned card.
_PARTITION_LABEL = "gpu_partition_id"
_NO_PARTITION = "NA"


class AMDDMETelemetryCollector(BaseMetricsCollectorMixin[TelemetryRecord]):
    """Scrapes AMD GPU telemetry from an AMD Device Metrics Exporter endpoint.

    The remote counterpart to the amdsmi collector: DME serves Prometheus text
    over HTTP, so the benchmark host needs no ROCm install to monitor the GPUs.

    Args:
        dcgm_url: URL of the AMD DME endpoint (e.g., "http://<amd-exporter-ip>:5000/metrics")
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        reachability_timeout: Timeout in seconds for reachability checks (default: from Environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    @classmethod
    def validate_environment(cls) -> None:
        """Remote HTTP collector, so there is no local environment to validate."""

    def __init__(
        self,
        dcgm_url: str,
        *,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        reachability_timeout: float = Environment.GPU.REACHABILITY_TIMEOUT,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "telemetry_collector",
    ) -> None:
        self._scaling_factors = SCALING_FACTORS
        super().__init__(
            endpoint_url=dcgm_url,
            collection_interval=collection_interval,
            reachability_timeout=reachability_timeout,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )

    async def _collect_and_process_metrics(self) -> None:
        fetch_result = await self._fetch_metrics_text()
        if fetch_result.is_duplicate:
            return
        records = self._parse_metrics_to_records(fetch_result.text)
        await self._send_records_via_callback(records)

    # TelemetryRecord field -> the DME metrics that report it, most preferred
    # first. DME exports gpu_package_power and gpu_power_usage each only when
    # the device reports it, so some parts expose one and some both; the order
    # makes the choice independent of the order Prometheus families arrive in.
    _METRIC_SOURCES: dict[str, tuple[str, ...]] = {
        "amd_power": ("gpu_package_power", "gpu_power_usage"),
        "amd_energy_consumption": ("gpu_energy_consumed",),
        "amd_gfx_activity": ("gpu_gfx_activity",),
        "amd_umc_activity": ("gpu_umc_activity",),
        "amd_memory_used": ("gpu_used_vram",),
        "amd_memory_free": ("gpu_free_vram",),
        "amd_memory_total": ("gpu_total_vram",),
        "amd_temperature": ("gpu_junction_temperature",),
        "amd_memory_temperature": ("gpu_memory_temperature",),
        "amd_ecc_uncorrectable": ("gpu_ecc_uncorrect_total",),
    }

    # DME metric name -> (field, rank). Lower rank wins a field.
    _METRIC_FIELDS: dict[str, tuple[str, int]] = {
        metric: (field, rank)
        for field, metrics in _METRIC_SOURCES.items()
        for rank, metric in enumerate(metrics)
    }

    # gpu_clock carries its clock in labels. clock_index is the position in the
    # device's clock list, not a clock domain, so the first clock of each type
    # is taken rather than a fixed index. Current DME normalises clock_type to
    # lower case without the GPU_CLOCK_TYPE_ prefix; older releases did not.
    _CLOCK_FIELDS: dict[str, str] = {
        "system": "amd_gfx_clock",
        "memory": "amd_mem_clock",
    }
    _CLOCK_TYPE_PREFIX = "GPU_CLOCK_TYPE_"

    @staticmethod
    def _gpu_uuid_from(labels: dict, gpu_index: int) -> str:
        """A UUID that stays distinct across the partitions of one card.

        Every partition of a partitioned GPU carries the parent card's
        ``serial_number``, so the serial alone collapses CPX, DPX and QPX
        partitions onto a single identity. The partition id disambiguates them
        and is absent, or reads ``NA``, on an unpartitioned card, where the
        bare serial is kept so existing identities do not move.
        """
        serial = labels.get("serial_number")
        if not serial:
            return f"amd-gpu-{gpu_index}"
        partition = labels.get(_PARTITION_LABEL)
        if partition and partition != _NO_PARTITION:
            return f"{serial}-p{partition}"
        return serial

    @staticmethod
    def _gpu_index_from(labels: dict) -> int | None:
        gpu_id = labels.get("gpu_id")
        if gpu_id is None:
            return None
        try:
            return int(gpu_id)
        except ValueError:
            return None

    @classmethod
    def _canonical_metric_name(cls, name: str) -> str | None:
        """The DME metric this sample name reports, allowing an exporter prefix.

        A deployment may namespace its metric names (``amd_gpu_package_power``
        for ``gpu_package_power``). An exact match wins; otherwise the longest
        known name that the sample name ends with, on an underscore boundary,
        is used. Matching on the boundary keeps ``gpu_used_vram`` from claiming
        a hypothetical ``x_gpu_used_vram_bytes``, which does not end with it.
        """
        if name in cls._METRIC_FIELDS or name == "gpu_clock":
            return name
        candidates = [*cls._METRIC_FIELDS, "gpu_clock"]
        matches = [c for c in candidates if name.endswith(f"_{c}")]
        return max(matches, key=len) if matches else None

    @classmethod
    def _resolve(cls, sample: Sample) -> tuple[str, int] | None:
        """The (field, rank) this sample would set, or None if nothing uses it."""
        name = cls._canonical_metric_name(sample.name)
        if name is None:
            return None
        mapped = cls._METRIC_FIELDS.get(name)
        if mapped is not None:
            return mapped
        if name != "gpu_clock":
            return None
        labels = sample.labels
        clock_type = labels.get("clock_type", "")
        clock_type = clock_type.removeprefix(cls._CLOCK_TYPE_PREFIX).lower()
        field = cls._CLOCK_FIELDS.get(clock_type)
        if field is None:
            return None
        try:
            return field, int(labels.get("clock_index", ""))
        except ValueError:
            return None

    def _ingest_sample(
        self,
        sample: Sample,
        gpu_data: dict[int, dict[str, tuple[int, float]]],
        gpu_metadata: dict[int, GpuMetadata],
    ) -> None:
        """Fold one Prometheus sample into the per-GPU accumulators.

        A GPU is registered only once one of its samples maps to a field, so an
        exporter reporting nothing this collector uses produces no record
        rather than an empty one every scrape.
        """
        if not is_finite_value(sample.value):
            return
        resolved = self._resolve(sample)
        if resolved is None:
            return
        labels = sample.labels
        gpu_index = self._gpu_index_from(labels)
        if gpu_index is None:
            return

        field, rank = resolved
        metrics = gpu_data.setdefault(gpu_index, {})
        held = metrics.get(field)
        if held is not None and held[0] <= rank:
            return
        metrics[field] = (rank, float(sample.value))

        if gpu_index not in gpu_metadata:
            gpu_metadata[gpu_index] = GpuMetadata(
                gpu_index=gpu_index,
                gpu_model_name=labels.get("card_model", "Unknown AMD GPU"),
                gpu_uuid=self._gpu_uuid_from(labels, gpu_index),
                pci_bus_id=None,
                device=None,
                hostname=labels.get("hostname"),
                namespace=labels.get("namespace"),
                pod_name=labels.get("pod"),
                platform=AMD_GPU_TELEMETRY_PLATFORM,
            )

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse one DME scrape into a TelemetryRecord per GPU.

        Returns an empty list for an empty or unparseable payload. A GPU whose
        metrics fail validation is dropped from this scrape alone, with a
        warning, rather than taking the other GPUs' records with it.
        """
        if not metrics_data.strip():
            return []

        current_timestamp = time.time_ns()
        gpu_data: dict[int, dict[str, tuple[int, float]]] = {}
        gpu_metadata: dict[int, GpuMetadata] = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                for sample in family.samples:
                    self._ingest_sample(sample, gpu_data, gpu_metadata)
        except ValueError as e:
            self.warning(f"Failed to parse Prometheus metrics - invalid format: {e}")
            return []

        records = []
        for gpu_index, ranked in gpu_data.items():
            metrics = {field: value for field, (_rank, value) in ranked.items()}
            try:
                record = TelemetryRecord(
                    timestamp_ns=current_timestamp,
                    # The redacted form: this is a hierarchy key, it reaches
                    # metric tags and exports, and it must not carry any
                    # credentials embedded in the exporter URL.
                    telemetry_source_url=self._display_url,
                    **gpu_metadata[gpu_index].model_dump(),
                    telemetry_data=TelemetryMetrics(
                        **self._apply_scaling_factors(metrics)
                    ),
                )
            except ValidationError as e:
                self.warning(
                    f"Dropping GPU {gpu_index} from this sample; its metrics "
                    f"failed validation: {e}"
                )
                continue
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """Convert DME's native units: energy uJ -> MJ, VRAM MiB -> GB."""
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics

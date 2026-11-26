# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""npu-exporter (Prometheus) telemetry collector for Huawei Ascend NPUs.

Scrapes the Ascend ``npu-exporter`` DaemonSet's Prometheus endpoint and maps its
``npu_chip_info_*`` families onto the same vendor-namespaced ``ascend_*`` fields
that :class:`NPUSMITelemetryCollector` writes.

Why this exists alongside the npu-smi collector: the exporter runs on the node
and already owns device access through the Ascend device plugin, so scraping it
needs no ``npu-smi`` binary, no driver mount and no ``huawei.com/Ascend910``
allocation inside AIPerf. That makes it the only viable Ascend source when every
NPU is already allocated to inference pods (AIPerf would otherwise sit Pending),
and it is the source that works from an operator-managed controller pod.

Observed wire format (npu-exporter v7.3.0, Ascend 910C/A3)::

    npu_chip_info_power{container_name="vllm",id="0",model_name="Ascend910-Ascend-V1",
      namespace="...",pcie_bus_info="0000:9D:00.0",pod_name="...",
      vdie_id="E29EF66C-..."} 92.9 1795486097636

Parsing is line-oriented rather than delegated to ``prometheus_client`` because
the exporter also publishes info-style families (``npu_chip_info_name``,
``npu_chip_info_serial_number``) whose values are not numbers; the strict parser
raises on those and would drop the entire scrape.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.mixins import (
    BaseMetricsCollectorMixin,
    TErrorCallback,
    TRecordCallback,
)
from aiperf.common.models import GpuMetadata, TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.constants import (
    ASCEND_AICORE_UTILIZATION_FIELD,
    ASCEND_GPU_TELEMETRY_PLATFORM,
    ASCEND_HEALTH_FIELD,
    ASCEND_MEMORY_USED_FIELD,
    ASCEND_MEMORY_UTILIZATION_FIELD,
    ASCEND_POWER_FIELD,
    ASCEND_TEMPERATURE_FIELD,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["NpuExporterTelemetryCollector"]

# npu-exporter family name -> TelemetryMetrics field. Kept deliberately equal to
# the npu-smi collector's field set so the two Ascend sources are interchangeable
# for every downstream consumer (accumulator, console, CSV/JSONL, dashboard and
# the energy-efficiency analyzer).
NPU_EXPORTER_TO_FIELD_MAPPING: dict[str, str] = {
    "npu_chip_info_power": ASCEND_POWER_FIELD,
    "npu_chip_info_temperature": ASCEND_TEMPERATURE_FIELD,
    "npu_chip_info_utilization": ASCEND_AICORE_UTILIZATION_FIELD,
    "npu_chip_info_hbm_utilization": ASCEND_MEMORY_UTILIZATION_FIELD,
    "npu_chip_info_hbm_used_memory": ASCEND_MEMORY_USED_FIELD,
    "npu_chip_info_health_status": ASCEND_HEALTH_FIELD,
}

# ``npu_chip_info_hbm_used_memory`` is reported in MB; the field is GB, matching
# the npu-smi collector's MiB -> GB conversion.
SCALING_FACTORS: dict[str, float] = {
    ASCEND_MEMORY_USED_FIELD: 1.0 / 1024.0,
}

# ``name{labels} value [timestamp]`` — the exporter always attaches a
# millisecond timestamp, which is accepted and ignored (records are stamped with
# the scrape time so all chips in one scrape share a timestamp).
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>\S+)"
    r"(?:\s+-?\d+)?\s*$"
)
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_labels(raw: str | None) -> dict[str, str]:
    """Parse a Prometheus label set (``a="1",b="2"``) into a dict."""
    if not raw:
        return {}
    return {match.group(1): match.group(2) for match in _LABEL_RE.finditer(raw)}


def _iter_samples(text: str) -> Iterator[tuple[str, dict[str, str], float]]:
    """Yield ``(family, labels, value)`` for numeric samples only.

    Non-numeric sample lines (info-style families such as
    ``npu_chip_info_name``) are skipped rather than failing the scrape.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] == "#":
            continue
        match = _SAMPLE_RE.match(stripped)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        yield match.group("name"), _parse_labels(match.group("labels")), value


def _coerce_health(value: float) -> float:
    """Normalize ``npu_chip_info_health_status`` to the 1.0/0.0 snapshot.

    npu-exporter v7.x reports ``1`` for a healthy chip; anything lower is treated
    as a degraded snapshot, matching what the npu-smi collector emits.
    """
    return 1.0 if value >= 1 else 0.0


def _finalize_metrics(raw_metrics: dict[str, float]) -> TelemetryMetrics:
    """Apply unit scaling and health normalization to one chip's raw samples."""
    scaled = raw_metrics.copy()
    for field_name, factor in SCALING_FACTORS.items():
        if field_name in scaled:
            scaled[field_name] *= factor
    if ASCEND_HEALTH_FIELD in scaled:
        scaled[ASCEND_HEALTH_FIELD] = _coerce_health(scaled[ASCEND_HEALTH_FIELD])
    return TelemetryMetrics(**scaled)


class NpuExporterTelemetryCollector(BaseMetricsCollectorMixin[TelemetryRecord]):
    """Collects Ascend NPU telemetry from an ``npu-exporter`` Prometheus endpoint.

    HTTP counterpart to :class:`NPUSMITelemetryCollector`: no local binary, no
    driver mount and no device allocation required.

    Features:
        - One record per chip, keyed by ``vdie_id`` (falling back to ``id``)
        - Automatic pod/namespace attribution from the exporter's labels
        - Tolerant parsing: info-style (non-numeric) families are skipped
        - Same ``ascend_*`` field set as the npu-smi collector

    Args:
        dcgm_url: Full metrics URL of one node's exporter, e.g.
            ``http://10.233.224.133:8082/metrics``. Each node runs its own
            exporter, so multi-node coverage means one URL per node.
        collection_interval: Interval in seconds between scrapes.
        reachability_timeout: Timeout in seconds for the reachability probe.
        record_callback: Async callback invoked with collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Async callback invoked on collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    @classmethod
    def validate_environment(cls) -> None:
        """Remote HTTP collector - no local environment to validate."""

    def __init__(
        self,
        dcgm_url: str,
        *,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        reachability_timeout: float = Environment.GPU.REACHABILITY_TIMEOUT,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "npu_exporter_collector",
    ) -> None:
        super().__init__(
            endpoint_url=dcgm_url,
            collection_interval=collection_interval,
            reachability_timeout=reachability_timeout,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )

    async def _collect_and_process_metrics(self) -> None:
        """Fetch one scrape, parse it, and dispatch the records."""
        fetch_result = await self._fetch_metrics_text()
        if fetch_result.is_duplicate:
            return
        records = self._parse_metrics_to_records(fetch_result.text)
        await self._send_records_via_callback(records)

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Group ``npu_chip_info_*`` samples per chip into telemetry records.

        Args:
            metrics_data: Raw Prometheus exposition text from npu-exporter.

        Returns:
            One ``TelemetryRecord`` per chip that reported at least one mapped
            signal. Chips with no usable value are omitted rather than emitted
            as zeros.
        """
        if not metrics_data.strip():
            return []

        timestamp_ns = time.time_ns()
        metadata_by_key, metrics_by_key = self._group_samples(metrics_data)

        records: list[TelemetryRecord] = []
        for chip_key, raw_metrics in metrics_by_key.items():
            telemetry_data = _finalize_metrics(raw_metrics)
            if not telemetry_data.model_fields_set:
                continue
            records.append(
                TelemetryRecord(
                    timestamp_ns=timestamp_ns,
                    telemetry_source_url=self._display_url,
                    **metadata_by_key[chip_key].model_dump(),
                    telemetry_data=telemetry_data,
                )
            )
        return records

    def _group_samples(
        self, metrics_data: str
    ) -> tuple[dict[str, GpuMetadata], dict[str, dict[str, float]]]:
        """Bucket mapped samples by chip, keeping the first metadata seen."""
        metadata_by_key: dict[str, GpuMetadata] = {}
        metrics_by_key: dict[str, dict[str, float]] = {}
        for family, labels, value in _iter_samples(metrics_data):
            field_name = NPU_EXPORTER_TO_FIELD_MAPPING.get(family)
            if field_name is None:
                continue
            chip_key = self._chip_key(labels)
            if chip_key is None:
                continue
            if chip_key not in metadata_by_key:
                metadata_by_key[chip_key] = self._build_metadata(labels)
            metrics_by_key.setdefault(chip_key, {})[field_name] = value
        return metadata_by_key, metrics_by_key

    @staticmethod
    def _chip_key(labels: dict[str, str]) -> str | None:
        """Stable per-chip key: ``vdie_id``, falling back to the chip ``id``."""
        vdie_id = labels.get("vdie_id")
        if vdie_id:
            return vdie_id
        chip_id = labels.get("id")
        return f"id-{chip_id}" if chip_id else None

    @staticmethod
    def _build_metadata(labels: dict[str, str]) -> GpuMetadata:
        """Build static GPU metadata from the exporter's per-chip labels."""
        chip_id = labels.get("id", "")
        try:
            gpu_index = int(chip_id)
        except ValueError:
            gpu_index = 0
        return GpuMetadata(
            gpu_index=gpu_index,
            gpu_uuid=labels.get("vdie_id") or f"NPU-{chip_id or gpu_index}",
            gpu_model_name=labels.get("model_name") or "Ascend NPU",
            pci_bus_id=labels.get("pcie_bus_info"),
            device=f"npu{gpu_index}",
            hostname=None,
            namespace=labels.get("namespace"),
            pod_name=labels.get("pod_name"),
            platform=ASCEND_GPU_TELEMETRY_PLATFORM,
        )

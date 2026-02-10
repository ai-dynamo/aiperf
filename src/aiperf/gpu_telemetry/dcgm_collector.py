# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.environment import Environment
from aiperf.common.mixins import (
    BaseMetricsCollectorMixin,
    TErrorCallback,
    TRecordCallback,
)
from aiperf.common.models import GpuMetadata, TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.constants import DCGM_TO_FIELD_MAPPING

__all__ = ["DCGMTelemetryCollector"]

# Unit conversion scaling factors for DCGM metrics
SCALING_FACTORS = {
    "energy_consumption": 1e-9,  # mJ -> MJ
    "gpu_memory_used": 1.048576e-3,  # MiB -> GB
    "sm_utilization": 100,  # ratio (0-1) -> percentage (0-100)
    "power_violation": 1e-3,  # ns -> µs
}


class DCGMTelemetryCollector(BaseMetricsCollectorMixin[TelemetryRecord]):
    """Collects GPU telemetry metrics from DCGM exporter HTTP endpoints.

    Async collector that fetches GPU metrics from DCGM Prometheus exporter and converts
    them to TelemetryRecord objects. Extends BaseMetricsCollectorMixin for HTTP
    collection patterns and uses prometheus_client for robust metric parsing.

    This is the default collector type for GPU telemetry when DCGM is available.
    For local GPU monitoring without DCGM, see PyNVMLTelemetryCollector.

    Features:
        - Async HTTP collection with aiohttp
        - DCGM Prometheus format parsing
        - GPU metadata extraction (UUID, model, hostname)
        - Automatic unit scaling (e.g., milliwatts to watts)
        - Callback-based record delivery

    Args:
        dcgm_url: URL of the DCGM metrics endpoint (e.g., "http://localhost:9400/metrics")
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        reachability_timeout: Timeout in seconds for reachability checks (default: from Environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    def __init__(
        self,
        dcgm_url: str,
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
        """Collect metrics from DCGM endpoint and process them into TelemetryRecord objects.

        Implements the abstract method from BaseMetricsCollectorMixin.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from DCGM endpoint (via mixin's _fetch_metrics_text)
        2. Parses Prometheus-format data into TelemetryRecord objects
        3. Sends records via callback (via mixin's _send_records_via_callback)

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        fetch_result = await self._fetch_metrics_text()
        if fetch_result.is_duplicate:
            return
        records = self._parse_metrics_to_records(fetch_result.text)
        await self._send_records_via_callback(records)

    def _parse_metrics_to_records(self, metrics_data: str) -> list[TelemetryRecord]:
        """Parse DCGM metrics text into TelemetryRecord objects using prometheus_client.

        Processes Prometheus exposition format metrics from DCGM exporter:
        1. Parses metric families using prometheus_client parser
        2. Extracts GPU metadata (UUID, model name, hostname, etc.) from labels
        3. Maps DCGM metric names to TelemetryRecord field names
        4. Applies scaling factors to convert units (e.g., milliwatts to watts)
        5. Aggregates metrics by GPU index into TelemetryRecord objects

        Skips non-finite values (NaN, inf) and metrics without valid GPU index.

        Args:
            metrics_data: Raw metrics text from DCGM exporter in Prometheus format

        Returns:
            list[TelemetryRecord]: List of TelemetryRecord objects, one per GPU with valid data.
                Returns empty list if metrics_data is empty or parsing fails.
        """
        if not metrics_data.strip():
            return []

        current_timestamp = time.time_ns()
        gpu_data = {}
        gpu_metadata = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                for sample in family.samples:
                    metric_name = sample.name
                    labels = sample.labels
                    value = sample.value

                    # Skip non-finite values early (value != value checks for NaN)
                    if isinstance(value, float) and (
                        value != value or value in (float("inf"), float("-inf"))
                    ):
                        continue

                    gpu_index = labels.get("gpu")
                    if gpu_index is not None:
                        try:
                            gpu_index = int(gpu_index)
                        except ValueError:
                            continue
                    else:
                        continue

                    if gpu_index not in gpu_metadata:
                        gpu_metadata[gpu_index] = GpuMetadata(
                            gpu_index=gpu_index,
                            gpu_model_name=labels.get("modelName"),
                            gpu_uuid=labels.get("UUID"),
                            pci_bus_id=labels.get("pci_bus_id"),
                            device=labels.get("device"),
                            hostname=labels.get("Hostname"),
                            namespace=labels.get("namespace"),
                            pod_name=labels.get("pod"),
                        )

                    base_metric_name = metric_name.removesuffix("_total")
                    if base_metric_name in DCGM_TO_FIELD_MAPPING:
                        field_name = DCGM_TO_FIELD_MAPPING[base_metric_name]
                        gpu_data.setdefault(gpu_index, {})[field_name] = value
        except ValueError as e:
            self.warning(f"Failed to parse Prometheus metrics - invalid format: {e}")
            return []

        records = []
        for gpu_index, metrics in gpu_data.items():
            metadata = gpu_metadata.get(gpu_index)
            if metadata is None:
                self.warning(f"No metadata found for GPU {gpu_index}")
                continue
            scaled_metrics = self._apply_scaling_factors(metrics)

            record = TelemetryRecord(
                timestamp_ns=current_timestamp,
                dcgm_url=self.endpoint_url,
                **metadata.model_dump(),
                telemetry_data=TelemetryMetrics(**scaled_metrics),
            )
            records.append(record)

        return records

    def _apply_scaling_factors(self, metrics: dict) -> dict:
        """Apply scaling factors to convert raw DCGM units to display units.

        Converts metrics from DCGM's native units to human-readable units:
        - Energy: millijoules -> megajoules
        - Memory: MiB -> GB
        - SM utilization: ratio (0-1) -> percentage (0-100)

        Only applies scaling to metrics present in the input dict. None values are preserved.

        Args:
            metrics: Dict of metric_name -> raw_value from DCGM

        Returns:
            dict: New dict with scaled values ready for display. Unscaled metrics are copied as-is.
        """
        scaled_metrics = metrics.copy()
        for metric, factor in self._scaling_factors.items():
            if metric in scaled_metrics and scaled_metrics[metric] is not None:
                scaled_metrics[metric] *= factor
        return scaled_metrics

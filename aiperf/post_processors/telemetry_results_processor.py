# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import TelemetryHierarchy, TelemetryRecord
from aiperf.common.protocols import (
    TelemetryResultsProcessorProtocol,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_RESULTS)
class TelemetryResultsProcessor(BaseMetricsProcessor):
    """Process individual TelemetryRecord objects into hierarchical storage."""

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        """
        Initialize the TelemetryResultsProcessor, preparing internal storage and metric unit mappings.
        
        Parameters:
            user_config (UserConfig): Configuration for the current user/session used by the processor; passed to the base class initializer.
        
        Detailed behavior:
            - Creates an empty TelemetryHierarchy to store incoming telemetry records in a hierarchical structure.
            - Initializes a mapping of telemetry metric names to their display units (e.g., "gpu_power_usage" -> "W", "energy_consumption" -> "MJ").
        """
        super().__init__(user_config=user_config, **kwargs)

        self._telemetry_hierarchy = TelemetryHierarchy()

        self._metric_units = {
            "gpu_power_usage": "W",
            "energy_consumption": "MJ",
            "gpu_utilization": "%",
            "gpu_memory_used": "GB",
            "sm_clock_frequency": "MHz",
            "memory_clock_frequency": "MHz",
            "memory_temperature": "°C",
            "gpu_temperature": "°C",
        }

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """
        Add a telemetry record to the processor's hierarchical storage.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU metrics and hierarchical metadata (for example `dcgm_url`, `gpu_uuid`, and metric samples).
        """

        if self.is_trace_enabled:
            self.trace(
                f"Processing telemetry for GPU {record.gpu_uuid} from {record.dcgm_url}"
            )

        self._telemetry_hierarchy.add_record(record)

    async def summarize(self) -> list[MetricResult]:
        """
        Produce MetricResult objects for each tracked metric of every GPU for display and export.
        
        Constructs per-metric tags and headers (including a sanitized DCGM endpoint identifier and GPU index/UUID prefix) and retrieves a MetricResult from the stored telemetry for each combination.
        
        Returns:
            list[MetricResult]: MetricResult objects, one per GPU per tracked metric. Tags are formatted for dashboard filtering using the sanitized DCGM URL and GPU identifiers.
        """

        results = []

        for dcgm_url, gpu_data in self._telemetry_hierarchy.dcgm_endpoints.items():
            for gpu_uuid, telemetry_data in gpu_data.items():
                gpu_index = telemetry_data.metadata.gpu_index

                # Iterate through available metrics from the metric units we track
                for metric_name in self._metric_units:
                    try:
                        dcgm_tag = (
                            dcgm_url.replace(":", "_")
                            .replace("/", "_")
                            .replace(".", "_")
                        )
                        tag = f"{metric_name}_dcgm_{dcgm_tag}_gpu{gpu_index}_{gpu_uuid[:8]}"

                        metric_display = metric_name.replace("_", " ").title()
                        header = (
                            f"{metric_display} (GPU {gpu_index}, {gpu_uuid[:8]}...)"
                        )

                        unit = self._metric_units.get(metric_name, "")

                        result = telemetry_data.get_metric_result(
                            metric_name, tag, header, unit
                        )
                        results.append(result)
                    except Exception:
                        continue

        return results

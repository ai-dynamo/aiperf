# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import TelemetryHierarchy, TelemetryRecord
from aiperf.common.protocols import (
    TelemetryResultsProcessorProtocol,
)
from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_RESULTS)
class TelemetryResultsProcessor(BaseMetricsProcessor):
    """Process individual TelemetryRecord objects into hierarchical storage."""

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)

        self._telemetry_hierarchy = TelemetryHierarchy()

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record into hierarchical storage.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """

        self._telemetry_hierarchy.add_record(record)

    async def summarize(self) -> list[MetricResult]:
        """Generate MetricResult list for real-time display and final export.

        This method is called by RecordsManager for:
        1. Final results generation when profiling completes
        2. [AIP-355] TODO: @ilana-n [FUTURE] real-time dashboard updates (every DEFAULT_REALTIME_METRICS_INTERVAL)
          when user-set flag is enabled

        Returns:
            List of MetricResult objects, one per GPU per metric type.
            Tags follow hierarchical naming pattern for dashboard filtering.
        """

        results = []

        for dcgm_url, gpu_data in self._telemetry_hierarchy.dcgm_endpoints.items():
            for gpu_uuid, telemetry_data in gpu_data.items():
                gpu_index = telemetry_data.metadata.gpu_index

                for (
                    metric_display,
                    metric_name,
                    unit_enum,
                ) in GPU_TELEMETRY_METRICS_CONFIG:
                    try:
                        dcgm_tag = (
                            dcgm_url.replace(":", "_")
                            .replace("/", "_")
                            .replace(".", "_")
                        )
                        # Use first 12 chars of UUID for readability while maintaining uniqueness
                        tag = f"{metric_name}_dcgm_{dcgm_tag}_gpu{gpu_index}_{gpu_uuid[:12]}"

                        header = (
                            f"{metric_display} (GPU {gpu_index}, {gpu_uuid[:12]}...)"
                        )

                        unit = unit_enum.value

                        result = telemetry_data.get_metric_result(
                            metric_name, tag, header, unit
                        )
                        results.append(result)
                    except NoMetricValue:
                        self.debug(
                            f"No data available for metric '{metric_name}' on GPU {gpu_uuid[:12]} from {dcgm_url}"
                        )
                        continue
                    except Exception as e:
                        self.exception(
                            f"Unexpected error generating metric result for '{metric_name}' on GPU {gpu_uuid[:12]} from {dcgm_url}: {e}"
                        )
                        continue

        return results

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.common.protocols import TelemetryResultsProcessorProtocol
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(TelemetryResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.TELEMETRY_EXPORT)
class TelemetryExportResultsProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[TelemetryRecord]
):
    """Exports per-record GPU telemetry data to JSONL files.

    This processor streams each TelemetryRecord as it arrives from the TelemetryManager,
    writing one JSON line per GPU per collection cycle. The output format supports
    multi-endpoint and multi-GPU time series analysis.

    Each line contains:
        - timestamp_ns: Collection timestamp in nanoseconds
        - dcgm_url: DCGM endpoint URL for filtering by endpoint
        - gpu_uuid: Unique GPU identifier
        - gpu_index: GPU index on the host
        - hostname: Host machine name
        - gpu_model_name: GPU model string
        - telemetry_data: Complete metrics snapshot (power, utilization, memory, etc.)
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ):
        output_file: Path = user_config.output.profile_export_gpu_telemetry_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self.info(f"GPU telemetry export enabled: {self.output_file}")

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record by writing it to JSONL.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        try:
            await self.buffered_write(record)
        except Exception as e:
            self.error(f"Failed to write GPU telemetry record: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

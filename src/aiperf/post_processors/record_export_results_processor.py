# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import AIPERF_DEV_MODE, DEFAULT_RECORD_EXPORT_BATCH_SIZE
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ExportLevel, ResultsProcessorType
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.common.protocols import ResultsProcessorProtocol
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.RECORD_EXPORT)
class RecordExportResultsProcessor(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[MetricRecordInfo]
):
    """Exports per-record metrics to JSONL with display unit conversion and filtering."""

    def __init__(
        self,
        service_id: str,
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        export_level = user_config.output.export_level
        if export_level not in (ExportLevel.RECORDS, ExportLevel.RAW):
            raise PostProcessorDisabled(
                f"Record export results processor is disabled for export level {export_level}"
            )

        output_file = user_config.output.profile_export_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        # Initialize parent classes with the output file
        super().__init__(
            output_file=output_file,
            batch_size=DEFAULT_RECORD_EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self.show_internal = (
            AIPERF_DEV_MODE and service_config.developer.show_internal_metrics
        )
        self.info(f"Record metrics export enabled: {self.output_file}")

    async def process_result(self, record_data: MetricRecordsData) -> None:
        try:
            metric_dict = MetricRecordDict(record_data.metrics)
            display_metrics = metric_dict.to_display_dict(
                MetricRegistry, self.show_internal
            )
            if not display_metrics:
                return

            record_info = MetricRecordInfo(
                metadata=record_data.metadata,
                metrics=display_metrics,
                error=record_data.error,
            )

            # Write using the buffered writer mixin (handles batching and flushing)
            await self.buffered_write(record_info)

        except Exception as e:
            self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

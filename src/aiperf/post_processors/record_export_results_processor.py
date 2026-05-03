# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


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
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            user_config=user_config,
            **kwargs,
        )

        self.show_internal = (
            Environment.DEV.MODE and Environment.DEV.SHOW_INTERNAL_METRICS
        )
        self.show_experimental = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        self.export_http_trace = user_config.output.export_http_trace
        self.info(f"Record metrics export enabled: {self.output_file}")
        if self.export_http_trace:
            self.info("HTTP trace export enabled (--export-http-trace)")

    async def process_result(self, record_data: MetricRecordsData) -> None:
        try:
            metric_dict = MetricRecordDict(record_data.metrics)
            display_metrics = metric_dict.to_display_dict(
                MetricRegistry, self.show_internal, self.show_experimental
            )
            # Skip records with no displayable metrics UNLESS they have an error
            # (error records should always be exported for debugging/analysis)
            if not display_metrics and not record_data.error:
                return

            # Convert trace data to export format (wall-clock timestamps) if enabled
            export_trace_data = None
            if self.export_http_trace and record_data.trace_data:
                export_trace_data = record_data.trace_data.to_export()

            record_info = MetricRecordInfo(
                metadata=record_data.metadata,
                metrics=display_metrics,
                trace_data=export_trace_data,
                error=record_data.error,
            )

            # Write using the buffered writer mixin (handles batching and flushing)
            await self.buffered_write(record_info)

        except Exception as e:
            self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

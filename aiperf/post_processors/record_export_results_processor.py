# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aiofiles

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import AIPERF_DEV_MODE
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ExportLevel, ResultsProcessorType
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.hooks import on_stop
from aiperf.common.messages.inference_messages import MetricRecordsMessage
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.common.protocols import ResultsProcessorProtocol
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.RECORD_EXPORT)
class RecordExportResultsProcessor(BaseMetricsProcessor):
    """Exports per-record metrics to JSONL with display unit conversion and filtering."""

    def __init__(
        self,
        service_id: str,
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(user_config=user_config, **kwargs)
        export_level = user_config.output.export_level
        export_file_path = user_config.output.export_file_path
        self.enabled = export_level in (ExportLevel.RECORDS, ExportLevel.RAW)

        if not self.enabled:
            return

        self.output_file = user_config.output.artifact_directory / export_file_path
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.record_count = 0
        self.show_internal = (
            AIPERF_DEV_MODE and service_config.developer.show_internal_metrics
        )
        if self.enabled:
            self.info(f"Record metrics export enabled: {self.output_file}")
            self.output_file.unlink(missing_ok=True)

    async def process_result(self, message: MetricRecordsMessage) -> None:
        if not self.enabled:
            return

        record_dicts = [MetricRecordDict(result) for result in message.results]
        for record_dict in record_dicts:
            try:
                display_metrics = record_dict.to_display_dict(
                    MetricRegistry, self.show_internal
                )
                if not display_metrics:
                    continue

                record_info = MetricRecordInfo(
                    record_id=message.record_id,
                    metadata=message.metadata,
                    metrics=display_metrics,
                )
                json_str = record_info.model_dump_json()

                async with aiofiles.open(
                    self.output_file, mode="a", encoding="utf-8"
                ) as f:
                    await f.write(json_str)
                    await f.write("\n")

                self.record_count += 1
                if self.record_count % 100 == 0:
                    self.debug(f"Wrote {self.record_count} record metrics")

            except Exception as e:
                self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

    @on_stop
    async def _shutdown(self) -> None:
        if self.enabled:
            self.info(
                f"RecordExportResultsProcessor: {self.record_count} records written to {self.output_file}"
            )

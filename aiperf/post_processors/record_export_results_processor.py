# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import aiofiles

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ExportLevel, ResultsProcessorType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.messages.inference_messages import MetricRecordsData
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
        export_file_path = user_config.output.profile_export_file
        if export_level not in (ExportLevel.RECORDS, ExportLevel.RAW):
            raise PostProcessorDisabled(
                f"Record export results processor is disabled for export level {export_level}"
            )

        self.output_file = user_config.output.artifact_directory / export_file_path
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.record_count = 0
        self.show_internal = Environment.DEV_MODE and Environment.SHOW_INTERNAL_METRICS
        self.show_experimental = (
            Environment.DEV_MODE and Environment.SHOW_EXPERIMENTAL_METRICS
        )
        self.info(f"Record metrics export enabled: {self.output_file}")
        self.output_file.unlink(missing_ok=True)

        # File handle for persistent writes with batching
        self._file_handle = None
        self._buffer: list[str] = []
        self._batch_size = Environment.RECORD_EXPORT_BATCH_SIZE
        self._buffer_lock = asyncio.Lock()

    @on_init
    async def _open_file(self) -> None:
        """Open a persistent file handle for writing."""
        self._file_handle = await aiofiles.open(
            self.output_file, mode="w", encoding="utf-8"
        )

    async def process_result(self, record_data: MetricRecordsData) -> None:
        try:
            metric_dict = MetricRecordDict(record_data.metrics)
            display_metrics = metric_dict.to_display_dict(
                MetricRegistry, self.show_internal, self.show_experimental
            )
            if not display_metrics:
                return

            record_info = MetricRecordInfo(
                metadata=record_data.metadata,
                metrics=display_metrics,
                error=record_data.error,
            )
            json_str = record_info.model_dump_json()

            buffer_to_flush = None
            async with self._buffer_lock:
                self._buffer.append(json_str)
                self.record_count += 1

                if len(self._buffer) >= self._batch_size:
                    buffer_to_flush = self._buffer
                    self._buffer = []

            if buffer_to_flush:
                await self._flush_buffer(buffer_to_flush)

        except Exception as e:
            self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

    async def _flush_buffer(self, buffer_to_flush: list[str]) -> None:
        """Write buffered records to disk."""
        if not buffer_to_flush:
            return

        try:
            self.debug(lambda: f"Flushing {len(buffer_to_flush)} records to file")
            await self._file_handle.write("\n".join(buffer_to_flush))
            await self._file_handle.write("\n")
            await self._file_handle.flush()
        except Exception as e:
            self.error(f"Failed to flush buffer: {e}")
            raise

    @on_stop
    async def _shutdown(self) -> None:
        async with self._buffer_lock:
            buffer_to_flush = self._buffer
            self._buffer = []

        try:
            await self._flush_buffer(buffer_to_flush)
        except Exception as e:
            self.error(f"Failed to flush remaining buffer during shutdown: {e}")

        if self._file_handle is not None:
            try:
                await self._file_handle.close()
            except Exception as e:
                self.error(f"Failed to close file handle during shutdown: {e}")
            finally:
                self._file_handle = None

        self.info(
            f"RecordExportResultsProcessor: {self.record_count} records written to {self.output_file}"
        )

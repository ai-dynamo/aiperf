# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aiofiles

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import AIPERF_DEV_MODE, DEFAULT_RECORD_EXPORT_BATCH_SIZE
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ExportLevel, ResultsProcessorType
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
        self.show_internal = (
            AIPERF_DEV_MODE and service_config.developer.show_internal_metrics
        )
        self.info(f"Record metrics export enabled: {self.output_file}")
        self.output_file.unlink(missing_ok=True)

        # File handle for persistent writes
        self._file_handle = None
        self._buffer: list[str] = []
        self._batch_size = DEFAULT_RECORD_EXPORT_BATCH_SIZE

    @on_init
    async def _open_file(self) -> None:
        """Open a persistent file handle for writing."""
        self._file_handle = await aiofiles.open(
            self.output_file, mode="w", encoding="utf-8"
        )

    async def _flush_buffer(self) -> None:
        """Write buffered records to disk."""
        if not self._buffer:
            return

        try:
            self.debug(lambda: f"Flushing {len(self._buffer)} records to file")
            await self._file_handle.write("\n".join(self._buffer))
            await self._file_handle.flush()
            self._buffer.clear()
        except Exception as e:
            self.error(f"Failed to flush buffer: {e}")
            raise

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
            json_str = record_info.model_dump_json()

            self._buffer.append(json_str)

            self.record_count += 1

            # Flush buffer when batch size is reached
            if len(self._buffer) >= self._batch_size:
                await self._flush_buffer()

        except Exception as e:
            self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

    @on_stop
    async def _shutdown(self) -> None:
        # Flush any remaining buffered records
        try:
            await self._flush_buffer()
            # Add a newline to the end of the file
            await self._file_handle.write("\n")
            await self._file_handle.flush()
        except Exception as e:
            self.error(f"Failed to flush remaining buffer during shutdown: {e}")

        # Close the file handle
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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class RecordExportJSONLWriter(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[MetricRecordInfo]
):
    """Exports per-record metrics to JSONL with display unit conversion and filtering.

    Registered as a ``stream_exporter``: writes each record to the on-disk
    JSONL sink as it arrives, with no end-of-run aggregation. Self-disables
    when ``artifacts.export_level`` is not ``RECORDS`` or ``RAW``.
    """

    def __init__(
        self,
        service_id: str,
        run: BenchmarkRun,
        **kwargs,
    ):
        export_level = run.cfg.artifacts.export_level
        if export_level not in (ExportLevel.RECORDS, ExportLevel.RAW):
            raise PostProcessorDisabled(
                f"Record export JSONL writer is disabled for export level {export_level}"
            )

        output_file = run.cfg.artifacts.profile_export_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            run=run,
            **kwargs,
        )

        self.show_internal = (
            Environment.DEV.MODE and Environment.DEV.SHOW_INTERNAL_METRICS
        )
        self.show_experimental = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        self.export_http_trace = run.cfg.artifacts.trace
        self.info(f"Record metrics export enabled: {self.output_file}")
        if self.export_http_trace:
            self.info("HTTP trace export enabled (--export-http-trace)")

    async def process_record(self, record_data: MetricRecordsData) -> None:
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
                spec_decode_acceptance=record_data.spec_decode_acceptance,
                error=record_data.error,
            )

            # Write using the buffered writer mixin (handles batching and flushing)
            await self.buffered_write(record_info)

        except Exception as e:  # noqa: BLE001 - per-record; skip bad record and continue
            self.error(f"Failed to write record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """No aggregation needed for JSONL export."""
        return []

    async def finalize(self) -> None:
        """Flush the JSONL writer at end-of-run.

        Called by RecordsManager after the final summarize() and before
        publishing the records-result message. Without this, downstream
        consumers can see results_exported=True before this writer's
        @on_stop _close_file fires — opening a window where /api/results
        serves a partial profile_export.jsonl.
        """
        await self._close_file()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.metric_records_wire import MetricRecordsData
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class RecordExportJSONLWriter(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[MetricRecordInfo]
):
    """Exports per-record metrics to JSONL with display unit conversion and filtering."""

    def __init__(
        self,
        service_id: str,
        run: BenchmarkRun,
        **kwargs,
    ):
        # Check if records export is enabled (records list is not False/empty)
        config = run.cfg
        artifacts = config.artifacts
        records_enabled = artifacts.records and artifacts.records is not False
        raw_enabled = artifacts.raw
        if not records_enabled and not raw_enabled:
            raise PostProcessorDisabled(
                "Record export JSONL writer is disabled (artifacts.records is not enabled)"
            )
        if (
            isinstance(artifacts.records, list)
            and "jsonl" not in artifacts.records
            and not raw_enabled
        ):
            raise PostProcessorDisabled(
                "JSONL record export disabled: 'jsonl' not in artifacts.records"
            )

        # Build output file path from artifacts config
        output_file = artifacts.profile_export_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        # Initialize parent classes with the output file
        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            flush_interval=Environment.RECORD.EXPORT_FLUSH_INTERVAL,
            run=run,
            **kwargs,
        )

        self.show_internal = (
            Environment.DEV.MODE and Environment.DEV.SHOW_INTERNAL_METRICS
        )
        self.show_experimental = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        self.export_http_trace = config.artifacts.trace
        self.export_per_chunk_data = config.artifacts.per_chunk_data
        self.info(f"Record metrics export enabled: {self.output_file}")
        if self.export_http_trace:
            self.info("HTTP trace export enabled (artifacts.trace)")
        if self.export_per_chunk_data:
            self.info("Per-chunk data export enabled (artifacts.per_chunk_data)")

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

            # Filter out list-valued metrics (per-chunk arrays) unless explicitly enabled
            if not self.export_per_chunk_data:
                display_metrics = {
                    k: v
                    for k, v in display_metrics.items()
                    if not isinstance(v.value, list)
                }

            # Convert trace data to export format (wall-clock timestamps) if enabled.
            # trace_data is a native msgspec Struct (BaseTraceData / AioHttpTraceData)
            # on the wire; call its to_export() directly.
            export_trace_data = None
            if self.export_http_trace and record_data.trace_data:
                export_trace_data = record_data.trace_data.to_export()

            from aiperf.common.metric_records_wire import wire_error_to_domain_error

            record_info = MetricRecordInfo(
                metadata=record_data.metadata,
                metrics=display_metrics,
                trace_data=export_trace_data,
                error=wire_error_to_domain_error(record_data.error),
            )

            # Write using the buffered writer mixin (handles batching and flushing)
            await self.buffered_write(record_info)

        except Exception as e:  # noqa: BLE001 - per-record; skip bad record and continue
            self.error(f"Failed to write record metrics: {e}")

    # ``RecordExportJSONLWriter`` is dual-registered in ``plugins.yaml`` as
    # both ``results_processor`` (legacy ``ResultsProcessorProtocol``) and
    # ``stream_exporter`` (new ``StreamExporterProtocol``). The two protocols
    # use different method names — ``process_result`` vs ``process_record`` —
    # for the same single-arg ``MetricRecordsData`` input. Aliasing keeps both
    # records-manager dispatch paths working with one implementation.
    process_result = process_record

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. For this processor, we don't need to summarize anything."""
        return []

    async def finalize(self) -> None:
        """Flush the JSONL writer at end-of-run.

        Called by RecordsManager._process_results AFTER the final summarize()
        and BEFORE publishing ProcessRecordsResultMessage. Without this,
        the operator's progress poll sees results_exported=True (set by the
        controller after marker write) before this processor's @on_stop
        _close_file fires — opening a window where /api/results/list serves
        a partial profile_export.jsonl.
        """
        await self._close_file()

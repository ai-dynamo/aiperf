# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-RecordProcessor JSONL shard writer + aggregator for computed metric records.

Shards the computed-record JSONL export the same way raw records and output
fragments are already sharded on this branch: each parallel ``RecordProcessor``
runs its own observer and writes an independent
``<artifacts.dir>/records_shards/records_{id}.jsonl`` shard (no cross-processor
contention), and a ``data_exporter`` aggregator concatenates the shards into the
final ``profile_export.jsonl`` at profile completion. This replaces the single
``RecordExportJSONLWriter`` that ran on the lone RecordsManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.record_models import MetricRecordInfo
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.shard_writer import ShardAggregatorMixin, ShardWriterMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.post_processors.record_observer_context import RecordObserverContext

# Export levels that emit the per-record computed JSONL. summary emits nothing.
_RECORD_EXPORT_LEVELS = (ExportLevel.RECORDS, ExportLevel.RAW)


class RecordShardJSONLWriter(
    ShardWriterMixin, BufferedJSONLWriterMixin[MetricRecordInfo]
):
    """Writes computed per-record metrics to a per-processor JSONL shard.

    A ``record_observer`` loaded once per ``RecordProcessor`` and fed every
    record via ``observe(ctx)``; it pulls the computed ``MetricRecordsData`` off
    ``ctx.metrics``. Each processor writes to
    ``<artifacts.dir>/records_shards/records_{id}.jsonl`` so writers never
    contend; ``RecordShardJSONLAggregator`` merges the shards into
    ``profile_export.jsonl`` at profile completion.
    """

    def __init__(
        self,
        service_id: str | None,
        run: BenchmarkRun,
        **kwargs,
    ):
        export_level = run.cfg.artifacts.export_level
        if export_level not in _RECORD_EXPORT_LEVELS:
            raise PostProcessorDisabled(
                f"Record shard JSONL writer is disabled for export level {export_level}"
            )

        output_file = self.shard_output_file(
            run.cfg.artifacts.dir,
            OutputDefaults.RECORDS_SHARDS_FOLDER,
            prefix="records",
            ext="jsonl",
            service_id=service_id,
        )

        self.show_internal = (
            Environment.DEV.MODE and Environment.DEV.SHOW_INTERNAL_METRICS
        )
        self.show_experimental = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        self.export_http_trace = run.cfg.artifacts.trace

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            service_id=service_id,
            run=run,
            **kwargs,
        )

        self.info(f"Record shard JSONL writer enabled: {self.output_file}")
        if self.export_http_trace:
            self.info("HTTP trace export enabled (--export-http-trace)")

    async def observe(self, ctx: RecordObserverContext) -> None:
        """Persist the computed per-record metrics for a single record."""
        record_data = ctx.metrics
        if record_data is None:
            return
        try:
            metric_dict = MetricRecordDict(record_data.metrics)
            display_metrics = metric_dict.to_display_dict(
                MetricRegistry, self.show_internal, self.show_experimental
            )
            # Skip records with no displayable metrics UNLESS they carry an error
            # (error records are always exported for debugging/analysis).
            if not display_metrics and not record_data.error:
                return

            export_trace_data = None
            if self.export_http_trace and record_data.trace_data:
                export_trace_data = record_data.trace_data.to_export()

            record_info = MetricRecordInfo(
                metadata=record_data.metadata,
                metrics=display_metrics,
                trace_data=export_trace_data,
                error=record_data.error,
            )
            await self.buffered_write(record_info)

        except Exception as e:  # noqa: BLE001 - per-record; skip bad record and continue
            self.error(f"Failed to write record metrics: {e}")


class RecordShardJSONLAggregator(ShardAggregatorMixin):
    """Merges per-processor record JSONL shards into ``profile_export.jsonl``."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        artifacts = exporter_config.cfg.artifacts
        if artifacts.export_level not in _RECORD_EXPORT_LEVELS:
            raise DataExporterDisabled(
                f"Record shard JSONL aggregator is disabled for export level {artifacts.export_level}"
            )
        self.output_file = artifacts.profile_export_jsonl_file
        self.shard_dir = (
            artifacts.artifact_directory / OutputDefaults.RECORDS_SHARDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Record Metrics (JSONL)",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Concatenate every ``records_*.jsonl`` shard into the final export file."""
        count = await self._concat_shards(
            self.shard_dir, "records_*.jsonl", self.output_file
        )
        self.info(f"Aggregated {count} record metrics to {self.output_file}")

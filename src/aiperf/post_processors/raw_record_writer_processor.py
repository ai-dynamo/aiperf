# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson

from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.finite import scrub_non_finite
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ParsedResponseRecord,
    RawRecordInfo,
)
from aiperf.common.redact import redact_headers
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.post_processors.shard_writer import ShardAggregatorMixin, ShardWriterMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.post_processors.record_observer_context import RecordObserverContext


class RawRecordWriterProcessor(
    ShardWriterMixin, BufferedJSONLWriterMixin[RawRecordInfo]
):
    """Writes raw request/response data with per-record metrics to JSONL files.

    Each RecordProcessor instance writes to its own file to avoid contention
    and enable efficient parallel I/O in distributed setups.

    File format: JSONL (newline-delimited JSON)
    One complete record per line for streaming efficiency.
    """

    def __init__(
        self,
        service_id: str | None,
        run: BenchmarkRun,
        **kwargs,
    ):
        self.service_id = service_id or "processor"
        self.run = run

        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            raise PostProcessorDisabled(
                f"RawRecordWriter processor is disabled for export level {self.run.cfg.artifacts.export_level}"
            )

        output_file = self.shard_output_file(
            self.run.cfg.artifacts.dir,
            OutputDefaults.RAW_RECORDS_FOLDER,
            prefix="raw_records",
            ext="jsonl",
            service_id=service_id,
        )

        # Initialize the buffered writer mixin
        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.RAW_EXPORT_BATCH_SIZE,
            service_id=service_id,
            run=run,
            **kwargs,
        )

        # Counter of records dropped by the fast-path due to non-JSON
        # payload_bytes or serialisation failures. Exposed so operators can
        # see silent-drop volume instead of it hiding behind a log line.
        self.dropped_record_count: int = 0

        self.info(
            f"RawRecordWriter initialized: {self.output_file} - "
            "FULL request/response data will be exported (files may be large)"
        )

    def _build_export_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RawRecordInfo:
        """Build the export record for a single record.

        ``inference_client`` canonicalises ``payload_bytes`` on every live
        request before transport dispatch, so the exporter reads it directly
        and splices it into the JSONL line via ``orjson.Fragment`` in
        ``buffered_write``. Records that never reached transport carry no
        ``payload_bytes``; those export with ``payload=None`` and rely on the
        attached ``error`` field for replay context (matches the v1
        null-payload invariant).
        """
        ctx = record.request.request_info
        payload_bytes = ctx.payload_bytes if ctx is not None else None

        return RawRecordInfo(
            metadata=metadata,
            start_perf_ns=record.request.start_perf_ns,
            payload=None,
            payload_bytes=payload_bytes,
            request_headers=redact_headers(record.request.request_headers),
            response_headers=None,
            status=record.request.status,
            responses=record.request.responses,
            error=record.request.error,
        )

    async def buffered_write(self, record: RawRecordInfo) -> None:
        """Serialise + buffer a ``RawRecordInfo``.

        Fast path: when ``record.payload_bytes`` is set, splice the bytes
        verbatim into the JSONL line via ``orjson.Fragment`` so the exporter
        never decodes-then-re-encodes the wire payload. Falls back to the
        mixin's generic ``model_dump``-based serialisation when
        ``payload_bytes`` is absent — the only surviving case is a
        pre-transport error record (``_build_export_record`` sets
        ``payload=None, payload_bytes=None`` when the enriched
        ``RecordContext`` carries no ``payload_bytes``).

        ``payload_bytes`` is spliced verbatim with no per-record JSON
        re-parse: the bytes are either produced by ``orjson.dumps`` upstream
        (valid by construction) or loaded from a raw dataset that was already
        parsed at dataset-load time, so re-validating every record here would
        only reintroduce the decode cost the ``Fragment`` path exists to avoid.
        """
        if record.payload_bytes is None:
            await super().buffered_write(record)
            return

        try:
            # Scrub before splicing: scrub_non_finite keeps main's "null on
            # disk = absent" invariant on the metadata/responses fields. The
            # Fragment is inserted afterwards so the wire-exact payload bytes
            # pass through untouched.
            dumped = scrub_non_finite(record.model_dump(exclude_none=True, mode="json"))
            # ``payload_bytes`` carries the wire-exact JSON; substitute it
            # in place of the (absent) ``payload`` dict so orjson emits the
            # pre-encoded bytes with zero re-parsing.
            dumped["payload"] = orjson.Fragment(record.payload_bytes)
            json_bytes = orjson.dumps(dumped)

            buffer_to_flush = None
            self._buffer.append(json_bytes)
            self.lines_written += 1
            if len(self._buffer) >= self._batch_size:
                buffer_to_flush = self._buffer
                self._buffer = []
            if buffer_to_flush:
                # Register the flush task in ``_flush_tasks`` (mirroring the
                # parent mixin) so ``_close_file`` awaits it on shutdown.
                # Without this the fire-and-forget task lives only in
                # ``self.tasks``, which ``_stop_all_tasks`` cancels before the
                # file is closed — losing the whole batch on stop.
                task = self.execute_async(self._flush_buffer(buffer_to_flush))
                self._flush_tasks.add(task)
                task.add_done_callback(self._flush_tasks.discard)
        except Exception as e:
            self.error(f"Failed to write raw record: {e!r}")
            self.dropped_record_count += 1

    async def observe(self, ctx: RecordObserverContext) -> None:
        """Write the raw request/response data for a single record."""
        # Build export record with full parsed record
        record_export = self._build_export_record(ctx.record, ctx.metadata)

        # Write using the buffered writer mixin (handles batching and flushing)
        await self.buffered_write(record_export)


class RawRecordAggregator(ShardAggregatorMixin):
    """Merges per-processor raw-record JSONL shards into the final raw export."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        if self.exporter_config.cfg.artifacts.export_level != ExportLevel.RAW:
            raise DataExporterDisabled(
                f"RawRecordAggregator is disabled for export level {self.exporter_config.cfg.artifacts.export_level}"
            )
        self.output_file = exporter_config.cfg.artifacts.profile_export_raw_jsonl_file
        self.shard_dir = (
            exporter_config.cfg.artifacts.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Raw Records",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Concatenate every ``raw_records_*.jsonl`` shard into the final export."""
        count = await self._concat_shards(
            self.shard_dir, "raw_records_*.jsonl", self.output_file
        )
        self.info(f"Aggregated {count} raw records to {self.output_file}")

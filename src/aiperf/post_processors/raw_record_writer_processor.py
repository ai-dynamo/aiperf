# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

import contextlib

import aiofiles
import orjson

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.mixins import AIPerfLoggerMixin, BufferedJSONLWriterMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ParsedResponseRecord,
    RawRecordInfo,
)
from aiperf.common.redact import redact_headers
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class RawRecordWriterProcessor(BufferedJSONLWriterMixin[RawRecordInfo]):
    """Writes raw request/response data with per-record metrics to JSONL files.

    Each RecordProcessor instance writes to its own file to avoid contention
    and enable efficient parallel I/O in distributed setups.

    File format: JSONL (newline-delimited JSON)
    One complete record per line for streaming efficiency.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs,
    ):
        self.service_id = service_id or "processor"
        self.user_config = user_config

        if self.user_config.output.export_level != ExportLevel.RAW:
            raise PostProcessorDisabled(
                f"RawRecordWriter processor is disabled for export level {self.user_config.output.export_level}"
            )

        # Construct output file path: raw_records/raw_records_processor_{id}.jsonl
        output_dir = (
            user_config.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each processor writes to its own file - avoids locking/contention
        # Sanitize service_id for filename (replace special chars)
        safe_id = self.service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_file = output_dir / f"raw_records_{safe_id}.jsonl"

        # Initialize the buffered writer mixin
        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.RAW_EXPORT_BATCH_SIZE,
            service_id=service_id,
            user_config=user_config,
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
        request before transport dispatch, so the exporter reads it
        directly and splices it into the JSONL line via ``orjson.Fragment``
        in ``buffered_write``. Error records that never reached transport
        carry no ``payload_bytes`` — those export with ``payload=None`` and
        rely on the attached ``error`` field for replay context.
        """
        request_info = record.request.request_info
        payload_bytes = request_info.payload_bytes if request_info else None
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
            cache_bust_marker=request_info.cache_bust_marker if request_info else None,
            cache_bust_target=request_info.cache_bust_target if request_info else None,
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
            dumped = record.model_dump(exclude_none=True, mode="json")
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

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> None:
        """Process a single record."""
        # Build export record with full parsed record
        record_export = self._build_export_record(record, metadata)

        # Write using the buffered writer mixin (handles batching and flushing)
        await self.buffered_write(record_export)


class RawRecordAggregator(AIPerfLoggerMixin):
    """Aggregator for raw records."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        if self.exporter_config.user_config.output.export_level != ExportLevel.RAW:
            raise DataExporterDisabled(
                f"RawRecordAggregator is disabled for export level {self.exporter_config.user_config.output.export_level}"
            )
        self.output_file = (
            exporter_config.user_config.output.profile_export_raw_jsonl_file
        )
        self.output_dir = (
            exporter_config.user_config.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Raw Records",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Aggregate the raw records."""
        if self.exporter_config.user_config.output.export_level != ExportLevel.RAW:
            return

        raw_record_files = list(self.output_dir.glob("raw_records_*.jsonl"))
        if not raw_record_files:
            return

        self.output_file.unlink(missing_ok=True)
        self.info(
            f"Aggregating {len(raw_record_files)} raw record files from {self.output_dir} to {self.output_file}"
        )
        record_count = 0
        async with aiofiles.open(self.output_file, "w") as export_file:
            for file in raw_record_files:
                async with aiofiles.open(file) as f:
                    async for line in f:
                        if line.strip():
                            record_count += 1
                            await export_file.write(line)
                file.unlink(missing_ok=True)

        with contextlib.suppress(OSError):
            self.output_dir.rmdir()

        self.info(f"Aggregated {record_count} raw records to {self.output_file}")

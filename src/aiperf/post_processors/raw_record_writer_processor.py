# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import orjson

from aiperf.common.enums import ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.mixins import AIPerfLoggerMixin, BufferedJSONLWriterMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ModelEndpointInfo,
    ParsedResponseRecord,
    RawRecordInfo,
    RawRecordSummaryInfo,
)
from aiperf.common.models.record_models import RecordContext, RequestInfo
from aiperf.common.redact import redact_headers
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.raw_record_summary import build_raw_record_summary_info

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class RawRecordSummaryWriter(BufferedJSONLWriterMixin[RawRecordSummaryInfo]):
    """Writes compact raw-record summaries alongside full raw exports."""


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
        run: BenchmarkRun,
        **kwargs,
    ):
        self.service_id = service_id or "processor"
        self.run = run

        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            raise PostProcessorDisabled(
                f"RawRecordWriter processor is disabled for export level {self.run.cfg.artifacts.export_level}"
            )

        # Construct output file path: raw_records/raw_records_processor_{id}.jsonl
        output_dir = self.run.cfg.artifacts.dir / OutputDefaults.RAW_RECORDS_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each processor writes to its own file - avoids locking/contention
        # Sanitize service_id for filename (replace special chars)
        safe_id = self.service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_file = output_dir / f"raw_records_{safe_id}.jsonl"
        summary_output_file = output_dir / f"raw_record_summaries_{safe_id}.jsonl"

        self._model_endpoint = ModelEndpointInfo.from_run(run)
        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, self._model_endpoint.endpoint.type
        )
        self._endpoint = EndpointClass(model_endpoint=self._model_endpoint)

        # Initialize the buffered writer mixin
        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.RAW_EXPORT_BATCH_SIZE,
            service_id=service_id,
            run=run,
            **kwargs,
        )

        self._summary_writer = RawRecordSummaryWriter(
            output_file=summary_output_file,
            batch_size=Environment.RECORD.RAW_EXPORT_BATCH_SIZE,
            service_id=service_id,
            run=run,
            **kwargs,
        )
        self.attach_child_lifecycle(self._summary_writer)

        self.info(
            f"RawRecordWriter initialized: {self.output_file} - "
            "FULL request/response data will be exported (files may be large)"
        )

    @property
    def summary_output_file(self) -> Path:
        """Path to this processor's compact raw summary JSONL file."""
        return self._summary_writer.output_file

    def _build_export_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RawRecordInfo:
        """Build the export record for a single record."""

        # The record arrives carrying a slim ``RecordContext`` (down-cast on
        # the worker side by ``inference_client._enrich_request_record``); the
        # transport-only ``model_endpoint`` was stripped to save ZMQ bytes.
        # Re-attach the locally-known ``model_endpoint`` so the endpoint
        # plugin's ``format_payload`` has what it needs.
        ctx = record.request.request_info
        if ctx is not None:
            ctx_fields = {
                k: v
                for k, v in ctx.model_dump().items()
                if k in RecordContext.model_fields
            }
            request_info = RequestInfo(
                **ctx_fields,
                model_endpoint=self._model_endpoint,
            )
        else:
            # Fallback for records without complete request_info
            # (extremely rare; would indicate an upstream bug).
            request_info = RequestInfo(
                model_endpoint=self._model_endpoint,
                turns=record.request.turns,
                turn_index=metadata.turn_index or 0,
                credit_num=metadata.session_num,
                credit_phase=metadata.benchmark_phase,
                x_request_id=metadata.x_request_id or "",
                x_correlation_id=metadata.x_correlation_id or "",
                conversation_id=metadata.conversation_id or "",
            )

        payload = (
            orjson.loads(request_info.payload_bytes)
            if request_info.payload_bytes is not None
            else self._endpoint.format_payload(request_info)
        )
        return RawRecordInfo(
            metadata=metadata,
            start_perf_ns=record.request.start_perf_ns,
            payload=payload,
            request_headers=redact_headers(record.request.request_headers),
            response_headers=None,
            status=record.request.status,
            responses=record.request.responses,
            error=record.request.error,
        )

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> None:
        """Process a single record."""
        # Build export record with full parsed record
        record_export = self._build_export_record(record, metadata)

        # Write using the buffered writer mixin (handles batching and flushing)
        await self.buffered_write(record_export)

        summary_export = build_raw_record_summary_info(record, metadata)
        await self._summary_writer.buffered_write(summary_export)


class RawRecordAggregator(AIPerfLoggerMixin):
    """Aggregator for raw records."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        if self.exporter_config.cfg.artifacts.export_level != ExportLevel.RAW:
            raise DataExporterDisabled(
                f"RawRecordAggregator is disabled for export level {self.exporter_config.cfg.artifacts.export_level}"
            )
        self.output_file = exporter_config.cfg.artifacts.profile_export_raw_jsonl_file
        self.summary_output_file = (
            exporter_config.cfg.artifacts.profile_export_raw_summary_jsonl_file
        )
        self.output_dir = (
            exporter_config.cfg.artifacts.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Raw Records",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Aggregate the raw records."""
        if self.exporter_config.cfg.artifacts.export_level != ExportLevel.RAW:
            return

        raw_record_files = list(self.output_dir.glob("raw_records_*.jsonl"))
        raw_summary_files = list(self.output_dir.glob("raw_record_summaries_*.jsonl"))
        if not raw_record_files and not raw_summary_files:
            return

        record_count = await self._aggregate_files(
            raw_record_files,
            self.output_file,
            "raw records",
        )
        summary_count = await self._aggregate_files(
            raw_summary_files,
            self.summary_output_file,
            "raw record summaries",
        )

        with contextlib.suppress(OSError):
            self.output_dir.rmdir()

        self.info(f"Aggregated {record_count} raw records to {self.output_file}")
        if summary_count:
            self.info(
                f"Aggregated {summary_count} raw record summaries to {self.summary_output_file}"
            )

    async def _aggregate_files(
        self,
        files: list[Path],
        output_file: Path,
        label: str,
    ) -> int:
        if not files:
            return 0

        output_file.unlink(missing_ok=True)
        self.info(
            f"Aggregating {len(files)} {label} files from {self.output_dir} to {output_file}"
        )
        record_count = 0
        async with aiofiles.open(output_file, "w") as export_file:
            for file in files:
                async with aiofiles.open(file) as f:
                    async for line in f:
                        if line.strip():
                            record_count += 1
                            await export_file.write(line)
                file.unlink(missing_ok=True)
        return record_count

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import aiofiles

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.metric_records_wire import MetricRecordMetadata
from aiperf.common.mixins import AIPerfLoggerMixin, BufferedJSONLWriterMixin
from aiperf.common.models import ParsedResponseRecord, RawRecordInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.redact import redact_headers
from aiperf.config.defaults import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


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

        if not self.run.cfg.artifacts.raw:
            raise PostProcessorDisabled(
                "RawRecordWriter processor is disabled (artifacts.raw is not enabled)"
            )

        # Construct output file path: raw_records/raw_records_processor_{id}.jsonl
        config = run.cfg
        output_dir = config.artifacts.dir / OutputDefaults.RAW_RECORDS_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each processor writes to its own file - avoids locking/contention
        # Sanitize service_id for filename (replace special chars)
        safe_id = self.service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_file = output_dir / f"raw_records_{safe_id}.jsonl"

        EndpointClass = plugins.get_class(PluginType.ENDPOINT, config.endpoint.type)
        self._endpoint = EndpointClass(run=run)

        # Initialize the buffered writer mixin
        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.RAW_EXPORT_BATCH_SIZE,
            flush_interval=Environment.RECORD.EXPORT_FLUSH_INTERVAL,
            service_id=service_id,
            run=run,
            **kwargs,
        )

        self.info(
            f"RawRecordWriter initialized: {self.output_file} - "
            "FULL request/response data will be exported (files may be large)"
        )

    def _build_export_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RawRecordInfo:
        """Build the export record for a single record."""

        payload = getattr(record.request, "raw_payload", None)

        # Use existing request_info if available, otherwise create minimal one
        request_info = record.request.request_info
        if payload is None and request_info is None:
            # Fallback for records without complete request_info
            # This should rarely happen after proper request_info propagation
            request_info = RequestInfo(
                turns=record.request.turns,
                turn_index=metadata.turn_index or 0,
                credit_num=metadata.request_num
                if metadata.request_num is not None
                else metadata.session_num,
                credit_phase=metadata.benchmark_phase,
                x_request_id=metadata.x_request_id or "",
                x_correlation_id=metadata.x_correlation_id or "",
                conversation_id=metadata.conversation_id or "",
            )

        if payload is None:
            payload = self._endpoint.format_payload(request_info)
        return RawRecordInfo(
            metadata=metadata,
            start_perf_ns=record.request.start_perf_ns,
            payload=payload,
            request_headers=redact_headers(record.request.request_headers),
            response_headers=None,
            status=record.request.status,
            responses=[r for r in record.request.responses if r is not None],
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


class RawRecordAggregator(AIPerfLoggerMixin):
    """Aggregator for raw records."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        if not self.exporter_config.config.artifacts.raw:
            raise DataExporterDisabled(
                "RawRecordAggregator is disabled (artifacts.raw is not enabled)"
            )
        # Build output file path from artifacts config
        artifacts = exporter_config.config.artifacts
        self.output_file = artifacts.profile_export_raw_jsonl_file
        self.output_dir = artifacts.dir / OutputDefaults.RAW_RECORDS_FOLDER

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Raw Records",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Aggregate the raw records."""
        if not self.exporter_config.config.artifacts.raw:
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

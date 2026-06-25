# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import orjson

from aiperf.common.constants import NANOS_PER_MILLIS
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
    RawRecordSummaryNvext,
)
from aiperf.common.models.record_models import RecordContext, RequestInfo
from aiperf.common.redact import redact_headers
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

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

    @staticmethod
    def _chunk_ms(start_perf_ns: int, chunk_perf_ns: int) -> float | None:
        delta_ns = chunk_perf_ns - start_perf_ns
        if delta_ns < 0:
            return None
        return delta_ns / NANOS_PER_MILLIS

    @staticmethod
    def _extract_finish_reason(packet: dict[str, Any]) -> str | None:
        finish_reason = packet.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason

        choices = packet.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None
        finish_reason = first_choice.get("finish_reason")
        return (
            finish_reason if isinstance(finish_reason, str) and finish_reason else None
        )

    @staticmethod
    def _extract_request_id(packet: dict[str, Any]) -> str | None:
        request_id = packet.get("request_id") or packet.get("id")
        return request_id if isinstance(request_id, str) else None

    @staticmethod
    def _extract_nvext(packet: dict[str, Any]) -> RawRecordSummaryNvext | None:
        nvext = packet.get("nvext")
        if not isinstance(nvext, dict):
            return None

        timing = nvext.get("timing")
        worker_id = nvext.get("worker_id")
        if not isinstance(timing, dict):
            timing = None
        worker_id = str(worker_id) if worker_id is not None else None

        if timing is None and worker_id is None:
            return None
        return RawRecordSummaryNvext(timing=timing, worker_id=worker_id)

    @staticmethod
    def _merge_nvext_summary(
        current_timing: dict[str, Any] | None,
        current_worker_id: str | None,
        packet_nvext: RawRecordSummaryNvext | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if packet_nvext is None:
            return current_timing, current_worker_id
        return (
            packet_nvext.timing if packet_nvext.timing is not None else current_timing,
            packet_nvext.worker_id
            if packet_nvext.worker_id is not None
            else current_worker_id,
        )

    def _chunk_offsets_ms(
        self, start_perf_ns: int, chunk_perf_ns: list[int]
    ) -> tuple[float | None, float | None, float | None]:
        if not chunk_perf_ns:
            return None, None, None

        first_chunk_ms = self._chunk_ms(start_perf_ns, chunk_perf_ns[0])
        last_chunk_ms = self._chunk_ms(start_perf_ns, chunk_perf_ns[-1])
        if (
            first_chunk_ms is None
            or last_chunk_ms is None
            or last_chunk_ms < first_chunk_ms
        ):
            return first_chunk_ms, last_chunk_ms, None
        return first_chunk_ms, last_chunk_ms, last_chunk_ms - first_chunk_ms

    def _build_summary_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RawRecordSummaryInfo:
        chunk_perf_ns: list[int] = []
        request_id = None
        finish_reason = None
        nvext_timing = None
        nvext_worker_id = None

        for response in record.request.responses:
            text = response.get_text()
            if text not in (None, "", "[DONE]"):
                chunk_perf_ns.append(response.perf_ns)

            packet = response.get_json()
            if not isinstance(packet, dict):
                continue

            packet_request_id = self._extract_request_id(packet)
            if request_id is None and packet_request_id is not None:
                request_id = packet_request_id

            if packet_finish_reason := self._extract_finish_reason(packet):
                finish_reason = packet_finish_reason

            nvext_timing, nvext_worker_id = self._merge_nvext_summary(
                nvext_timing,
                nvext_worker_id,
                self._extract_nvext(packet),
            )

        first_chunk_ms, last_chunk_ms, stream_decode_ms = self._chunk_offsets_ms(
            record.request.start_perf_ns,
            chunk_perf_ns,
        )

        nvext_summary = None
        if nvext_timing is not None or nvext_worker_id is not None:
            nvext_summary = RawRecordSummaryNvext(
                timing=nvext_timing,
                worker_id=nvext_worker_id,
            )

        return RawRecordSummaryInfo(
            metadata=metadata,
            request_id=request_id,
            status=record.request.status,
            data_chunk_count=len(chunk_perf_ns),
            finish_reason=finish_reason,
            first_chunk_ms=first_chunk_ms,
            last_chunk_ms=last_chunk_ms,
            stream_decode_ms=stream_decode_ms,
            nvext=nvext_summary,
        )

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

        summary_export = self._build_summary_record(record, metadata)
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

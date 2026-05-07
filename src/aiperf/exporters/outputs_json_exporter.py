# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import aiofiles
import orjson

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import CreditPhase
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.record_models import MetricRecordInfo, RawRecordInfo
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class OutputsJsonExporter(AIPerfLoggerMixin):
    """Exports per-request output metadata and response text to outputs.json.

    When raw records are available (--export-level raw), includes the full
    response text for downstream safety/accuracy evaluation. Otherwise,
    includes metrics only.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._user_config = exporter_config.user_config
        self._file_path = self._user_config.output.outputs_json_file
        self._jsonl_path = self._user_config.output.profile_export_jsonl_file
        self._raw_records_dir = (
            self._user_config.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging."""
        return FileExportInfo(
            export_type="Outputs JSON",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        """Read per-request records and write outputs.json with response text when available."""
        if not self._jsonl_path.exists():
            self.debug(
                f"JSONL file not found, skipping outputs.json export: {self._jsonl_path}"
            )
            return

        # Load raw records for response text (if available)
        raw_responses = await asyncio.to_thread(self._load_raw_responses)

        records: list[dict] = await asyncio.to_thread(
            self._read_and_parse_records, raw_responses
        )
        records.sort(key=lambda r: r["session_num"])

        output = {
            "schema_version": "1.0",
            "data": records,
        }

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        content = orjson.dumps(output, option=orjson.OPT_INDENT_2)
        async with aiofiles.open(self._file_path, "wb") as f:
            await f.write(content)
        self.info(f"Exported {len(records)} records to {self._file_path}")

    def _load_raw_responses(self) -> dict[str, str]:
        """Load response text from raw record files, keyed by session_num:turn_index."""
        responses: dict[str, str] = {}
        if not self._raw_records_dir.exists():
            return responses

        for raw_file in self._raw_records_dir.glob("*.jsonl"):
            with open(raw_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = RawRecordInfo.model_validate_json(line)
                    except (ValueError, KeyError) as e:
                        self.debug(f"Skipping malformed raw record: {e}")
                        continue
                    if raw.metadata.benchmark_phase != CreditPhase.PROFILING:
                        continue
                    text = self._extract_response_text(raw)
                    if text:
                        key = (
                            f"{raw.metadata.session_num}:{raw.metadata.turn_index or 0}"
                        )
                        responses[key] = text

        return responses

    def _read_and_parse_records(self, raw_responses: dict[str, str]) -> list[dict]:
        """Read JSONL and parse profiling records (runs in thread pool)."""
        records: list[dict] = []
        with open(self._jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = MetricRecordInfo.model_validate_json(line)
                if record.metadata.benchmark_phase != CreditPhase.PROFILING:
                    continue
                entry = self._build_output_entry(record)
                key = f"{record.metadata.session_num}:{record.metadata.turn_index or 0}"
                entry["response_text"] = raw_responses.get(key)
                records.append(entry)
        return records

    @staticmethod
    def _extract_response_text(raw: RawRecordInfo) -> str | None:
        """Extract concatenated response text from raw record responses."""
        parts: list[str] = []
        for resp in raw.responses:
            if hasattr(resp, "text") and resp.text:
                parts.append(resp.text)
            elif hasattr(resp, "data") and resp.data:
                parts.append(str(resp.data))
        return "".join(parts) if parts else None

    @staticmethod
    def _build_output_entry(record: MetricRecordInfo) -> dict:
        """Extract relevant fields from a MetricRecordInfo into the outputs.json schema."""
        metrics: dict = {}
        for key in ("output_token_count", "output_sequence_length", "request_latency"):
            if key in record.metrics:
                metrics[key] = record.metrics[key].value

        return {
            "session_num": record.metadata.session_num,
            "conversation_id": record.metadata.conversation_id,
            "turn_index": record.metadata.turn_index,
            "x_request_id": record.metadata.x_request_id,
            "request_start_ns": record.metadata.request_start_ns,
            "request_end_ns": record.metadata.request_end_ns,
            "metrics": metrics,
            "response_text": None,
            "error": record.error.model_dump() if record.error else None,
        }

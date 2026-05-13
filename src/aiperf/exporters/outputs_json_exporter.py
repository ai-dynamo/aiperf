# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregator that merges output fragments with metrics into final outputs.json."""

import aiofiles
import orjson

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.record_models import MetricRecordInfo
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class OutputsJsonExporter(AIPerfLoggerMixin):
    """Aggregates per-processor output fragment files and merges with metrics from profile_export.jsonl.

    Self-disables unless --export-outputs-json is set.
    """

    _METRIC_ALLOWLIST = (
        "output_token_count",
        "output_sequence_length",
        "request_latency",
    )

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._user_config = exporter_config.user_config

        if not self._user_config.output.export_outputs_json:
            raise DataExporterDisabled(
                "OutputsJsonExporter is disabled (--export-outputs-json not set)"
            )

        self._file_path = self._user_config.output.outputs_json_file
        self._jsonl_path = self._user_config.output.profile_export_jsonl_file
        self._fragments_dir = (
            self._user_config.output.artifact_directory
            / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging."""
        return FileExportInfo(
            export_type="Outputs JSON",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        """Read fragment files, merge with metrics, and write final outputs.json."""
        fragment_files = list(self._fragments_dir.glob("output_fragments_*.jsonl"))
        if not fragment_files:
            self.debug("No output fragment files found, skipping outputs.json export")
            return

        fragments = self._read_fragments(fragment_files)
        metrics_map = self._build_metrics_map()

        records: list[dict] = []
        for frag in fragments:
            key = f"{frag['session_num']}:{frag.get('turn_index', 0)}"
            metrics = metrics_map.get(key, {})
            entry = {
                "session_num": frag["session_num"],
                "conversation_id": frag.get("conversation_id"),
                "turn_index": frag.get("turn_index"),
                "x_request_id": frag.get("x_request_id"),
                "request_start_ns": frag.get("request_start_ns"),
                "request_end_ns": frag.get("request_end_ns"),
                "metrics": metrics,
                "response_text": frag.get("response_text"),
            }
            records.append(entry)

        records.sort(key=lambda r: (r["session_num"], r.get("turn_index") or 0))

        output = {
            "schema_version": "1.0",
            "data": records,
        }

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        content = orjson.dumps(output, option=orjson.OPT_INDENT_2)
        async with aiofiles.open(self._file_path, "wb") as f:
            await f.write(content)

        self.info(f"Exported {len(records)} records to {self._file_path}")

        self._cleanup_fragments(fragment_files)

    def _read_fragments(self, fragment_files: list) -> list[dict]:
        """Read all fragment JSONL files and return parsed dicts."""
        fragments: list[dict] = []
        for file in fragment_files:
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fragments.append(orjson.loads(line))
        return fragments

    def _build_metrics_map(self) -> dict[str, dict]:
        """Read profile_export.jsonl and build a metrics map keyed by session_num:turn_index."""
        metrics_map: dict[str, dict] = {}
        if not self._jsonl_path.exists():
            self.debug("profile_export.jsonl not found, metrics will be empty")
            return metrics_map

        with open(self._jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = MetricRecordInfo.model_validate_json(line)
                if record.metadata.benchmark_phase != CreditPhase.PROFILING:
                    continue
                key = f"{record.metadata.session_num}:{record.metadata.turn_index or 0}"
                metrics: dict = {}
                for metric_key in self._METRIC_ALLOWLIST:
                    if metric_key in record.metrics:
                        metrics[metric_key] = record.metrics[metric_key].value
                metrics_map[key] = metrics

        return metrics_map

    def _cleanup_fragments(self, fragment_files: list) -> None:
        """Remove fragment files and directory."""
        for file in fragment_files:
            file.unlink(missing_ok=True)
        try:
            self._fragments_dir.rmdir()
        except OSError:
            self.debug(
                f"Could not remove fragments directory (may not be empty): {self._fragments_dir}"
            )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregator that concatenates per-processor output fragments into final outputs.json."""

from pathlib import Path
from typing import Any

import aiofiles
import orjson

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import scrub_non_finite
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo

JsonObject = dict[str, Any]


class OutputsJsonExporter(AIPerfLoggerMixin):
    """Aggregates per-processor output fragment files into the final outputs.json.

    Each fragment already carries its allowlisted per-request metrics (captured in
    display units by OutputsJsonRecordProcessor), so this exporter performs no
    metrics join and does not depend on the records JSONL export being enabled.

    Self-disables unless --export-outputs-json is set.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cfg = exporter_config.cfg

        if not self._cfg.artifacts.export_outputs_json:
            raise DataExporterDisabled(
                "OutputsJsonExporter is disabled (--export-outputs-json not set)"
            )

        self._file_path = self._cfg.artifacts.outputs_json_file
        self._fragments_dir = (
            self._cfg.artifacts.artifact_directory
            / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging."""
        return FileExportInfo(
            export_type="Outputs JSON",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        """Read fragment files (each self-contained) and write the final outputs.json."""
        fragment_files: list[Path] = list(
            self._fragments_dir.glob("output_fragments_*.jsonl")
        )
        if not fragment_files:
            self.debug("No output fragment files found, skipping outputs.json export")
            return

        fragments = await self._read_fragments(fragment_files)

        records: list[JsonObject] = []
        for frag in fragments:
            entry = {
                "session_num": frag["session_num"],
                "conversation_id": frag.get("conversation_id"),
                "turn_index": frag.get("turn_index"),
                "x_request_id": frag.get("x_request_id"),
                "request_start_ns": frag.get("request_start_ns"),
                "request_end_ns": frag.get("request_end_ns"),
                "metrics": frag.get("metrics") or {},
                "response_text": frag.get("response_text"),
            }
            records.append(entry)

        records.sort(key=lambda r: (r["session_num"], r.get("turn_index") or 0))

        output = {
            "schema_version": "1.0",
            "data": records,
        }

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        content = orjson.dumps(scrub_non_finite(output), option=orjson.OPT_INDENT_2)
        async with aiofiles.open(self._file_path, "wb") as f:
            await f.write(content)

        self.info(f"Exported {len(records)} records to {self._file_path}")

        self._cleanup_fragments(fragment_files)

    async def _read_fragments(self, fragment_files: list[Path]) -> list[JsonObject]:
        """Read all fragment JSONL files, skipping any unparseable lines.

        A crashed processor can leave a half-written trailing line; one bad line
        must not sink the whole export, so decode errors and non-object lines are
        logged and skipped.
        """
        fragments: list[JsonObject] = []
        for file in fragment_files:
            async with aiofiles.open(file) as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fragment = orjson.loads(line)
                    except orjson.JSONDecodeError as e:
                        self.warning(
                            f"Skipping unparseable fragment line in {file}: {e}"
                        )
                        continue
                    if not isinstance(fragment, dict):
                        self.warning(
                            f"Skipping non-object fragment line in {file}: {line!r}"
                        )
                        continue
                    fragments.append(fragment)
        return fragments

    def _cleanup_fragments(self, fragment_files: list[Path]) -> None:
        """Remove fragment files and directory."""
        for file in fragment_files:
            file.unlink(missing_ok=True)
        try:
            self._fragments_dir.rmdir()
        except OSError:
            self.debug(
                f"Could not remove fragments directory (may not be empty): {self._fragments_dir}"
            )

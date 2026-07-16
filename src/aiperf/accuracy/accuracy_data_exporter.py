# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo

_CSV_COLUMNS = ["task", "total", "passed", "unparsed", "accuracy_rate", "unparsed_rate"]


class AccuracyDataExporter(AIPerfLoggerMixin):
    """Data exporter for accuracy benchmarking results.

    Exports the per-task accuracy summary (plus an OVERALL row) to CSV for
    offline analysis, sourced from the structured ``AccuracySummary`` delivered
    on the dedicated accuracy channel.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        accuracy_cfg = exporter_config.cfg.accuracy
        if accuracy_cfg is None or not accuracy_cfg.enabled:
            raise DataExporterDisabled(
                "Accuracy data exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

        artifact_dir = Path(exporter_config.cfg.artifacts.artifact_directory)
        self._csv_path = artifact_dir / "accuracy_results.csv"

    def get_export_info(self) -> FileExportInfo:
        """Return the export path for the accuracy CSV written by ``export``."""
        return FileExportInfo(
            export_type="accuracy_csv",
            file_path=self._csv_path,
        )

    async def export(self) -> None:
        """Write the per-task accuracy summary to CSV at the path from ``get_export_info``.

        Columns: task, total, passed, unparsed, accuracy_rate, unparsed_rate.
        One row per task plus a final OVERALL row. Does nothing when no accuracy
        summary was delivered.
        """
        summary = self.exporter_config.accuracy_results
        if summary is None:
            return

        rows = summary.to_csv()
        await asyncio.to_thread(self._write_csv, rows)
        self.info(f"Accuracy results exported to {self._csv_path}")

    def _write_csv(self, rows: list[dict[str, Any]]) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

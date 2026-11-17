# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from aiperf.accuracy.models import (
    ACCURACY_GRADER_ERROR_TAG,
    ACCURACY_GRADER_ERROR_TASK_TAG_PREFIX,
    ACCURACY_METRIC_PREFIX,
    ACCURACY_OVERALL_TAG,
    ACCURACY_TASK_TAG_PREFIX,
    ACCURACY_UNPARSED_TAG,
    ACCURACY_UNPARSED_TASK_TAG_PREFIX,
)
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo

AccuracyCsvRow = tuple[
    str, int, int, int, str, int
]  # (task, correct, total, unparsed, accuracy, grader_errors)


class AccuracyDataExporter(AIPerfLoggerMixin):
    """Data exporter for accuracy benchmarking results.

    Exports per-task accuracy summary to CSV for offline analysis.
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
        """Write per-task accuracy summary to CSV at the path from ``get_export_info``.

        Columns: task, correct, total, unparsed, accuracy (4 decimal places),
        grader_errors (appended last so positional readers keep working). Rows are
        emitted for each ``accuracy.task.*`` metric plus a final OVERALL row.
        Does nothing if no ``accuracy.*`` metrics are present in results.
        """
        results = self.exporter_config.results
        if results is None or results.records is None:
            return

        accuracy_metrics = [
            r for r in results.records if r.tag.startswith(ACCURACY_METRIC_PREFIX)
        ]
        if not accuracy_metrics:
            return

        unparsed_overall = next(
            (m for m in accuracy_metrics if m.tag == ACCURACY_UNPARSED_TAG), None
        )
        unparsed_by_task: dict[str, int] = {
            m.tag.removeprefix(ACCURACY_UNPARSED_TASK_TAG_PREFIX): int(m.sum or 0)
            for m in accuracy_metrics
            if m.tag.startswith(ACCURACY_UNPARSED_TASK_TAG_PREFIX)
        }
        grader_errors_overall = next(
            (m for m in accuracy_metrics if m.tag == ACCURACY_GRADER_ERROR_TAG), None
        )
        grader_errors_by_task: dict[str, int] = {
            m.tag.removeprefix(ACCURACY_GRADER_ERROR_TASK_TAG_PREFIX): int(m.sum or 0)
            for m in accuracy_metrics
            if m.tag.startswith(ACCURACY_GRADER_ERROR_TASK_TAG_PREFIX)
        }

        rows: list[AccuracyCsvRow] = []
        for m in accuracy_metrics:
            if m.tag == ACCURACY_OVERALL_TAG:
                task_name = "OVERALL"
                unparsed = int(unparsed_overall.sum or 0) if unparsed_overall else 0
                grader_errors = (
                    int(grader_errors_overall.sum or 0) if grader_errors_overall else 0
                )
            elif m.tag.startswith(ACCURACY_TASK_TAG_PREFIX):
                task_name = m.tag.removeprefix(ACCURACY_TASK_TAG_PREFIX)
                unparsed = unparsed_by_task.get(task_name, 0)
                grader_errors = grader_errors_by_task.get(task_name, 0)
            else:
                continue
            rows.append(
                (
                    task_name,
                    int(m.sum or 0),
                    int(m.count or 0),
                    unparsed,
                    f"{m.current:.4f}" if m.current is not None else "",
                    grader_errors,
                )
            )

        await asyncio.to_thread(self._write_csv, rows)
        self.info(f"Accuracy results exported to {self._csv_path}")

    def _write_csv(self, rows: list[AccuracyCsvRow]) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["task", "correct", "total", "unparsed", "accuracy", "grader_errors"]
            )
            for row in rows:
                writer.writerow(row)

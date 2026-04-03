# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class AccuracyDataExporter(AIPerfLoggerMixin):
    """Data exporter for accuracy benchmarking results.

    Exports per-task accuracy summary to CSV for offline analysis.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        if not exporter_config.user_config.accuracy.enabled:
            raise DataExporterDisabled(
                "Accuracy data exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

        artifact_dir = Path(exporter_config.user_config.output.artifact_directory)
        self._csv_path = artifact_dir / "accuracy_results.csv"

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="accuracy_csv",
            file_path=self._csv_path,
        )

    async def export(self) -> None:
        results = self.exporter_config.results
        if results is None or results.records is None:
            return

        accuracy_metrics = [
            r for r in results.records if r.tag.startswith("accuracy.")
        ]
        if not accuracy_metrics:
            return

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "correct", "total", "accuracy"])

            for m in accuracy_metrics:
                if m.tag == "accuracy.overall":
                    task_name = "OVERALL"
                elif m.tag.startswith("accuracy.task."):
                    task_name = m.tag.removeprefix("accuracy.task.")
                else:
                    continue

                writer.writerow(
                    [
                        task_name,
                        int(m.sum or 0),
                        int(m.count or 0),
                        f"{m.current:.4f}" if m.current is not None else "",
                    ]
                )

        self.info(f"Accuracy results exported to {self._csv_path}")

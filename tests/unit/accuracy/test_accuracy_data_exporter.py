# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

from aiperf.accuracy.accuracy_data_exporter import AccuracyDataExporter
from aiperf.accuracy.models import AccuracySummary, TaskAccuracyStats
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_cfg(accuracy: dict | None = None):
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy=accuracy,
    ).cfg


def _make_summary() -> AccuracySummary:
    return AccuracySummary(
        total_evaluated=10,
        total_passed=8,
        accuracy_rate=0.8,
        overall_unparsed=1,
        grader_name="mmlu",
        per_task={
            "algebra": TaskAccuracyStats(
                total=5, passed=3, unparsed=1, accuracy_rate=0.6, unparsed_rate=0.2
            ),
            "history": TaskAccuracyStats(
                total=5, passed=5, unparsed=0, accuracy_rate=1.0, unparsed_rate=0.0
            ),
        },
    )


def _make_exporter(
    tmp_path: Path, summary: AccuracySummary | None
) -> AccuracyDataExporter:
    exporter_config = ExporterConfig(
        cfg=_make_cfg({"benchmark": AccuracyBenchmarkType.MMLU}),
        results=None,
        telemetry_results=None,
        accuracy_results=summary,
    )
    exporter = AccuracyDataExporter(exporter_config=exporter_config)
    exporter._csv_path = tmp_path / "accuracy_results.csv"
    return exporter


@pytest.mark.asyncio
class TestAccuracyDataExporterExport:
    async def test_export_writes_task_rows_and_overall(self, tmp_path: Path) -> None:
        exporter = _make_exporter(tmp_path, _make_summary())

        await exporter.export()

        rows = list(csv.reader(exporter._csv_path.open()))
        assert rows[0] == [
            "task",
            "total",
            "passed",
            "unparsed",
            "accuracy_rate",
            "unparsed_rate",
        ]
        # Per-task rows are sorted by name.
        assert rows[1] == ["algebra", "5", "3", "1", "0.6", "0.2"]
        assert rows[2] == ["history", "5", "5", "0", "1.0", "0.0"]
        assert rows[3] == ["OVERALL", "10", "8", "1", "0.8", "0.1"]

    async def test_export_does_nothing_when_summary_is_none(
        self, tmp_path: Path
    ) -> None:
        exporter = _make_exporter(tmp_path, None)

        await exporter.export()

        assert not exporter._csv_path.exists()

    async def test_constructor_raises_when_accuracy_disabled(
        self, tmp_path: Path
    ) -> None:
        exporter_config = ExporterConfig(
            cfg=_make_cfg(None),
            results=None,
            telemetry_results=None,
            accuracy_results=_make_summary(),
        )
        with pytest.raises(DataExporterDisabled):
            AccuracyDataExporter(exporter_config=exporter_config)

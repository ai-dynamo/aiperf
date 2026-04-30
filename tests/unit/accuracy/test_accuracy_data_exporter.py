# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

from aiperf.accuracy.accuracy_data_exporter import AccuracyDataExporter
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_exporter(tmp_path: Path, records: list[MetricResult]) -> AccuracyDataExporter:
    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
    )
    exporter_config = ExporterConfig(
        user_config=user_config,
        service_config=ServiceConfig(),
        results=ProfileResults(
            records=records,
            completed=len(records),
            start_ns=0,
            end_ns=1,
        ),
        telemetry_results=None,
    )
    exporter = AccuracyDataExporter(exporter_config=exporter_config)
    exporter._csv_path = tmp_path / "accuracy_results.csv"
    return exporter


def _make_metric(tag: str, correct: int, total: int, accuracy: float) -> MetricResult:
    return MetricResult(
        tag=tag,
        header=tag,
        unit="ratio",
        sum=correct,
        count=total,
        current=accuracy,
    )


@pytest.mark.asyncio
class TestAccuracyDataExporterExport:
    async def test_export_writes_overall_and_task_rows(self, tmp_path: Path) -> None:
        records = [
            _make_metric("accuracy.overall", correct=8, total=10, accuracy=0.8),
            _make_metric("accuracy.task.algebra", correct=3, total=5, accuracy=0.6),
            _make_metric("accuracy.task.history", correct=5, total=5, accuracy=1.0),
            _make_metric("accuracy.unparsed", correct=1, total=10, accuracy=0.1),
            _make_metric(
                "accuracy.unparsed.task.algebra", correct=1, total=5, accuracy=0.2
            ),
            _make_metric(
                "accuracy.unparsed.task.history", correct=0, total=5, accuracy=0.0
            ),
        ]
        exporter = _make_exporter(tmp_path, records)

        await exporter.export()

        rows = list(csv.reader(exporter._csv_path.open()))
        assert rows[0] == ["task", "correct", "total", "unparsed", "accuracy"]
        assert rows[1] == ["OVERALL", "8", "10", "1", "0.8000"]
        assert rows[2] == ["algebra", "3", "5", "1", "0.6000"]
        assert rows[3] == ["history", "5", "5", "0", "1.0000"]

    async def test_export_skips_non_accuracy_metrics(self, tmp_path: Path) -> None:
        records = [
            _make_metric("request_latency", correct=0, total=100, accuracy=0.0),
            _make_metric("accuracy.overall", correct=4, total=10, accuracy=0.4),
        ]
        exporter = _make_exporter(tmp_path, records)

        await exporter.export()

        rows = list(csv.reader(exporter._csv_path.open()))
        assert len(rows) == 2  # header + overall only
        assert rows[1][0] == "OVERALL"

    async def test_export_does_nothing_when_no_accuracy_metrics(
        self, tmp_path: Path
    ) -> None:
        records = [_make_metric("request_latency", correct=0, total=10, accuracy=0.0)]
        exporter = _make_exporter(tmp_path, records)

        await exporter.export()

        assert not exporter._csv_path.exists()

    async def test_export_does_nothing_when_records_is_none(
        self, tmp_path: Path
    ) -> None:
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            ),
            accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
        )
        exporter_config = ExporterConfig(
            user_config=user_config,
            service_config=ServiceConfig(),
            results=ProfileResults(records=None, completed=0, start_ns=0, end_ns=1),
            telemetry_results=None,
        )
        exporter = AccuracyDataExporter(exporter_config=exporter_config)
        exporter._csv_path = tmp_path / "accuracy_results.csv"

        await exporter.export()

        assert not exporter._csv_path.exists()

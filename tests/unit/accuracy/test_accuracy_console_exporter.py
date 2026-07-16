# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from aiperf.accuracy.accuracy_console_exporter import AccuracyConsoleExporter
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_exporter(records: list[MetricResult] | None) -> AccuracyConsoleExporter:
    cfg = make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    ).cfg
    results = (
        ProfileResults(records=records, completed=0, start_ns=0, end_ns=1)
        if records is not None
        else None
    )
    exporter_config = ExporterConfig(
        cfg=cfg,
        results=results,
        telemetry_results=None,
    )
    return AccuracyConsoleExporter(exporter_config=exporter_config)


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
class TestAccuracyConsoleExporterExport:
    async def test_prints_table_with_task_and_overall_rows(self) -> None:
        exporter = _make_exporter(
            records=[
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
        )
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "algebra" in output
        assert "history" in output
        assert "OVERALL" in output
        assert "Unparsed" in output

    async def test_no_output_when_results_is_none(self) -> None:
        exporter = _make_exporter(records=None)
        console = MagicMock()
        await exporter.export(console)
        console.print.assert_not_called()

    async def test_no_output_when_records_is_none(self) -> None:
        exporter = _make_exporter(records=None)
        exporter.exporter_config.results = ProfileResults(
            records=None, completed=0, start_ns=0, end_ns=1
        )
        console = MagicMock()
        await exporter.export(console)
        console.print.assert_not_called()

    async def test_no_output_when_no_accuracy_metrics(self) -> None:
        exporter = _make_exporter(
            records=[_make_metric("throughput", correct=0, total=100, accuracy=0.0)]
        )
        console = MagicMock()
        await exporter.export(console)
        console.print.assert_not_called()

    async def test_overall_row_omitted_when_no_overall_metric(self) -> None:
        exporter = _make_exporter(
            records=[
                _make_metric("accuracy.task.algebra", correct=3, total=5, accuracy=0.6),
            ]
        )
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "OVERALL" not in output
        assert "algebra" in output

    async def test_accuracy_formatted_as_percentage(self) -> None:
        exporter = _make_exporter(
            records=[
                _make_metric("accuracy.task.algebra", correct=3, total=5, accuracy=0.6),
            ]
        )
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        assert "60.00%" in buf.getvalue()

    async def test_warns_when_all_responses_unparsed(self) -> None:
        """Smoke-test J regression: when 100% of responses fail to parse,
        the exporter must surface a loud diagnostic so users do not
        mistake mock-server / misconfigured-endpoint output for real
        accuracy=0% results."""
        exporter = _make_exporter(
            records=[
                _make_metric("accuracy.overall", correct=0, total=5, accuracy=0.0),
                _make_metric(
                    "accuracy.task.abstract_algebra",
                    correct=0,
                    total=5,
                    accuracy=0.0,
                ),
                _make_metric("accuracy.unparsed", correct=5, total=5, accuracy=1.0),
                _make_metric(
                    "accuracy.unparsed.task.abstract_algebra",
                    correct=5,
                    total=5,
                    accuracy=1.0,
                ),
            ]
        )
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "Warning" in output
        assert "unparsed" in output
        assert "inference server" in output

    async def test_no_warning_when_partial_unparsed(self) -> None:
        """Mixed parsed/unparsed runs are normal — the diagnostic must
        only fire on the 100%-unparsed pathology."""
        exporter = _make_exporter(
            records=[
                _make_metric("accuracy.overall", correct=2, total=5, accuracy=0.4),
                _make_metric("accuracy.task.algebra", correct=2, total=5, accuracy=0.4),
                _make_metric("accuracy.unparsed", correct=2, total=5, accuracy=0.4),
                _make_metric(
                    "accuracy.unparsed.task.algebra", correct=2, total=5, accuracy=0.4
                ),
            ]
        )
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "Warning" not in output
        assert "inference server" not in output

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from aiperf.accuracy.accuracy_console_exporter import AccuracyConsoleExporter
from aiperf.accuracy.models import AccuracySummary, TaskAccuracyStats
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_exporter(summary: AccuracySummary | None) -> AccuracyConsoleExporter:
    cfg = make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    ).cfg
    exporter_config = ExporterConfig(
        cfg=cfg,
        results=None,
        telemetry_results=None,
        accuracy_results=summary,
    )
    return AccuracyConsoleExporter(exporter_config=exporter_config)


def _task(passed: int, total: int, unparsed: int) -> TaskAccuracyStats:
    return TaskAccuracyStats(
        total=total,
        passed=passed,
        unparsed=unparsed,
        accuracy_rate=passed / total if total else 0.0,
        unparsed_rate=unparsed / total if total else 0.0,
    )


@pytest.mark.asyncio
class TestAccuracyConsoleExporterExport:
    async def test_prints_table_with_task_and_overall_rows(self) -> None:
        summary = AccuracySummary(
            total_evaluated=10,
            total_passed=8,
            accuracy_rate=0.8,
            overall_unparsed=1,
            per_task={
                "algebra": _task(passed=3, total=5, unparsed=1),
                "history": _task(passed=5, total=5, unparsed=0),
            },
        )
        exporter = _make_exporter(summary)
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "algebra" in output
        assert "history" in output
        assert "OVERALL" in output
        assert "Unparsed" in output

    async def test_no_output_when_summary_is_none(self) -> None:
        exporter = _make_exporter(None)
        console = MagicMock()
        await exporter.export(console)
        console.print.assert_not_called()

    async def test_overall_row_omitted_when_no_evaluations(self) -> None:
        summary = AccuracySummary(
            total_evaluated=0,
            total_passed=0,
            accuracy_rate=0.0,
            overall_unparsed=0,
            per_task={"algebra": _task(passed=3, total=5, unparsed=0)},
        )
        exporter = _make_exporter(summary)
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "OVERALL" not in output
        assert "algebra" in output

    async def test_accuracy_formatted_as_percentage(self) -> None:
        summary = AccuracySummary(
            total_evaluated=5,
            total_passed=3,
            accuracy_rate=0.6,
            overall_unparsed=0,
            per_task={"algebra": _task(passed=3, total=5, unparsed=0)},
        )
        exporter = _make_exporter(summary)
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        assert "60.00%" in buf.getvalue()

    async def test_warns_when_all_responses_unparsed(self) -> None:
        """Smoke-test regression: when 100% of responses fail to parse,
        the exporter must surface a loud diagnostic so users do not
        mistake mock-server / misconfigured-endpoint output for real
        accuracy=0% results."""
        summary = AccuracySummary(
            total_evaluated=5,
            total_passed=0,
            accuracy_rate=0.0,
            overall_unparsed=5,
            per_task={"abstract_algebra": _task(passed=0, total=5, unparsed=5)},
        )
        exporter = _make_exporter(summary)
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "Warning" in output
        assert "unparsed" in output
        assert "inference server" in output

    async def test_no_warning_when_partial_unparsed(self) -> None:
        """Mixed parsed/unparsed runs are normal - the diagnostic must
        only fire on the 100%-unparsed pathology."""
        summary = AccuracySummary(
            total_evaluated=5,
            total_passed=2,
            accuracy_rate=0.4,
            overall_unparsed=2,
            per_task={"algebra": _task(passed=2, total=5, unparsed=2)},
        )
        exporter = _make_exporter(summary)
        buf = io.StringIO()
        console = Console(file=buf, highlight=False)
        await exporter.export(console)

        output = buf.getvalue()
        assert "Warning" not in output
        assert "inference server" not in output

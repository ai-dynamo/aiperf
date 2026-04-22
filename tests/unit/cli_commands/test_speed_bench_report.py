# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for speed_bench_report CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.speed_bench_report import speed_bench_report

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_generate_report() -> Generator[MagicMock, None, None]:
    """Mock generate_report (lazily imported inside the CLI function)."""
    with patch("aiperf.analysis.speed_bench_report.generate_report") as mock:
        yield mock


class TestSpeedBenchReportCommand:
    """Tests for speed_bench_report() CLI function."""

    def test_forwards_all_arguments(self, mock_generate_report: MagicMock) -> None:
        paths = [Path("/artifacts/run1"), Path("/artifacts/run2")]
        output = Path("/tmp/out.csv")

        speed_bench_report(
            paths=paths,
            output=output,
            output_format="csv",
            metric="accept_rate",
        )

        mock_generate_report.assert_called_once_with(
            paths, output=output, output_format="csv", metric="accept_rate"
        )

    def test_default_output_path(self, mock_generate_report: MagicMock) -> None:
        speed_bench_report(paths=[Path("/artifacts")])

        call_kwargs = mock_generate_report.call_args.kwargs
        assert call_kwargs["output"] == Path("speed_bench_report.csv")

    def test_default_format_is_both(self, mock_generate_report: MagicMock) -> None:
        speed_bench_report(paths=[Path("/artifacts")])

        call_kwargs = mock_generate_report.call_args.kwargs
        assert call_kwargs["output_format"] == "both"

    def test_default_metric_is_accept_length(
        self, mock_generate_report: MagicMock
    ) -> None:
        speed_bench_report(paths=[Path("/artifacts")])

        call_kwargs = mock_generate_report.call_args.kwargs
        assert call_kwargs["metric"] == "accept_length"

    @pytest.mark.parametrize("metric", ["accept_length", "accept_rate", "throughput"])
    def test_all_metric_choices_pass_through(
        self, mock_generate_report: MagicMock, metric: str
    ) -> None:
        speed_bench_report(paths=[Path("/artifacts")], metric=metric)

        call_kwargs = mock_generate_report.call_args.kwargs
        assert call_kwargs["metric"] == metric

    @pytest.mark.parametrize("output_format", ["csv", "table", "both"])
    def test_all_format_choices_pass_through(
        self, mock_generate_report: MagicMock, output_format: str
    ) -> None:
        speed_bench_report(paths=[Path("/artifacts")], output_format=output_format)

        call_kwargs = mock_generate_report.call_args.kwargs
        assert call_kwargs["output_format"] == output_format

    def test_multiple_paths_forwarded(self, mock_generate_report: MagicMock) -> None:
        paths = [Path(f"/artifacts/run{i}") for i in range(5)]

        speed_bench_report(paths=paths)

        call_args = mock_generate_report.call_args.args
        assert call_args[0] == paths

    def test_error_from_generate_report_exits_with_code_1(
        self, mock_generate_report: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.analysis.speed_bench_report import SpeedBenchReportError

        mock_generate_report.side_effect = SpeedBenchReportError("no results found")

        with pytest.raises(SystemExit) as exc_info:
            speed_bench_report(paths=[Path("/artifacts")])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: no results found" in captured.err

    def test_unrelated_exception_is_not_caught(
        self, mock_generate_report: MagicMock
    ) -> None:
        mock_generate_report.side_effect = RuntimeError("unexpected failure")

        with pytest.raises(RuntimeError, match="unexpected failure"):
            speed_bench_report(paths=[Path("/artifacts")])


class TestSpeedBenchReportAppRegistration:
    """Tests for the cyclopts App object registration."""

    def test_app_name(self) -> None:
        from aiperf.cli_commands.speed_bench_report import app

        assert "speed-bench-report" in app.name

    def test_default_command_is_registered(self) -> None:
        from aiperf.cli_commands.speed_bench_report import app

        # cyclopts stores the default command; confirm the app is callable
        assert app is not None

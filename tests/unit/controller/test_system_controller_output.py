# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``SystemControllerOutputMixin``.

Covers the console / panel rendering helpers that are bundled as a mixin so
``SystemController`` itself stays under the file-size budget. Each test
exercises the mixin against a minimal ``_FakeHost`` shaped like
``SystemController``: only the attributes the mixin actually reads (results,
exit_errors, run config, memory tracker, ``_was_cancelled`` /
``_results_exported`` flags) are populated.

We do NOT assert byte-exact rich rendering: rich's terminal output mixes ANSI
sequences and box drawing that change with version/width. Instead we capture
to ``StringIO`` (or assert on call args) and check for the load-bearing
substrings — the panel title, the warning words, the metric value, the file
path — that the mixin promises to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.memory_tracker import MemoryTracker
from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.common.models.metric_result_models import (
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.controller.system_controller_output import SystemControllerOutputMixin

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFileInfo:
    """Mimic ``ExporterManager.get_exported_file_infos()`` items."""

    def __init__(self, export_type: str, file_path: Path) -> None:
        self.export_type = export_type
        self.file_path = file_path


class _FakeExporterManager:
    """Mimic ``ExporterManager`` enough for the output mixin."""

    def __init__(
        self,
        file_infos: list[_FakeFileInfo] | None = None,
    ) -> None:
        self._file_infos = file_infos or []
        self.export_console = AsyncMock()
        self.export_data = AsyncMock()

    def get_exported_file_infos(self) -> list[_FakeFileInfo]:
        return self._file_infos


@dataclass
class _FakeHost(SystemControllerOutputMixin):
    """Minimal SystemController-shaped host that the mixin writes through.

    Only carries the attributes the mixin reads. The dataclass machinery
    keeps construction terse for tests.
    """

    artifacts_dir: Path = field(default_factory=lambda: Path("/tmp/aiperf-test"))
    cli_command: str | None = "aiperf profile -m test-model"
    _exit_errors: list[ExitErrorInfo] = field(default_factory=list)
    _profile_results: ProcessRecordsResult | None = None
    _telemetry_results: Any = None
    _server_metrics_results: Any = None
    _was_cancelled: bool = False
    _results_exported: bool = False
    _exporter_manager: Any = None
    _memory_tracker: MemoryTracker = field(default_factory=MemoryTracker)
    _controller_pss_at_start: int | None = None

    def __post_init__(self) -> None:
        self.run = SimpleNamespace(
            cli_command=self.cli_command,
            cfg=SimpleNamespace(
                artifacts=SimpleNamespace(
                    dir=self.artifacts_dir,
                ),
            ),
        )


def _make_profile_results(
    *,
    duration_avg: float | None = 12.345,
) -> ProcessRecordsResult:
    """Build a ProcessRecordsResult with a benchmark-duration metric."""
    records: list[MetricResult] = []
    if duration_avg is not None:
        records.append(
            MetricResult(
                tag="benchmark_duration",
                header="Benchmark Duration",
                unit="sec",
                avg=duration_avg,
            )
        )
    return ProcessRecordsResult(
        results=ProfileResults(
            completed=10,
            start_ns=0,
            end_ns=int(duration_avg * 1e9) if duration_avg is not None else 0,
            records=records,
        )
    )


# ---------------------------------------------------------------------------
# _print_cancel_warning / _print_force_quit_warning
# ---------------------------------------------------------------------------


class TestPrintCancelWarning:
    """Cancellation banner rendered on first Ctrl+C."""

    def test_cancel_warning_renders_panel_with_key_messaging(self) -> None:
        from rich.panel import Panel

        host = _FakeHost()
        captured: list[Any] = []

        class _CapturingConsole:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                self.file = StringIO()

            def print(self, *args: Any, **_kw: Any) -> None:
                captured.extend(args)

        with patch(
            "aiperf.controller.system_controller_output.Console",
            new=_CapturingConsole,
        ):
            host._print_cancel_warning()

        panels = [c for c in captured if isinstance(c, Panel)]
        assert len(panels) == 1
        panel = panels[0]
        # Panel.renderable carries the rich-markup body; rich's str() formats
        # the *object* (we want the source markup instead).
        body = str(panel.renderable)
        title = str(panel.title)
        assert "BENCHMARK CANCELLED" in body
        assert "Cancellation in Progress" in title
        assert "force quit" in body.lower()
        assert panel.border_style == "yellow"

    def test_cancel_warning_flushes_stderr_console(self) -> None:
        host = _FakeHost()
        with patch(
            "aiperf.controller.system_controller_output.Console"
        ) as mock_console_cls:
            mock_console = MagicMock()
            mock_console_cls.return_value = mock_console
            host._print_cancel_warning()

        # Console constructed pointing at stderr with terminal forcing
        kwargs = mock_console_cls.call_args.kwargs
        assert kwargs.get("force_terminal") is True
        # mixin must flush before returning so the banner is visible even if
        # the process is about to be killed.
        mock_console.file.flush.assert_called_once()
        assert mock_console.print.call_count >= 2  # blank line + panel + blank line


class TestPrintForceQuitWarning:
    """Force-quit banner rendered on second Ctrl+C."""

    def test_force_quit_warning_renders_red_panel(self) -> None:
        host = _FakeHost()
        sink = StringIO()

        captured: list[Any] = []

        class _CapturingConsole:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                self.file = sink

            def print(self, *args: Any, **_kw: Any) -> None:
                captured.extend(args)

        with patch(
            "aiperf.controller.system_controller_output.Console",
            new=_CapturingConsole,
        ):
            host._print_force_quit_warning()

        # Panel content is the second print call (first/last are spacers)
        from rich.panel import Panel

        panels = [c for c in captured if isinstance(c, Panel)]
        assert len(panels) == 1
        panel = panels[0]
        assert panel.border_style == "red"
        assert "Force Quit" in str(panel.title)


# ---------------------------------------------------------------------------
# _print_exit_errors_and_log_file
# ---------------------------------------------------------------------------


class TestPrintExitErrorsAndLogFile:
    """Exit errors panel + log-file footer."""

    def test_no_exit_errors_still_prints_log_file(self) -> None:
        host = _FakeHost(_exit_errors=[])
        captured_console = MagicMock()
        with (
            patch(
                "aiperf.controller.system_controller_output.Console",
                return_value=captured_console,
            ),
            patch(
                "aiperf.controller.system_controller_output.print_exit_errors"
            ) as mock_pee,
        ):
            host._print_exit_errors_and_log_file()

        # print_exit_errors invoked even with empty list (the helper itself
        # decides whether to render; the mixin's contract is "always call").
        mock_pee.assert_called_once()
        # Log-file line printed.
        printed = " ".join(
            str(c.args[0]) for c in captured_console.print.call_args_list if c.args
        )
        assert "Log File" in printed
        captured_console.file.flush.assert_called_once()

    def test_with_exit_errors_passes_them_through(self) -> None:
        err = ExitErrorInfo(
            error_details=ErrorDetails(message="boom"),
            operation="Boom",
            service_id="svc-0",
        )
        host = _FakeHost(_exit_errors=[err])
        with (
            patch(
                "aiperf.controller.system_controller_output.Console",
                return_value=MagicMock(),
            ),
            patch(
                "aiperf.controller.system_controller_output.print_exit_errors"
            ) as mock_pee,
        ):
            host._print_exit_errors_and_log_file()

        mock_pee.assert_called_once()
        # First positional arg is the list of errors
        passed_errors = mock_pee.call_args.args[0]
        assert passed_errors == [err]


# ---------------------------------------------------------------------------
# _print_log_file_info / _print_cli_command / _print_exported_file_infos
# ---------------------------------------------------------------------------


class TestPrintHelpers:
    """Small per-line helpers used by the post-benchmark summary."""

    def test_log_file_info_uses_run_artifacts_dir(self, tmp_path: Path) -> None:
        host = _FakeHost(artifacts_dir=tmp_path)
        console = MagicMock()
        host._print_log_file_info(console)

        printed = console.print.call_args.args[0]
        assert "Log File" in printed
        # artifacts_dir / logs / aiperf.log appears resolved in the message
        assert str(tmp_path.resolve()) in printed
        assert "aiperf.log" in printed

    def test_cli_command_renders_when_present(self) -> None:
        host = _FakeHost(cli_command="aiperf profile --concurrency 8")
        console = MagicMock()
        host._print_cli_command(console)
        printed = console.print.call_args.args[0]
        assert "CLI Command" in printed
        assert "aiperf profile --concurrency 8" in printed

    def test_cli_command_falls_back_to_n_a_when_missing(self) -> None:
        host = _FakeHost(cli_command=None)
        console = MagicMock()
        host._print_cli_command(console)
        printed = console.print.call_args.args[0]
        assert "N/A" in printed

    def test_exported_file_infos_lists_each_file(self, tmp_path: Path) -> None:
        host = _FakeHost()
        em = _FakeExporterManager(
            file_infos=[
                _FakeFileInfo("JSON", tmp_path / "out.json"),
                _FakeFileInfo("CSV", tmp_path / "out.csv"),
            ]
        )
        console = MagicMock()
        host._print_exported_file_infos(em, console)

        assert console.print.call_count == 2
        rendered = " ".join(c.args[0] for c in console.print.call_args_list)
        assert "JSON" in rendered and "CSV" in rendered
        assert str((tmp_path / "out.json").resolve()) in rendered
        assert str((tmp_path / "out.csv").resolve()) in rendered

    def test_exported_file_infos_empty_list_prints_nothing(self) -> None:
        host = _FakeHost()
        console = MagicMock()
        host._print_exported_file_infos(_FakeExporterManager(file_infos=[]), console)
        console.print.assert_not_called()


# ---------------------------------------------------------------------------
# _print_benchmark_duration
# ---------------------------------------------------------------------------


class TestPrintBenchmarkDuration:
    """Benchmark-duration line inside the summary block."""

    def test_renders_average_duration_when_metric_present(self) -> None:
        host = _FakeHost(_profile_results=_make_profile_results(duration_avg=42.5))
        console = MagicMock()
        host._print_benchmark_duration(console)
        printed = console.print.call_args.args[0]
        assert "Benchmark Duration" in printed
        assert "42.50" in printed  # f"{:.2f}"
        assert "sec" in printed
        assert "(cancelled early)" not in printed

    def test_appends_cancelled_marker_when_cancelled(self) -> None:
        host = _FakeHost(
            _profile_results=_make_profile_results(duration_avg=1.0),
            _was_cancelled=True,
        )
        console = MagicMock()
        host._print_benchmark_duration(console)
        printed = console.print.call_args.args[0]
        assert "(cancelled early)" in printed

    def test_skips_when_duration_metric_missing(self) -> None:
        host = _FakeHost(_profile_results=_make_profile_results(duration_avg=None))
        console = MagicMock()
        host._print_benchmark_duration(console)
        console.print.assert_not_called()


# ---------------------------------------------------------------------------
# _print_post_benchmark_info_and_metrics
# ---------------------------------------------------------------------------


class TestPrintPostBenchmarkInfoAndMetrics:
    """Top-level summary orchestrator."""

    @pytest.mark.asyncio
    async def test_uses_existing_exporter_when_already_exported(
        self, tmp_path: Path
    ) -> None:
        em = _FakeExporterManager(
            file_infos=[_FakeFileInfo("JSON", tmp_path / "x.json")],
        )
        host = _FakeHost(
            artifacts_dir=tmp_path,
            _profile_results=_make_profile_results(duration_avg=2.0),
            _results_exported=True,
            _exporter_manager=em,
        )

        with patch("aiperf.controller.system_controller_output.Console") as cls:
            cls.return_value = MagicMock(width=120)
            await host._print_post_benchmark_info_and_metrics()

        # Already-exported branch must NOT re-build the exporter manager.
        em.export_data.assert_not_awaited()
        em.export_console.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_builds_exporter_when_not_yet_exported(self, tmp_path: Path) -> None:
        host = _FakeHost(
            artifacts_dir=tmp_path,
            _profile_results=_make_profile_results(duration_avg=2.0),
            _results_exported=False,
            _exporter_manager=None,
        )

        fake_em = _FakeExporterManager(
            file_infos=[_FakeFileInfo("JSON", tmp_path / "x.json")],
        )

        with (
            patch("aiperf.controller.system_controller_output.Console") as cls,
            patch(
                "aiperf.exporters.exporter_manager.ExporterManager",
                return_value=fake_em,
            ),
        ):
            cls.return_value = MagicMock(width=120)
            await host._print_post_benchmark_info_and_metrics()

        fake_em.export_data.assert_awaited_once()
        fake_em.export_console.assert_awaited_once()
        assert host._results_exported is True
        assert host._exporter_manager is fake_em

    @pytest.mark.asyncio
    async def test_widens_narrow_console_to_minimum_100(self, tmp_path: Path) -> None:
        em = _FakeExporterManager()
        host = _FakeHost(
            artifacts_dir=tmp_path,
            _profile_results=_make_profile_results(duration_avg=1.0),
            _results_exported=True,
            _exporter_manager=em,
        )
        narrow_console = MagicMock(width=80)

        with patch(
            "aiperf.controller.system_controller_output.Console",
            return_value=narrow_console,
        ):
            await host._print_post_benchmark_info_and_metrics()

        assert narrow_console.width == 100

    @pytest.mark.asyncio
    async def test_emits_cancelled_notice_when_cancelled(self, tmp_path: Path) -> None:
        em = _FakeExporterManager()
        host = _FakeHost(
            artifacts_dir=tmp_path,
            _profile_results=_make_profile_results(duration_avg=1.0),
            _results_exported=True,
            _exporter_manager=em,
            _was_cancelled=True,
        )
        console = MagicMock(width=120)

        with patch(
            "aiperf.controller.system_controller_output.Console",
            return_value=console,
        ):
            await host._print_post_benchmark_info_and_metrics()

        printed = " ".join(
            str(c.args[0]) for c in console.print.call_args_list if c.args
        )
        assert "cancelled early" in printed.lower()


# ---------------------------------------------------------------------------
# _print_process_memory_summary
# ---------------------------------------------------------------------------


class TestPrintProcessMemorySummary:
    """Memory-tracker summary at shutdown."""

    def test_records_controller_startup_pss_when_present(self) -> None:
        tracker = MemoryTracker()
        host = _FakeHost(
            _memory_tracker=tracker,
            _controller_pss_at_start=12345,
        )
        with patch.object(tracker, "print_summary") as mock_print:
            host._print_process_memory_summary()

        # Startup recording was made, plus a SHUTDOWN capture for the
        # controller process itself.
        snapshots = tracker.snapshots
        assert "SystemController" in snapshots
        snap = snapshots["SystemController"]
        assert snap.startup is not None
        assert snap.startup.pss == 12345
        # capture(SHUTDOWN) happens regardless of whether psutil yields a
        # reading — the snapshot is created either way.
        mock_print.assert_called_once_with(title="AIPerf Process Memory")

    def test_skips_startup_recording_when_pss_unavailable(self) -> None:
        tracker = MemoryTracker()
        host = _FakeHost(
            _memory_tracker=tracker,
            _controller_pss_at_start=None,
        )
        with patch.object(tracker, "print_summary"):
            host._print_process_memory_summary()

        # No startup reading recorded.
        snap = tracker.snapshots.get("SystemController")
        # capture() inside print_process_memory_summary creates the snapshot
        # entry when reading own memory; startup remains None.
        if snap is not None:
            assert snap.startup is None

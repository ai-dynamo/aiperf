# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param
from rich.console import Console

from aiperf.common.models import MetricResult, ProfileResults
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin.enums import (
    EndpointType,
)
from tests.unit.conftest import make_run_from_cli

ANSI_ESCAPE_PREFIX = "\x1b["
STYLED_LINE = "console-artifact-line"


class _StyledConsoleExporter:
    """Real console exporter double that prints one styled line."""

    def __init__(self, exporter_config: ExporterConfig) -> None:
        self._exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        console.print(STYLED_LINE, style="bold red")


def _make_manager(
    sample_records: list[MetricResult], cfg: CLIConfig
) -> ExporterManager:
    return ExporterManager(
        results=ProfileResults(
            records=sample_records,
            start_ns=0,
            end_ns=0,
            completed=0,
            was_cancelled=False,
            error_summary=[],
        ),
        run=make_run_from_cli(cfg),
        telemetry_results=None,
    )


@pytest.fixture
def endpoint_config():
    return CLIConfig(
        endpoint_type=EndpointType.CHAT, streaming=True, model_names=["gpt2"]
    )


@pytest.fixture
def output_config(tmp_path):
    """Returns the artifact directory path used by mock_cfg."""
    return tmp_path


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="Latency",
            unit="ms",
            avg=10.0,
            header="test-header",
        )
    ]


@pytest.fixture
def mock_cfg(endpoint_config, output_config):
    config = CLIConfig(
        **endpoint_config.model_dump(exclude_unset=True),
        artifact_directory=output_config,
    )
    return config


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(
        self, endpoint_config, output_config, sample_records, mock_cfg
    ):
        # Create a mock exporter instance
        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_class = MagicMock(return_value=mock_instance)

        # Create a mock PluginEntry for iter_all
        mock_entry = MagicMock()
        mock_entry.name = "mock_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[(mock_entry, mock_class)],
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                run=make_run_from_cli(mock_cfg),
                telemetry_results=None,
            )
            await manager.export_data()

        mock_class.assert_called_once()
        mock_instance.export.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_runs_mlflow_after_other_data_exporters(
        self, endpoint_config, output_config, sample_records, mock_cfg
    ):
        execution_order: list[str] = []

        async def _export_csv() -> None:
            execution_order.append("csv")

        async def _export_mlflow() -> None:
            execution_order.append("mlflow")

        csv_instance = MagicMock()
        csv_instance.export = AsyncMock(side_effect=_export_csv)
        csv_instance.is_deferred = False
        csv_class = MagicMock(return_value=csv_instance)
        csv_entry = MagicMock()
        csv_entry.name = "csv"

        mlflow_instance = MagicMock()
        mlflow_instance.export = AsyncMock(side_effect=_export_mlflow)
        mlflow_instance.is_deferred = True
        mlflow_class = MagicMock(return_value=mlflow_instance)
        mlflow_entry = MagicMock()
        mlflow_entry.name = "mlflow"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[
                (mlflow_entry, mlflow_class),
                (csv_entry, csv_class),
            ],
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                run=make_run_from_cli(mock_cfg),
                telemetry_results=None,
            )
            await manager.export_data()

        assert execution_order == ["csv", "mlflow"]
        csv_instance.export.assert_awaited_once()
        mlflow_instance.export.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_console(
        self, endpoint_config, output_config, sample_records, mock_cfg
    ):
        # Create mock exporter instances for each console exporter type
        mock_instances = []
        mock_classes = []
        mock_entries = []

        for i in range(2):  # Simulate two console exporters
            instance = MagicMock()
            instance.export = AsyncMock()
            mock_class = MagicMock(return_value=instance)
            mock_entry = MagicMock()
            mock_entry.name = f"mock_exporter_{i}"

            mock_instances.append(instance)
            mock_classes.append(mock_class)
            mock_entries.append(mock_entry)

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=list(zip(mock_entries, mock_classes, strict=False)),
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                run=make_run_from_cli(mock_cfg),
                telemetry_results=None,
            )
            # Non-terminal console renders the exporter loop exactly once, so the
            # assert_*_once assertions stay deterministic regardless of pytest -s.
            await manager.export_console(Console(file=io.StringIO()))

        for mock_class, mock_instance in zip(
            mock_classes, mock_instances, strict=False
        ):
            mock_class.assert_called_once()
            mock_instance.export.assert_awaited_once()


class TestExportConsoleArtifactAndStyling:
    """Pins for the console txt artifact write and the tty-gated styled replay."""

    @pytest.mark.asyncio
    async def test_write_console_txt_writes_plain_artifact_via_asyncio_to_thread(
        self, sample_records, mock_cfg, monkeypatch: pytest.MonkeyPatch
    ):
        real_to_thread = asyncio.to_thread
        to_thread_calls: list[tuple[Any, tuple, dict]] = []

        async def _recording_to_thread(func: Any, /, *args: Any, **kwargs) -> Any:
            to_thread_calls.append((func, args, kwargs))
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(
            "aiperf.exporters.exporter_manager.asyncio.to_thread",
            _recording_to_thread,
        )

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[
                (SimpleNamespace(name="styled_console"), _StyledConsoleExporter)
            ],
        ):
            manager = _make_manager(sample_records, mock_cfg)
            await manager.export_console(Console(file=io.StringIO()))

        txt_path = manager._run.cfg.artifacts.profile_export_console_txt_file
        assert txt_path.exists(), "console txt artifact was not written"
        content = txt_path.read_text(encoding="utf-8")
        assert STYLED_LINE in content
        assert ANSI_ESCAPE_PREFIX not in content, (
            "console txt artifact must be plain text"
        )

        write_text_calls = [
            (func, args, kwargs)
            for func, args, kwargs in to_thread_calls
            if getattr(func, "__name__", "") == "write_text"
        ]
        assert len(write_text_calls) == 1, (
            "console txt artifact write must be offloaded via asyncio.to_thread"
        )
        func, args, kwargs = write_text_calls[0]
        assert func.__self__ == txt_path
        assert STYLED_LINE in args[0]
        assert kwargs == {"encoding": "utf-8"}

    @pytest.mark.asyncio
    async def test_export_console_non_terminal_replay_has_no_ansi_escapes(
        self, sample_records, mock_cfg
    ):
        buffer = io.StringIO()
        console = Console(file=buffer)
        assert not console.is_terminal

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[
                (SimpleNamespace(name="styled_console"), _StyledConsoleExporter)
            ],
        ):
            manager = _make_manager(sample_records, mock_cfg)
            await manager.export_console(console)

        replayed = buffer.getvalue()
        assert STYLED_LINE in replayed
        assert ANSI_ESCAPE_PREFIX not in replayed, (
            "non-tty console replay must be plain text, not ANSI-styled"
        )

    @pytest.mark.asyncio
    async def test_export_console_forced_terminal_replay_preserves_styles(
        self, sample_records, mock_cfg
    ):
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True)
        assert console.is_terminal

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[
                (SimpleNamespace(name="styled_console"), _StyledConsoleExporter)
            ],
        ):
            manager = _make_manager(sample_records, mock_cfg)
            await manager.export_console(console)

        replayed = buffer.getvalue()
        assert STYLED_LINE in replayed
        assert ANSI_ESCAPE_PREFIX in replayed, (
            "terminal console replay must preserve ANSI styling"
        )

    @pytest.mark.parametrize(
        "console_kwargs",
        [
            param({"force_terminal": True, "no_color": True}, id="no-color"),
            param(
                {"force_terminal": True, "color_system": None},
                id="no-color-system",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_export_console_terminal_without_color_replays_plain_text(
        self, sample_records, mock_cfg, console_kwargs
    ):
        buffer = io.StringIO()
        console = Console(file=buffer, **console_kwargs)
        assert console.is_terminal
        assert console.no_color or console.color_system is None

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[
                (SimpleNamespace(name="styled_console"), _StyledConsoleExporter)
            ],
        ):
            manager = _make_manager(sample_records, mock_cfg)
            await manager.export_console(console)

        replayed = buffer.getvalue()
        assert STYLED_LINE in replayed
        assert ANSI_ESCAPE_PREFIX not in replayed, (
            "color-disabled terminal replay must be plain text, not ANSI-styled"
        )

    @pytest.mark.asyncio
    async def test_export_console_renders_live_output_at_terminal_width_on_tty(
        self, endpoint_config, output_config, sample_records, mock_cfg
    ):
        captured_widths: list[int] = []

        async def _capture_width(*, console: Console) -> None:
            captured_widths.append(console.width)

        instance = MagicMock()
        instance.export = AsyncMock(side_effect=_capture_width)
        mock_class = MagicMock(return_value=instance)
        mock_entry = MagicMock()
        mock_entry.name = "mock_console_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[(mock_entry, mock_class)],
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                run=make_run_from_cli(mock_cfg),
                telemetry_results=None,
            )
            await manager.export_console(
                Console(force_terminal=True, width=100, file=io.StringIO())
            )

        # The recording console for the .txt artifact stays pinned at the fixed
        # export width (140); the live TTY render must also happen at the
        # terminal's own width (100) — the latter fails on the buggy code.
        assert 140 in captured_widths
        assert 100 in captured_widths

    @pytest.mark.asyncio
    async def test_export_console_replays_fixed_width_on_non_tty(
        self, endpoint_config, output_config, sample_records, mock_cfg
    ):
        captured_widths: list[int] = []

        async def _capture_width(*, console: Console) -> None:
            captured_widths.append(console.width)

        instance = MagicMock()
        instance.export = AsyncMock(side_effect=_capture_width)
        mock_class = MagicMock(return_value=instance)
        mock_entry = MagicMock()
        mock_entry.name = "mock_console_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            return_value=[(mock_entry, mock_class)],
        ):
            manager = ExporterManager(
                results=ProfileResults(
                    records=sample_records,
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                    was_cancelled=False,
                    error_summary=[],
                ),
                run=make_run_from_cli(mock_cfg),
                telemetry_results=None,
            )
            # Non-tty: single render at the fixed export width, then the recorded
            # text is replayed verbatim (no second render at terminal width).
            await manager.export_console(Console(file=io.StringIO()))

        assert captured_widths == [140]

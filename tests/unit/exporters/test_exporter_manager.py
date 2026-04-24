# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_manager import ExporterManager


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


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(self, sample_records, config):
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
                config=config,
                telemetry_results=None,
            )
            await manager.export_data()

        mock_class.assert_called_once()
        mock_instance.export.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_console(self, sample_records, config):
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
                config=config,
                telemetry_results=None,
            )
            await manager.export_console(Console())

        for mock_class, mock_instance in zip(
            mock_classes, mock_instances, strict=False
        ):
            mock_class.assert_called_once()
            mock_instance.export.assert_awaited_once()


class TestConsoleExportToFile:
    """Verify export_console writes .txt and .ansi files."""

    @pytest.mark.asyncio
    async def test_writes_txt_and_ansi_files(self, sample_records, config, tmp_path):
        config.artifacts.dir = tmp_path

        mock_instance = MagicMock()

        async def fake_export(console):
            console.print("[bold red]Hello[/bold red] world")

        mock_instance.export = AsyncMock(side_effect=fake_export)
        mock_class = MagicMock(return_value=mock_instance)
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
                config=config,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO()))

        txt_file = tmp_path / "profile_export_console.txt"
        ansi_file = tmp_path / "profile_export_console.ansi"

        assert txt_file.exists(), ".txt file not created"
        assert ansi_file.exists(), ".ansi file not created"

        txt_content = txt_file.read_text()
        ansi_content = ansi_file.read_text()

        assert "Hello" in txt_content
        assert "world" in txt_content
        assert "\x1b[" not in txt_content

        assert "\x1b[" in ansi_content
        assert "Hello" in ansi_content

    @pytest.mark.asyncio
    async def test_file_write_failure_does_not_crash(self, sample_records, config):
        config.artifacts.dir = Path("/nonexistent/path/that/should/fail")

        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_class = MagicMock(return_value=mock_instance)
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
                config=config,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO()))

    @pytest.mark.asyncio
    async def test_stdout_still_receives_output(self, sample_records, config, tmp_path):
        config.artifacts.dir = tmp_path

        mock_instance = MagicMock()

        async def fake_export(console):
            console.print("visible output")

        mock_instance.export = AsyncMock(side_effect=fake_export)
        mock_class = MagicMock(return_value=mock_instance)
        mock_entry = MagicMock()
        mock_entry.name = "mock_exporter"

        stdout_capture = StringIO()

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
                config=config,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=stdout_capture))

        assert "visible output" in stdout_capture.getvalue()

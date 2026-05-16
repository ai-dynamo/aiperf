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
        mock_class.__name__ = "MockExporter"

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
                config=config.benchmark,
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
                config=config.benchmark,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO()))

        for mock_class, mock_instance in zip(
            mock_classes, mock_instances, strict=False
        ):
            mock_class.assert_called_once()
            assert mock_instance.export.await_count == 2

    @pytest.mark.asyncio
    async def test_export_passes_steady_state_and_energy_results(
        self, sample_records, config
    ):
        """ExporterConfig sees the steady_state and energy_efficiency results we pass in."""
        sentinel_steady = MagicMock(name="SteadyStateSummary")
        sentinel_energy = MagicMock(name="EnergyEfficiencySummary")

        manager = ExporterManager(
            results=ProfileResults(
                records=sample_records,
                start_ns=0,
                end_ns=0,
                completed=0,
                was_cancelled=False,
                error_summary=[],
            ),
            config=config.benchmark,
            telemetry_results=None,
            steady_state_results=sentinel_steady,
            energy_efficiency_results=sentinel_energy,
        )

        assert manager._exporter_config.steady_state_results is sentinel_steady
        assert manager._exporter_config.energy_efficiency_results is sentinel_energy


class TestConsoleExportToFile:
    """Verify export_console writes the .txt artifact."""

    @pytest.mark.asyncio
    async def test_writes_txt_file(self, sample_records, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path

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
                config=config.benchmark,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO()))

        txt_file = tmp_path / "profile_export_console.txt"
        ansi_file = tmp_path / "profile_export_console.ansi"

        assert txt_file.exists(), ".txt file not created"
        assert not ansi_file.exists(), ".ansi file should no longer be written"

        txt_content = txt_file.read_text()

        assert "Hello" in txt_content
        assert "world" in txt_content
        assert "\x1b[" not in txt_content

    @pytest.mark.asyncio
    async def test_file_write_failure_does_not_crash(self, sample_records, config):
        config.benchmark.artifacts.dir = Path("/nonexistent/path/that/should/fail")

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
                config=config.benchmark,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO()))

    @pytest.mark.asyncio
    async def test_recording_console_uses_fixed_width_140(
        self, sample_records, config, tmp_path
    ):
        """Non-tty live console and the .txt recording both render at 140 cols."""
        config.benchmark.artifacts.dir = tmp_path

        captured_widths: list[int] = []

        async def fake_export(console):
            captured_widths.append(console.width)

        mock_instance = MagicMock()
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
                config=config.benchmark,
                telemetry_results=None,
            )
            await manager.export_console(Console(file=StringIO(), width=80))

        assert captured_widths == [140, 140]

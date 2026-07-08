# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_config import FileExportInfo
from aiperf.exporters.exporter_manager import ExporterFailure, ExporterManager
from aiperf.plugin.enums import PluginType


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


def _iter_all_by_type(mapping):
    """Build a plugins.iter_all side_effect returning per-PluginType entries.

    export_data() iterates both DATA_EXPORTER (files) and CONSOLE_EXPORTER
    (the recorded .txt artifact), so tests must route each plugin type to
    the intended mock exporters.
    """

    def _side_effect(plugin_type):
        return mapping.get(plugin_type, [])

    return _side_effect


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(self, sample_records, config, tmp_path):
        config.benchmark.artifacts.dir = tmp_path

        # Create a mock exporter instance
        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_instance.is_deferred = False
        mock_class = MagicMock(return_value=mock_instance)
        mock_class.__name__ = "MockExporter"

        # Create a mock PluginEntry for iter_all
        mock_entry = MagicMock()
        mock_entry.name = "mock_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {PluginType.DATA_EXPORTER: [(mock_entry, mock_class)]}
            ),
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
    async def test_deferred_exporters_run_after_local(
        self, sample_records, config, tmp_path
    ):
        """Deferred exporters (wandb/mlflow uploads) must not race the local
        file writers whose artifacts they upload: stage 2 starts only after
        stage 1 (and the console .txt artifact) fully completes."""
        config.benchmark.artifacts.dir = tmp_path
        events: list[str] = []

        class LocalExporter:
            is_deferred = False

            def __init__(self, exporter_config) -> None:
                pass

            def get_export_info(self) -> FileExportInfo:
                return FileExportInfo(export_type="Local", file_path=Path("local.json"))

            async def export(self) -> None:
                events.append("local_start")
                await asyncio.sleep(0.01)
                events.append("local_end")

        class DeferredExporter(LocalExporter):
            is_deferred = True

            async def export(self) -> None:
                events.append("deferred_start")
                events.append("deferred_end")

        local_entry = MagicMock()
        local_entry.name = "local_exporter"
        deferred_entry = MagicMock()
        deferred_entry.name = "deferred_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {
                    PluginType.DATA_EXPORTER: [
                        (deferred_entry, DeferredExporter),
                        (local_entry, LocalExporter),
                    ]
                }
            ),
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

        assert events == ["local_start", "local_end", "deferred_start", "deferred_end"]

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

        # export_console only renders to the live console; the .txt recording
        # pass runs during export_data() instead.
        for mock_class, mock_instance in zip(
            mock_classes, mock_instances, strict=False
        ):
            mock_class.assert_called_once()
            assert mock_instance.export.await_count == 1

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


class TestExporterFailurePropagation:
    """export_data() must SURFACE per-exporter failures, not only log them.

    Regression: when the disk fills mid-export, a data exporter raises
    OSError(ENOSPC) after leaving a truncated artifact. The old code gathered
    tasks with ``return_exceptions=True`` and only logged the exception in a
    done-callback, so ``export_data()`` returned None and the caller stamped a
    lying K8s results-ready marker over corrupt files.
    """

    def _profile_results(self, sample_records) -> ProfileResults:
        return ProfileResults(
            records=sample_records,
            start_ns=0,
            end_ns=0,
            completed=0,
            was_cancelled=False,
            error_summary=[],
        )

    @pytest.mark.asyncio
    async def test_export_data_surfaces_local_exporter_failure(
        self, sample_records, config, tmp_path
    ):
        config.benchmark.artifacts.dir = tmp_path

        class ENOSPCExporter:
            is_deferred = False

            def __init__(self, exporter_config) -> None:
                pass

            def get_export_info(self) -> FileExportInfo:
                return FileExportInfo(
                    export_type="JSON Export",
                    file_path=tmp_path / "profile_export_aiperf.json",
                )

            async def export(self) -> None:
                # Mimic a truncated write followed by a full disk.
                (tmp_path / "profile_export_aiperf.json").write_text('{"metr')
                raise OSError(28, "No space left on device")

        entry = MagicMock()
        entry.name = "json_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {PluginType.DATA_EXPORTER: [(entry, ENOSPCExporter)]}
            ),
        ):
            manager = ExporterManager(
                results=self._profile_results(sample_records),
                config=config.benchmark,
                telemetry_results=None,
            )
            failures = await manager.export_data()

        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, ExporterFailure)
        assert failure.exporter == "ENOSPCExporter"
        assert failure.is_deferred is False
        assert isinstance(failure.error, OSError)
        assert failure.error.errno == 28

    @pytest.mark.asyncio
    async def test_export_data_returns_empty_on_success(
        self, sample_records, config, tmp_path
    ):
        """A clean export surfaces no failures (empty list, never None)."""
        config.benchmark.artifacts.dir = tmp_path

        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_instance.is_deferred = False
        mock_class = MagicMock(return_value=mock_instance)
        mock_class.__name__ = "OkExporter"
        entry = MagicMock()
        entry.name = "ok_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {PluginType.DATA_EXPORTER: [(entry, mock_class)]}
            ),
        ):
            manager = ExporterManager(
                results=self._profile_results(sample_records),
                config=config.benchmark,
                telemetry_results=None,
            )
            failures = await manager.export_data()

        assert failures == []

    @pytest.mark.asyncio
    async def test_export_data_tags_deferred_failure_and_keeps_ordering(
        self, sample_records, config, tmp_path
    ):
        """A deferred (remote-upload) failure is tagged is_deferred=True and a
        surviving local exporter still runs; failures preserve local-before-
        deferred ordering."""
        config.benchmark.artifacts.dir = tmp_path
        events: list[str] = []

        class LocalOk:
            is_deferred = False

            def __init__(self, exporter_config) -> None:
                pass

            def get_export_info(self) -> FileExportInfo:
                return FileExportInfo(export_type="Local", file_path=Path("local.json"))

            async def export(self) -> None:
                events.append("local")

        class DeferredBoom:
            is_deferred = True

            def __init__(self, exporter_config) -> None:
                pass

            def get_export_info(self) -> FileExportInfo:
                return FileExportInfo(
                    export_type="Deferred", file_path=Path("remote.json")
                )

            async def export(self) -> None:
                events.append("deferred")
                raise RuntimeError("wandb upload failed")

        local_entry = MagicMock()
        local_entry.name = "local"
        deferred_entry = MagicMock()
        deferred_entry.name = "deferred"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {
                    PluginType.DATA_EXPORTER: [
                        (deferred_entry, DeferredBoom),
                        (local_entry, LocalOk),
                    ]
                }
            ),
        ):
            manager = ExporterManager(
                results=self._profile_results(sample_records),
                config=config.benchmark,
                telemetry_results=None,
            )
            failures = await manager.export_data()

        assert events == ["local", "deferred"]
        assert len(failures) == 1
        assert failures[0].exporter == "DeferredBoom"
        assert failures[0].is_deferred is True
        assert isinstance(failures[0].error, RuntimeError)


class TestConsoleExportToFile:
    """Verify export_data records and writes the .txt artifact.

    The .txt is written during export_data() (not export_console) so it is
    on disk BEFORE the K8s results-ready marker is stamped by the caller.
    """

    @pytest.mark.asyncio
    async def test_export_data_writes_txt_file(self, sample_records, config, tmp_path):
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
            side_effect=_iter_all_by_type(
                {PluginType.CONSOLE_EXPORTER: [(mock_entry, mock_class)]}
            ),
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

        txt_file = tmp_path / "profile_export_console.txt"
        ansi_file = tmp_path / "profile_export_console.ansi"

        assert txt_file.exists(), ".txt file not created"
        assert not ansi_file.exists(), ".ansi file should no longer be written"

        txt_content = txt_file.read_text()

        assert "Hello" in txt_content
        assert "world" in txt_content
        assert "\x1b[" not in txt_content

    @pytest.mark.asyncio
    async def test_export_console_does_not_write_txt_file(
        self, sample_records, config, tmp_path
    ):
        """The live console pass must not re-write the .txt after the ready
        marker; only export_data() records it."""
        config.benchmark.artifacts.dir = tmp_path

        mock_instance = MagicMock()
        mock_instance.export = AsyncMock()
        mock_class = MagicMock(return_value=mock_instance)
        mock_entry = MagicMock()
        mock_entry.name = "mock_exporter"

        with patch(
            "aiperf.exporters.exporter_manager.plugins.iter_all",
            side_effect=_iter_all_by_type(
                {PluginType.CONSOLE_EXPORTER: [(mock_entry, mock_class)]}
            ),
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

        assert not (tmp_path / "profile_export_console.txt").exists()

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
            side_effect=_iter_all_by_type(
                {PluginType.CONSOLE_EXPORTER: [(mock_entry, mock_class)]}
            ),
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

    @pytest.mark.asyncio
    async def test_recording_console_uses_fixed_width_140(
        self, sample_records, config, tmp_path
    ):
        """The .txt recording (export_data) and a non-tty live console
        (export_console) both render at 140 cols."""
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
            side_effect=_iter_all_by_type(
                {PluginType.CONSOLE_EXPORTER: [(mock_entry, mock_class)]}
            ),
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
            await manager.export_console(Console(file=StringIO(), width=80))

        assert captured_widths == [140, 140]

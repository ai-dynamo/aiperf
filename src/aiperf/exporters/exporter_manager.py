# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console

from aiperf.common.exceptions import (
    ArtifactPublisherDisabled,
    ConsoleExporterDisabled,
    DataExporterDisabled,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.protocols import (
    ArtifactPublisherProtocol,
    ConsoleExporterProtocol,
    DataExporterProtocol,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import DataExporterType, PluginType

if TYPE_CHECKING:
    from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary
    from aiperf.config import BenchmarkConfig, BenchmarkRun
    from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(
        self,
        *,
        results: ProfileResults,
        config: BenchmarkConfig,
        telemetry_results: TelemetryExportData | None,
        server_metrics_results: ServerMetricsResults | None = None,
        steady_state_results: SteadyStateSummary | None = None,
        energy_efficiency_results: EnergyEfficiencySummary | None = None,
        run: BenchmarkRun | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._results = results
        self._config = config
        self._run = run
        self._tasks: set[asyncio.Task] = set()
        self._exporter_config = ExporterConfig(
            results=self._results,
            config=self._config,
            telemetry_results=telemetry_results,
            server_metrics_results=server_metrics_results,
            steady_state_results=steady_state_results,
            energy_efficiency_results=energy_efficiency_results,
            run=run,
        )
        self._exported_file_infos: dict[str, FileExportInfo] = {}

    def _task_done_callback(self, task: asyncio.Task) -> None:
        self.debug(lambda: f"Task done: {task}")
        if task.exception():
            self.error(f"Error exporting records: {task.exception()}")
        else:
            self.debug(f"Exported records: {task.result()}")
        self._tasks.discard(task)

    async def export_data(self) -> None:
        """Export data files using all registered data exporters.

        Also populates exported_file_infos so callers can read file paths
        without re-instantiating exporters.
        """
        self.info("Exporting all records")

        for exporter in self._instantiate_data_exporters():
            self.debug(f"Creating task for exporter: {exporter.__class__.__name__}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting all records completed")

    def _instantiate_data_exporters(self) -> list[DataExporterProtocol]:
        """Instantiate all enabled data exporters, collecting file infos along the way."""
        exporters: list[DataExporterProtocol] = []
        self._exported_file_infos = {}

        for exporter_entry, ExporterClass in plugins.iter_all(PluginType.DATA_EXPORTER):
            if exporter_entry.name == DataExporterType.SERVER_METRICS_PARQUET:
                # TODO: Until the exporters move to the records manager, we need to skip the
                # parquet exporter here, as it requires the server metrics accumulator to be available.
                continue

            try:
                exporter: DataExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except DataExporterDisabled:
                self.debug(
                    f"Data exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:  # noqa: BLE001 - per-exporter; skip bad plugin and continue
                self.error(f"Error creating data exporter: {e!r}")
                continue

            exporters.append(exporter)
            self._exported_file_infos[ExporterClass.__name__] = (
                exporter.get_export_info()
            )

        return exporters

    def get_exported_file_infos(self) -> list[FileExportInfo]:
        """Get the file infos for all exported files (legacy list-shaped API)."""
        return list(self.exported_file_infos.values())

    @property
    def exported_file_infos(self) -> dict[str, FileExportInfo]:
        """File infos collected during export_data() or populated on access.

        Returns dict mapping exporter class name to FileExportInfo.
        After export_data() has run, returns the cached dict. If export_data()
        hasn't been called yet, instantiates exporters to collect the infos.
        """
        if not self._exported_file_infos:
            self._instantiate_data_exporters()
        return self._exported_file_infos

    async def export_console(self, console: Console) -> None:
        self.info("Exporting console data")

        # Without a tty, Rich falls back to a default width that's typically
        # too narrow for our metrics tables; pin it to 140 to match the
        # .txt artifact recording.
        if not console.is_terminal:
            console = Console(file=console.file, width=140)

        # Fixed-width recording used only to capture the .txt artifact;
        # live output goes to `console` directly at terminal width.
        recording_console = Console(
            record=True,
            file=__import__("io").StringIO(),
            force_terminal=True,
            width=140,
        )

        for exporter_entry, ExporterClass in plugins.iter_all(
            PluginType.CONSOLE_EXPORTER
        ):
            try:
                exporter: ConsoleExporterProtocol = ExporterClass(
                    exporter_config=self._exporter_config
                )
            except ConsoleExporterDisabled:
                self.debug(
                    f"Console exporter {exporter_entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:  # noqa: BLE001 - per-exporter; skip bad plugin and continue
                self.error(f"Error creating console exporter: {e!r}")
                continue

            self.debug(f"Creating task for exporter: {exporter_entry.name}")
            live_task = asyncio.create_task(exporter.export(console=console))
            file_task = asyncio.create_task(exporter.export(console=recording_console))
            self._tasks.add(live_task)
            self._tasks.add(file_task)
            live_task.add_done_callback(self._task_done_callback)
            file_task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        self._write_console_files(recording_console)

        self.debug("Exporting console data completed")

    def _write_console_files(self, recording_console: Console) -> None:
        """Write recorded console output to the .txt artifact."""
        try:
            txt_path = self._config.artifacts.profile_export_console_txt_file
            plain_text = recording_console.export_text(styles=False, clear=False)
            txt_path.write_text(plain_text, encoding="utf-8")
            self.debug(f"Console export written to {txt_path}")
        except (OSError, ValueError) as e:
            self.warning(f"Failed to write console export file: {e}")

    async def publish_artifacts(self, artifacts: list[FileExportInfo]) -> None:
        """Publish artifacts to all registered artifact publishers.

        Iterates over all ARTIFACT_PUBLISHER plugins, instantiates each, and
        runs publish() concurrently. Errors are isolated per-publisher.
        """
        self.info("Publishing artifacts to remote storage")

        if not hasattr(PluginType, "ARTIFACT_PUBLISHER"):
            self.debug("No artifact_publisher category registered, skipping")
            return

        for entry, PublisherClass in plugins.iter_all(PluginType.ARTIFACT_PUBLISHER):
            try:
                publisher: ArtifactPublisherProtocol = PublisherClass(
                    exporter_config=self._exporter_config
                )
            except ArtifactPublisherDisabled:
                self.debug(
                    f"Artifact publisher {entry.name} is disabled and will not be used"
                )
                continue
            except Exception as e:  # noqa: BLE001 - per-publisher; skip bad plugin
                self.error(f"Error creating artifact publisher: {e!r}")
                continue

            self.debug(f"Creating task for artifact publisher: {entry.name}")
            task = asyncio.create_task(publisher.publish(artifacts))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Artifact publishing completed")

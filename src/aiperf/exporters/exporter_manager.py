# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console

from aiperf.common.exceptions import (
    ConsoleExporterDisabled,
    DataExporterDisabled,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.protocols import ConsoleExporterProtocol, DataExporterProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import DataExporterType, PluginType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(
        self,
        results: ProfileResults,
        config: BenchmarkConfig,
        telemetry_results: TelemetryExportData | None,
        server_metrics_results: ServerMetricsResults | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._results = results
        self._config = config
        self._tasks: set[asyncio.Task] = set()
        self._exporter_config = ExporterConfig(
            results=self._results,
            config=self._config,
            telemetry_results=telemetry_results,
            server_metrics_results=server_metrics_results,
        )

    def _task_done_callback(self, task: asyncio.Task) -> None:
        self.debug(lambda: f"Task done: {task}")
        if task.exception():
            self.error(f"Error exporting records: {task.exception()}")
        else:
            self.debug(f"Exported records: {task.result()}")
        self._tasks.discard(task)

    async def export_data(self) -> None:
        self.info("Exporting all records")

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

            self.debug(f"Creating task for exporter: {exporter_entry.name}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting all records completed")

    def get_exported_file_infos(self) -> list[FileExportInfo]:
        """Get the file infos for all exported files."""
        file_infos = []
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

            file_infos.append(exporter.get_export_info())
        return file_infos

    async def export_console(self, console: Console) -> None:
        self.info("Exporting console data")

        recording_console = Console(
            record=True,
            file=__import__("io").StringIO(),
            force_terminal=True,
            width=console.width or 100,
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
            task = asyncio.create_task(exporter.export(console=recording_console))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        self._write_console_files(recording_console)

        ansi_output = recording_console.export_text(styles=True)
        if ansi_output.strip():
            console.file.write(ansi_output)
            console.file.flush()

        self.debug("Exporting console data completed")

    def _write_console_files(self, recording_console: Console) -> None:
        """Write recorded console output to .txt and .ansi files."""
        try:
            artifacts = self._config.artifacts
            txt_path = artifacts.profile_export_console_txt_file
            ansi_path = artifacts.profile_export_console_ansi_file

            plain_text = recording_console.export_text(styles=False, clear=False)
            ansi_text = recording_console.export_text(styles=True, clear=False)

            txt_path.write_text(plain_text, encoding="utf-8")
            ansi_path.write_text(ansi_text, encoding="utf-8")

            self.debug(f"Console export written to {txt_path} and {ansi_path}")
        except (OSError, ValueError) as e:
            self.warning(f"Failed to write console export files: {e}")

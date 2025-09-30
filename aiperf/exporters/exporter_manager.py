# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from rich.console import Console

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.factories import ConsoleExporterFactory, DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults, TelemetryResults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(
        self,
        results: ProfileResults,
        input_config: UserConfig,
        service_config: ServiceConfig,
        telemetry_results: TelemetryResults | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ExporterManager with profiling results and configuration used to construct exporter instances.
        
        Parameters:
            results (ProfileResults): Collected profiling results to be exported.
            input_config (UserConfig): User-provided configuration that influences exporter behaviour and output.
            service_config (ServiceConfig): Service-level configuration required by exporters.
            telemetry_results (TelemetryResults | None): Optional telemetry data to include in exporter configuration.
            **kwargs: Additional keyword arguments forwarded to the superclass initializer.
        """
        super().__init__(**kwargs)
        self._results = results
        self._input_config = input_config
        self._tasks: set[asyncio.Task] = set()
        self._service_config = service_config
        self._exporter_config = ExporterConfig(
            results=self._results,
            user_config=self._input_config,
            service_config=self._service_config,
            telemetry_results=telemetry_results,
        )

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        Handle completion of an asyncio Task created by ExporterManager.
        
        This callback logs the task result or any exception raised, and removes the task from the manager's internal tracking set.
        Parameters:
            task (asyncio.Task): The completed task whose outcome will be logged and which will be removed from self._tasks.
        """
        self.debug(lambda: f"Task done: {task}")
        if task.exception():
            self.error(f"Error exporting records: {task.exception()}")
        else:
            self.debug(f"Exported records: {task.result()}")
        self._tasks.discard(task)

    async def export_data(self) -> None:
        self.info("Exporting all records")

        for exporter_type in DataExporterFactory.get_all_class_types():
            exporter = DataExporterFactory.create_instance(
                exporter_type, exporter_config=self._exporter_config
            )
            self.debug(f"Creating task for exporter: {exporter_type}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting all records completed")

    def get_exported_file_infos(self) -> list[FileExportInfo]:
        """Get the file infos for all exported files."""
        file_infos = []
        for exporter_type in DataExporterFactory.get_all_class_types():
            exporter = DataExporterFactory.create_instance(
                exporter_type, exporter_config=self._exporter_config
            )
            file_infos.append(exporter.get_export_info())
        return file_infos

    async def export_console(self, console: Console) -> None:
        self.info("Exporting console data")

        for exporter_type in ConsoleExporterFactory.get_all_class_types():
            exporter = ConsoleExporterFactory.create_instance(
                exporter_type, exporter_config=self._exporter_config
            )
            self.debug(f"Creating task for exporter: {exporter_type}")
            task = asyncio.create_task(exporter.export(console=console))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.debug("Exporting console data completed")

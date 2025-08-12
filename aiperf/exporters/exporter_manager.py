# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from rich.console import Console

from aiperf.common.config import UserConfig
from aiperf.common.factories import ConsoleExporterFactory, DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig


class ExporterManager(AIPerfLoggerMixin):
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(self, results: ProfileResults, input_config: UserConfig, **kwargs):
        super().__init__(**kwargs)
        self._results = results
        self._input_config = input_config
        self._tasks: set[asyncio.Task] = set()
        self._exporter_config = ExporterConfig(
            results=self._results,
            user_config=self._input_config,
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

        for exporter_type in DataExporterFactory.get_all_class_types():
            exporter = DataExporterFactory.create_instance(
                exporter_type, exporter_config=self._exporter_config
            )
            self.debug(f"Creating task for exporter: {exporter_type}")
            task = asyncio.create_task(exporter.export())
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.debug("Exporting all records completed")

    async def export_console(self, console: Console, width: int | None = None) -> None:
        self.info("Exporting console data")

        for exporter_type in ConsoleExporterFactory.get_all_class_types():
            exporter = ConsoleExporterFactory.create_instance(
                exporter_type, exporter_config=self._exporter_config
            )
            self.debug(f"Creating task for exporter: {exporter_type}")
            task = asyncio.create_task(exporter.export(console=console, width=width))
            self._tasks.add(task)
            task.add_done_callback(self._task_done_callback)

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.debug("Exporting console data completed")

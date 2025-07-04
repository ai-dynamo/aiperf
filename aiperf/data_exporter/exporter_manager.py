# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from aiperf.common.config import UserConfig
from aiperf.common.factories import DataExporterFactory
from aiperf.data_exporter.exporter_config import ExporterConfig
from aiperf.progress import ProfileResultsMessage


class ExporterManager:
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(self, results: ProfileResultsMessage, input_config: UserConfig):
        self._results = results
        self._input_config = input_config
        self._exporter_classes = DataExporterFactory.get_all_classes()

    async def export_all(self) -> None:
        tasks: list[asyncio.Task] = []
        for exporter_class in self._exporter_classes:
            exporter_config = ExporterConfig(
                results=self._results,
                input_config=self._input_config,
            )
            exporter = exporter_class(exporter_config)
            task = asyncio.create_task(exporter.export())
            tasks.append(task)

        await asyncio.gather(*tasks)

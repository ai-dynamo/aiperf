# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime

import aiofiles
from pydantic import BaseModel

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models import ErrorDetailsCount, MetricResult
from aiperf.data_exporter.exporter_config import ExporterConfig


class JsonExportData(BaseModel):
    """Data to be exported to a JSON file."""

    input_config: UserConfig | None = None
    records: dict[str, MetricResult] | None = None
    was_cancelled: bool | None = None
    errors_by_type: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@DataExporterFactory.register(DataExporterType.JSON)
class JsonExporter:
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing JsonExporter with config: %s", exporter_config)
        self._results = exporter_config.results
        self._output_directory = exporter_config.input_config.output.artifact_directory
        self._input_config = exporter_config.input_config

    async def export(self) -> None:
        filename = self._output_directory / "profile_export_aiperf.json"
        self._output_directory.mkdir(parents=True, exist_ok=True)

        start_time = (
            datetime.fromtimestamp(self._results.start_ns / NANOS_PER_SECOND)
            if self._results.start_ns
            else None
        )
        end_time = (
            datetime.fromtimestamp(self._results.end_ns / NANOS_PER_SECOND)
            if self._results.end_ns
            else None
        )

        export_data = JsonExportData(
            input_config=self._input_config,
            records={record.tag: record for record in self._results.records},
            was_cancelled=self._results.was_cancelled,
            errors_by_type=self._results.errors_by_type,
            start_time=start_time,
            end_time=end_time,
        )

        self.logger.debug("Exporting data to JSON file: %s", export_data)
        export_data_json = export_data.model_dump_json(indent=2, exclude_unset=True)
        async with aiofiles.open(filename, "w") as f:
            await f.write(export_data_json)

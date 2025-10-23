# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aiofiles

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models import MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.common.types import MetricTagT
from aiperf.exporters.base_json_exporter import BaseJsonExporter
from aiperf.exporters.display_units_utils import convert_all_metrics_to_display_units
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@DataExporterFactory.register(DataExporterType.JSON)
@implements_protocol(DataExporterProtocol)
class JsonExporter(BaseJsonExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
<<<<<<< HEAD
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._input_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._file_path = exporter_config.user_config.output.profile_export_json_file
=======
        super().__init__(exporter_config, **kwargs)
        self.debug(lambda: f"Initializing JsonExporter with config: {exporter_config}")
>>>>>>> b3ba190c (feat: Add support for timeslice metrics JSON and CSV outputs)

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        """Export inference and telemetry data to JSON file.

        Creates a JSON file containing:
        - Input configuration
        - Inference metric results (converted to display units)
        - Telemetry data with statistical summaries per endpoint/GPU
        - Error summaries
        - Timestamps

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to JSON file: {self._file_path}")

        try:
            json_content = self._generate_json_content(
                self._results.records, self._telemetry_results
            )

            async with aiofiles.open(self._file_path, "w", encoding="utf-8") as f:
                await f.write(json_content)

        except Exception as e:
            self.error(f"Failed to export CSV to {self._file_path}: {e}")
            raise

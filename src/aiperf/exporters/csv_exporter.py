# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import aiofiles

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.base_csv_exporter import BaseCsvExporter
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@DataExporterFactory.register(DataExporterType.CSV)
@implements_protocol(DataExporterProtocol)
class CsvExporter(BaseCsvExporter):
    """Exports records to a CSV file in a legacy, two-section format."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self.debug(lambda: f"Initializing CsvExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._metric_registry = MetricRegistry
        self._file_path = exporter_config.user_config.output.profile_export_csv_file
        self._percentile_keys = _percentile_keys_from(STAT_KEYS)

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="CSV Export",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        """Export inference and telemetry data to CSV file.

        Creates a CSV file with three sections:
        1. Request metrics (with percentiles)
        2. System metrics (single values)
        3. GPU telemetry metrics (if available)

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to CSV file: {self._file_path}")

        try:
            csv_content = self._generate_csv_content(
                self._results.records, self._telemetry_results
            )

            async with aiofiles.open(
                self._file_path, "w", newline="", encoding="utf-8"
            ) as f:
                await f.write(csv_content)

        except Exception as e:
            self.error(f"Failed to export CSV to {self._file_path}: {e}")
            raise

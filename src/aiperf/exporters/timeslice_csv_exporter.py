# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aiofiles

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.base_csv_exporter import BaseCsvExporter
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@DataExporterFactory.register(DataExporterType.TIMESLICE_CSV)
@implements_protocol(DataExporterProtocol)
class TimesliceCsvExporter(BaseCsvExporter):
    """Exports timeslice metrics to individual CSV files.

    Creates separate CSV files for each timeslice in the format:
    {artifact_directory}/timeslices/profile_export_aiperf_{index}.csv
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)

        self.debug(
            lambda: f"Initializing TimesliceCsvExporter with config: {exporter_config}"
        )

        if not self._results.timeslice_metric_results:
            raise DataExporterDisabled(
                "TimesliceCsvExporter disabled: no timeslice metric results found"
            )

        # Use configured subdirectory name
        self._timeslices_dir = (
            self._output_directory / OutputDefaults.TIMESLICES_SUBDIRECTORY
        )

        # Extract base filename from configured CSV path
        # e.g., "profile_export_aiperf.csv" -> "profile_export_aiperf"
        # or user's custom "my_export.csv" -> "my_export"
        self._base_filename = (
            exporter_config.user_config.output._profile_export_csv_file.stem
        )

        self.debug(
            lambda: f"Initialized TimesliceCsvExporter: base={self._base_filename}, dir={self._timeslices_dir}"
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice CSV Export",
            file_path=self._timeslices_dir,
        )

    async def export(self) -> None:
        """Export timeslice metrics to individual CSV files.

        Creates a CSV file for each timeslice using the BaseCsvExporter format.
        Each file contains request metrics, system metrics (no telemetry).

        Raises:
            Exception: If file writing fails
        """
        self._timeslices_dir.mkdir(parents=True, exist_ok=True)

        self.info(
            f"Exporting {len(self._results.timeslice_metric_results)} timeslice CSV files to {self._timeslices_dir}"
        )

        for counter, timeslice_index in enumerate(
            sorted(self._results.timeslice_metric_results.keys())
        ):
            # Use the configured base filename
            file_path = self._timeslices_dir / f"{self._base_filename}_{counter}.csv"

            metric_results = self._results.timeslice_metric_results[timeslice_index]

            try:
                # Generate CSV content (no telemetry for timeslices)
                csv_content = self._generate_csv_content(
                    metric_results, telemetry_results=None
                )

                self.debug(f"Exporting timeslice {counter} to {file_path.name}")

                async with aiofiles.open(
                    file_path, "w", newline="", encoding="utf-8"
                ) as f:
                    await f.write(csv_content)

            except Exception as e:
                self.error(f"Failed to export CSV to {file_path}: {e}")
                raise

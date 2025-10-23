# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aiofiles

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.factories import DataExporterFactory
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.base_json_exporter import BaseJsonExporter
from aiperf.exporters.display_units_utils import convert_all_metrics_to_display_units
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@DataExporterFactory.register(DataExporterType.TIMESLICE_JSON)
@implements_protocol(DataExporterProtocol)
class TimesliceJsonExporter(BaseJsonExporter):
    """
    A class to export timeslice metrics to JSON files.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self.debug(
            lambda: f"Initializing TimesliceJsonExporter with config: {exporter_config}"
        )

        if not self._results.timeslice_metric_results:
            raise DataExporterDisabled(
                "TimesliceJsonExporter disabled: no timeslice metric results found"
            )

        # Use configured subdirectory name
        self._timeslices_dir = (
            self._output_directory / OutputDefaults.TIMESLICES_SUBDIRECTORY
        )

        # Extract base filename from configured JSON path
        # e.g., "profile_export_aiperf.json" -> "profile_export_aiperf"
        # or user's custom "my_export.json" -> "my_export"
        self._base_filename = (
            exporter_config.user_config.output._profile_export_json_file.stem
        )

        self.debug(
            lambda: f"Initialized TimesliceJsonExporter: base={self._base_filename}, dir={self._timeslices_dir}"
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice JSON Export",
            file_path=self._timeslices_dir,
        )

    async def export(self) -> None:
        """Export timeslice metrics to individual JSON files.

        Creates a JSON file for each timeslice using the BaseJsonExporter format.
        Each file contains request metrics, system metrics (no telemetry).

        Raises:
            Exception: If file writing fails
        """
        self._timeslices_dir.mkdir(parents=True, exist_ok=True)

        self.info(
            f"Exporting {len(self._results.timeslice_metric_results)} timeslice JSON files to {self._timeslices_dir}"
        )

        for counter, timeslice_index in enumerate(
            sorted(self._results.timeslice_metric_results.keys())
        ):
            # Use the configured base filename
            file_path = self._timeslices_dir / f"{self._base_filename}_{counter}.json"

            try:
                metric_results = self._results.timeslice_metric_results[timeslice_index]

                # Convert to display units (e.g., ms instead of ns)
                metric_results_converted_units = convert_all_metrics_to_display_units(
                    metric_results, self._metric_registry
                )

                # Generate JSON content (no telemetry for timeslices)
                json_content = self._generate_json_content(
                    metric_results_converted_units, telemetry_results=None
                )

                self.debug(f"Exporting timeslice {counter} to {file_path.name}")

                async with aiofiles.open(
                    file_path, "w", newline="", encoding="utf-8"
                ) as f:
                    await f.write(json_content)

            except Exception as e:
                self.error(f"Failed to export JSON to {file_path}: {e}")
                raise

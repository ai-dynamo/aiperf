# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models.export_models import (
    TimesliceCollectionExportData,
    TimesliceData,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter


class TimesliceMetricsJsonExporter(MetricsJsonExporter):
    """Exports all timeslice metrics to a single JSON file.

    Creates one JSON file containing an array of all timeslices in the format:
    {
        "timeslices": [
            {"timeslice_index": 0, "metric_1": {...}, "metric_2": {...}},
            {"timeslice_index": 1, "metric_1": {...}, "metric_2": {...}}
        ],
        "input_config": {...}
    }
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)

        if not self._results.timeslice_metric_results:
            raise DataExporterDisabled(
                "TimesliceMetricsJsonExporter disabled: no timeslice metric results found"
            )

        # Override file path for timeslice-specific output
        self._file_path = (
            exporter_config.user_config.output.profile_export_timeslices_json_file
        )
        self.trace_or_debug(
            lambda: f"Initializing TimesliceMetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing TimesliceMetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate single JSON with all timeslices in an array.

        Uses instance data member self._results.timeslice_metric_results.

        Returns:
            str: JSON content with all timeslices
        """
        timeslices_list = []

        for timeslice_index in sorted(self._results.timeslice_metric_results.keys()):
            metric_results = self._results.timeslice_metric_results[timeslice_index]

            # Reuse base class helper to prepare metrics
            prepared_json_metrics = self._prepare_metrics_for_json(metric_results)

            # Create timeslice object with dynamic metrics
            timeslice = TimesliceData(timeslice_index=timeslice_index)
            for tag, json_result in prepared_json_metrics.items():
                setattr(timeslice, tag, json_result)

            timeslices_list.append(timeslice)

        # Create collection with metadata
        export_data = TimesliceCollectionExportData(
            timeslices=timeslices_list,
            input_config=self._user_config,
        )

        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

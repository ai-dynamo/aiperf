# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import scrub_non_finite
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
            {"start_ns": ..., "end_ns": ..., "metric_1": {...}, ...},
            {"start_ns": ..., "end_ns": ..., "metric_1": {...}, ...}
        ],
        "input_config": {...}
    }

    Slice ordering is conveyed by position in the array — there is no explicit
    timeslice_index field, matching the server-metrics BaseTimeslice wire
    format. Reads ``ProfileResults.timeslices`` produced by the
    MetricsAccumulator engine.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)

        if not self._results or not self._results.timeslices:
            raise DataExporterDisabled(
                "TimesliceMetricsJsonExporter disabled: no timeslice metric results found"
            )

        # Override file path for timeslice-specific output
        self._file_path = (
            exporter_config.cfg.artifacts.profile_export_timeslices_json_file
        )
        self.trace_or_debug(
            lambda: (
                f"Initializing TimesliceMetricsJsonExporter with config: {exporter_config}"
            ),
            lambda: (
                f"Initializing TimesliceMetricsJsonExporter with file path: {self._file_path}"
            ),
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate single JSON with all timeslices in an array.

        Uses instance data member self._results.timeslices.

        Returns:
            str: JSON content with all timeslices
        """
        timeslices_list = []

        # Slices are stored in chronological order. Position == slice index.
        for ts in self._results.timeslices:
            prepared_json_metrics = self._prepare_metrics_for_json(
                ts.metric_results.values()
            )

            timeslice = TimesliceData(
                start_ns=ts.start_ns,
                end_ns=ts.end_ns,
                is_complete=ts.is_complete,
            )
            for tag, json_result in prepared_json_metrics.items():
                setattr(timeslice, tag, json_result)

            timeslices_list.append(timeslice)

        # Create collection with metadata
        export_data = TimesliceCollectionExportData(
            timeslices=timeslices_list,
            input_config=self._cfg,
        )

        return orjson.dumps(
            scrub_non_finite(
                export_data.model_dump(
                    mode="json", exclude_unset=True, exclude_none=True
                )
            ),
            option=orjson.OPT_INDENT_2,
        ).decode("utf-8")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from decimal import Decimal

from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class ArchetypeMetricsCsvExporter(MetricsBaseExporter):
    """Exports per-archetype metrics to a single CSV file in tidy/long format.

    Creates one CSV file with all archetypes in a tidy data format:
        Archetype,Metric,Unit,Stat,Value
        image-only,Request Latency,ms,avg,30.0
        image-only,Request Latency,ms,p95,55.0
        video-only,Request Latency,ms,avg,890.0
        ...

    This format is optimal for data science tools (pandas, R, Tableau, etc.)
    and matches the shape of TimesliceMetricsCsvExporter.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        archetype_results = getattr(self._results, "archetype_metric_results", None)
        if not archetype_results:
            raise DataExporterDisabled(
                "ArchetypeMetricsCsvExporter disabled: no archetype metric results found"
            )

        self._file_path = (
            exporter_config.user_config.output.profile_export_archetypes_csv_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ArchetypeMetricsCsvExporter with config: {exporter_config}",
            lambda: f"Initializing ArchetypeMetricsCsvExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Archetype CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate tidy/long format CSV content from all archetypes.

        Uses instance data member self._results.archetype_metric_results.

        Returns:
            str: Complete CSV content in tidy format
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(["Archetype", "Metric", "Unit", "Stat", "Value"])

        archetype_results = self._results.archetype_metric_results
        for archetype_name in sorted(archetype_results.keys()):
            metric_results_list = archetype_results[archetype_name]
            prepared_metrics = self._prepare_metrics(metric_results_list)

            for tag, metric in sorted(prepared_metrics.items()):
                metric_name = metric.header or tag
                unit = metric.unit or ""

                for stat in STAT_KEYS:
                    value = getattr(metric, stat, None)
                    if value is not None:
                        writer.writerow(
                            [
                                archetype_name,
                                metric_name,
                                unit,
                                stat,
                                self._format_number(value),
                            ]
                        )

        return buf.getvalue()

    def _format_number(self, value) -> str:
        """Format a number for CSV output. Mirrors TimesliceMetricsCsvExporter."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.2f}"
        return str(value)

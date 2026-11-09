# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from decimal import Decimal

from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class TimesliceMetricsCsvExporter(MetricsBaseExporter):
    """Exports timeslice metrics to a single CSV file in tidy/long format.

    Creates one CSV file with all timeslices in a tidy data format:
        Timeslice,Start_NS,End_NS,Metric,Unit,Stat,Value
        0,...,...,Request Latency,ms,avg,45.2
        0,...,...,Request Latency,ms,min,12.1
        1,...,...,Request Latency,ms,avg,48.5
        ...

    This format is optimal for data science tools (pandas, R, Tableau, etc.)
    System metrics (with single values) use stat='avg'. Reads
    ``ProfileResults.timeslices`` produced by the MetricsAccumulator engine.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        if not self._results or not self._results.timeslices:
            raise DataExporterDisabled(
                "TimesliceMetricsCsvExporter disabled: no timeslice metric results found"
            )

        # Extract base filename from configured CSV path
        self._file_path = (
            exporter_config.cfg.artifacts.profile_export_timeslices_csv_file
        )
        self.trace_or_debug(
            lambda: (
                f"Initializing TimesliceMetricsCsvExporter with config: {exporter_config}"
            ),
            lambda: (
                f"Initializing TimesliceMetricsCsvExporter with file path: {self._file_path}"
            ),
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Timeslice CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate tidy/long format CSV content from all timeslices.

        Uses instance data member self._results.timeslices.

        Returns:
            str: Complete CSV content in tidy format
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Write header
        writer.writerow(
            ["Timeslice", "Start_NS", "End_NS", "Metric", "Unit", "Stat", "Value"]
        )

        # Slices are stored in chronological order. Position == slice index.
        for timeslice_index, ts in enumerate(self._results.timeslices):
            # Convert to display units and filter exportable metrics
            prepared_metrics = self._prepare_metrics(ts.metric_results.values())

            # Write rows for each metric
            for tag, metric in sorted(prepared_metrics.items()):
                metric_name = metric.header or tag
                unit = metric.unit or ""

                # Write a row for each stat that has a value
                for stat in STAT_KEYS:
                    value = getattr(metric, stat, None)
                    if value is not None:
                        writer.writerow(
                            [
                                timeslice_index,
                                ts.start_ns,
                                ts.end_ns,
                                metric_name,
                                unit,
                                stat,
                                self._format_number(value),
                            ]
                        )

        return buf.getvalue()

    def _format_number(self, value) -> str:
        """Format a number for CSV output."""
        if value is None:
            return ""
        # Handle bools explicitly (bool is a subclass of int)
        if isinstance(value, bool):
            return str(value)
        # Integers (covers built-in int and other Integral implementations)
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        # Real numbers (covers built-in float and many Real implementations) and Decimal
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.2f}"

        return str(value)

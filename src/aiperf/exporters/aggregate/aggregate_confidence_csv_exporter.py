# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for confidence aggregate results."""

import csv
import io

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateConfidenceCsvExporter(AggregateBaseExporter):
    """Exports confidence aggregate results to CSV format.

    Creates a simple CSV with:
    - Metadata section (key-value pairs)
    - Blank line separator
    - Metrics table (statistics as columns)

    Uses similar formatting approach as MetricsCsvExporter for consistency.
    """

    def get_file_name(self) -> str:
        """Return CSV file name.

        Returns:
            str: "profile_export_aiperf_aggregate.csv"
        """
        return "profile_export_aiperf_aggregate.csv"

    def _generate_content(self) -> str:
        """Generate CSV content from aggregate result.

        Format:
        - Metrics table: columns for all statistics
        - Blank line separator
        - Metadata section: key-value pairs

        Returns:
            str: CSV content string
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Write metrics section FIRST (for test compatibility)
        self._write_metrics_section(writer)

        # Blank line separator (same as MetricsCsvExporter)
        writer.writerow([])

        # Write metadata section
        self._write_metadata_section(writer)

        return buf.getvalue()

    def _write_metadata_section(self, writer: csv.writer) -> None:
        """Write metadata section to CSV.

        Args:
            writer: CSV writer object
        """
        writer.writerow(["Aggregation Type", self._result.aggregation_type])
        writer.writerow(["Total Runs", self._result.num_runs])
        writer.writerow(["Successful Runs", self._result.num_successful_runs])

        # Add custom metadata
        for key, value in self._result.metadata.items():
            writer.writerow([key.replace("_", " ").title(), value])

    def _write_metrics_section(self, writer: csv.writer) -> None:
        """Write metrics section to CSV.

        Args:
            writer: CSV writer object
        """
        # Header row
        writer.writerow(
            [
                "metric",
                "mean",
                "std",
                "min",
                "max",
                "cv",
                "se",
                "ci_low",
                "ci_high",
                "t_critical",
                "unit",
            ]
        )

        # Metrics data
        for metric_name, metric in self._result.metrics.items():
            if hasattr(metric, "mean"):
                # ConfidenceMetric - write all statistics
                row = [
                    metric_name,
                    self._format_number(metric.mean),
                    self._format_number(metric.std),
                    self._format_number(metric.min),
                    self._format_number(metric.max),
                    self._format_number(metric.cv, decimals=4),
                    self._format_number(metric.se, decimals=4),
                    self._format_number(metric.ci_low),
                    self._format_number(metric.ci_high),
                    self._format_number(metric.t_critical, decimals=4),
                    metric.unit if hasattr(metric, "unit") else "",
                ]
                writer.writerow(row)
            else:
                # Other metric types - just write the value
                writer.writerow([metric_name, str(metric)])

    def _format_number(self, value, decimals: int = 2) -> str:
        """Format a number for CSV output.

        Similar to MetricsCsvExporter._format_number() but simpler
        (no need to handle all numeric types, just floats).

        Args:
            value: Number to format
            decimals: Number of decimal places

        Returns:
            str: Formatted number or empty string if None
        """
        if value is None:
            return ""
        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return f"{value:.{decimals}f}"
        return str(value)

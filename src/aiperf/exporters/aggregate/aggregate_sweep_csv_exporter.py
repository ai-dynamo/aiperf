# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for sweep aggregate results."""

import csv
import io

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepCsvExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to CSV format.

    Creates a CSV with multiple sections:
    - Per-combination metrics table (one row per parameter combination)
    - Blank line separator
    - Best configurations section
    - Pareto optimal points section
    - Metadata section

    Uses similar formatting approach as AggregateConfidenceCsvExporter for consistency.
    """

    def get_file_name(self) -> str:
        """Return CSV file name.

        Returns:
            str: "profile_export_aiperf_sweep.csv"
        """
        return "profile_export_aiperf_sweep.csv"

    def _generate_content(self) -> str:
        """Generate CSV content from sweep aggregate result.

        Format: Multiple sections separated by blank lines:
        1. Per-combination metrics table (one row per parameter combination)
        2. Best configurations section
        3. Pareto optimal points section
        4. Metadata section

        Returns:
            str: CSV content string
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Get sweep parameters from metadata
        sweep_parameters = self._result.metadata.get("sweep_parameters", [])
        param_names = [p["name"] for p in sweep_parameters]

        # Section 1: Per-combination metrics table
        per_combination_metrics = self._result.metrics

        if per_combination_metrics:
            # Build header: param1, param2, ..., metric1_mean, metric1_std, ...
            header = param_names.copy()

            # Get all metric names from the first combination
            if per_combination_metrics:
                first_combo = per_combination_metrics[0]
                metric_names = sorted(first_combo.get("metrics", {}).keys())
            else:
                metric_names = []

            # Add columns for each metric's statistics
            for metric_name in metric_names:
                header.extend(
                    [
                        f"{metric_name}_mean",
                        f"{metric_name}_std",
                        f"{metric_name}_min",
                        f"{metric_name}_max",
                        f"{metric_name}_cv",
                    ]
                )

            writer.writerow(header)

            # Write data rows (one row per parameter combination)
            for combo_entry in per_combination_metrics:
                parameters = combo_entry.get("parameters", {})
                metrics = combo_entry.get("metrics", {})

                # Start row with parameter values
                row = [parameters.get(param_name, "") for param_name in param_names]

                # Add metric statistics
                for metric_name in metric_names:
                    metric_data = metrics.get(metric_name, {})
                    if isinstance(metric_data, dict):
                        row.extend(
                            [
                                self._format_number(metric_data.get("mean")),
                                self._format_number(metric_data.get("std")),
                                self._format_number(metric_data.get("min")),
                                self._format_number(metric_data.get("max")),
                                self._format_number(metric_data.get("cv"), decimals=4),
                            ]
                        )
                    else:
                        # If not a dict, fill with empty values
                        row.extend(["", "", "", "", ""])

                writer.writerow(row)

        # Section 2: Best Configurations
        writer.writerow([])  # Blank line
        writer.writerow(["Best Configurations"])
        best_configs = self._result.metadata.get("best_configurations", {})
        if best_configs:
            # Header with all parameter names
            header = ["Configuration"] + param_names + ["Metric", "Unit"]
            writer.writerow(header)

            for config_name, config_data in best_configs.items():
                # Format config name: best_throughput -> Best Throughput
                formatted_name = config_name.replace("_", " ").title()
                parameters = config_data.get("parameters", {})

                # Build row with parameter values
                row = [formatted_name]
                row.extend(
                    [parameters.get(param_name, "") for param_name in param_names]
                )
                row.extend(
                    [
                        self._format_number(config_data.get("metric")),
                        config_data.get("unit", ""),
                    ]
                )
                writer.writerow(row)

        # Section 3: Pareto Optimal Points
        writer.writerow([])  # Blank line
        writer.writerow(["Pareto Optimal Points"])
        pareto_optimal = self._result.metadata.get("pareto_optimal", [])
        if pareto_optimal:
            # Header with all parameter names
            writer.writerow(param_names)
            for combo_params in pareto_optimal:
                row = [combo_params.get(param_name, "") for param_name in param_names]
                writer.writerow(row)
        else:
            writer.writerow(["None"])

        # Section 4: Metadata
        writer.writerow([])  # Blank line
        writer.writerow(["Metadata"])
        writer.writerow(["Field", "Value"])
        writer.writerow(["Aggregation Type", self._result.aggregation_type])
        writer.writerow(["Sweep Parameters", ", ".join(param_names)])
        writer.writerow(
            ["Number of Combinations", self._result.metadata.get("num_combinations", 0)]
        )
        writer.writerow(["Number of Profile Runs", self._result.num_runs])
        writer.writerow(["Number of Successful Runs", self._result.num_successful_runs])

        return buf.getvalue()

    def _format_number(self, value: float | int | None, decimals: int = 2) -> str:
        """Format a number for CSV output.

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

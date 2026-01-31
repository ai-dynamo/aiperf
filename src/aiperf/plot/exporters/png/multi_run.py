# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run PNG exporter for comparison plots.

Generates static PNG images comparing multiple profiling runs.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.record_models import MetricResult
from aiperf.plot.constants import DEFAULT_PERCENTILE, NON_METRIC_KEYS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import flatten_config
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig, PlotSpec
from aiperf.plot.exporters.png.base import BasePNGExporter
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


class MultiRunPNGExporter(BasePNGExporter):
    """
    PNG exporter for multi-run comparison plots.

    Generates static PNG images comparing multiple profiling runs:
    1. Pareto curve (latency vs throughput)
    2. TTFT vs Throughput
    3. Throughput per User vs Concurrency
    4. Token Throughput per GPU vs Latency (conditional on telemetry)
    5. Token Throughput per GPU vs Interactivity (conditional on telemetry)
    """

    def export(
        self,
        runs: list[RunData],
        available_metrics: dict,
        plot_specs: list[PlotSpec],
        classification_config: ExperimentClassificationConfig | None = None,
    ) -> list[Path]:
        """
        Export multi-run comparison plots as PNG files.

        Args:
            runs: List of RunData objects with aggregated metrics
            available_metrics: Dictionary with display_names and units for metrics
            plot_specs: List of plot specifications defining which plots to generate

        Returns:
            List of Path objects for generated PNG files
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = self._runs_to_dataframe(runs, available_metrics, classification_config)

        generated_files = []

        for spec in plot_specs:
            try:
                if not self._can_generate_plot(spec, df):
                    self.debug(f"Skipping {spec.name} - required columns not available")
                    continue

                fig = self._create_plot_from_spec(spec, df, available_metrics)

                path = self.output_dir / spec.filename
                self._export_figure(fig, path)
                self.debug(f"Generated {spec.filename}")
                generated_files.append(path)

            except Exception as e:
                self.error(f"Failed to generate {spec.name}: {e}")

        self._create_summary_file(generated_files)

        return generated_files

    def _can_generate_plot(self, spec: PlotSpec, df: pd.DataFrame) -> bool:
        """
        Check if a plot can be generated based on column availability.

        Args:
            spec: Plot specification
            df: DataFrame with aggregated metrics

        Returns:
            True if the plot can be generated, False otherwise
        """
        for metric in spec.metrics:
            if metric.name not in df.columns and metric.name != "concurrency":
                return False
        return True

    def _create_plot_from_spec(
        self, spec: PlotSpec, df: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """
        Create a plot figure from a plot specification using the factory pattern.

        Args:
            spec: Plot specification
            df: DataFrame with aggregated metrics
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            Plotly figure object
        """
        HandlerClass = plugins.get_class(PluginType.PLOT, spec.plot_type)
        handler = HandlerClass(plot_generator=self.plot_generator)

        return handler.create_plot(spec, df, available_metrics)

    def _runs_to_dataframe(
        self,
        runs: list[RunData],
        available_metrics: dict,
        classification_config: ExperimentClassificationConfig | None = None,
    ) -> pd.DataFrame:
        """
        Convert list of run data into a DataFrame for plotting.

        Extracts all configuration fields to support arbitrary swept parameter analysis.

        Args:
            runs: List of RunData objects
            available_metrics: Dictionary with display_names and units

        Returns:
            DataFrame with columns for metrics, metadata, and all config fields
        """
        rows = []
        for run in runs:
            row = {}

            row["run_name"] = run.metadata.run_name
            row["model"] = run.metadata.model or "Unknown"
            row["concurrency"] = run.metadata.concurrency or 1
            row["request_count"] = run.metadata.request_count
            row["duration_seconds"] = run.metadata.duration_seconds
            row["experiment_type"] = run.metadata.experiment_type
            row["experiment_group"] = run.metadata.experiment_group
            if run.metadata.endpoint_type:
                row["endpoint_type"] = run.metadata.endpoint_type

            if "input_config" in run.aggregated:
                config = run.aggregated["input_config"]
                flattened = flatten_config(config)
                row.update(flattened)

            for key, value in run.aggregated.items():
                if key in NON_METRIC_KEYS:
                    continue

                if isinstance(value, MetricResult):
                    if (
                        hasattr(value, DEFAULT_PERCENTILE)
                        and getattr(value, DEFAULT_PERCENTILE) is not None
                    ):
                        row[key] = getattr(value, DEFAULT_PERCENTILE)
                    elif value.avg is not None:
                        row[key] = value.avg
                elif isinstance(value, dict) and "unit" in value and value is not None:
                    if DEFAULT_PERCENTILE in value:
                        row[key] = value[DEFAULT_PERCENTILE]
                    elif "avg" in value:
                        row[key] = value["avg"]
                    elif "value" in value:
                        row[key] = value["value"]

            # Extract server metrics from server_metrics_aggregated
            if run.server_metrics_aggregated:
                for metric_name, endpoint_data in run.server_metrics_aggregated.items():
                    # Aggregate across ALL endpoints and label combinations
                    # This ensures consistent behavior regardless of label cardinality
                    values = []
                    metric_type = None
                    total_combinations = 0

                    for _endpoint_url, labels_dict in endpoint_data.items():
                        for _labels_key, series_data in labels_dict.items():
                            total_combinations += 1
                            stats = series_data.get("stats")

                            if stats is None:
                                # Static value (no variation) - use the value directly
                                static_value = series_data.get("value")
                                if static_value is not None:
                                    values.append(static_value)
                                continue

                            # Extract metric type (same for all series)
                            if metric_type is None:
                                metric_type = series_data.get(
                                    "type", PrometheusMetricType.UNKNOWN
                                )

                            # Extract appropriate stat based on metric type
                            if metric_type == PrometheusMetricType.COUNTER:
                                # Use rate for counters
                                if hasattr(stats, "rate") and stats.rate is not None:
                                    values.append(stats.rate)
                                elif (
                                    isinstance(stats, dict)
                                    and stats.get("rate") is not None
                                ):
                                    values.append(stats["rate"])
                            else:
                                # Use avg for gauge/histogram
                                if hasattr(stats, "avg") and stats.avg is not None:
                                    values.append(stats.avg)
                                elif (
                                    isinstance(stats, dict)
                                    and stats.get("avg") is not None
                                ):
                                    values.append(stats["avg"])

                    # Aggregate all values
                    if values:
                        # Use sum for counters (total rate), average for others
                        if metric_type == PrometheusMetricType.COUNTER:
                            row[metric_name] = sum(
                                values
                            )  # Sum rates across all labels/endpoints
                        else:
                            row[metric_name] = sum(values) / len(
                                values
                            )  # Average across labels/endpoints

                        # Warn if multiple combinations exist (potential semantic issue)
                        if total_combinations > 1:
                            self.debug(
                                f"Server metric '{metric_name}' has {total_combinations} "
                                f"endpoint+label combinations - aggregated to single value "
                                f"({'sum' if metric_type == PrometheusMetricType.COUNTER else 'average'})"
                            )

            rows.append(row)

        df = pd.DataFrame(rows)

        if "experiment_group" in df.columns:
            if classification_config and classification_config.group_display_names:
                df["group_display_name"] = (
                    df["experiment_group"]
                    .map(classification_config.group_display_names)
                    .fillna(df["experiment_group"])
                )
            else:
                df["group_display_name"] = df["experiment_group"]

        if "experiment_group" in df.columns:
            unique_groups = df["experiment_group"].unique()
            self.info(
                f"DataFrame has {len(unique_groups)} unique experiment_groups: {sorted(unique_groups)}"
            )

        if "experiment_type" in df.columns:
            unique_types = df["experiment_type"].unique()
            self.info(
                f"DataFrame has {len(unique_types)} unique experiment_types: {sorted(unique_types)}"
            )

        return df

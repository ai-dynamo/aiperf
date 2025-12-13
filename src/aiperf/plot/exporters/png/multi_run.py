# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run PNG exporter for comparison plots.

Generates static PNG images comparing multiple profiling runs.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from aiperf.common.models.record_models import MetricResult
from aiperf.plot.constants import DEFAULT_PERCENTILE, NON_METRIC_KEYS
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import flatten_config
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig, PlotSpec
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory
from aiperf.plot.exporters.png.base import BasePNGExporter


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
        handler = PlotTypeHandlerFactory.create_instance(
            spec.plot_type,
            plot_generator=self.plot_generator,
        )

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

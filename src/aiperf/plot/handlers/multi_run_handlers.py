# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run plot type handlers.

Handlers for creating comparison plots from multiple profiling runs.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.constants import DEFAULT_PERCENTILE
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotSpec, PlotType
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory


class BaseMultiRunHandler:
    """
    Base class for multi-run plot handlers.

    Provides common functionality for working with multi-run DataFrames.
    """

    def __init__(self, plot_generator: PlotGenerator) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
        """
        self.plot_generator = plot_generator

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        display_name = None
        unit = ""

        if "display_names" in available_metrics or "units" in available_metrics:
            display_name = available_metrics.get("display_names", {}).get(metric_name)
            unit = available_metrics.get("units", {}).get(metric_name, "")

        if not display_name and metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")

        if display_name:
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name
        return metric_name

    def _extract_experiment_types(
        self, data: pd.DataFrame, group_by: str | None
    ) -> dict[str, str] | None:
        """
        Extract experiment types from DataFrame for experiment groups color assignment.

        Args:
            data: DataFrame with aggregated metrics
            group_by: Column name to group by

        Returns:
            Dictionary mapping group values to experiment_type, or None
        """
        if not group_by or group_by not in data.columns:
            return None

        if "experiment_type" not in data.columns:
            return None

        experiment_types = {}
        for group_val in data[group_by].unique():
            group_df = data[data[group_by] == group_val]
            experiment_types[group_val] = group_df["experiment_type"].iloc[0]

        return experiment_types

    def _extract_group_display_names(
        self, data: pd.DataFrame, group_by: str | None
    ) -> dict[str, str] | None:
        """
        Extract group display names from DataFrame for legend labels.

        Args:
            data: DataFrame with aggregated metrics
            group_by: Column name to group by

        Returns:
            Dictionary mapping group values to display names, or None
        """
        if not group_by or group_by not in data.columns:
            return None

        # Only use group_display_name when grouping by experiment_group
        # For other groupings (e.g., 'model'), use the actual group value
        if group_by != "experiment_group":
            return None

        if "group_display_name" not in data.columns:
            return None

        display_names = {}
        for group_val in data[group_by].unique():
            group_df = data[data[group_by] == group_val]
            display_names[group_val] = group_df["group_display_name"].iloc[0]

        return display_names


@PlotTypeHandlerFactory.register(PlotType.PARETO)
class ParetoHandler(BaseMultiRunHandler):
    """Handler for Pareto curve plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if Pareto plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a Pareto curve plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        experiment_types = self._extract_experiment_types(data, spec.group_by)
        group_display_names = self._extract_group_display_names(data, spec.group_by)

        return self.plot_generator.create_pareto_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
            group_display_names=group_display_names,
        )


@PlotTypeHandlerFactory.register(PlotType.SCATTER_LINE)
class ScatterLineHandler(BaseMultiRunHandler):
    """Handler for scatter line plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if scatter line plot can be generated."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter line plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        if x_metric.name == "concurrency":
            x_label = "Concurrency Level"
        else:
            x_label = self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            )

        y_label = self._get_metric_label(
            y_metric.name, y_metric.stat or "avg", available_metrics
        )

        experiment_types = self._extract_experiment_types(data, spec.group_by)
        group_display_names = self._extract_group_display_names(data, spec.group_by)

        return self.plot_generator.create_scatter_line_plot(
            df=data,
            x_metric=x_metric.name,
            y_metric=y_metric.name,
            label_by=spec.label_by,
            group_by=spec.group_by,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
            group_display_names=group_display_names,
        )

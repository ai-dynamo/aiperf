# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-run plot type handlers.

Handlers for creating comparison plots from multiple profiling runs.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.constants import DEFAULT_PERCENTILE
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotSpec
from aiperf.plot.models.uncertainty import BenchmarkPoint


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


def _build_uncertainty_points(
    data: "pd.DataFrame",
    x_col: str,
    y_col: str,
    *,
    group_col: str | list[str] | None,
    label_col: str | None,
    ci_level: float,
) -> list[BenchmarkPoint]:
    """Build a list of BenchmarkPoint from a grouped DataFrame using t-distribution CIs.

    For each group bucket, computes (mean, ci_half) for x and y and constructs
    one BenchmarkPoint. CIs use a Student-t critical value at ci_level with
    df = n - 1; for n < 2, CI half-widths collapse to 0.

    Args:
        data: DataFrame with metric columns.
        x_col: Column name for x-axis metric.
        y_col: Column name for y-axis metric.
        group_col: Column name OR list of candidate column names. If a list is
            given, the first name that exists in data.columns is used; if none
            match (or None), all rows form a single bucket.
        label_col: Optional column whose mode-per-bucket becomes BenchmarkPoint.label.
        ci_level: Confidence level in (0, 1), typically 0.90, 0.95, or 0.99.
    """
    import numpy as np
    from scipy import stats as scipy_stats

    # Normalize list-valued group_col to a single column name
    if isinstance(group_col, list):
        group_col = next((col for col in group_col if col in data.columns), None)

    if group_col and group_col in data.columns:
        groups: list = sorted(data[group_col].dropna().unique())
    else:
        groups = [None]

    points: list[BenchmarkPoint] = []
    for group in groups:
        group_df = (
            data[data[group_col] == group] if group is not None and group_col else data
        )
        group_df = group_df.dropna(subset=[x_col, y_col])
        if group_df.empty:
            continue

        x_vals = group_df[x_col].values.astype(float)
        y_vals = group_df[y_col].values.astype(float)
        n = len(x_vals)
        x_mean = float(np.mean(x_vals))
        y_mean = float(np.mean(y_vals))

        if n >= 2:
            t_crit = scipy_stats.t.ppf((1 + ci_level) / 2, df=n - 1)
            x_ci_half = t_crit * float(np.std(x_vals, ddof=1) / np.sqrt(n))
            y_ci_half = t_crit * float(np.std(y_vals, ddof=1) / np.sqrt(n))
        else:
            x_ci_half = 0.0
            y_ci_half = 0.0

        label_val = None
        if label_col and label_col in group_df.columns:
            label_mode = group_df[label_col].dropna().mode()
            label_val = str(label_mode.iloc[0]) if not label_mode.empty else None

        points.append(
            BenchmarkPoint(
                x_mean=x_mean,
                y_mean=y_mean,
                x_ci_low=x_mean - x_ci_half,
                x_ci_high=x_mean + x_ci_half,
                y_ci_low=y_mean - y_ci_half,
                y_ci_high=y_mean + y_ci_half,
                cov_xy=None,
                label=label_val,
                n_runs=n,
            )
        )

    return points


class LatencyThroughputUncertaintyHandler(BaseMultiRunHandler):
    """Handler for latency-throughput uncertainty plots."""

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame) -> bool:
        """Check if required metric columns exist in the DataFrame."""
        for metric in spec.metrics:
            if metric.name not in data.columns and metric.name != "concurrency":
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: pd.DataFrame, available_metrics: dict
    ) -> go.Figure:
        """Build LatencyThroughputUncertaintyData from DataFrame and delegate to PlotGenerator."""
        from aiperf.plot.models.uncertainty import (
            LatencyThroughputUncertaintyData,
            UncertaintySeries,
        )

        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        ci_level = spec.ci_level
        if ci_level not in {0.90, 0.95, 0.99}:
            ci_level = 0.95

        # Series-level grouping (e.g., model or request_count)
        series_col = spec.group_by
        if isinstance(series_col, list):
            series_col = series_col[0] if series_col else None

        # Point-level grouping (operating points within each series)
        point_col = "concurrency" if "concurrency" in data.columns else None

        # Build series
        series_list: list[UncertaintySeries] = []
        if series_col and series_col in data.columns and series_col != point_col:
            for series_val in sorted(data[series_col].dropna().unique(), key=str):
                series_df = data[data[series_col] == series_val]
                points = _build_uncertainty_points(
                    series_df,
                    x_metric.name,
                    y_metric.name,
                    group_col=point_col,
                    label_col=spec.label_by,
                    ci_level=ci_level,
                )
                if points:
                    series_name = f"{series_col} = {series_val}"
                    series_list.append(
                        UncertaintySeries(
                            name=series_name,
                            points=points,
                        )
                    )
        else:
            # Single series — group by concurrency for operating points
            points = _build_uncertainty_points(
                data,
                x_metric.name,
                y_metric.name,
                group_col=point_col or series_col,
                label_col=spec.label_by,
                ci_level=ci_level,
            )
            if points:
                series_list.append(UncertaintySeries(name="Mean", points=points))

        uncertainty_data = LatencyThroughputUncertaintyData(
            series=series_list,
            confidence_level=ci_level,
            title=spec.title,
            x_label=self._get_metric_label(
                x_metric.name, x_metric.stat or DEFAULT_PERCENTILE, available_metrics
            ),
            y_label=self._get_metric_label(
                y_metric.name, y_metric.stat or "avg", available_metrics
            ),
            group_by=series_col,
        )

        return self.plot_generator.create_uncertainty_plot(uncertainty_data)

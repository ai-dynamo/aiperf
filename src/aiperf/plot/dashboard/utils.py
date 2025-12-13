# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions shared between dashboard builder and callbacks.

This module provides helper functions for creating plot containers,
generating plots, and other operations used by both layout builder
and interactive callbacks.
"""

import logging
import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    PlotTheme,
)
from aiperf.plot.dashboard.styling import get_theme_colors
from aiperf.plot.metric_names import get_metric_display_name

_logger = logging.getLogger(__name__)


def extract_metric_value(run, metric_name: str, stat: str = "p50") -> float | None:
    """
    Extract a metric value from a run with specified stat.

    Args:
        run: RunData object
        metric_name: Name of the metric (e.g., 'time_to_first_token')
        stat: Statistic to extract (e.g., 'p50', 'avg', 'max')

    Returns:
        Metric value or None if not available
    """
    metric = run.get_metric(metric_name)
    if metric is None:
        return None

    if hasattr(metric, "stats"):
        # MetricResult object
        return getattr(metric.stats, stat, None)
    elif isinstance(metric, dict):
        # Dict format
        return metric.get(stat)

    return None


def create_plot_container_component(
    plot_id: str,
    figure,
    theme: PlotTheme,
    resizable: bool = True,
    size: int = 400,
    size_class: str = "half",
    visible: bool = True,
) -> html.Div:
    """
    Create a plot container with settings icon and resize handle.

    Shared between builder (initial render) and callbacks (dynamic updates).

    Args:
        plot_id: Unique ID for the plot
        figure: Plotly figure object
        theme: Plot theme
        resizable: Whether to show resize handle
        size: Minimum height for plot container in pixels
        size_class: Grid size class ("half" for 50%, "full" for 100%)
        visible: Whether the plot should be visible (False = display: none)

    Returns:
        Dash HTML Div containing the plot
    """
    colors = get_theme_colors(theme)

    # Calculate height based on size_class to maintain aspect ratio
    # Half-width (50%): base height, Full-width (100%): 2× height
    calculated_height = size * 2 if size_class == "full" else size

    container_style = {
        "position": "relative",
        "min-height": f"{calculated_height}px",
        "width": "100%",
        "box-sizing": "border-box",
        "overflow": "visible",
        "background": colors["paper"],
        "border-radius": "8px",
        "border": f"1px solid {colors['border']}",
        "display": "block" if visible else "none",
    }

    # Settings button (⚙️) for configuring plot
    settings_button = html.Button(
        "⚙",
        id={"type": "settings-plot-btn", "index": plot_id},
        className="plot-settings-btn",
        title="Edit plot settings",
    )

    # Hide plot button (👁️) for hiding plot from grid
    hide_button = html.Button(
        "👁",
        id={"type": "hide-plot-btn-direct", "index": plot_id},
        className="plot-hide-btn",
        title="Hide plot",
    )

    # Resize handle (click to toggle half/full)
    resize_handle = []
    if resizable:
        resize_handle = [
            html.Button(
                "⇲",
                id={"type": "resize-handle", "index": plot_id},
                className="resize-handle",
                title="Click to toggle size",
                style={
                    "background": "none",
                    "border": "none",
                    "padding": "4px",
                },
            )
        ]

    return html.Div(
        [
            settings_button,
            hide_button,
            *resize_handle,
            dcc.Graph(
                id={"type": "plot-graph", "index": plot_id},
                figure=figure,
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "modeBarButtonsToRemove": [
                        "select2d",
                        "lasso2d",
                        "autoScale2d",
                        "pan2d",
                    ],
                },
                style={"height": "100%", "width": "100%"},
            ),
        ],
        id={"type": "plot-container", "index": plot_id},
        className=f"plot-container size-{size_class}",
        style=container_style,
    )


def get_plot_title(plot_id: str, plot_configs: dict | None = None) -> str:
    """
    Get display title for a plot ID.

    Args:
        plot_id: Plot ID (e.g., 'pareto', 'custom-ttft-vs-latency')
        plot_configs: Dict of all plot configs (default and custom)

    Returns:
        Human-readable plot title
    """
    # Check plot_configs FIRST (works for both default and custom)
    if plot_configs and plot_id in plot_configs:
        config = plot_configs[plot_id]
        return config.get("title", plot_id.replace("-", " ").title())

    # Fallback: format the plot_id
    return plot_id.replace("-", " ").title()


def add_run_idx_to_figure(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Post-process figure to add run_idx to customdata for config viewer.

    This adapter ensures that plots generated by core PlotGenerator can work
    with the config modal click handler. It uses coordinate-based matching to
    handle both single-trace and multi-trace (grouped) plots correctly.

    Args:
        fig: Plotly Figure object from PlotGenerator
        df: DataFrame with run_idx column and metric columns

    Returns:
        Updated Figure with run_idx enriched in customdata
    """
    if "run_idx" not in df.columns or df.empty:
        return fig

    # Identify metric columns (exclude metadata)
    metric_cols = [
        c
        for c in df.columns
        if c not in ["model", "concurrency", "run_idx", "run_name"]
    ]

    if len(metric_cols) < 2:
        return fig

    x_metric, y_metric = metric_cols[0], metric_cols[1]

    for trace in fig.data:
        # Skip traces without x/y data
        if not hasattr(trace, "x") or not hasattr(trace, "y"):
            continue

        # Skip shadow traces (hoverinfo='skip')
        if hasattr(trace, "hoverinfo") and trace.hoverinfo == "skip":
            continue

        # Create customdata from scratch if it doesn't exist
        new_customdata = []

        # Match each trace point to DataFrame row by coordinates
        for i, (x_val, y_val) in enumerate(zip(trace.x, trace.y, strict=True)):
            # Find matching row in DataFrame (with tolerance for float comparison)
            matches = df[
                (abs(df[x_metric] - x_val) < 0.001)
                & (abs(df[y_metric] - y_val) < 0.001)
            ]

            if not matches.empty:
                run_idx = int(matches.iloc[0]["run_idx"])

                # Check if existing customdata needs to be preserved
                if (
                    hasattr(trace, "customdata")
                    and trace.customdata is not None
                    and i < len(trace.customdata)
                ):
                    existing_data = trace.customdata[i]

                    # Enrich existing customdata with run_idx
                    if isinstance(existing_data, str):
                        new_customdata.append(
                            {"text": existing_data, "run_idx": run_idx}
                        )
                    elif isinstance(existing_data, dict):
                        existing_data["run_idx"] = run_idx
                        new_customdata.append(existing_data)
                    else:
                        new_customdata.append({"run_idx": run_idx})
                else:
                    # Create new customdata with just run_idx
                    new_customdata.append({"run_idx": run_idx})
            else:
                # No match found
                if (
                    hasattr(trace, "customdata")
                    and trace.customdata is not None
                    and i < len(trace.customdata)
                ):
                    # Keep original customdata if it exists
                    new_customdata.append(trace.customdata[i])
                else:
                    # Create empty customdata
                    new_customdata.append({})

        if new_customdata:
            trace.customdata = new_customdata

    return fig


def _convert_to_numeric(value: Any, context: str = "") -> float | int | None:
    """
    Convert a value to numeric type (int or float) with error handling.

    Args:
        value: Value to convert (could be str, int, float, None, etc.)
        context: Context string for error logging (e.g., "metric=latency, stat=p50")

    Returns:
        Numeric value (int or float) or None if conversion fails or value is None
    """
    if value is None:
        return None

    if isinstance(value, int | float):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return value
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        if isinstance(value, str) and "." not in value:
            return int(value)
        return float(value)
    except (ValueError, TypeError):
        _logger.warning(f"Could not convert value to numeric: {value!r} ({context})")
        return None


def runs_to_dataframe(
    runs: list, x_metric: str, x_stat: str, y_metric: str, y_stat: str
) -> dict:
    """
    Convert list of RunData to DataFrame for plotting.

    Args:
        runs: List of RunData objects
        x_metric: Name of x-axis metric
        x_stat: Requested statistic for x-axis (p50, p90, avg, etc.)
        y_metric: Name of y-axis metric
        y_stat: Requested statistic for y-axis

    Returns:
        Dict with keys:
            - df: DataFrame with columns: x_metric, y_metric, model, concurrency, run_idx, run_name
            - x_stat_actual: Stat used for x-axis (same as x_stat)
            - y_stat_actual: Stat used for y-axis (same as y_stat)
            - warnings: List of warning messages
    """
    data = []
    excluded_runs = []
    warnings = []

    for idx, run in enumerate(runs):
        x_val = extract_metric_value(run, x_metric, x_stat)
        y_val = extract_metric_value(run, y_metric, y_stat)

        if x_val is not None and y_val is not None:
            data.append(
                {
                    x_metric: _convert_to_numeric(
                        x_val, f"x_metric={x_metric}, stat={x_stat}"
                    ),
                    y_metric: _convert_to_numeric(
                        y_val, f"y_metric={y_metric}, stat={y_stat}"
                    ),
                    "model": run.metadata.model or "Unknown",
                    "concurrency": _convert_to_numeric(
                        run.metadata.concurrency, "concurrency"
                    )
                    or 0,
                    "run_idx": idx,
                    "run_name": run.metadata.run_name,
                    "experiment_type": run.metadata.experiment_type,
                    "experiment_group": run.metadata.experiment_group,
                }
            )
        else:
            excluded_runs.append(f"{run.metadata.run_name or f'run_{idx}'}")

    if excluded_runs:
        warnings.append(
            f"{len(excluded_runs)} of {len(runs)} runs excluded (missing metrics)"
        )
        excluded_preview = ", ".join(excluded_runs[:5])
        if len(excluded_runs) > 5:
            excluded_preview += " ..."
        _logger.warning(
            f"{len(excluded_runs)} of {len(runs)} runs excluded due to missing metrics. "
            f"Excluded: {excluded_preview}. Plotting with {len(data)} runs."
        )

    x_stat_actual = x_stat
    y_stat_actual = y_stat

    unique_warnings = list(dict.fromkeys(warnings))

    df = pd.DataFrame(data)

    if not df.empty:
        df[x_metric] = pd.to_numeric(df[x_metric], errors="coerce")
        df[y_metric] = pd.to_numeric(df[y_metric], errors="coerce")
        df["concurrency"] = pd.to_numeric(df["concurrency"], errors="coerce")
        df["run_idx"] = df["run_idx"].astype(int)

        x_nan_count = df[x_metric].isna().sum()
        y_nan_count = df[y_metric].isna().sum()
        if x_nan_count > 0 or y_nan_count > 0:
            unique_warnings.append(
                f"Type conversion: {x_nan_count} invalid {x_metric} values, "
                f"{y_nan_count} invalid {y_metric} values (converted to NaN)"
            )

    return {
        "df": df,
        "x_stat_actual": x_stat_actual,
        "y_stat_actual": y_stat_actual,
        "warnings": unique_warnings,
    }


def get_available_stats_for_metric(runs: list, metric_name: str) -> list[str]:
    """
    Get list of available stats for a given metric across all runs.

    Args:
        runs: List of RunData objects
        metric_name: Name of the metric to check

    Returns:
        List of available stat keys (e.g., ["avg", "p50", "p90"])
    """
    if not runs:
        return ALL_STAT_KEYS

    # Special case: concurrency is metadata, not a metric
    if metric_name == "concurrency":
        return ["value"]

    # Sample first run to get available stats
    first_run = runs[0]

    # Handle derived metrics
    if metric_name == "output_token_throughput_per_user":
        # Based on output_token_throughput
        metric = first_run.get_metric("output_token_throughput")
    elif metric_name == "output_token_throughput_per_gpu":
        # Try direct metric first, fallback to base throughput
        metric = first_run.get_metric("output_token_throughput_per_gpu")
        if metric is None:
            metric = first_run.get_metric("output_token_throughput")
    else:
        # Standard metric
        metric = first_run.get_metric(metric_name)

    if metric is None:
        return ALL_STAT_KEYS

    # Extract available stats
    if isinstance(metric, dict):
        return [k for k in metric if k != "unit"]
    else:
        # MetricResult object - check which stat attributes exist and are not None
        return [
            stat for stat in ALL_STAT_KEYS if getattr(metric, stat, None) is not None
        ]


def prepare_timeseries_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Prepare DataFrame for time-series plotting by ensuring x-axis column exists.

    Handles the case where request_number column is missing and DataFrame
    uses default RangeIndex. Converts index to a proper column to avoid
    KeyError when plotting functions access df[x_col].

    Args:
        df: DataFrame with per-request data

    Returns:
        Tuple of (prepared_df, x_column_name) where:
        - prepared_df: DataFrame with guaranteed x-axis column
        - x_column_name: Name of the x-axis column to use
    """
    if "request_number" in df.columns:
        return df, "request_number"

    # Check if index has a meaningful name
    if df.index.name and df.index.name != "index":
        # Reset index to make it a column
        df_copy = df.reset_index()
        return df_copy, df.index.name

    # Default RangeIndex case: convert index to request_number column
    df_copy = df.copy()
    df_copy["request_number"] = df_copy.index
    return df_copy, "request_number"


# Stats that can be suffixed to metrics (e.g., inter_chunk_latency_avg)
SINGLE_RUN_STAT_SUFFIXES = ["avg", "p50", "p95", "std", "min", "max", "range"]


def get_single_run_metrics_with_stats(
    columns: list[str], excluded_columns: list[str]
) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Process DataFrame columns to extract base metrics and their available stats.

    Groups compound metrics (e.g., inter_chunk_latency_avg, inter_chunk_latency_p50)
    into a single base metric with available stat options.

    Args:
        columns: List of column names from DataFrame
        excluded_columns: List of column names to exclude

    Returns:
        Tuple of:
        - List of metric options for dropdown (base metrics only)
        - Dict mapping base metric name to list of available stats
    """
    metric_stats: dict[str, list[str]] = {}
    simple_metrics: list[str] = []

    for col in columns:
        if col in excluded_columns:
            continue

        # Check if this column has a stat suffix
        found_stat = None
        base_metric = None
        for stat in SINGLE_RUN_STAT_SUFFIXES:
            suffix = f"_{stat}"
            if col.endswith(suffix):
                base_metric = col[: -len(suffix)]
                found_stat = stat
                break

        if base_metric and found_stat:
            # This is a compound metric with stat suffix
            if base_metric not in metric_stats:
                metric_stats[base_metric] = []
            metric_stats[base_metric].append(found_stat)
        else:
            # Simple metric without stat suffix
            simple_metrics.append(col)

    # Build dropdown options
    options: list[dict] = []

    # Add simple metrics first
    for metric in simple_metrics:
        options.append({"label": get_metric_display_name(metric), "value": metric})

    # Add compound metrics (base names only)
    for base_metric in sorted(metric_stats.keys()):
        options.append(
            {"label": get_metric_display_name(base_metric), "value": base_metric}
        )

    # For simple metrics, set their stats to just "avg"
    all_metric_stats = {metric: ["avg"] for metric in simple_metrics}
    all_metric_stats.update(metric_stats)

    return options, all_metric_stats


def get_stat_options_for_single_run_metric(
    metric_name: str, metric_stats: dict[str, list[str]]
) -> list[dict]:
    """
    Get stat dropdown options for a specific metric.

    Args:
        metric_name: Base metric name
        metric_stats: Dict mapping metric names to available stats

    Returns:
        List of dropdown options for stats
    """
    stats = metric_stats.get(metric_name, ["avg"])

    # Define label mapping
    stat_labels = {
        "avg": "Average",
        "p50": "p50 (Median)",
        "p95": "p95",
        "std": "Std Dev",
        "min": "Minimum",
        "max": "Maximum",
        "range": "Range",
    }

    # Order stats consistently
    ordered_stats = ["avg", "p50", "p95", "std", "min", "max", "range"]
    options = []
    for stat in ordered_stats:
        if stat in stats:
            options.append({"label": stat_labels.get(stat, stat), "value": stat})

    return options if options else [{"label": "Average", "value": "avg"}]


def resolve_single_run_column_name(
    metric_name: str, stat: str | None, metric_stats: dict[str, list[str]]
) -> str:
    """
    Resolve the actual DataFrame column name from metric + stat.

    For compound metrics like inter_chunk_latency, combines metric_stat.
    For simple metrics, returns the metric name directly.

    Args:
        metric_name: Base metric name
        stat: Selected stat (may be None or "avg" for simple metrics)
        metric_stats: Dict mapping metric names to available stats

    Returns:
        Actual column name to use for DataFrame access
    """
    stats = metric_stats.get(metric_name, ["avg"])

    # If metric has only "avg" and it's a simple metric, return metric_name
    if stats == ["avg"]:
        return metric_name

    # For compound metrics, combine metric + stat
    if stat and stat in stats:
        return f"{metric_name}_{stat}"

    # Fallback: return first available stat
    return f"{metric_name}_{stats[0]}" if stats else metric_name

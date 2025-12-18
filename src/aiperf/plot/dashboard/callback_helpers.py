# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for dashboard callbacks.

This module consolidates shared logic used across callback functions,
reducing duplication and making callbacks more generic per plot type.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.plot.constants import ALL_STAT_KEYS, STAT_LABELS
from aiperf.plot.dashboard.utils import (
    get_single_run_metrics_with_stats,
    resolve_single_run_column_name,
)
from aiperf.plot.metric_names import (
    get_gpu_metrics,
    get_metric_display_name,
    get_metric_display_name_with_unit,
)

if TYPE_CHECKING:
    from aiperf.plot.core.data_loader import RunData

MULTI_RUN_STAT_OPTIONS = [
    {"label": STAT_LABELS.get(stat, stat), "value": stat} for stat in ALL_STAT_KEYS
]


def filter_stat_options_by_available(
    all_options: list[dict],
    available_stats: list[str],
) -> list[dict]:
    """
    Filter stat options to only include those available for a metric.

    Args:
        all_options: Full list of stat options with label/value dicts.
        available_stats: List of stat keys that are available.

    Returns:
        Filtered list of options containing only available stats.
    """
    if not available_stats:
        return all_options
    return [opt for opt in all_options if opt["value"] in available_stats]


def select_best_stat(
    available_stats: list[str],
    current_stat: str | None,
    preference_order: list[str] | None = None,
) -> str | None:
    """
    Select the best stat from available options with smart fallback.

    Args:
        available_stats: List of available stat keys.
        current_stat: Currently selected stat (may be None).
        preference_order: Preferred stats in order. Defaults to common stats.

    Returns:
        Best stat to use, or None if no stats available.
    """
    if not available_stats:
        return None

    if len(available_stats) == 1:
        return available_stats[0]

    if current_stat and current_stat in available_stats:
        return current_stat

    if preference_order is None:
        preference_order = ["p50", "avg", "p90", "p95", "p99"]

    for pref in preference_order:
        if pref in available_stats:
            return pref

    return available_stats[0]


@dataclass
class SingleRunFieldConfig:
    """Configuration for single-run modal field visibility and options."""

    x_axis_visible: bool
    stat_visible: bool
    y2_visible: bool
    y2_label_visible: bool
    x_axis_options: list[dict]
    x_axis_default: str | None


def get_single_run_field_config(
    plot_type: str,
    slice_duration: float | None = None,
) -> SingleRunFieldConfig:
    """
    Get field visibility and options based on plot type.

    Args:
        plot_type: The selected plot type (scatter, area, timeslice, etc.).
        slice_duration: Optional slice duration for timeslice label.

    Returns:
        Configuration specifying which fields to show and their options.
    """
    if plot_type in ("scatter", "area"):
        return SingleRunFieldConfig(
            x_axis_visible=True,
            stat_visible=False,
            y2_visible=False,
            y2_label_visible=False,
            x_axis_options=[
                {"label": "Request Number", "value": "request_number"},
                {"label": "Timestamp (s)", "value": "timestamp_s"},
            ],
            x_axis_default=None,
        )

    if plot_type == "timeslice":
        label = f"Timeslice ({slice_duration}s)" if slice_duration else "Timeslice"
        return SingleRunFieldConfig(
            x_axis_visible=True,
            stat_visible=False,
            y2_visible=False,
            y2_label_visible=False,
            x_axis_options=[{"label": label, "value": "Timeslice"}],
            x_axis_default="Timeslice",
        )

    if plot_type == "dual_axis":
        return SingleRunFieldConfig(
            x_axis_visible=True,
            stat_visible=False,
            y2_visible=True,
            y2_label_visible=True,
            x_axis_options=[{"label": "Timestamp (s)", "value": "timestamp_s"}],
            x_axis_default="timestamp_s",
        )

    if plot_type == "request_timeline":
        return SingleRunFieldConfig(
            x_axis_visible=True,
            stat_visible=False,
            y2_visible=False,
            y2_label_visible=False,
            x_axis_options=[{"label": "Timestamp (s)", "value": "timestamp_s"}],
            x_axis_default="timestamp_s",
        )

    # Default fallback
    return SingleRunFieldConfig(
        x_axis_visible=True,
        stat_visible=False,
        y2_visible=False,
        y2_label_visible=False,
        x_axis_options=[{"label": "Request Number", "value": "request_number"}],
        x_axis_default=None,
    )


def field_config_to_outputs(config: SingleRunFieldConfig) -> tuple:
    """
    Convert field config to callback output tuple for create modal (6 outputs).

    Args:
        config: The field configuration.

    Returns:
        Tuple of (x_axis_style, stat_style, y2_style, y2_label_style, x_options, x_default).
    """
    return (
        {"display": "block" if config.x_axis_visible else "none"},
        {"display": "block" if config.stat_visible else "none"},
        {"display": "block" if config.y2_visible else "none"},
        {"display": "block" if config.y2_label_visible else "none"},
        config.x_axis_options,
        config.x_axis_default,
    )


def field_config_to_edit_outputs(
    config: SingleRunFieldConfig,
    current_x_axis: str | None = None,
) -> tuple:
    """
    Convert field config to callback output tuple for edit modal (5 outputs).

    The edit modal has no y2_label field and preserves current x_axis for scatter/area.

    Args:
        config: The field configuration.
        current_x_axis: Current x-axis value to preserve if valid.

    Returns:
        Tuple of (x_axis_style, stat_style, y2_style, x_options, x_value).
    """
    x_value = config.x_axis_default
    if current_x_axis is not None:
        valid_values = [opt["value"] for opt in config.x_axis_options]
        if current_x_axis in valid_values:
            x_value = current_x_axis

    return (
        {"display": "block" if config.x_axis_visible else "none"},
        {"display": "block" if config.stat_visible else "none"},
        {"display": "block" if config.y2_visible else "none"},
        config.x_axis_options,
        x_value,
    )


def get_single_run_y_metric_options(
    run: "RunData",
    plot_type: str,
    excluded_columns: list[str],
    fallback_metrics: list[dict] | None = None,
) -> list[dict]:
    """
    Get Y-metric dropdown options based on plot type.

    For timeslice plots, returns metrics from timeslice data.
    For other plots, returns metrics from request data plus GPU metrics.

    Args:
        run: The RunData containing requests, timeslices, and gpu_telemetry.
        plot_type: The selected plot type.
        excluded_columns: Columns to exclude from request metrics.
        fallback_metrics: Optional fallback metrics if run data unavailable.

    Returns:
        List of metric options with label/value dicts.
    """
    if plot_type == "timeslice":
        if run.timeslices is not None and not run.timeslices.empty:
            timeslice_metrics = run.timeslices["Metric"].unique().tolist()
        else:
            timeslice_metrics = []
        return [{"label": m, "value": m} for m in timeslice_metrics if m != "Timeslice"]

    # For non-timeslice plots, use request metrics + GPU metrics
    if run.requests is not None and not run.requests.empty:
        options, _ = get_single_run_metrics_with_stats(
            list(run.requests.columns), excluded_columns
        )
        # Add GPU metrics if available
        if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
            plottable_gpu_metrics = set(get_gpu_metrics())
            gpu_metrics = [
                {"label": get_metric_display_name_with_unit(col), "value": col}
                for col in run.gpu_telemetry.columns
                if col in plottable_gpu_metrics
            ]
            if gpu_metrics:
                options.append(
                    {
                        "label": "── GPU Metrics ──",
                        "value": "_gpu_divider",
                        "disabled": True,
                    }
                )
                options.extend(gpu_metrics)
        return options

    return fallback_metrics or []


def select_metric_value(
    options: list[dict],
    current_value: str | None,
) -> str | None:
    """
    Select a metric value, preserving current if valid.

    Args:
        options: List of metric options with value keys.
        current_value: Current selected value to preserve if valid.

    Returns:
        The current value if valid, first option if not, or None if empty.
    """
    if not options:
        return None
    if current_value and any(o["value"] == current_value for o in options):
        return current_value
    return options[0]["value"]


SINGLE_RUN_TITLE_SUFFIXES = {
    "scatter": "Across Requests",
    "area": "Across Time",
    "request_timeline": "Across Time",
    "timeslice": "Across Time Slices",
    "dual_axis": "Over Time",
}


def build_single_run_plot_config(
    plot_type: str,
    x_axis: str,
    y_metric: str,
    y_stat: str | None,
    metric_stats: dict[str, list[str]],
    y2_metric: str | None = None,
    y2_label: str | None = None,
    custom_title: str | None = None,
    custom_x_label: str | None = None,
    custom_y_label: str | None = None,
    size: str = "half",
    is_default: bool = False,
) -> dict:
    """
    Build a single-run plot configuration dictionary.

    Args:
        plot_type: Type of plot (scatter, area, timeslice, dual_axis, etc.).
        x_axis: X-axis column name.
        y_metric: Y-axis metric (base name or display name for timeslice).
        y_stat: Selected statistic for the metric.
        metric_stats: Dict mapping metric base names to available stats.
        y2_metric: Secondary Y-axis metric (for dual_axis).
        y2_label: Custom label for secondary Y-axis.
        custom_title: Optional custom plot title.
        custom_x_label: Optional custom X-axis label.
        custom_y_label: Optional custom Y-axis label.
        size: Plot size ("half" or "full").
        is_default: Whether this is a default plot.

    Returns:
        Plot configuration dictionary.
    """
    # Resolve actual column name and generate title
    if plot_type == "timeslice":
        actual_column = y_metric
        default_title = f"{y_metric} {SINGLE_RUN_TITLE_SUFFIXES['timeslice']}"
    elif plot_type == "dual_axis":
        actual_column = resolve_single_run_column_name(y_metric, y_stat, metric_stats)
        y_display = get_metric_display_name(y_metric)
        y2_display = get_metric_display_name(y2_metric) if y2_metric else ""
        default_title = (
            f"{y_display} and {y2_display} {SINGLE_RUN_TITLE_SUFFIXES['dual_axis']}"
        )
    else:
        actual_column = resolve_single_run_column_name(y_metric, y_stat, metric_stats)
        suffix = SINGLE_RUN_TITLE_SUFFIXES.get(plot_type, f"({plot_type})")
        default_title = f"{get_metric_display_name(y_metric)} {suffix}"

    # Use custom title if provided
    title = (
        custom_title.strip() if custom_title and custom_title.strip() else default_title
    )

    # Build base config
    config = {
        "plot_type": plot_type,
        "x_axis": x_axis,
        "y_metric": actual_column,
        "y_metric_base": y_metric,
        "y_stat": y_stat,
        "is_default": is_default,
        "size": size,
        "title": title,
        "x_label": custom_x_label.strip() if custom_x_label else "",
        "y_label": custom_y_label.strip() if custom_y_label else "",
        "mode": "single_run",
    }

    # Add type-specific fields
    if plot_type == "timeslice":
        config["stat"] = y_stat or "avg"
        config["source"] = "timeslices"

    if plot_type == "dual_axis" and y2_metric:
        config["y2_metric"] = y2_metric
        config["y2_label"] = y2_label.strip() if y2_label else ""
        config["source"] = "dual"

    return config

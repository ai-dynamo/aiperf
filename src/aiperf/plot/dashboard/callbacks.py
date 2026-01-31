# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dash callbacks for interactive dashboard features.

This module contains all callback functions that handle user interactions
and update the dashboard dynamically.
"""

import io
import json
import logging
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from ruamel.yaml import YAML

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    CUMULATIVE_METRIC_PATTERNS,
    NVIDIA_DARK,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_WHITE,
    PLOT_FONT_FAMILY,
    STAT_LABELS,
    PlotTheme,
)
from aiperf.plot.core.data_loader import DataLoader, RunData
from aiperf.plot.core.data_preparation import (
    prepare_timeslice_metrics,
)
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotSpec, PlotType
from aiperf.plot.dashboard.builder import EXCLUDED_METRIC_COLUMNS, DashboardBuilder
from aiperf.plot.dashboard.cache import (
    CacheKey,
    compute_config_hash,
    compute_runs_hash,
    get_plot_cache,
)
from aiperf.plot.dashboard.callback_helpers import (
    MULTI_RUN_STAT_OPTIONS,
    build_single_run_plot_config,
    field_config_to_edit_outputs,
    field_config_to_outputs,
    get_single_run_field_config,
    get_single_run_y_metric_options,
    select_metric_value,
)
from aiperf.plot.dashboard.styling import (
    get_button_style,
    get_header_style,
    get_label_style,
    get_main_area_style,
    get_sidebar_style,
    get_theme_colors,
)
from aiperf.plot.dashboard.utils import (
    add_run_idx_to_figure,
    create_plot_container_component,
    get_available_stats_for_metric,
    get_plot_title,
    get_single_run_metrics_with_stats,
    get_stat_options_for_single_run_metric,
    prepare_timeseries_dataframe,
    resolve_single_run_column_name,
    runs_to_dataframe,
)
from aiperf.plot.metric_names import (
    get_all_metric_display_names,
    get_gpu_metrics,
    get_metric_display_name,
    get_metric_display_name_with_unit,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

_logger = logging.getLogger(__name__)

# Global cache for PlotGenerator instances (one per theme)
_PLOT_GENERATOR_CACHE: dict[PlotTheme, PlotGenerator] = {}

# Module-level caches for drill-down single-run data
# These are populated by handle_url_routing and used by update_grid_children
_drill_down_run_cache: dict[str, RunData] = {}
_drill_down_specs_cache: dict[str, list[PlotSpec]] = {}
_drill_down_plot_config_cache: dict[str, PlotConfig] = {}


def _get_plot_generator(theme: PlotTheme) -> PlotGenerator:
    """
    Get or create a PlotGenerator for the specified theme.

    Uses a global cache to avoid recreating PlotGenerator instances.

    Args:
        theme: Plot theme

    Returns:
        PlotGenerator instance for the theme
    """
    if theme not in _PLOT_GENERATOR_CACHE:
        _PLOT_GENERATOR_CACHE[theme] = PlotGenerator(theme=theme)
    return _PLOT_GENERATOR_CACHE[theme]


def _get_current_theme(theme_data: dict, default_theme: PlotTheme) -> PlotTheme:
    """
    Extract current theme from theme store data.

    Args:
        theme_data: Theme store data dict
        default_theme: Default theme to use if not found in data

    Returns:
        Current PlotTheme
    """
    theme_str = theme_data.get("theme", default_theme.value)
    return PlotTheme(theme_str)


def _generate_multirun_figure(
    filtered_runs: list[RunData],
    plot_config_dict: dict,
    theme: PlotTheme,
) -> tuple[go.Figure | None, list[str]]:
    """
    Generate a multi-run plot figure for the specified theme.

    This helper function encapsulates the plot generation logic so it can be
    called for both current and opposite themes during caching.

    Args:
        filtered_runs: List of RunData objects (already filtered by selection)
        plot_config_dict: Plot configuration dictionary
        theme: Theme to use for plot generation

    Returns:
        Tuple of (figure or None, list of warnings)
    """
    plot_gen = _get_plot_generator(theme)

    x_metric = plot_config_dict.get("x_metric")
    x_stat = plot_config_dict.get("x_stat", "p50")
    y_metric = plot_config_dict.get("y_metric")
    y_stat = plot_config_dict.get("y_stat", "avg")
    log_scale = plot_config_dict.get("log_scale", "none")
    autoscale = plot_config_dict.get("autoscale", "none")
    plot_type = plot_config_dict.get("plot_type", "scatter_line")
    label_by = plot_config_dict.get("label_by", "concurrency")
    group_by = plot_config_dict.get("group_by", "model")
    title = plot_config_dict.get(
        "title", plot_config_dict.get("plot_id", "Plot").replace("-", " ").title()
    )

    result = runs_to_dataframe(filtered_runs, x_metric, x_stat, y_metric, y_stat)
    df = result["df"]
    x_stat_actual = result["x_stat_actual"]
    y_stat_actual = result["y_stat_actual"]
    plot_warnings = result["warnings"]

    if df.empty:
        return None, plot_warnings

    actual_label_by = None if label_by == "none" else label_by
    actual_group_by = None if group_by == "none" else group_by

    if actual_label_by and actual_label_by not in df.columns:
        actual_label_by = "concurrency" if "concurrency" in df.columns else None

    if actual_group_by and actual_group_by not in df.columns:
        actual_group_by = "model" if "model" in df.columns else None

    experiment_types = None
    if (
        "experiment_type" in df.columns
        and actual_group_by
        and actual_group_by in df.columns
    ):
        experiment_types = {
            g: df[df[actual_group_by] == g]["experiment_type"].iloc[0]
            for g in df[actual_group_by].unique()
        }

    x_label = (
        plot_config_dict.get("x_label")
        or f"{get_metric_display_name(x_metric)} ({x_stat_actual})"
    )
    y_label = (
        plot_config_dict.get("y_label")
        or f"{get_metric_display_name(y_metric)} ({y_stat_actual})"
    )

    fig = None
    if plot_type == "pareto":
        fig = plot_gen.create_pareto_plot(
            df,
            x_metric,
            y_metric,
            label_by=actual_label_by,
            group_by=actual_group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
        )
    elif plot_type == "scatter_line":
        fig = plot_gen.create_scatter_line_plot(
            df,
            x_metric,
            y_metric,
            label_by=actual_label_by,
            group_by=actual_group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
        )
    elif plot_type == "scatter":
        fig = plot_gen.create_scatter_line_plot(
            df,
            x_metric,
            y_metric,
            label_by=actual_label_by,
            group_by=actual_group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            mode="markers",
            experiment_types=experiment_types,
        )
    elif plot_type == "bar":
        fig = plot_gen.create_multi_run_bar_chart(
            df=df,
            x_metric=x_metric,
            y_metric=y_metric,
            group_by=actual_group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
    else:
        fig = plot_gen.create_scatter_line_plot(
            df,
            x_metric,
            y_metric,
            label_by=actual_label_by,
            group_by=actual_group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
        )

    if fig:
        fig = add_run_idx_to_figure(fig, df)

        if log_scale in ("x", "both"):
            fig.update_xaxes(type="log")
        if log_scale in ("y", "both"):
            fig.update_yaxes(type="log")

        # Apply autoscale settings (default is "none" which starts both axes at 0)
        x_rangemode = "normal" if autoscale in ("x", "both") else "tozero"
        y_rangemode = "normal" if autoscale in ("y", "both") else "tozero"
        fig.update_xaxes(rangemode=x_rangemode)
        fig.update_yaxes(rangemode=y_rangemode)

    return fig, plot_warnings


def generate_plot_from_spec(
    spec: PlotSpec,
    runs: list[RunData],
    stat_overrides: dict[str, str],
    theme: PlotTheme,
) -> tuple[go.Figure, pd.DataFrame] | None:
    """
    Generate plot dynamically from PlotSpec configuration.

    Args:
        spec: PlotSpec object from config
        runs: List of RunData objects
        stat_overrides: Dict mapping metric names to selected stats
        theme: Plot theme

    Returns:
        Tuple of (figure, dataframe) or None if plot cannot be generated
    """
    plot_gen = _get_plot_generator(theme)

    # Extract x and y metrics from spec
    x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
    y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

    if not x_metric_spec or not y_metric_spec:
        return None

    # Use stat overrides if available, otherwise use spec defaults
    x_stat = stat_overrides.get(x_metric_spec.name, x_metric_spec.stat or "p50")
    y_stat = stat_overrides.get(y_metric_spec.name, y_metric_spec.stat or "avg")

    # Create DataFrame
    result = runs_to_dataframe(
        runs, x_metric_spec.name, x_stat, y_metric_spec.name, y_stat
    )
    df = result["df"]

    if df.empty:
        return None

    # Generate plot based on type
    title = (
        spec.title
        or f"{y_metric_spec.name.replace('_', ' ').title()} vs {x_metric_spec.name.replace('_', ' ').title()}"
    )
    label_by = spec.label_by or "concurrency"
    group_by = spec.group_by or "model"

    # Use custom axis labels if provided, otherwise auto-generate
    x_label = (
        spec.x_label or f"{get_metric_display_name(x_metric_spec.name)} ({x_stat})"
    )
    y_label = (
        spec.y_label or f"{get_metric_display_name(y_metric_spec.name)} ({y_stat})"
    )

    # Extract experiment_types mapping for color assignment
    experiment_types = None
    if "experiment_type" in df.columns and group_by:
        group_col = group_by[0] if isinstance(group_by, list) else group_by
        if group_col in df.columns:
            experiment_types = {
                g: df[df[group_col] == g]["experiment_type"].iloc[0]
                for g in df[group_col].unique()
            }

    if spec.plot_type == PlotType.PARETO:
        fig = plot_gen.create_pareto_plot(
            df,
            x_metric_spec.name,
            y_metric_spec.name,
            label_by=label_by,
            group_by=group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
        )
    elif spec.plot_type == PlotType.SCATTER_LINE:
        fig = plot_gen.create_scatter_line_plot(
            df,
            x_metric_spec.name,
            y_metric_spec.name,
            label_by=label_by,
            group_by=group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            experiment_types=experiment_types,
        )
    elif spec.plot_type == PlotType.SCATTER:
        fig = plot_gen.create_scatter_line_plot(
            df,
            x_metric_spec.name,
            y_metric_spec.name,
            label_by=label_by,
            group_by=group_by,
            title=title,
            x_label=x_label,
            y_label=y_label,
            mode="markers",
            experiment_types=experiment_types,
        )
    else:
        return None

    return fig, df


def _safe_register_callback(callback_name: str, registration_func):
    """
    Safely register a callback with error handling.

    Wraps callback registration to catch and log errors without stopping
    other callbacks from registering.

    Args:
        callback_name: Display name for logging
        registration_func: Function to call for registration

    Returns:
        True if successful, False otherwise
    """
    try:
        registration_func()
        return True
    except Exception as e:
        _logger.error(f"Failed to register {callback_name}: {e!r}")
        return False


def _register_single_run_only_callbacks(
    app: dash.Dash,
    runs: list[RunData],
    theme: PlotTheme,
) -> None:
    """Register callbacks exclusive to single-run mode."""
    register_single_run_callbacks(app, runs, theme)
    register_single_run_custom_plot_callbacks(app, runs, VisualizationMode.SINGLE_RUN)
    register_single_run_plot_edit_callbacks(app, runs)


def _register_multi_run_only_callbacks(
    app: dash.Dash,
    runs: list[RunData],
    run_dirs: list[Path],
    loader: DataLoader,
    plot_config: PlotConfig,
) -> None:
    """Register callbacks exclusive to multi-run mode."""
    register_custom_plot_callbacks(app, runs, plot_config)
    register_multi_run_plot_edit_callbacks(app, runs)
    register_run_count_badge_callback(app)
    register_drill_down_callbacks(app, runs, run_dirs, loader, plot_config)

    # Also register single-run callbacks for drill-down support
    register_single_run_custom_plot_callbacks(app, runs, VisualizationMode.SINGLE_RUN)
    register_single_run_plot_edit_callbacks(app, runs)


def register_all_callbacks(
    app: dash.Dash,
    runs: list[RunData],
    run_dirs: list[Path],
    mode: VisualizationMode,
    theme: PlotTheme,
    plot_config: PlotConfig,
    loader: DataLoader,
):
    """
    Register all dashboard callbacks.

    Args:
        app: Dash application instance
        runs: List of RunData objects
        run_dirs: List of run directory paths for lazy loading
        mode: Visualization mode
        theme: Plot theme
        plot_config: PlotConfig instance
        loader: DataLoader instance for lazy loading per-request data
    """
    # Register common callbacks (mode-agnostic)
    register_theme_callback(app)
    register_version_check_callback(app, plot_config, mode)
    register_size_migration_callback(app)
    register_layout_theme_callbacks(app)
    register_modal_theme_callbacks(app, mode)
    register_sidebar_widgets_theme_callback(app)
    register_sidebar_components_theme_callback(app)
    register_config_modal_callback(app, runs, plot_config, mode)
    register_hide_show_plot_callbacks(app)
    register_export_png_callback(app, runs, mode, theme, plot_config)
    register_layout_control_callbacks(app, mode, plot_config, theme)
    register_context_menu_callbacks(app)
    register_toast_notifications_callbacks(app)
    register_nested_run_selector_callbacks(app, runs)
    register_resize_toggle_callbacks(app)
    register_sidebar_sync_callback(app, theme)
    register_sidebar_toggle_callback(app, theme)
    register_collapsible_sections_callback(app)

    # Register mode-specific callbacks
    if mode == VisualizationMode.SINGLE_RUN:
        _register_single_run_only_callbacks(app, runs, theme)
    else:
        _register_multi_run_only_callbacks(app, runs, run_dirs, loader, plot_config)

    # Register dynamic grid callback (unified but mode-aware)
    register_dynamic_grid_callback(app, runs, mode, theme, plot_config)


def register_single_run_callbacks(
    app: dash.Dash, runs: list[RunData], theme: PlotTheme
):
    """
    Register callbacks for single-run analysis mode.

    Note: Individual plot callbacks (TTFT, ITL, latency, throughput) have been
    removed as they are now handled by the unified caching architecture:
    - Initial rendering: _render_single_run_plots() in grid callback
    - Theme switching: update_figures_on_theme_change() callback
    """
    # No individual plot callbacks needed - handled by caching architecture
    pass


def register_theme_callback(app: dash.Dash):
    """Register callback for theme toggle."""

    @app.callback(
        Output("theme-store", "data"),
        [Input("theme-toggle", "value")],
        [State("theme-store", "data")],
    )
    def toggle_theme(is_dark, current_theme_data):
        """Toggle between light and dark theme."""
        new_theme = "dark" if is_dark else "light"
        return {"theme": new_theme}

    @app.callback(
        Output("theme-toggle", "label"),
        [Input("theme-store", "data")],
    )
    def update_toggle_label(theme_data):
        """Update toggle button label based on current theme."""
        current_theme = theme_data.get("theme", "light")
        return "Light" if current_theme == "dark" else "Dark"


def register_version_check_callback(
    app: dash.Dash, plot_config: PlotConfig, mode: VisualizationMode
):
    """
    Check localStorage version and auto-migrate if config changed.

    This callback runs on first load to detect if the config.yaml has changed
    since the last session. If changed, it merges localStorage custom plots
    with new config defaults.

    Args:
        app: Dash application instance
        plot_config: PlotConfig instance with YAML config
        mode: Visualization mode (MULTI_RUN or SINGLE_RUN)
    """

    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input("config-version-store", "data"),
        State("plot-state-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def check_config_version(current_version, plot_state):
        """Reset state if config version changed."""
        if not plot_state:
            raise PreventUpdate

        stored_version = plot_state.get("config_version")

        # Version matches - keep localStorage as-is (preserves hidden plots)
        if stored_version == current_version:
            raise PreventUpdate

        # Version mismatch - reset to pure config defaults
        _logger.debug(f"Config version mismatch: {stored_version} -> {current_version}")

        # Get current defaults from config
        if mode == VisualizationMode.MULTI_RUN:
            specs = plot_config.get_multi_run_plot_specs()
        else:
            specs = plot_config.get_single_run_plot_specs()

        default_plot_ids = [spec.name for spec in specs]

        # Rebuild plot_configs with ONLY defaults (no custom plots)
        new_configs = {}
        if mode == VisualizationMode.MULTI_RUN:
            for spec in specs:
                # Extract x and y metric specs
                x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
                y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

                if x_metric_spec and y_metric_spec:
                    plot_type_val = (
                        spec.plot_type.value if spec.plot_type else "scatter_line"
                    )
                    new_configs[spec.name] = {
                        "x_metric": x_metric_spec.name,
                        "x_stat": x_metric_spec.stat or "p50",
                        "y_metric": y_metric_spec.name,
                        "y_stat": y_metric_spec.stat or "avg",
                        "log_scale": "none",
                        "is_default": True,
                        "plot_type": plot_type_val,
                        "size": "half",  # All plots reset to half-width on config change
                        "label_by": spec.label_by or "concurrency",
                        "group_by": spec.group_by or "model",
                        "title": spec.title or spec.name.replace("-", " ").title(),
                    }
        else:
            # Single-run mode: full config for edit modal support
            plot_type_map = {
                "timeslice": "timeslice",
                "scatter": "scatter",
                "area": "area",
                "dual_axis": "dual_axis",
                "scatter_with_percentiles": "scatter",
                "request_timeline": "scatter",
            }
            for spec in specs:
                plot_type_val = plot_type_map.get(spec.plot_type.value, "scatter")
                if plot_type_val == "timeslice":
                    x_axis = "Timeslice"
                else:
                    x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
                    x_axis = x_metric.name if x_metric else "request_number"
                y_metric = next((m for m in spec.metrics if m.axis == "y"), None)

                new_configs[spec.name] = {
                    "is_default": True,
                    "size": "half",
                    "mode": "single_run",
                    "plot_type": plot_type_val,
                    "x_axis": x_axis,
                    "y_metric": y_metric.name if y_metric else "",
                    "y_metric_base": y_metric.name if y_metric else "",
                    "stat": y_metric.stat if y_metric and y_metric.stat else "avg",
                    "source": y_metric.source.value if y_metric else "requests",
                    "title": spec.title or spec.name.replace("-", " ").title(),
                }

        # Preserve slice_duration from existing state
        slice_duration = plot_state.get("slice_duration")

        return {
            "visible_plots": default_plot_ids,
            "hidden_plots": [],
            "plot_configs": new_configs,
            "config_version": current_version,
            "slice_duration": slice_duration,
        }


def register_size_migration_callback(app: dash.Dash):
    """
    Migrate old size values (small/medium/large) to new size values (half/full).

    This callback runs on app load to detect and migrate old localStorage data
    from the previous three-size system back to the two-size system.

    Migration mapping:
        "medium" -> "half" (50% width)
        "large" -> "full" (100% width)
        "small" -> "half" (fallback to 50% width)

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input("plot-state-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def migrate_size_values(plot_state):
        """Migrate old size values to new format."""
        if not plot_state:
            raise PreventUpdate

        # Check if already migrated
        if plot_state.get("_size_migrated"):
            raise PreventUpdate

        plot_configs = plot_state.get("plot_configs", {})
        needs_migration = False

        # Check if any plot has old size values
        for _plot_id, config in plot_configs.items():
            size = config.get("size", "half")
            if size in ("small", "medium", "large"):
                needs_migration = True
                break

        if not needs_migration:
            raise PreventUpdate

        # Migrate size values
        migrated_configs = {}
        for plot_id, config in plot_configs.items():
            size = config.get("size", "half")
            if size == "medium":
                new_size = "half"
            elif size == "large":
                new_size = "full"
            elif size == "small":
                new_size = "half"
            else:
                new_size = size

            migrated_configs[plot_id] = {**config, "size": new_size}

        return {
            **plot_state,
            "plot_configs": migrated_configs,
            "_size_migrated": True,
        }


def register_config_modal_callback(
    app: dash.Dash,
    runs: list[RunData],
    plot_config: PlotConfig,
    mode: VisualizationMode,
):
    """
    Register callback for config viewer modal.

    Handles:
    - Click on plot point → open modal with run config
    - Close button → close modal
    - Hover tooltip with run summary (via customdata in plots)

    Args:
        app: Dash application instance
        runs: List of RunData objects
        plot_config: PlotConfig instance
        mode: Visualization mode
    """
    # Use pattern-matching callback to handle ALL plots (default + custom)
    # This ensures both default plots and custom plots can trigger the config modal

    @app.callback(
        [
            Output("config-modal", "is_open"),
            Output("config-modal-header", "children"),
            Output("config-modal-summary", "children"),
            Output("config-modal-yaml", "children"),
            Output("current-run-idx-store", "data"),
        ],
        [
            Input({"type": "plot-graph", "index": dash.ALL}, "clickData"),
            Input("btn-close-config-modal", "n_clicks"),
        ],
        [State("config-modal", "is_open"), State("theme-store", "data")],
    )
    def toggle_config_modal(all_plot_clicks, close_clicks, is_open_state, theme_data):
        """Open modal with run config on plot point click."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        ctx = dash.callback_context

        if not ctx.triggered:
            return False, "", "", "", dash.no_update

        trigger_id = ctx.triggered[0]["prop_id"]
        trigger_value = ctx.triggered[0]["value"]

        # Close button clicked
        if "close" in trigger_id or "btn-close-config-modal" in trigger_id:
            return False, "", "", "", dash.no_update

        # Get the clicked data from whichever plot was clicked
        # With pattern-matching, trigger_value is the clickData from the clicked plot
        if not trigger_value or not isinstance(trigger_value, dict):
            return False, "", "", "", dash.no_update

        # Extract point data
        if "points" not in trigger_value or not trigger_value["points"]:
            return False, "", "", "", dash.no_update

        point = trigger_value["points"][0]

        # Additional validation: Ensure the click has actual coordinate data
        # This helps prevent button overlay clicks from triggering the modal
        if not point or "x" not in point or "y" not in point:
            return False, "", "", "", dash.no_update

        # Get run index from customdata
        if "customdata" not in point:
            return False, "", "", "No run data available", dash.no_update

        customdata = point["customdata"]

        # Handle both dict and list customdata formats
        if isinstance(customdata, dict):
            run_idx = customdata.get("run_idx", 0)
        elif isinstance(customdata, list) and len(customdata) > 0:
            run_idx = customdata[0] if isinstance(customdata[0], int) else 0
        else:
            return False, "", "", "Invalid run data", dash.no_update

        # Validate run index
        if run_idx < 0 or run_idx >= len(runs):
            return False, "", "", f"Invalid run index: {run_idx}", dash.no_update

        run = runs[run_idx]

        # Format header
        model = run.metadata.model or "Unknown Model"
        concurrency = run.metadata.concurrency or "N/A"
        header = f"{model} - Concurrency {concurrency}"

        # Create summary cards with key metrics
        summary_cards = []

        # Metadata card
        duration = run.metadata.duration_seconds

        # If duration not in metadata, try getting from aggregated metrics
        if duration is None or duration == 0:
            dur_metric = run.get_metric("benchmark_duration")
            if dur_metric:
                if hasattr(dur_metric, "stats"):
                    duration = getattr(dur_metric.stats, "avg", None)
                elif isinstance(dur_metric, dict):
                    duration = dur_metric.get("avg")

        duration = duration or 0
        request_count = run.metadata.request_count or 0
        endpoint_type = run.metadata.endpoint_type or "Unknown"

        summary_cards.append(
            html.Div(
                [
                    html.Div(
                        "Run Details",
                        style={
                            "font-weight": "600",
                            "margin-bottom": "8px",
                            "font-size": "11px",
                            "color": colors["text"],
                        },
                    ),
                    html.Div(
                        f"Duration: {duration:.1f}s",
                        style={
                            "font-size": "11px",
                            "margin-bottom": "4px",
                            "color": colors["text"],
                        },
                    ),
                    html.Div(
                        f"Requests: {request_count}",
                        style={
                            "font-size": "11px",
                            "margin-bottom": "4px",
                            "color": colors["text"],
                        },
                    ),
                    html.Div(
                        f"Endpoint: {endpoint_type}",
                        style={"font-size": "11px", "color": colors["text"]},
                    ),
                ],
                style={
                    "flex": "1",
                    "padding": "12px",
                    "background": colors["paper"],
                    "border-radius": "6px",
                    "margin-right": "8px",
                    "border": f"1px solid {colors['border']}",
                },
            )
        )

        # Key metrics card
        metric_lines = []
        for metric_name in [
            "request_latency",
            "time_to_first_token",
            "inter_token_latency",
            "request_throughput",
        ]:
            metric = run.get_metric(metric_name)
            if metric and hasattr(metric, "stats"):
                display_name = metric_name.replace("_", " ").title()
                p50 = getattr(metric.stats, "p50", None)
                if p50 is not None:
                    metric_lines.append(
                        html.Div(
                            f"{display_name}: {p50:.2f}",
                            style={
                                "font-size": "11px",
                                "margin-bottom": "4px",
                                "color": colors["text"],
                            },
                        )
                    )

        if metric_lines:
            summary_cards.append(
                html.Div(
                    [
                        html.Div(
                            "Key Metrics (p50)",
                            style={
                                "font-weight": "600",
                                "margin-bottom": "8px",
                                "font-size": "11px",
                                "color": colors["text"],
                            },
                        )
                    ]
                    + metric_lines,
                    style={
                        "flex": "1",
                        "padding": "12px",
                        "background": colors["paper"],
                        "border-radius": "6px",
                        "border": f"1px solid {colors['border']}",
                    },
                )
            )

        summary_section = html.Div(
            summary_cards,
            style={"display": "flex", "gap": "8px"},
        )

        # Format config as YAML
        try:
            config_dict = run.aggregated.get("input_config", {})

            # Use ruamel.yaml for pretty formatting
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 80

            stream = io.StringIO()
            yaml.dump(config_dict, stream)
            yaml_str = stream.getvalue()

        except Exception as e:
            yaml_str = f"Error formatting config: {e}\n\nRaw data:\n{config_dict}"

        return True, header, summary_section, yaml_str, run_idx


def register_hide_show_plot_callbacks(app: dash.Dash):
    """
    Register callbacks for hiding and showing plots.

    Handles:
    - Click X button → hide plot (CSS toggle for instant response)
    - Click eye icon → hide plot (CSS toggle for instant response)
    - Click Show button → restore hidden plot
    - Update hidden plots list in sidebar
    - State persistence for hidden_plots

    Args:
        app: Dash application instance
    """

    # Consolidated CSS toggle for instant hide/show - handles all button types
    @app.callback(
        Output({"type": "plot-container", "index": dash.MATCH}, "style"),
        Input({"type": "remove-plot-btn", "index": dash.MATCH}, "n_clicks"),
        Input({"type": "hide-plot-btn-direct", "index": dash.MATCH}, "n_clicks"),
        Input({"type": "show-plot-btn", "index": dash.MATCH}, "n_clicks"),
        State({"type": "plot-container", "index": dash.MATCH}, "style"),
        prevent_initial_call=True,
    )
    def toggle_plot_visibility_css(
        remove_clicks, hide_clicks, show_clicks, current_style
    ):
        """Toggle plot visibility via CSS - instant, no grid rebuild."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"]
        new_style = dict(current_style) if current_style else {}

        if "show-plot-btn" in trigger_id:
            new_style["display"] = "block"
        else:
            new_style["display"] = "none"

        return new_style

    # State persistence callback for X button (updates plot-state-store for persistence)
    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input({"type": "remove-plot-btn", "index": dash.ALL}, "n_clicks"),
        State("plot-state-store", "data"),
        prevent_initial_call=True,
    )
    def hide_plot(n_clicks_list, plot_state):
        """Hide a plot when X button is clicked."""
        # Validate that at least one button was clicked
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Find which button was clicked
        trigger_id = ctx.triggered[0]["prop_id"]
        if "remove-plot-btn" not in trigger_id:
            raise PreventUpdate

        # Extract plot ID from button ID
        button_id = json.loads(trigger_id.split(".")[0])
        plot_id = button_id["index"]

        # Move plot from visible to hidden - create NEW lists to trigger Dash change detection
        visible = list(plot_state.get("visible_plots", []))
        hidden = list(plot_state.get("hidden_plots", []))

        if plot_id in visible:
            visible.remove(plot_id)
            hidden.append(plot_id)
        else:
            _logger.warning(f"plot_id '{plot_id}' not found in visible list")

        return {
            "visible_plots": visible,
            "hidden_plots": hidden,
            "plot_configs": dict(plot_state.get("plot_configs", {})),
            "config_version": plot_state.get("config_version"),
        }

    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input({"type": "show-plot-btn", "index": dash.ALL}, "n_clicks"),
        State("plot-state-store", "data"),
        prevent_initial_call=True,
    )
    def show_plot(n_clicks_list, plot_state):
        """Show a hidden plot when Show button is clicked."""
        ctx = dash.callback_context

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"]
        trigger_value = ctx.triggered[0]["value"]

        # Skip if no actual button click (value is None or 0)
        # This happens when buttons are added/removed from the page
        if not trigger_value or trigger_value == 0:
            raise PreventUpdate

        if "show-plot-btn" not in trigger_id:
            raise PreventUpdate

        # Extract plot ID
        button_id = json.loads(trigger_id.split(".")[0])
        plot_id = button_id["index"]

        # Move plot from hidden to visible - create NEW lists to trigger Dash change detection
        visible = list(plot_state.get("visible_plots", []))
        hidden = list(plot_state.get("hidden_plots", []))

        if plot_id in hidden:
            hidden.remove(plot_id)
            visible.append(plot_id)

        return {
            "visible_plots": visible,
            "hidden_plots": hidden,
            "plot_configs": dict(plot_state.get("plot_configs", {})),
            "config_version": plot_state.get("config_version"),
        }

    @app.callback(
        Output("hidden-plots-list", "children"),
        [Input("plot-state-store", "data"), Input("theme-store", "data")],
    )
    def update_hidden_plots_list(plot_state, theme_data):
        """Update the list of hidden plots in sidebar."""
        hidden = plot_state.get("hidden_plots", [])
        plot_configs = plot_state.get("plot_configs", {})
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        if not hidden:
            return html.Div(
                "No deleted plots",
                style={
                    "font-size": "11px",
                    "color": NVIDIA_GRAY,
                    "font-style": "italic",
                    "font-family": PLOT_FONT_FAMILY,
                },
            )

        children = []
        for plot_id in hidden:
            plot_name = get_plot_title(plot_id, plot_configs)

            children.append(
                html.Div(
                    [
                        html.Span(
                            plot_name,
                            style={
                                "font-size": "11px",
                                "color": colors["text"],
                                "font-family": PLOT_FONT_FAMILY,
                                "flex": "1",
                                "min-width": "0",
                                "overflow": "hidden",
                                "text-overflow": "ellipsis",
                                "white-space": "nowrap",
                            },
                            title=plot_name,
                        ),
                        html.Button(
                            "Show",
                            id={"type": "show-plot-btn", "index": plot_id},
                            n_clicks=0,
                            style={
                                "padding": "2px 8px",
                                "font-size": "10px",
                                "background": colors["paper"],
                                "color": colors["text"],
                                "border": f"1px solid {NVIDIA_GREEN}",
                                "border-radius": "4px",
                                "cursor": "pointer",
                                "font-family": PLOT_FONT_FAMILY,
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "margin-bottom": "6px",
                        "padding": "4px",
                        "background": colors["paper"],
                        "border-radius": "4px",
                    },
                )
            )

        return children

    # Callback for direct hide button on plot (eye icon)
    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input({"type": "hide-plot-btn-direct", "index": dash.ALL}, "n_clicks"),
        State("plot-state-store", "data"),
        prevent_initial_call=True,
    )
    def hide_plot_direct(n_clicks_list, plot_state):
        """Hide plot directly from plot container (eye icon button)."""
        ctx = dash.callback_context

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"]
        trigger_value = ctx.triggered[0]["value"]

        if not trigger_value or trigger_value == 0:
            raise PreventUpdate

        if "hide-plot-btn-direct" not in trigger_id:
            raise PreventUpdate

        button_id = json.loads(trigger_id.split(".")[0])
        plot_id = button_id["index"]

        visible = list(plot_state.get("visible_plots", []))
        hidden = list(plot_state.get("hidden_plots", []))

        if plot_id in visible:
            visible.remove(plot_id)
            if plot_id not in hidden:
                hidden.append(plot_id)

        new_state = {
            "visible_plots": [p for p in visible],
            "hidden_plots": [p for p in hidden],
            "plot_configs": {
                k: dict(v) for k, v in plot_state.get("plot_configs", {}).items()
            },
            "config_version": plot_state.get("config_version"),
        }

        return new_state


def register_export_png_callback(
    app: dash.Dash,
    runs: list[RunData],
    mode: VisualizationMode,
    theme: PlotTheme,
    plot_config: PlotConfig,
):
    """
    Register callback for exporting visible plots as PNG bundle with size selection.

    Args:
        app: Dash application instance
        runs: List of RunData objects
        mode: Visualization mode
        theme: Plot theme
        plot_config: Plot configuration containing plot specs
    """

    SIZE_MAPPING = {
        "small": (960, 540),
        "medium": (1920, 1080),
        "large": (2880, 1620),
    }

    FONT_SCALE_MAPPING = {
        "small": 1.0,
        "medium": 1.5,
        "large": 2.0,
    }

    # Get plot specs based on mode
    if mode == VisualizationMode.MULTI_RUN:
        plot_specs = plot_config.get_multi_run_plot_specs()
    else:
        plot_specs = plot_config.get_single_run_plot_specs()

    # Build available_metrics dict (reuse existing function at line 3848)
    available_metrics = _build_available_metrics_dict(plot_specs)

    def _scale_figure_fonts(fig: go.Figure, scale: float) -> go.Figure:
        """
        Scale all font sizes in a figure by the given factor.

        Args:
            fig: Plotly figure object
            scale: Font scale multiplier (1.0 = base size)

        Returns:
            The figure with scaled fonts
        """
        if scale == 1.0:
            return fig
        fig.update_layout(
            title_font_size=int(18 * scale),
            font_size=int(10 * scale),
            legend_font_size=int(11 * scale),
            xaxis_title_font_size=int(12 * scale),
            yaxis_title_font_size=int(12 * scale),
        )
        return fig

    def _convert_figure_to_png(
        fig: go.Figure, filename: str, width: int, height: int, font_scale: float = 1.0
    ) -> tuple[str, bytes | str, str]:
        """
        Convert Plotly figure to PNG or HTML fallback.

        Thread-safe worker function for parallel PNG conversion.

        Args:
            fig: Plotly figure object
            filename: Base filename (without extension)
            width: PNG width in pixels
            height: PNG height in pixels
            font_scale: Font scale multiplier for readability

        Returns:
            Tuple of (filename, content, format) where:
            - filename: Base filename
            - content: PNG bytes or HTML string
            - format: "png" or "html"
        """
        try:
            _scale_figure_fonts(fig, font_scale)
            img_bytes = fig.to_image(format="png", width=width, height=height)
            return (filename, img_bytes, "png")
        except Exception as e:
            # Fallback to HTML if kaleido/Chrome unavailable
            _logger.warning(
                f"PNG conversion failed for {filename}, falling back to HTML: {e}"
            )
            html_str = fig.to_html(include_plotlyjs="cdn", config={"responsive": True})
            return (filename, html_str.encode(), "html")

    def _export_single_run_default_plot(
        plot_id: str,
        plot_specs: list,
        run: RunData,
        plot_gen,
        available_metrics: dict,
    ):
        """
        Export default single-run plot using handler factory pattern.

        Args:
            plot_id: Plot identifier (matches PlotSpec.name)
            plot_specs: List of PlotSpec objects
            run: RunData object
            plot_gen: PlotGenerator instance
            available_metrics: Dictionary with metric metadata

        Returns:
            Plotly figure or None if plot cannot be generated
        """
        # Find spec by name
        spec = next((s for s in plot_specs if s.name == plot_id), None)
        if not spec:
            _logger.warning(f"No spec found for plot '{plot_id}'")
            return None

        # Use handler factory (same as PNG exporter line 121)
        try:
            HandlerClass = plugins.get_class(PluginType.PLOT, spec.plot_type)
            handler = HandlerClass(plot_generator=plot_gen, logger=None)
            return handler.create_plot(spec, run, available_metrics)
        except Exception as e:
            _logger.error(f"Error creating plot '{plot_id}': {e}")
            return None

    def generate_export_filename_single_run(plot_id: str, plot_config: dict) -> str:
        """
        Generate filename for single-run plot export.

        Args:
            plot_id: Plot identifier
            plot_config: Plot configuration dict

        Returns:
            Safe filename string
        """
        is_default = plot_config.get("is_default", True)

        if is_default:
            # Use plot_id for default plots (e.g., "ttft_over_time")
            return plot_id.replace("-", "_")
        else:
            # Build descriptive name for custom plots
            plot_type = plot_config.get("plot_type", "plot")
            y_metric = plot_config.get("y_metric", "")
            x_axis = plot_config.get("x_axis", "")

            parts = [plot_type]
            if y_metric:
                parts.append(y_metric)
            if x_axis and x_axis != "request_number":
                parts.append(x_axis)

            filename = "_".join(parts)
            return filename.replace("-", "_").replace("/", "_").replace(":", "_")

    def generate_export_filename(plot_id: str, plot_config: dict) -> str:
        """Generate clean, descriptive filename from plot config."""
        plot_type = plot_config.get("plot_type", "plot")
        x_metric = plot_config.get("x_metric", "")
        y_metric = plot_config.get("y_metric", "")
        x_stat = plot_config.get("x_stat", "")
        y_stat = plot_config.get("y_stat", "")

        # Build parts: [plot_type, x_metric, x_stat, "vs", y_metric, y_stat]
        parts = [plot_type]
        if x_metric:
            parts.append(x_metric)
            if x_stat and x_stat != "value":
                parts.append(x_stat)
        if y_metric:
            parts.append("vs")
            parts.append(y_metric)
            if y_stat and y_stat != "value":
                parts.append(y_stat)

        filename = "_".join(parts)
        # Sanitize for filesystem
        return filename.replace("-", "_").replace("/", "_").replace(":", "_")

    @app.callback(
        Output("download-png-bundle", "data"),
        Input("btn-export-png", "n_clicks"),
        [
            State("export-format-selector", "value"),
            State("export-size-selector", "value"),
            State("plot-state-store", "data"),
            State("run-selector", "value"),
            State("theme-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def export_plots_bundle(
        n_clicks,
        export_format,
        export_size,
        plot_state,
        selected_runs,
        theme_data,
    ):
        """Export all visible plots as PNG or HTML files in a ZIP archive."""
        # USE_PARALLEL_EXPORT: Set to False to use old sequential version for comparison
        USE_PARALLEL_EXPORT = True

        if USE_PARALLEL_EXPORT:
            return export_plots_bundle_parallel(
                n_clicks,
                export_format,
                export_size,
                plot_state,
                selected_runs,
                theme_data,
            )

        # ORIGINAL SEQUENTIAL VERSION (kept for rollback if needed)
        if not n_clicks:
            raise PreventUpdate

        visible_plots = plot_state.get("visible_plots", [])
        if not visible_plots:
            raise PreventUpdate

        # Get export format (default to PNG)
        export_format = export_format or "png"

        # Get width, height, and font scale from size selector (only used for PNG)
        if export_format == "png":
            export_size = export_size or "medium"
            width, height = SIZE_MAPPING.get(export_size, (1920, 1080))
            font_scale = FONT_SCALE_MAPPING.get(export_size, 1.5)
        else:
            # HTML export doesn't use fixed dimensions
            width, height = None, None
            font_scale = 1.0

        filtered_runs = [runs[i] for i in selected_runs] if selected_runs else runs
        current_theme = _get_current_theme(theme_data, theme)
        plot_gen = _get_plot_generator(current_theme)

        # Reset color registry to ensure consistent colors with CLI export
        plot_gen.reset_color_registry()

        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        html_fallback_used = [False]

        plot_configs = plot_state.get("plot_configs", {})

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            if mode == VisualizationMode.SINGLE_RUN:
                # Single-run export - use drill-down cache if available
                if "current" in _drill_down_run_cache:
                    run = _drill_down_run_cache["current"]
                else:
                    run = runs[0]

                for plot_id in visible_plots:
                    config = plot_configs.get(plot_id)
                    if not config:
                        continue

                    try:
                        is_default = config.get("is_default", True)

                        if is_default:
                            # Export default plot from PlotSpec
                            fig = _export_single_run_default_plot(
                                plot_id, plot_specs, run, plot_gen, available_metrics
                            )
                        else:
                            # Export custom plot from config
                            fig = _generate_custom_single_run_plot(
                                config, run, plot_gen, current_theme
                            )

                        if not fig:
                            continue

                        # Generate filename
                        safe_filename = generate_export_filename_single_run(
                            plot_id, config
                        )

                        # Export to ZIP
                        if export_format == "png":
                            try:
                                _scale_figure_fonts(fig, font_scale)
                                img_bytes = fig.to_image(
                                    format="png", width=width, height=height
                                )
                                zip_file.writestr(f"{safe_filename}.png", img_bytes)
                            except Exception as e:
                                print(
                                    f"⚠️ PNG conversion failed for {safe_filename}, falling back to HTML: {e}"
                                )
                                html_str = fig.to_html(
                                    include_plotlyjs="cdn", config={"responsive": True}
                                )
                                zip_file.writestr(
                                    f"{safe_filename}.html", html_str.encode()
                                )
                                html_fallback_used[0] = True
                        else:
                            html_str = fig.to_html(
                                include_plotlyjs="cdn", config={"responsive": True}
                            )
                            zip_file.writestr(
                                f"{safe_filename}.html", html_str.encode()
                            )

                    except Exception as e:
                        _logger.warning(f"Failed to export plot {plot_id}: {e}")
                        continue

            else:
                # Multi-run export - Generate and add each visible plot dynamically using plot_configs
                for plot_id in visible_plots:
                    config = plot_configs.get(plot_id)
                    if not config:
                        continue

                    try:
                        # Get config values
                        x_metric = config.get("x_metric")
                        y_metric = config.get("y_metric")
                        x_stat = config.get("x_stat", "p50")
                        y_stat = config.get("y_stat", "avg")
                        plot_type = config.get("plot_type", "scatter_line")
                        log_scale = config.get("log_scale", "none")
                        title = config.get("title", "")
                        label_by = config.get("label_by", "concurrency")
                        group_by = config.get("group_by", "model")

                        if not x_metric or not y_metric:
                            continue

                        # Generate DataFrame
                        result = runs_to_dataframe(
                            filtered_runs, x_metric, x_stat, y_metric, y_stat
                        )
                        df = result["df"]

                        if df.empty:
                            continue

                        # Extract experiment_types mapping for color assignment
                        experiment_types = None
                        if "experiment_type" in df.columns and group_by:
                            group_col = (
                                group_by[0] if isinstance(group_by, list) else group_by
                            )
                            if group_col in df.columns:
                                experiment_types = {
                                    g: df[df[group_col] == g]["experiment_type"].iloc[0]
                                    for g in df[group_col].unique()
                                }

                        # Generate figure based on plot_type
                        if plot_type == "pareto":
                            fig = plot_gen.create_pareto_plot(
                                df,
                                x_metric,
                                y_metric,
                                label_by=label_by,
                                group_by=group_by,
                                title=title,
                                experiment_types=experiment_types,
                            )
                        elif plot_type == "scatter_line":
                            fig = plot_gen.create_scatter_line_plot(
                                df,
                                x_metric,
                                y_metric,
                                label_by=label_by,
                                group_by=group_by,
                                title=title,
                                mode="lines+markers",
                                experiment_types=experiment_types,
                            )
                        elif plot_type == "scatter":
                            fig = plot_gen.create_scatter_line_plot(
                                df,
                                x_metric,
                                y_metric,
                                label_by=label_by,
                                group_by=group_by,
                                title=title,
                                mode="markers",
                                experiment_types=experiment_types,
                            )
                        elif plot_type == "bar":
                            fig = plot_gen.create_multi_run_bar_chart(
                                df=df,
                                x_metric=x_metric,
                                y_metric=y_metric,
                                group_by=group_by,
                                title=title,
                            )
                        else:
                            # Skip unsupported plot types
                            continue

                        # Apply log scale
                        if log_scale in ("x", "both"):
                            fig.update_xaxes(type="log")
                        if log_scale in ("y", "both"):
                            fig.update_yaxes(type="log")

                        # Export to ZIP with descriptive filename
                        safe_filename = generate_export_filename(plot_id, config)
                        if export_format == "png":
                            try:
                                _scale_figure_fonts(fig, font_scale)
                                img_bytes = fig.to_image(
                                    format="png", width=width, height=height
                                )
                                zip_file.writestr(f"{safe_filename}.png", img_bytes)
                            except Exception as e:
                                print(
                                    f"⚠️ PNG conversion failed for {safe_filename}, falling back to HTML: {e}"
                                )
                                html_str = fig.to_html(
                                    include_plotlyjs="cdn", config={"responsive": True}
                                )
                                zip_file.writestr(
                                    f"{safe_filename}.html", html_str.encode()
                                )
                                html_fallback_used[0] = True
                        else:
                            # HTML export - no try/catch needed, intentional format
                            html_str = fig.to_html(
                                include_plotlyjs="cdn", config={"responsive": True}
                            )
                            zip_file.writestr(
                                f"{safe_filename}.html", html_str.encode()
                            )

                    except Exception as e:
                        # Log error but continue with other plots
                        _logger.warning(f"Failed to export plot {plot_id}: {e}")
                        continue

            # Add README only if PNG export failed and fell back to HTML
            if export_format == "png" and html_fallback_used[0]:
                readme = """AIPerf Dashboard Plots - Exported as HTML

Chrome rendering failed, so PNG export was unavailable.
Plots have been exported as interactive HTML files instead.

To enable PNG export in the future, run:
    plotly_get_chrome

HTML files can be opened in any web browser and are fully interactive.
"""
                zip_file.writestr("README.txt", readme.encode())

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_format == "png":
            filename = f"aiperf_dashboard_png_{export_size}_{timestamp}.zip"
        else:
            filename = f"aiperf_dashboard_html_{timestamp}.zip"

        return dcc.send_bytes(zip_buffer.getvalue(), filename)

    def export_plots_bundle_parallel(
        n_clicks,
        export_format,
        export_size,
        plot_state,
        selected_runs,
        theme_data,
    ):
        """
        Export all visible plots as PNG or HTML files in a ZIP archive (PARALLEL VERSION).

        This version uses ThreadPoolExecutor to parallelize PNG conversion for better performance.
        """
        if not n_clicks:
            raise PreventUpdate

        visible_plots = plot_state.get("visible_plots", [])
        if not visible_plots:
            raise PreventUpdate

        # Get export format (default to PNG)
        export_format = export_format or "png"

        # Get width, height, and font scale from size selector (only used for PNG)
        if export_format == "png":
            export_size = export_size or "medium"
            width, height = SIZE_MAPPING.get(export_size, (1920, 1080))
            font_scale = FONT_SCALE_MAPPING.get(export_size, 1.5)
        else:
            # HTML export doesn't use fixed dimensions
            width, height = None, None
            font_scale = 1.0

        filtered_runs = [runs[i] for i in selected_runs] if selected_runs else runs
        current_theme = _get_current_theme(theme_data, theme)

        html_fallback_used = [False]
        plot_configs = plot_state.get("plot_configs", {})

        # Initialize cache for figure reuse
        cache = get_plot_cache()
        runs_hash = (
            "single"
            if mode == VisualizationMode.SINGLE_RUN
            else compute_runs_hash(selected_runs)
        )
        cache_hits = 0
        cache_misses = 0

        # Phase 1: Generate all figures (with cache lookup)
        figure_tasks = []  # List of (fig, filename) tuples
        _logger.info("Generating {len(visible_plots)} plots (with cache)...")

        if mode == VisualizationMode.SINGLE_RUN:
            # Single-run export - use drill-down cache if available
            if "current" in _drill_down_run_cache:
                run = _drill_down_run_cache["current"]
            else:
                run = runs[0]

            for plot_id in visible_plots:
                config = plot_configs.get(plot_id)
                if not config:
                    continue

                try:
                    config_hash = compute_config_hash(config)
                    cache_key = CacheKey(
                        plot_id=plot_id,
                        config_hash=config_hash,
                        runs_hash=runs_hash,
                        theme=current_theme,
                    )

                    # Try cache first
                    fig = cache.get(cache_key)

                    if fig is not None:
                        cache_hits += 1
                        _logger.debug("CACHE HIT (export): {plot_id}")
                    else:
                        cache_misses += 1
                        _logger.debug("CACHE MISS (export): {plot_id}")
                        fig = _generate_singlerun_figure(
                            plot_id, config, run, plot_specs, current_theme
                        )
                        if fig is not None:
                            cache.set(cache_key, fig)

                    if not fig:
                        continue

                    safe_filename = generate_export_filename_single_run(plot_id, config)
                    figure_tasks.append((fig, safe_filename))

                except Exception as e:
                    _logger.warning(f"Failed to generate plot {plot_id}: {e}")
                    continue

        else:
            # Multi-run mode
            for plot_id in visible_plots:
                config = plot_configs.get(plot_id)
                if not config:
                    continue

                try:
                    config_hash = compute_config_hash(config)
                    cache_key = CacheKey(
                        plot_id=plot_id,
                        config_hash=config_hash,
                        runs_hash=runs_hash,
                        theme=current_theme,
                    )

                    # Try cache first
                    fig = cache.get(cache_key)

                    if fig is not None:
                        cache_hits += 1
                        _logger.debug("CACHE HIT (export): {plot_id}")
                    else:
                        cache_misses += 1
                        _logger.debug("CACHE MISS (export): {plot_id}")
                        fig, _ = _generate_multirun_figure(
                            filtered_runs, config, current_theme
                        )
                        if fig is not None:
                            cache.set(cache_key, fig)

                    if not fig:
                        continue

                    safe_filename = generate_export_filename(plot_id, config)
                    figure_tasks.append((fig, safe_filename))

                except Exception as e:
                    _logger.warning(f"Failed to generate plot {plot_id}: {e}")
                    continue

        _logger.info("Export cache stats: hits={cache_hits}, misses={cache_misses}")

        # Phase 2: Convert to PNG in parallel (NEW!)
        conversion_results = []  # List of (filename, content, format) tuples

        if export_format == "png" and figure_tasks:
            _logger.info("Converting {len(figure_tasks)} plots to PNG in parallel...")

            max_workers = min(4, len(figure_tasks))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all conversion tasks
                future_to_task = {
                    executor.submit(
                        _convert_figure_to_png, fig, filename, width, height, font_scale
                    ): (
                        fig,
                        filename,
                    )
                    for fig, filename in figure_tasks
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_task):
                    try:
                        filename, content, format_type = future.result()
                        conversion_results.append((filename, content, format_type))

                        if format_type == "html":
                            html_fallback_used[0] = True

                        completed += 1
                        print(
                            f"  ✓ Converted {completed}/{len(figure_tasks)}: {filename}"
                        )

                    except Exception as e:
                        _, filename = future_to_task[future]
                        _logger.warning(f"Failed to convert {filename}: {e}")
                        continue

        else:
            # HTML export - no parallel conversion needed
            _logger.info("Exporting {len(figure_tasks)} plots as HTML...")
            for fig, filename in figure_tasks:
                html_str = fig.to_html(
                    include_plotlyjs="cdn", config={"responsive": True}
                )
                conversion_results.append((filename, html_str.encode(), "html"))

        # Phase 3: Write to ZIP sequentially (thread-safe)
        _logger.info("Writing {len(conversion_results)} files to ZIP...")
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename, content, format_type in conversion_results:
                if format_type == "png":
                    zip_file.writestr(f"{filename}.png", content)
                else:
                    zip_file.writestr(f"{filename}.html", content)

            # Add README if any PNG conversion fell back to HTML
            if export_format == "png" and html_fallback_used[0]:
                readme = """AIPerf Dashboard Plots - Exported as HTML

Chrome rendering failed, so PNG export was unavailable.
Plots have been exported as interactive HTML files instead.

To enable PNG export in the future, run:
    plotly_get_chrome

HTML files can be opened in any web browser and are fully interactive.
"""
                zip_file.writestr("README.txt", readme.encode())

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_format == "png":
            filename = f"aiperf_dashboard_png_{export_size}_{timestamp}.zip"
        else:
            filename = f"aiperf_dashboard_html_{timestamp}.zip"

        _logger.info("Export complete: {filename}")
        return dcc.send_bytes(zip_buffer.getvalue(), filename)

    @app.callback(
        [
            Output("export-size-selector", "disabled"),
            Output("export-size-label", "style"),
        ],
        Input("export-format-selector", "value"),
        prevent_initial_call=False,
    )
    def toggle_size_selector_state(export_format):
        """Disable size selector when HTML format is selected."""
        base_style = get_label_style(theme)
        if export_format == "html":
            return True, {**base_style, "opacity": "0.5", "cursor": "not-allowed"}
        else:
            return False, base_style


def register_layout_control_callbacks(
    app: dash.Dash, mode: VisualizationMode, plot_config: PlotConfig, theme: PlotTheme
):
    """
    Register callbacks for layout control buttons.

    Handles:
    - Reset layout to defaults
    - Clear saved layout from localStorage

    Args:
        app: Dash application instance
        mode: Visualization mode
        plot_config: PlotConfig instance with YAML config
    """

    @app.callback(
        Output("plot-state-store", "data"),
        Input("btn-reset-layout", "n_clicks"),
        State("plot-state-store", "data"),
    )
    def reset_layout(n_clicks, plot_state):
        """Reset plot layout to defaults from config (preserve custom plots)."""
        if not n_clicks:
            raise PreventUpdate
        _logger.debug("RESET LAYOUT CALLBACK FIRED")
        # Get defaults from config (NOT hardcoded!)
        if mode == VisualizationMode.MULTI_RUN:
            specs = plot_config.get_multi_run_plot_specs()
        else:
            specs = plot_config.get_single_run_plot_specs()

        default_visible = [spec.name for spec in specs]
        _logger.debug("Defaults from config: {default_visible}")

        # Rebuild plot_configs from YAML (reset all settings to YAML values)
        new_configs = {}
        if mode == VisualizationMode.MULTI_RUN:
            for spec in specs:
                # Extract x and y metric specs
                x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
                y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

                if x_metric_spec and y_metric_spec:
                    plot_type_val = (
                        spec.plot_type.value if spec.plot_type else "scatter_line"
                    )
                    new_configs[spec.name] = {
                        "x_metric": x_metric_spec.name,
                        "x_stat": x_metric_spec.stat or "p50",
                        "y_metric": y_metric_spec.name,
                        "y_stat": y_metric_spec.stat or "avg",
                        "log_scale": "none",
                        "is_default": True,
                        "plot_type": plot_type_val,
                        "size": "half",
                        "label_by": spec.label_by or "concurrency",
                        "group_by": spec.group_by or "model",
                        "title": spec.title or spec.name.replace("-", " ").title(),
                    }
        else:
            # Single-run mode: full config for edit modal support
            plot_type_map = {
                "timeslice": "timeslice",
                "scatter": "scatter",
                "area": "area",
                "dual_axis": "dual_axis",
                "scatter_with_percentiles": "scatter",
                "request_timeline": "scatter",
            }
            for spec in specs:
                plot_type_val = plot_type_map.get(spec.plot_type.value, "scatter")
                if plot_type_val == "timeslice":
                    x_axis = "Timeslice"
                else:
                    x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
                    x_axis = x_metric.name if x_metric else "request_number"
                y_metric = next((m for m in spec.metrics if m.axis == "y"), None)

                new_configs[spec.name] = {
                    "is_default": True,
                    "size": "half",
                    "mode": "single_run",
                    "plot_type": plot_type_val,
                    "x_axis": x_axis,
                    "y_metric": y_metric.name if y_metric else "",
                    "y_metric_base": y_metric.name if y_metric else "",
                    "stat": y_metric.stat if y_metric and y_metric.stat else "avg",
                    "source": y_metric.source.value if y_metric else "requests",
                    "title": spec.title or spec.name.replace("-", " ").title(),
                }

        _logger.debug("Final plot_configs: {list(new_configs.keys())}")

        # Get slice_duration for single-run mode
        slice_duration = plot_state.get("slice_duration")

        new_state = {
            "visible_plots": list(default_visible),
            "hidden_plots": [],
            "plot_configs": new_configs,
            "config_version": plot_state.get("config_version"),
            "slice_duration": slice_duration,
        }
        _logger.debug("Resetting to YAML defaults")
        return new_state


def register_sidebar_widgets_theme_callback(app: dash.Dash):
    """
    Register callback to update sidebar widget styles when theme changes.

    Updates dropdown classNames and run selector label styles to match the current theme.

    Args:
        app: Dash application instance
    """

    @app.callback(
        [
            Output({"type": "metric-stat-selector", "metric": dash.ALL}, "className"),
            Output("run-selector", "labelStyle"),
        ],
        [Input("theme-store", "data")],
    )
    def update_sidebar_widgets(theme_data):
        """Update sidebar widgets when theme changes."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)

        # Get number of metric stat selectors from context
        ctx = dash.callback_context
        num_dropdowns = len(ctx.outputs_list[0])

        # Dropdown className based on theme
        dropdown_class = "dark-dropdown" if current_theme == PlotTheme.DARK else ""
        dropdown_classes = [dropdown_class] * num_dropdowns

        # Run selector label style
        label_color = NVIDIA_GRAY if current_theme == PlotTheme.LIGHT else "#E0E0E0"
        label_style = {
            "display": "block",
            "margin": "4px 0",
            "color": label_color,
            "font-family": PLOT_FONT_FAMILY,
        }

        return dropdown_classes, label_style


def register_sidebar_components_theme_callback(app: dash.Dash):
    """
    Register callback to update sidebar card backgrounds and button styles when theme changes.

    Updates the visual card wrappers and action buttons in the sidebar to match current theme.

    Args:
        app: Dash application instance
    """

    @app.callback(
        [
            # Card wrappers
            Output("sidebar-export-card", "style"),
            Output("sidebar-layout-card", "style"),
            Output("sidebar-run-selector-card", "style"),
            # Buttons
            Output("btn-reset-layout", "style"),
            Output("btn-export-png", "style"),
        ],
        [Input("theme-store", "data")],
    )
    def update_sidebar_components(theme_data):
        """Update sidebar card backgrounds and buttons when theme changes."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        # Card style for visual separation (complete styling to match builder.py)
        card_style = {
            "background": colors["paper"],
            "border": f"1px solid {colors['border']}",
            "border-radius": "8px",
            "padding": "12px",
            "margin-bottom": "24px",
            "width": "100%",
            "box-sizing": "border-box",
            "display": "flex",
            "flex-direction": "column",
            "align-items": "stretch",
        }

        # Run selector card has no bottom margin (last item)
        run_selector_card_style = {
            **card_style,
            "margin-bottom": "0",
        }

        # Button styles using theme-aware helper
        reset_button_style = get_button_style(current_theme, "secondary")
        export_button_style = get_button_style(current_theme, "secondary")

        return (
            card_style,  # export card
            card_style,  # layout card
            run_selector_card_style,
            reset_button_style,
            export_button_style,
        )


def register_modal_theme_callbacks(app: dash.Dash, mode: VisualizationMode):
    """
    Register callbacks to update modal styles when theme changes.

    Args:
        app: Dash application instance
        mode: Visualization mode (determines which modals exist)
    """

    @app.callback(
        [
            Output("config-modal-header-container", "style"),
            Output("config-modal-body-container", "style"),
            Output("config-modal-footer-container", "style"),
            Output("config-modal-yaml-label", "style"),
            Output("config-modal-header", "style"),
            Output("config-modal-yaml", "style"),
            Output("config-modal", "className"),
        ],
        [Input("theme-store", "data")],
    )
    def update_config_modal_theme(theme_data):
        """Update config modal styles when theme changes."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        header_style = {
            "background-color": colors["paper"],
            "color": colors["text"],
            "border-bottom": f"1px solid {colors['border']}",
        }

        body_style = {
            "background-color": colors["background"],
            "color": colors["text"],
        }

        footer_style = {
            "background-color": colors["paper"],
            "border-top": f"1px solid {colors['border']}",
        }

        yaml_label_style = {
            "font-size": "12px",
            "font-weight": "600",
            "color": colors["text"],
            "margin-bottom": "8px",
            "font-family": PLOT_FONT_FAMILY,
        }

        header_text_style = {
            "font-size": "16px",
            "font-weight": "600",
            "color": colors["text"],
            "margin-bottom": "16px",
            "padding-bottom": "8px",
            "border-bottom": f"1px solid {colors['border']}",
        }

        yaml_style = {
            "background": colors["paper"],
            "color": colors["text"],
            "padding": "12px",
            "border-radius": "4px",
            "overflow-x": "auto",
            "font-family": "monospace",
            "font-size": "11px",
            "line-height": "1.5",
            "border": f"1px solid {colors['border']}",
            "max-height": "400px",
            "overflow-y": "auto",
        }

        modal_class = f"theme-{current_theme.value}"

        return (
            header_style,
            body_style,
            footer_style,
            yaml_label_style,
            header_text_style,
            yaml_style,
            modal_class,
        )

    # Multi-run mode specific modals
    if mode == VisualizationMode.MULTI_RUN:

        @app.callback(
            [
                Output("custom-plot-modal-header-container", "style"),
                Output("custom-plot-modal-body-container", "style"),
                Output("custom-plot-modal-footer-container", "style"),
                Output("custom-plot-modal", "className"),
            ],
            [Input("theme-store", "data")],
        )
        def update_custom_plot_modal_theme(theme_data):
            """Update custom plot modal styles when theme changes."""
            current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
            colors = get_theme_colors(current_theme)

            header_style = {
                "background-color": colors["paper"],
                "color": colors["text"],
                "border-bottom": f"1px solid {colors['border']}",
            }

            body_style = {
                "background-color": colors["background"],
                "color": colors["text"],
            }

            footer_style = {
                "background-color": colors["paper"],
                "border-top": f"1px solid {colors['border']}",
            }

            modal_class = f"theme-{current_theme.value}"

            return header_style, body_style, footer_style, modal_class

        @app.callback(
            [
                Output("edit-plot-modal-header-container", "style"),
                Output("edit-plot-modal-body-container", "style"),
                Output("edit-plot-modal-footer-container", "style"),
                Output("btn-save-as-new-plot", "style"),
                Output("btn-cancel-edit-plot", "style"),
                Output("edit-plot-modal", "className"),
            ],
            [Input("theme-store", "data")],
        )
        def update_edit_plot_modal_theme(theme_data):
            """Update edit plot modal styles when theme changes."""
            current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
            colors = get_theme_colors(current_theme)

            header_style = {
                "background-color": colors["paper"],
                "color": colors["text"],
                "border-bottom": f"1px solid {colors['border']}",
            }

            body_style = {
                "background-color": colors["background"],
                "color": colors["text"],
            }

            footer_style = {
                "background-color": colors["paper"],
                "border-top": f"1px solid {colors['border']}",
                "display": "flex",
                "flex-wrap": "nowrap",
                "justify-content": "space-between",
                "align-items": "center",
                "gap": "6px",
                "padding": "10px 12px",
            }

            save_as_new_style = {
                "background": colors["paper"],
                "color": colors["text"],
                "border": f"1px solid {NVIDIA_GREEN}",
                "padding": "4px 12px",
            }

            cancel_style = {
                "background": colors["paper"],
                "color": colors["text"],
                "border": f"1px solid {colors['border']}",
                "padding": "4px 12px",
            }

            modal_class = f"theme-{current_theme.value}"

            return (
                header_style,
                body_style,
                footer_style,
                save_as_new_style,
                cancel_style,
                modal_class,
            )

    # Single-run mode specific modals
    if mode == VisualizationMode.SINGLE_RUN:

        @app.callback(
            [
                Output("single-run-custom-plot-modal-header", "style"),
                Output("single-run-custom-plot-modal-body", "style"),
                Output("single-run-custom-plot-modal-footer", "style"),
                Output("single-run-custom-plot-modal", "className"),
            ],
            [Input("theme-store", "data")],
            prevent_initial_call=True,
        )
        def update_single_run_custom_plot_modal_theme(theme_data):
            """Update single-run custom plot modal styles when theme changes."""
            current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
            colors = get_theme_colors(current_theme)

            header_style = {
                "background-color": colors["paper"],
                "color": colors["text"],
                "border-bottom": f"1px solid {colors['border']}",
            }

            body_style = {
                "background-color": colors["background"],
                "color": colors["text"],
            }

            footer_style = {
                "background-color": colors["paper"],
                "border-top": f"1px solid {colors['border']}",
            }

            modal_class = f"theme-{current_theme.value}"

            return header_style, body_style, footer_style, modal_class

        @app.callback(
            [
                Output("edit-sr-plot-modal-header", "style"),
                Output("edit-sr-plot-modal-body", "style"),
                Output("edit-sr-plot-modal-footer", "style"),
                Output("btn-sr-save-as-new", "style"),
                Output("btn-sr-cancel-edit", "style"),
                Output("edit-single-run-plot-modal", "className"),
                Output("edit-sr-plot-title", "style"),
            ],
            [Input("theme-store", "data")],
        )
        def update_edit_single_run_plot_modal_theme(theme_data):
            """Update single-run edit plot modal styles when theme changes."""
            current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
            colors = get_theme_colors(current_theme)

            header_style = {
                "background-color": colors["paper"],
                "color": colors["text"],
                "border-bottom": f"1px solid {colors['border']}",
            }

            body_style = {
                "background-color": colors["background"],
                "color": colors["text"],
            }

            footer_style = {
                "background-color": colors["paper"],
                "border-top": f"1px solid {colors['border']}",
                "display": "flex",
                "flex-wrap": "nowrap",
                "justify-content": "space-between",
                "align-items": "center",
                "gap": "6px",
                "padding": "10px 12px",
            }

            save_as_new_style = {
                "background": colors["paper"],
                "color": colors["text"],
                "border": f"1px solid {NVIDIA_GREEN}",
                "padding": "4px 12px",
            }

            cancel_style = {
                "background": colors["paper"],
                "color": colors["text"],
                "border": f"1px solid {colors['border']}",
                "padding": "4px 12px",
            }

            modal_class = f"theme-{current_theme.value}"

            title_input_style = {
                "margin-bottom": "12px",
                "font-size": "12px",
                "width": "100%",
                "background-color": colors["paper"],
                "color": colors["text"],
                "border": f"1px solid {colors['border']}",
                "padding": "6px 8px",
                "border-radius": "4px",
            }

            return (
                header_style,
                body_style,
                footer_style,
                save_as_new_style,
                cancel_style,
                modal_class,
                title_input_style,
            )


def register_custom_plot_callbacks(app: dash.Dash, runs: list, plot_config: PlotConfig):
    """
    Register callbacks for custom plot creation.

    Handles:
    - Open modal when + button clicked
    - Close modal on Cancel
    - Create custom plot and add to state

    Args:
        app: Dash application instance
        runs: List of run data
        plot_config: Plot configuration for accessing experiment classification settings
    """

    @app.callback(
        [
            Output("custom-plot-modal", "is_open", allow_duplicate=True),
            Output("custom-x-metric", "value"),
            Output("custom-y-metric", "value"),
            Output("custom-x-stat", "value"),
            Output("custom-y-stat", "value"),
            Output("custom-x-log-switch", "value"),
            Output("custom-y-log-switch", "value"),
            Output("custom-x-autoscale-switch", "value"),
            Output("custom-y-autoscale-switch", "value"),
            Output("custom-plot-type", "value"),
            Output("custom-label-by", "value"),
            Output("custom-group-by", "value"),
        ],
        Input("add-multirun-plot-slot", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_custom_plot_modal(n_clicks):
        """Open custom plot creation modal and reset all form fields."""
        if n_clicks and n_clicks > 0:
            # Check if experimental classification is enabled
            exp_class_config = plot_config.get_experiment_classification_config()
            default_group_by = (
                "experiment_group" if exp_class_config is not None else None
            )

            return (
                True,
                None,
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                None,
                None,
                default_group_by,
            )
        raise PreventUpdate

    @app.callback(
        Output("custom-plot-modal", "is_open", allow_duplicate=True),
        Input("btn-cancel-custom-plot", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_custom_plot_modal(n_clicks):
        """Close custom plot modal on cancel."""
        if n_clicks and n_clicks > 0:
            return False
        raise PreventUpdate

    @app.callback(
        [
            Output("custom-x-stat", "options"),
            Output("custom-x-stat", "value", allow_duplicate=True),
            Output("custom-y-stat", "options"),
            Output("custom-y-stat", "value", allow_duplicate=True),
        ],
        [
            Input("custom-x-metric", "value"),
            Input("custom-y-metric", "value"),
        ],
        [
            State("custom-x-stat", "value"),
            State("custom-y-stat", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_custom_stat_options(x_metric, y_metric, current_x_stat, current_y_stat):
        """Filter stat dropdown options based on available stats for selected metrics."""
        all_stat_options = MULTI_RUN_STAT_OPTIONS

        # Get available stats for each metric
        x_available_stats = (
            get_available_stats_for_metric(runs, x_metric) if x_metric else []
        )
        y_available_stats = (
            get_available_stats_for_metric(runs, y_metric) if y_metric else []
        )

        # Filter options to only those available
        x_options = (
            [opt for opt in all_stat_options if opt["value"] in x_available_stats]
            if x_available_stats
            else all_stat_options
        )
        y_options = (
            [opt for opt in all_stat_options if opt["value"] in y_available_stats]
            if y_available_stats
            else all_stat_options
        )

        # Smart fallback logic for X stat value
        x_value = None
        # Auto-select if only one option
        if len(x_options) == 1:
            x_value = x_options[0]["value"]
        elif x_available_stats:
            # Priority: current → p50 → avg → first available
            if current_x_stat and current_x_stat in x_available_stats:
                x_value = current_x_stat
            elif "p50" in x_available_stats:
                x_value = "p50"
            elif "avg" in x_available_stats:
                x_value = "avg"
            elif x_available_stats:
                x_value = x_available_stats[0]

        # Smart fallback logic for Y stat value
        y_value = None
        # Auto-select if only one option
        if len(y_options) == 1:
            y_value = y_options[0]["value"]
        elif y_available_stats:
            # Priority: current → avg → p50 → first available
            if current_y_stat and current_y_stat in y_available_stats:
                y_value = current_y_stat
            elif "avg" in y_available_stats:
                y_value = "avg"
            elif "p50" in y_available_stats:
                y_value = "p50"
            elif y_available_stats:
                y_value = y_available_stats[0]

        return x_options, x_value, y_options, y_value

    @app.callback(
        [
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("custom-plot-modal", "is_open", allow_duplicate=True),
        ],
        Input("btn-create-custom-plot", "n_clicks"),
        [
            State("custom-x-metric", "value"),
            State("custom-x-stat", "value"),
            State("custom-y-metric", "value"),
            State("custom-y-stat", "value"),
            State("custom-x-log-switch", "value"),
            State("custom-y-log-switch", "value"),
            State("custom-x-autoscale-switch", "value"),
            State("custom-y-autoscale-switch", "value"),
            State("custom-plot-type", "value"),
            State("custom-label-by", "value"),
            State("custom-group-by", "value"),
            State("custom-plot-title", "value"),
            State("custom-x-label", "value"),
            State("custom-y-label", "value"),
            State("plot-state-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def create_custom_plot(
        n_clicks,
        x_metric,
        x_stat,
        y_metric,
        y_stat,
        x_log,
        y_log,
        x_autoscale,
        y_autoscale,
        plot_type,
        label_by,
        group_by,
        custom_title,
        custom_x_label,
        custom_y_label,
        plot_state,
    ):
        """Create custom plot and add to state."""
        if not n_clicks or not x_metric or not y_metric:
            raise PreventUpdate

        # Use defaults if not selected
        x_stat = x_stat or "p50"
        y_stat = y_stat or "avg"

        # Convert switch booleans to string format
        if x_log and y_log:
            log_scale = "both"
        elif x_log:
            log_scale = "x"
        elif y_log:
            log_scale = "y"
        else:
            log_scale = "none"

        if x_autoscale and y_autoscale:
            autoscale = "both"
        elif x_autoscale:
            autoscale = "x"
        elif y_autoscale:
            autoscale = "y"
        else:
            autoscale = "none"
        plot_type = plot_type or "scatter_line"
        label_by = label_by or "concurrency"
        group_by = group_by or "model"

        # Generate unique ID for custom plot
        timestamp = int(time.time() * 1000)
        custom_id = (
            f"custom-{timestamp}-{plot_type}-{x_metric}-{x_stat}-vs-{y_metric}-{y_stat}"
        )

        # Use custom title if provided, otherwise auto-generate
        title = (
            custom_title.strip()
            if custom_title and custom_title.strip()
            else f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )

        # Add to plot_configs (unified storage for all plots)
        plot_configs = {**plot_state.get("plot_configs", {})}
        plot_configs[custom_id] = {
            "mode": "multi_run",
            "x_metric": x_metric,
            "x_stat": x_stat,
            "y_metric": y_metric,
            "y_stat": y_stat,
            "log_scale": log_scale,
            "autoscale": autoscale,
            "is_default": False,
            "plot_type": plot_type,
            "size": "half",
            "label_by": label_by,
            "group_by": group_by,
            "title": title,
            "x_label": custom_x_label.strip() if custom_x_label else "",
            "y_label": custom_y_label.strip() if custom_y_label else "",
        }

        # Add to visible plots (create new list to trigger state change)
        visible = list(plot_state.get("visible_plots", []))
        if custom_id not in visible:
            visible.append(custom_id)

        # Create NEW container objects to trigger Dash change detection
        new_state = {
            "visible_plots": visible,
            "hidden_plots": list(plot_state.get("hidden_plots", [])),
            "plot_configs": dict(plot_configs),
            "config_version": plot_state.get("config_version"),
        }

        return new_state, False


def register_single_run_custom_plot_callbacks(
    app: dash.Dash,
    runs: list[RunData],
    mode: VisualizationMode,
):
    """
    Register callbacks for single-run custom plot creation.

    Handles:
    - Modal field visibility based on plot type
    - Modal open/close
    - Custom plot creation for single-run mode

    Args:
        app: Dash application instance
        runs: List of RunData objects
        mode: Visualization mode (must be SINGLE_RUN)
    """

    # Field visibility callback
    @app.callback(
        [
            Output("single-run-x-axis-container", "style"),
            Output("single-run-stat-container", "style"),
            Output("single-run-y2-container", "style"),
            Output("single-run-y2-label-container", "style"),
            Output("single-run-x-axis", "options"),
            Output("single-run-x-axis", "value"),
        ],
        [Input("single-run-plot-type", "value")],
        [State("plot-state-store", "data")],
        prevent_initial_call=True,
    )
    def update_field_visibility(plot_type, plot_state):
        """Show/hide modal fields based on selected plot type."""
        slice_duration = plot_state.get("slice_duration") if plot_state else None
        config = get_single_run_field_config(plot_type, slice_duration)
        return field_config_to_outputs(config)

    # Y-metric dropdown update callback - switch between request and timeslice metrics
    @app.callback(
        [
            Output("single-run-y-metric", "options"),
            Output("single-run-y-metric", "value"),
        ],
        [Input("single-run-plot-type", "value")],
        [State("single-run-request-metrics-store", "data")],
        prevent_initial_call=True,
    )
    def update_y_metric_options(plot_type, request_metrics):
        """Update Y-metric options based on selected plot type."""
        run = _drill_down_run_cache.get("current", runs[0])
        options = get_single_run_y_metric_options(
            run, plot_type, EXCLUDED_METRIC_COLUMNS, request_metrics
        )
        return options, None

    # Y-axis stat dropdown update callback
    @app.callback(
        [
            Output("single-run-y-stat", "options"),
            Output("single-run-y-stat", "value"),
        ],
        [Input("single-run-y-metric", "value")],
        [State("single-run-metric-stats-store", "data")],
        prevent_initial_call=True,
    )
    def update_y_stat_options(y_metric, metric_stats):
        """Update Y-axis stat options based on selected metric."""
        if not y_metric:
            return [{"label": "Average", "value": "avg"}], None

        # Use drill-down cached run if available, otherwise fall back to runs[0]
        if "current" in _drill_down_run_cache:
            run = _drill_down_run_cache["current"]
        else:
            run = runs[0]

        # Compute metric_stats dynamically if run has per-request data
        if run.requests is not None and not run.requests.empty:
            _, computed_metric_stats = get_single_run_metrics_with_stats(
                list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
            )
            metric_stats = computed_metric_stats

        # Check if this is a timeslice metric (not in request metric_stats)
        if (
            metric_stats
            and y_metric not in metric_stats
            and run.timeslices is not None
            and not run.timeslices.empty
        ):
            metric_data = run.timeslices[run.timeslices["Metric"] == y_metric]
            available_stats = metric_data["Stat"].unique().tolist()
            if available_stats:
                options = [
                    {"label": STAT_LABELS.get(s, s), "value": s}
                    for s in ALL_STAT_KEYS
                    if s in available_stats
                ]
                if options:
                    # Auto-select if only one option
                    value = options[0]["value"] if len(options) == 1 else None
                    return options, value

        if not metric_stats:
            return [{"label": "Average", "value": "avg"}], None

        options = get_stat_options_for_single_run_metric(y_metric, metric_stats)

        # Auto-select if only one option
        value = options[0]["value"] if len(options) == 1 else None
        return options, value

    # Modal open (with reset for fields not controlled by other callbacks)
    @app.callback(
        [
            Output("single-run-custom-plot-modal", "is_open", allow_duplicate=True),
            Output("single-run-plot-type", "value"),
            Output("single-run-stat", "value"),
            Output("single-run-y2-metric", "value"),
            Output("single-run-plot-title", "value", allow_duplicate=True),
            Output("single-run-x-label", "value", allow_duplicate=True),
            Output("single-run-y-label", "value", allow_duplicate=True),
        ],
        Input("add-singlerun-plot-slot", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_single_run_modal(n_clicks):
        """Open single-run modal and reset form fields."""
        if n_clicks and n_clicks > 0:
            return True, None, None, None, "", "", ""
        raise PreventUpdate

    # Modal close
    @app.callback(
        Output("single-run-custom-plot-modal", "is_open", allow_duplicate=True),
        [
            Input("btn-create-single-run-custom-plot", "n_clicks"),
            Input("btn-cancel-single-run-custom-plot", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def close_single_run_modal(create_clicks, cancel_clicks):
        """Close single-run modal on create or cancel."""
        ctx = dash.callback_context
        if ctx.triggered:
            return False
        raise PreventUpdate

    # Create custom plot
    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        [Input("btn-create-single-run-custom-plot", "n_clicks")],
        [
            State("single-run-plot-type", "value"),
            State("single-run-x-axis", "value"),
            State("single-run-y-metric", "value"),
            State("single-run-y-stat", "value"),
            State("single-run-stat", "value"),
            State("single-run-y2-metric", "value"),
            State("single-run-y2-label", "value"),
            State("single-run-plot-title", "value"),
            State("single-run-x-label", "value"),
            State("single-run-y-label", "value"),
            State("single-run-metric-stats-store", "data"),
            State("plot-state-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def create_custom_plot(
        n_clicks,
        plot_type,
        x_axis,
        y_metric,
        y_stat,
        stat,
        y2_metric,
        y2_label,
        custom_title,
        custom_x_label,
        custom_y_label,
        metric_stats,
        current_state,
    ):
        """Create a new custom single-run plot."""
        if not n_clicks or not y_metric:
            raise PreventUpdate

        if plot_type == "dual_axis" and not y2_metric:
            raise PreventUpdate

        # Use drill-down cached run if available to compute metric_stats dynamically
        if "current" in _drill_down_run_cache:
            run = _drill_down_run_cache["current"]
            if run.requests is not None and not run.requests.empty:
                _, metric_stats = get_single_run_metrics_with_stats(
                    list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
                )

        # Generate unique plot ID
        plot_id = f"custom-{plot_type}-{int(time.time() * 1000)}"

        # For timeslice plots, always use "Timeslice" as x-axis
        if plot_type == "timeslice":
            x_axis = "Timeslice"

        # Build plot config using helper
        plot_config = build_single_run_plot_config(
            plot_type=plot_type,
            x_axis=x_axis,
            y_metric=y_metric,
            y_stat=y_stat,
            metric_stats=metric_stats or {},
            y2_metric=y2_metric,
            y2_label=y2_label,
            custom_title=custom_title,
            custom_x_label=custom_x_label,
            custom_y_label=custom_y_label,
            size="half",
            is_default=False,
        )

        # Update state
        visible_plots = current_state.get("visible_plots", [])
        plot_configs = current_state.get("plot_configs", {})

        visible_plots.append(plot_id)
        plot_configs[plot_id] = plot_config

        new_state = {
            **current_state,
            "visible_plots": visible_plots,
            "plot_configs": plot_configs,
            "config_version": int(time.time()),
        }

        return new_state


def register_single_run_plot_edit_callbacks(app: dash.Dash, runs: list):
    """
    Register callbacks for editing existing single-run plots.

    Handles:
    - Open edit modal when settings button clicked
    - Populate modal with current plot config
    - Update existing plot configuration
    - Save as new custom plot

    Args:
        app: Dash application instance
        runs: List of RunData objects
    """

    @app.callback(
        [
            # Single-run modal outputs
            Output("edit-single-run-plot-modal", "is_open", allow_duplicate=True),
            Output("edit-sr-plot-id-store", "data", allow_duplicate=True),
            Output("edit-sr-plot-type", "value", allow_duplicate=True),
            Output("edit-sr-x-axis", "value", allow_duplicate=True),
            Output("edit-sr-y-metric", "options", allow_duplicate=True),
            Output("edit-sr-y-metric", "value", allow_duplicate=True),
            Output("edit-sr-y-stat", "value", allow_duplicate=True),
            Output("edit-sr-stat", "value", allow_duplicate=True),
            Output("edit-sr-y2-metric", "value", allow_duplicate=True),
            Output("edit-sr-y2-label", "value", allow_duplicate=True),
            Output("edit-sr-plot-size", "value", allow_duplicate=True),
            Output("edit-sr-plot-title", "value", allow_duplicate=True),
            Output("edit-sr-x-label", "value", allow_duplicate=True),
            Output("edit-sr-y-label", "value", allow_duplicate=True),
            Output("edit-sr-original-y-metric-store", "data", allow_duplicate=True),
        ],
        Input(
            {"type": "settings-plot-btn", "index": dash.dependencies.ALL}, "n_clicks"
        ),
        [
            State("plot-state-store", "data"),
            State("edit-sr-request-metrics-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def open_plot_edit_modal(n_clicks_list, plot_state, request_metrics):
        """Open plot edit modal with current plot configuration."""

        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Get the plot_id from the button that was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        triggered_value = ctx.triggered[0]["value"]

        # Skip if no actual button click (value is None or 0)
        # This happens when buttons are added/removed from the page
        if not triggered_value or triggered_value == 0:
            raise PreventUpdate

        if not triggered_id or "settings-plot-btn" not in triggered_id:
            raise PreventUpdate

        # Parse the plot_id from triggered ID
        button_id_str = triggered_id.split(".")[0]
        button_id = json.loads(button_id_str)
        plot_id = button_id["index"]

        # Get plot configuration from state
        plot_configs = plot_state.get("plot_configs", {})
        plot_config = plot_configs.get(plot_id, {})

        if not plot_config:
            _logger.warning("No config found for plot_id: {plot_id}")
            raise PreventUpdate

        # Detect if this is a single-run or multi-run plot
        # Check mode field first (for default plots), fallback to x_axis check (for custom plots)
        mode = plot_config.get("mode")
        is_single_run = mode == "single_run" if mode else "x_axis" in plot_config

        if is_single_run:
            # Extract single-run config values
            plot_type_single = plot_config.get("plot_type", "scatter")
            x_axis = plot_config.get("x_axis", "request_number")
            # Use base metric if available (for compound metrics), otherwise use y_metric
            y_metric_single = plot_config.get("y_metric_base") or plot_config.get(
                "y_metric", ""
            )
            y_stat = plot_config.get("y_stat", "avg")
            stat = plot_config.get("stat", "avg")
            y2_metric = plot_config.get("y2_metric", "")
            y2_label = plot_config.get("y2_label", "")
            size = plot_config.get("size", "half")
            title = get_plot_title(plot_id, plot_configs)

            # Compute auto-generated axis labels for single-run
            x_axis_labels = {
                "request_number": "Request Number",
                "timestamp_s": "Time (s)",
                "Timeslice": "Timeslice (s)",
            }
            auto_x_label = x_axis_labels.get(x_axis, x_axis.replace("_", " ").title())
            auto_y_label = y_metric_single.replace("_", " ").title()
            x_label = plot_config.get("x_label") or auto_x_label
            y_label = plot_config.get("y_label") or auto_y_label

            print(
                f"DEBUG: Single-run config - plot_type={plot_type_single}, x_axis={x_axis}, y_metric={y_metric_single}, y_stat={y_stat}, stat={stat}, y2_metric={y2_metric}, size={size}, title={title}"
            )

            # Use drill-down cached run if available, otherwise fall back to runs[0]
            if "current" in _drill_down_run_cache:
                run = _drill_down_run_cache["current"]
            else:
                run = runs[0]

            # Determine y-metric options based on plot type
            if plot_type_single == "timeslice":
                if run.timeslices is not None and not run.timeslices.empty:
                    timeslice_metrics = run.timeslices["Metric"].unique().tolist()
                else:
                    timeslice_metrics = []
                y_metric_options = [
                    {"label": m, "value": m}
                    for m in timeslice_metrics
                    if m != "Timeslice"
                ]
            else:
                # Compute request metrics dynamically from run data
                if run.requests is not None and not run.requests.empty:
                    y_metric_options, _ = get_single_run_metrics_with_stats(
                        list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
                    )
                    # Add GPU metrics if available
                    if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
                        # Filter to only include plottable GPU metrics
                        plottable_gpu_metrics = set(get_gpu_metrics())
                        gpu_metrics = [
                            {
                                "label": get_metric_display_name_with_unit(col),
                                "value": col,
                            }
                            for col in run.gpu_telemetry.columns
                            if col in plottable_gpu_metrics
                        ]
                        if gpu_metrics:
                            y_metric_options.append(
                                {
                                    "label": "── GPU Metrics ──",
                                    "value": "_gpu_divider",
                                    "disabled": True,
                                }
                            )
                            y_metric_options.extend(gpu_metrics)
                else:
                    # Fall back to stored request_metrics
                    y_metric_options = request_metrics or []

            return (
                # Single-run modal (OPEN)
                True,  # is_open
                plot_id,  # plot-id-store
                plot_type_single,  # plot-type
                x_axis,  # x-axis
                y_metric_options,  # y-metric options
                y_metric_single,  # y-metric value
                y_stat,  # y-stat
                stat,  # stat (for timeslice)
                y2_metric,  # y2-metric
                y2_label,  # y2-label
                size,  # plot-size
                title,  # plot-title
                x_label,  # x-label
                y_label,  # y-label
                y_metric_single,  # original y-metric for save-as-new title detection
            )
        else:
            # Defensive: if this is a multi-run plot, let multi-run callback handle it
            print(
                f"DEBUG: Multi-run plot detected for {plot_id} - allowing multi-run callback to handle"
            )
            raise PreventUpdate

    # Y-axis stat dropdown update callback for edit modal
    @app.callback(
        [
            Output("edit-sr-y-stat", "options"),
            Output("edit-sr-y-stat", "value", allow_duplicate=True),
        ],
        [Input("edit-sr-y-metric", "value")],
        [
            State("edit-sr-metric-stats-store", "data"),
            State("edit-sr-y-stat", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_edit_y_stat_options(y_metric, metric_stats, current_y_stat):
        """Update Y-axis stat options based on selected metric in edit modal."""
        if not y_metric or not metric_stats:
            return [{"label": "Average", "value": "avg"}], "avg"

        options = get_stat_options_for_single_run_metric(y_metric, metric_stats)

        # Auto-select if only one option
        if len(options) == 1:
            return options, options[0]["value"]

        # Preserve current value if it's valid for this metric
        if current_y_stat and any(o["value"] == current_y_stat for o in options):
            return options, current_y_stat

        # Default to avg if available, otherwise first option
        default_value = (
            "avg" if any(o["value"] == "avg" for o in options) else options[0]["value"]
        )

        return options, default_value

    @app.callback(
        [
            Output("edit-sr-x-axis-container", "style"),
            Output("edit-sr-stat-container", "style"),
            Output("edit-sr-y2-container", "style"),
            Output("edit-sr-x-axis", "options"),
            Output("edit-sr-x-axis", "value", allow_duplicate=True),
        ],
        [Input("edit-sr-plot-type", "value")],
        [State("plot-state-store", "data"), State("edit-sr-x-axis", "value")],
        prevent_initial_call=True,
    )
    def update_edit_sr_field_visibility(plot_type, plot_state, current_x_axis):
        """Show/hide edit modal fields based on selected single-run plot type."""
        slice_duration = plot_state.get("slice_duration") if plot_state else None
        config = get_single_run_field_config(plot_type, slice_duration)
        return field_config_to_edit_outputs(config, current_x_axis)

    # Edit modal Y-metric dropdown update callback
    @app.callback(
        [
            Output("edit-sr-y-metric", "options", allow_duplicate=True),
            Output("edit-sr-y-metric", "value", allow_duplicate=True),
        ],
        [Input("edit-sr-plot-type", "value")],
        [
            State("edit-sr-request-metrics-store", "data"),
            State("edit-sr-y-metric", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_edit_y_metric_options(plot_type, request_metrics, current_value):
        """Update Y-metric options in edit modal based on selected plot type."""
        run = _drill_down_run_cache.get("current", runs[0])
        options = get_single_run_y_metric_options(
            run, plot_type, EXCLUDED_METRIC_COLUMNS, request_metrics
        )
        default_value = select_metric_value(options, current_value)
        return options, default_value

    @app.callback(
        [
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("edit-single-run-plot-modal", "is_open", allow_duplicate=True),
        ],
        Input("btn-sr-update-plot", "n_clicks"),
        [
            State("edit-sr-plot-id-store", "data"),
            State("edit-sr-plot-type", "value"),
            State("edit-sr-x-axis", "value"),
            State("edit-sr-y-metric", "value"),
            State("edit-sr-y-stat", "value"),
            State("edit-sr-stat", "value"),
            State("edit-sr-y2-metric", "value"),
            State("edit-sr-y2-label", "value"),
            State("edit-sr-plot-size", "value"),
            State("edit-sr-plot-title", "value"),
            State("edit-sr-x-label", "value"),
            State("edit-sr-y-label", "value"),
            State("edit-sr-metric-stats-store", "data"),
            State("plot-state-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_single_run_plot_config(
        n_clicks,
        plot_id,
        plot_type,
        x_axis,
        y_metric,
        y_stat,
        stat,
        y2_metric,
        y2_label,
        size,
        custom_title,
        custom_x_label,
        custom_y_label,
        metric_stats,
        plot_state,
    ):
        """Update existing single-run plot configuration."""
        if not n_clicks or not plot_id or not y_metric:
            raise PreventUpdate

        if plot_type == "dual_axis" and not y2_metric:
            raise PreventUpdate

        # Use drill-down cached run if available to compute metric_stats dynamically
        if "current" in _drill_down_run_cache:
            run = _drill_down_run_cache["current"]
            if run.requests is not None and not run.requests.empty:
                _, metric_stats = get_single_run_metrics_with_stats(
                    list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
                )

        # Get existing state
        old_visible = plot_state.get("visible_plots", [])
        old_hidden = plot_state.get("hidden_plots", [])
        old_configs = plot_state.get("plot_configs", {})
        existing_config = old_configs.get(plot_id, {})

        # For timeslice, use stat param; for others, use y_stat
        effective_stat = stat if plot_type == "timeslice" else y_stat

        # Build config using helper
        new_config = build_single_run_plot_config(
            plot_type=plot_type,
            x_axis=x_axis,
            y_metric=y_metric,
            y_stat=effective_stat,
            metric_stats=metric_stats or {},
            y2_metric=y2_metric,
            y2_label=y2_label,
            custom_title=custom_title,
            custom_x_label=custom_x_label,
            custom_y_label=custom_y_label,
            size=size or "half",
            is_default=existing_config.get("is_default", False),
        )

        # Create new plot_configs dict
        new_plot_configs = {k: v for k, v in old_configs.items()}
        new_plot_configs[plot_id] = new_config

        # Create new state object
        new_state = {
            "visible_plots": list(old_visible),
            "hidden_plots": list(old_hidden),
            "plot_configs": new_plot_configs,
            "config_version": int(time.time()),
        }

        # Invalidate cache for this plot (config changed)
        cache = get_plot_cache()
        cache.invalidate_plot(plot_id)
        _logger.debug("Cache invalidated for plot: {plot_id}")

        return new_state, False

    @app.callback(
        [
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("edit-single-run-plot-modal", "is_open", allow_duplicate=True),
        ],
        Input("btn-sr-save-as-new", "n_clicks"),
        [
            State("edit-sr-plot-type", "value"),
            State("edit-sr-x-axis", "value"),
            State("edit-sr-y-metric", "value"),
            State("edit-sr-y-stat", "value"),
            State("edit-sr-stat", "value"),
            State("edit-sr-y2-metric", "value"),
            State("edit-sr-y2-label", "value"),
            State("edit-sr-plot-size", "value"),
            State("edit-sr-plot-title", "value"),
            State("edit-sr-x-label", "value"),
            State("edit-sr-y-label", "value"),
            State("edit-sr-metric-stats-store", "data"),
            State("plot-state-store", "data"),
            State("edit-sr-original-y-metric-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def save_as_new_single_run_plot(
        n_clicks,
        plot_type,
        x_axis,
        y_metric,
        y_stat,
        stat,
        y2_metric,
        y2_label,
        size,
        custom_title,
        custom_x_label,
        custom_y_label,
        metric_stats,
        plot_state,
        original_y_metric,
    ):
        """Create new single-run plot from edited configuration."""
        if not n_clicks:
            raise PreventUpdate

        # Validate required fields
        if not y_metric:
            raise PreventUpdate

        # Use drill-down cached run if available to compute metric_stats dynamically
        if "current" in _drill_down_run_cache:
            run = _drill_down_run_cache["current"]
            if run.requests is not None and not run.requests.empty:
                _, metric_stats = get_single_run_metrics_with_stats(
                    list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
                )

        # Get existing state
        old_visible = plot_state.get("visible_plots", [])
        old_hidden = plot_state.get("hidden_plots", [])
        old_configs = plot_state.get("plot_configs", {})

        # Generate unique ID with timestamp
        timestamp = int(time.time() * 1000)
        custom_id = f"custom-{timestamp}-{plot_type}-{y_metric}"

        # Handle timeslice vs other plot types differently
        if plot_type == "timeslice":
            # y_metric is already a display name for timeslice
            actual_column = y_metric
            default_title = f"{y_metric} Across Time Slices"
        else:
            # Resolve actual column name for compound metrics (request data)
            actual_column = resolve_single_run_column_name(
                y_metric, y_stat, metric_stats or {}
            )
            # Generate title suffix based on plot type
            title_suffixes = {
                "scatter": "Across Requests",
                "area": "Across Time",
                "request_timeline": "Across Time",
            }
            suffix = title_suffixes.get(plot_type, f"({plot_type})")
            default_title = f"{y_metric.replace('_', ' ').title()} {suffix}"

        # Regenerate title if metric changed (user edited the metric)
        metric_changed = original_y_metric and y_metric != original_y_metric
        if custom_title and custom_title.strip() and not metric_changed:
            title = custom_title.strip()
        else:
            title = default_title

        # Build single-run config
        new_config = {
            "plot_type": plot_type,
            "x_axis": x_axis,
            "y_metric": actual_column,  # Store resolved column name or display name
            "y_metric_base": y_metric,  # Store base metric for editing
            "y_stat": y_stat,  # Store selected stat for editing
            "is_default": False,
            "size": size or "half",
            "title": title,
            "x_label": custom_x_label.strip() if custom_x_label else "",
            "y_label": custom_y_label.strip() if custom_y_label else "",
            "mode": "single_run",
        }

        # Add type-specific fields
        if plot_type == "timeslice":
            new_config["stat"] = stat or "avg"
            new_config["source"] = "timeslices"

        if plot_type == "dual_axis":
            if not y2_metric:
                raise PreventUpdate
            new_config["y2_metric"] = y2_metric
            new_config["y2_label"] = y2_label.strip() if y2_label else ""
            new_config["source"] = "dual"

        # Create new lists and dicts
        new_visible = list(old_visible)
        new_visible.append(custom_id)
        new_configs = {k: v for k, v in old_configs.items()}
        new_configs[custom_id] = new_config

        # Create new state object
        new_state = {
            "visible_plots": new_visible,
            "hidden_plots": list(old_hidden),
            "plot_configs": new_configs,
            "config_version": int(time.time()),
        }

        return new_state, False

    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input("btn-sr-hide-plot", "n_clicks"),
        [State("edit-sr-plot-id-store", "data"), State("plot-state-store", "data")],
        prevent_initial_call=True,
    )
    def hide_single_run_plot_update_state(n_clicks, plot_id, plot_state):
        """Hide single-run plot - update state only."""
        if not n_clicks or not plot_id:
            raise PreventUpdate

        # Move plot from visible to hidden
        visible = list(plot_state.get("visible_plots", []))
        hidden = list(plot_state.get("hidden_plots", []))

        if plot_id in visible:
            visible.remove(plot_id)
            if plot_id not in hidden:
                hidden.append(plot_id)

        # Create new state object
        new_state = {
            "visible_plots": [p for p in visible],
            "hidden_plots": [p for p in hidden],
            "plot_configs": {
                k: dict(v) for k, v in plot_state.get("plot_configs", {}).items()
            },
            "config_version": plot_state.get("config_version"),
        }

        return new_state

    @app.callback(
        Output("edit-single-run-plot-modal", "is_open", allow_duplicate=True),
        Input("btn-sr-hide-plot", "n_clicks"),
        prevent_initial_call=True,
    )
    def hide_single_run_plot_close_modal(n_clicks):
        """Hide single-run plot - close modal after state update."""
        if n_clicks:
            return False
        raise PreventUpdate

    @app.callback(
        Output("edit-single-run-plot-modal", "is_open", allow_duplicate=True),
        Input("btn-sr-cancel-edit", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_edit_single_run_plot_modal(n_clicks):
        """Close single-run edit plot modal on cancel."""
        if n_clicks and n_clicks > 0:
            return False
        raise PreventUpdate


def register_multi_run_plot_edit_callbacks(app: dash.Dash, runs: list):
    """
    Register callbacks for editing existing multi-run plots.

    Handles:
    - Open edit modal when settings button clicked
    - Populate modal with current plot config
    - Update existing plot configuration
    - Save as new custom plot

    Args:
        app: Dash application instance
        runs: List of RunData objects
    """

    @app.callback(
        [
            # Multi-run modal outputs
            Output("edit-plot-modal", "is_open", allow_duplicate=True),
            Output("edit-plot-id-store", "data", allow_duplicate=True),
            Output("edit-x-metric", "value", allow_duplicate=True),
            Output("edit-x-stat", "value", allow_duplicate=True),
            Output("edit-y-metric", "value", allow_duplicate=True),
            Output("edit-y-stat", "value", allow_duplicate=True),
            Output("edit-x-log-switch", "value", allow_duplicate=True),
            Output("edit-y-log-switch", "value", allow_duplicate=True),
            Output("edit-x-autoscale-switch", "value", allow_duplicate=True),
            Output("edit-y-autoscale-switch", "value", allow_duplicate=True),
            Output("edit-plot-type", "value", allow_duplicate=True),
            Output("edit-plot-size", "value", allow_duplicate=True),
            Output("edit-plot-title", "value", allow_duplicate=True),
            Output("edit-x-label", "value", allow_duplicate=True),
            Output("edit-y-label", "value", allow_duplicate=True),
            Output("edit-x-stat-warning", "children"),
            Output("edit-y-stat-warning", "children"),
            Output("edit-original-x-metric-store", "data", allow_duplicate=True),
            Output("edit-original-y-metric-store", "data", allow_duplicate=True),
        ],
        Input(
            {"type": "settings-plot-btn", "index": dash.dependencies.ALL}, "n_clicks"
        ),
        State("plot-state-store", "data"),
        prevent_initial_call=True,
    )
    def open_plot_edit_modal(n_clicks_list, plot_state):
        """Open multi-run plot edit modal with current plot configuration."""
        # Find which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Get the plot_id from the button that was clicked
        triggered_id = ctx.triggered[0]["prop_id"]
        triggered_value = ctx.triggered[0]["value"]

        # Skip if no actual button click (value is None or 0)
        if not triggered_value or triggered_value == 0:
            raise PreventUpdate

        if not triggered_id or "settings-plot-btn" not in triggered_id:
            raise PreventUpdate

        # Parse the plot_id from triggered ID
        button_id_str = triggered_id.split(".")[0]
        button_id = json.loads(button_id_str)
        plot_id = button_id["index"]

        # Get plot configuration from state
        plot_configs = plot_state.get("plot_configs", {})
        plot_config = plot_configs.get(plot_id, {})

        if not plot_config:
            _logger.warning("No config found for plot_id: {plot_id}")
            raise PreventUpdate

        # Detect if this is a single-run or multi-run plot
        # Check mode field first (for default plots), fallback to x_metric check (for custom plots)
        mode = plot_config.get("mode")
        is_multi_run = mode != "single_run" if mode else "x_metric" in plot_config

        if not is_multi_run:
            # Defensive: if this is a single-run plot, let single-run callback handle it
            print(
                f"DEBUG: Single-run plot detected for {plot_id} - allowing single-run callback to handle"
            )
            raise PreventUpdate

        # Extract multi-run config values
        x_metric = plot_config.get("x_metric")
        x_stat = plot_config.get("x_stat", "p50")
        y_metric = plot_config.get("y_metric")
        y_stat = plot_config.get("y_stat", "avg")
        log_scale = plot_config.get("log_scale", "none")
        autoscale = plot_config.get("autoscale", "none")
        plot_type = plot_config.get("plot_type", "scatter_line")
        size = plot_config.get("size", "full" if plot_type == "pareto" else "half")
        title = get_plot_title(plot_id, plot_configs)

        print(
            f"DEBUG: Multi-run config - x={x_metric}:{x_stat}, y={y_metric}:{y_stat}, log={log_scale}, plot_type={plot_type}, size={size}"
        )

        # Calculate actual stats being used (with fallbacks)
        result = runs_to_dataframe(runs, x_metric, x_stat, y_metric, y_stat)
        x_stat_actual = result["x_stat_actual"]
        y_stat_actual = result["y_stat_actual"]

        # Create warning messages if fallback occurred
        x_warning = ""
        y_warning = ""
        if x_stat_actual != x_stat:
            x_warning = f"⚠️ Using {x_stat_actual} ({x_stat} unavailable)"
        if y_stat_actual != y_stat:
            y_warning = f"⚠️ Using {y_stat_actual} ({y_stat} unavailable)"

        # Get axis labels - use custom if set, otherwise prefill with auto-generated
        auto_x_label = f"{get_metric_display_name(x_metric)} ({x_stat_actual})"
        auto_y_label = f"{get_metric_display_name(y_metric)} ({y_stat_actual})"
        x_label = plot_config.get("x_label") or auto_x_label
        y_label = plot_config.get("y_label") or auto_y_label

        print(
            f"DEBUG: Actual stats - x={x_stat_actual}, y={y_stat_actual}, warnings: x='{x_warning}', y='{y_warning}'"
        )

        # Convert log_scale/autoscale strings to individual booleans
        x_log = log_scale in ("x", "both")
        y_log = log_scale in ("y", "both")
        x_autoscale = autoscale in ("x", "both")
        y_autoscale = autoscale in ("y", "both")

        return (
            # Multi-run modal (OPEN)
            True,  # is_open
            plot_id,  # edit-plot-id-store
            x_metric,  # x-metric
            x_stat_actual,  # x-stat
            y_metric,  # y-metric
            y_stat_actual,  # y-stat
            x_log,  # x-log-switch
            y_log,  # y-log-switch
            x_autoscale,  # x-autoscale-switch
            y_autoscale,  # y-autoscale-switch
            plot_type,  # plot-type
            size,  # plot-size
            title,  # plot-title
            x_label,  # x-label
            y_label,  # y-label
            x_warning,  # x-stat-warning
            y_warning,  # y-stat-warning
            x_metric,  # original x-metric for save-as-new title detection
            y_metric,  # original y-metric for save-as-new title detection
        )

    @app.callback(
        Output("edit-plot-modal", "is_open", allow_duplicate=True),
        Input("btn-cancel-edit-plot", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_edit_plot_modal(n_clicks):
        """Close edit plot modal on cancel."""
        if n_clicks and n_clicks > 0:
            return False
        raise PreventUpdate

    @app.callback(
        [
            Output("edit-x-stat", "options"),
            Output("edit-x-stat", "value", allow_duplicate=True),
            Output("edit-y-stat", "options"),
            Output("edit-y-stat", "value", allow_duplicate=True),
        ],
        [
            Input("edit-x-metric", "value"),
            Input("edit-y-metric", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_edit_stat_options(x_metric, y_metric):
        """Filter stat dropdown options based on available stats for selected metrics."""
        all_stat_options = MULTI_RUN_STAT_OPTIONS

        # Get available stats for each metric
        x_available_stats = (
            get_available_stats_for_metric(runs, x_metric) if x_metric else []
        )
        y_available_stats = (
            get_available_stats_for_metric(runs, y_metric) if y_metric else []
        )

        # Filter options to only those available
        x_options = (
            [opt for opt in all_stat_options if opt["value"] in x_available_stats]
            if x_available_stats
            else all_stat_options
        )
        y_options = (
            [opt for opt in all_stat_options if opt["value"] in y_available_stats]
            if y_available_stats
            else all_stat_options
        )

        # Auto-select if only one option, otherwise preserve user selection
        x_value = x_options[0]["value"] if len(x_options) == 1 else dash.no_update
        y_value = y_options[0]["value"] if len(y_options) == 1 else dash.no_update

        return x_options, x_value, y_options, y_value

    @app.callback(
        [
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("edit-plot-modal", "is_open", allow_duplicate=True),
        ],
        Input("btn-update-plot", "n_clicks"),
        [
            State("edit-plot-id-store", "data"),
            State("edit-x-metric", "value"),
            State("edit-x-stat", "value"),
            State("edit-y-metric", "value"),
            State("edit-y-stat", "value"),
            State("edit-x-log-switch", "value"),
            State("edit-y-log-switch", "value"),
            State("edit-x-autoscale-switch", "value"),
            State("edit-y-autoscale-switch", "value"),
            State("edit-plot-type", "value"),
            State("edit-plot-size", "value"),
            State("edit-plot-title", "value"),
            State("edit-x-label", "value"),
            State("edit-y-label", "value"),
            State("plot-state-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_plot_config(
        n_clicks,
        plot_id,
        x_metric,
        x_stat,
        y_metric,
        y_stat,
        x_log,
        y_log,
        x_autoscale,
        y_autoscale,
        plot_type,
        size,
        custom_title,
        custom_x_label,
        custom_y_label,
        plot_state,
    ):
        """Update existing multi-run plot configuration."""
        _logger.debug("UPDATE_PLOT_CONFIG CALLBACK FIRED")
        _logger.debug("plot_id: {plot_id}")
        _logger.debug("n_clicks: {n_clicks}")

        if not n_clicks or not plot_id:
            _logger.debug("PreventUpdate - Missing required params")
            raise PreventUpdate

        # Validate multi-run fields
        if not x_metric or not y_metric:
            _logger.debug("PreventUpdate - Missing x_metric or y_metric")
            raise PreventUpdate

        # Get existing state
        old_visible = plot_state.get("visible_plots", [])
        old_hidden = plot_state.get("hidden_plots", [])
        old_configs = plot_state.get("plot_configs", {})
        existing_config = old_configs.get(plot_id, {})

        # Use custom title if provided
        if custom_title and custom_title.strip():
            new_title = custom_title.strip()
        else:
            new_title = f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"

        # Convert switch booleans to string format
        if x_log and y_log:
            log_scale = "both"
        elif x_log:
            log_scale = "x"
        elif y_log:
            log_scale = "y"
        else:
            log_scale = "none"

        if x_autoscale and y_autoscale:
            autoscale = "both"
        elif x_autoscale:
            autoscale = "x"
        elif y_autoscale:
            autoscale = "y"
        else:
            autoscale = "none"

        # Build multi-run config
        new_config = {
            "x_metric": x_metric,
            "x_stat": x_stat or "p50",
            "y_metric": y_metric,
            "y_stat": y_stat or "avg",
            "log_scale": log_scale,
            "autoscale": autoscale,
            "plot_type": plot_type or "scatter_line",
            "size": size or "half",
            "title": new_title,
            "x_label": custom_x_label.strip() if custom_x_label else "",
            "y_label": custom_y_label.strip() if custom_y_label else "",
            "label_by": existing_config.get("label_by", "concurrency"),
            "group_by": existing_config.get("group_by", "model"),
            "is_default": existing_config.get("is_default", False),
        }

        _logger.debug("Multi-run config: {new_config}")

        # Create completely NEW plot_configs dict
        new_plot_configs = {k: v for k, v in old_configs.items()}
        new_plot_configs[plot_id] = new_config

        # Create NEW container objects to trigger Dash change detection
        new_state = {
            "visible_plots": list(old_visible),
            "hidden_plots": list(old_hidden),
            "plot_configs": new_plot_configs,
            "config_version": plot_state.get("config_version"),
        }

        _logger.debug("Updated config for {plot_id}: {new_config}")

        # Invalidate cache for this plot (config changed)
        cache = get_plot_cache()
        cache.invalidate_plot(plot_id)
        _logger.debug("Cache invalidated for plot: {plot_id}")
        _logger.debug("Returning new state and closing modal")
        return new_state, False

    @app.callback(
        [
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("edit-plot-modal", "is_open", allow_duplicate=True),
        ],
        Input("btn-save-as-new-plot", "n_clicks"),
        [
            State("edit-x-metric", "value"),
            State("edit-x-stat", "value"),
            State("edit-y-metric", "value"),
            State("edit-y-stat", "value"),
            State("edit-x-log-switch", "value"),
            State("edit-y-log-switch", "value"),
            State("edit-x-autoscale-switch", "value"),
            State("edit-y-autoscale-switch", "value"),
            State("edit-plot-title", "value"),
            State("edit-x-label", "value"),
            State("edit-y-label", "value"),
            State("edit-plot-size", "value"),
            State("plot-state-store", "data"),
            State("edit-original-x-metric-store", "data"),
            State("edit-original-y-metric-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def save_as_new_plot(
        n_clicks,
        x_metric,
        x_stat,
        y_metric,
        y_stat,
        x_log,
        y_log,
        x_autoscale,
        y_autoscale,
        custom_title,
        custom_x_label,
        custom_y_label,
        size,
        plot_state,
        original_x_metric,
        original_y_metric,
    ):
        """Create new multi-run custom plot from edited configuration."""
        _logger.debug("SAVE_AS_NEW_PLOT CALLBACK FIRED")
        _logger.debug("n_clicks: {n_clicks}")

        if not n_clicks:
            _logger.debug("PreventUpdate - No click")
            raise PreventUpdate

        # Validate multi-run fields
        if not x_metric or not y_metric:
            _logger.debug("PreventUpdate - Missing x_metric or y_metric")
            raise PreventUpdate

        # Get existing state
        old_visible = plot_state.get("visible_plots", [])
        old_hidden = plot_state.get("hidden_plots", [])
        old_configs = plot_state.get("plot_configs", {})

        # Generate UNIQUE ID with timestamp
        timestamp = int(time.time() * 1000)
        custom_id = f"custom-{timestamp}-{x_metric}-{x_stat}-vs-{y_metric}-{y_stat}"
        _logger.debug("Generated UNIQUE custom_id: {custom_id}")

        # Regenerate title if metrics changed (user edited the metrics)
        metrics_changed = (original_x_metric and x_metric != original_x_metric) or (
            original_y_metric and y_metric != original_y_metric
        )
        if custom_title and custom_title.strip() and not metrics_changed:
            title = custom_title.strip()
        else:
            title = f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"

        # Convert switch booleans to string format
        if x_log and y_log:
            log_scale = "both"
        elif x_log:
            log_scale = "x"
        elif y_log:
            log_scale = "y"
        else:
            log_scale = "none"

        if x_autoscale and y_autoscale:
            autoscale = "both"
        elif x_autoscale:
            autoscale = "x"
        elif y_autoscale:
            autoscale = "y"
        else:
            autoscale = "none"

        # Build multi-run config
        new_config = {
            "mode": "multi_run",
            "x_metric": x_metric,
            "x_stat": x_stat or "p50",
            "y_metric": y_metric,
            "y_stat": y_stat or "avg",
            "log_scale": log_scale,
            "autoscale": autoscale,
            "is_default": False,
            "plot_type": "scatter_line",
            "label_by": "concurrency",
            "group_by": "model",
            "size": size or "half",
            "title": title,
            "x_label": custom_x_label.strip() if custom_x_label else "",
            "y_label": custom_y_label.strip() if custom_y_label else "",
        }

        _logger.debug("Multi-run config: {new_config}")

        # Create completely NEW plot_configs dict
        new_plot_configs = {k: v for k, v in old_configs.items()}
        new_plot_configs[custom_id] = new_config

        # Add to visible plots (always add since ID is unique)
        visible = list(old_visible)
        visible.append(custom_id)

        # Create NEW container objects to trigger Dash change detection
        new_state = {
            "visible_plots": visible,
            "hidden_plots": list(old_hidden),
            "plot_configs": new_plot_configs,
            "config_version": plot_state.get("config_version"),
        }

        _logger.debug("New state object IDs:")
        print(
            f"     visible_plots: {id(new_state['visible_plots'])} {'✅ NEW' if id(new_state['visible_plots']) != id(old_visible) else '❌ SAME'} - count: {len(new_state['visible_plots'])}"
        )
        print(
            f"     hidden_plots: {id(new_state['hidden_plots'])} {'✅ NEW' if id(new_state['hidden_plots']) != id(old_hidden) else '❌ SAME'}"
        )
        print(
            f"     plot_configs: {id(new_state['plot_configs'])} {'✅ NEW' if id(new_state['plot_configs']) != id(old_configs) else '❌ SAME'} - count: {len(new_state['plot_configs'])}"
        )
        _logger.debug("Added plot to visible_plots: {custom_id}")
        _logger.debug("visible_plots content: {new_state['visible_plots']}")
        _logger.debug("Returning new state and closing modal")
        return new_state, False

    # Callback 1: Update state only
    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input("btn-hide-plot-from-modal", "n_clicks"),
        [State("edit-plot-id-store", "data"), State("plot-state-store", "data")],
        prevent_initial_call=True,
    )
    def hide_plot_update_state(n_clicks, plot_id, plot_state):
        """Hide plot - update state only."""
        _logger.debug("HIDE_PLOT_UPDATE_STATE CALLBACK FIRED")
        _logger.debug("plot_id: {plot_id}")
        print(f"n_clicks: {n_clicks}")

        if not n_clicks or not plot_id:
            _logger.debug("Missing n_clicks or plot_id - PreventUpdate")
        raise PreventUpdate

        # Move plot from visible to hidden
        visible = list(plot_state.get("visible_plots", []))
        hidden = list(plot_state.get("hidden_plots", []))
        _logger.debug("Before: visible={visible}, hidden={hidden}")

        if plot_id in visible:
            visible.remove(plot_id)
            if plot_id not in hidden:
                hidden.append(plot_id)
            _logger.debug("Removed {plot_id} from visible, added to hidden")
        else:
            _logger.warning("plot_id '{plot_id}' not found in visible list!")

        _logger.debug("After: visible={visible}, hidden={hidden}")

        # Create completely NEW objects
        new_state = {
            "visible_plots": [p for p in visible],
            "hidden_plots": [p for p in hidden],
            "plot_configs": {
                k: dict(v) for k, v in plot_state.get("plot_configs", {}).items()
            },
            "config_version": plot_state.get("config_version"),
        }

        _logger.debug(
            f"Returning updated state: {len(visible)} visible, {len(hidden)} hidden"
        )
        return new_state

    # Callback 2: Close modal only
    @app.callback(
        Output("edit-plot-modal", "is_open", allow_duplicate=True),
        Input("btn-hide-plot-from-modal", "n_clicks"),
        prevent_initial_call=True,
    )
    def hide_plot_close_modal(n_clicks):
        """Hide plot - close modal after state update."""
        if n_clicks:
            print("🚪 Closing modal after hide")
            return False
        raise PreventUpdate

    # Debug callback to detect which modal button is clicked
    @app.callback(
        Output("edit-plot-id-store", "data", allow_duplicate=True),
        [
            Input("btn-hide-plot-from-modal", "n_clicks"),
            Input("btn-update-plot", "n_clicks"),
            Input("btn-save-as-new-plot", "n_clicks"),
            Input("btn-cancel-edit-plot", "n_clicks"),
        ],
        State("edit-plot-id-store", "data"),
        prevent_initial_call=True,
    )
    def detect_any_modal_button_click(
        hide_clicks, update_clicks, save_clicks, cancel_clicks, plot_id
    ):
        """Debug callback to detect which button was actually clicked."""
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            _logger.debug(f"BUTTON CLICK DETECTED: {button_id}")
            print(
                f"   hide={hide_clicks}, update={update_clicks}, save={save_clicks}, cancel={cancel_clicks}\n"
            )
        raise PreventUpdate  # Don't actually change anything


def register_nested_run_selector_callbacks(app: dash.Dash, runs: list):
    """
    Register callbacks for nested run selector functionality.

    Handles:
    - Group checkbox toggling all runs in a group
    - Individual run selection updating group checkbox
    - Aggregating nested selections into flat run-selector value

    Args:
        app: Dash application instance
        runs: List of RunData objects
    """

    # Callback: Aggregate nested selections into single run-selector value
    @app.callback(
        Output("run-selector", "value", allow_duplicate=True),
        Input({"type": "run-selector-nested", "index": dash.ALL}, "value"),
        prevent_initial_call=True,
    )
    def aggregate_nested_selections(nested_values):
        """Aggregate all nested checklist selections into flat list."""
        if not nested_values:
            raise PreventUpdate

        # Flatten all selected indices from all groups
        all_selected = []
        for group_selections in nested_values:
            if group_selections:
                all_selected.extend(group_selections)

        return sorted(list(set(all_selected)))

    # Callback: When group checkbox clicked, select/deselect all runs in that group
    @app.callback(
        Output({"type": "run-selector-nested", "index": dash.MATCH}, "value"),
        Input({"type": "group-selector", "index": dash.MATCH}, "value"),
        State({"type": "run-selector-nested", "index": dash.MATCH}, "options"),
        prevent_initial_call=True,
    )
    def toggle_group_selection(group_checked, run_options):
        """Select/deselect all runs when group checkbox is clicked."""
        if group_checked:
            # Group is checked - select all runs in group
            return [opt["value"] for opt in run_options]
        else:
            # Group is unchecked - deselect all runs in group
            return []

    # Callback: When individual runs change, update group checkbox accordingly
    @app.callback(
        Output(
            {"type": "group-selector", "index": dash.MATCH},
            "value",
            allow_duplicate=True,
        ),
        Input({"type": "run-selector-nested", "index": dash.MATCH}, "value"),
        State({"type": "run-selector-nested", "index": dash.MATCH}, "options"),
        State({"type": "group-selector", "index": dash.MATCH}, "id"),
        State({"type": "group-selector", "index": dash.MATCH}, "value"),
        prevent_initial_call=True,
    )
    def update_group_checkbox(
        selected_runs, run_options, group_id, current_group_value
    ):
        """Update group checkbox based on selected runs."""
        if not run_options:
            raise PreventUpdate

        all_run_indices = [opt["value"] for opt in run_options]
        selected_runs = selected_runs or []

        # Count how many runs are selected
        num_selected = len([idx for idx in all_run_indices if idx in selected_runs])
        total_runs = len(all_run_indices)

        # Determine group checkbox state
        group_name = group_id["index"]

        if num_selected == 0:
            # No runs selected - uncheck group
            return []
        elif num_selected == total_runs:
            # All runs selected - check group
            return [group_name]
        else:
            # Partial selection (INDETERMINATE STATE)
            # Keep current group checkbox state to avoid triggering toggle_group_selection
            raise PreventUpdate

    # Callback: Toggle run group collapsible sections
    @app.callback(
        Output({"type": "run-group-content", "index": dash.MATCH}, "style"),
        Output({"type": "run-group-arrow", "index": dash.MATCH}, "children"),
        Input({"type": "run-group-header", "index": dash.MATCH}, "n_clicks"),
        State({"type": "run-group-content", "index": dash.MATCH}, "style"),
        prevent_initial_call=True,
    )
    def toggle_run_group(n_clicks, current_style):
        """Toggle visibility of run group collapsible sections."""
        if not n_clicks:
            raise PreventUpdate

        is_visible = current_style.get("display", "none") == "block"
        new_style = {"display": "none" if is_visible else "block"}
        new_arrow = "▶" if is_visible else "▼"

        return new_style, new_arrow


def register_resize_toggle_callbacks(app: dash.Dash):
    """
    Register callback for toggling plot size between half and full.

    Simple click-to-toggle resize functionality (no drag tracking).

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output("plot-state-store", "data", allow_duplicate=True),
        Input({"type": "resize-handle", "index": dash.dependencies.ALL}, "n_clicks"),
        State("plot-state-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_plot_size(n_clicks_list, plot_state):
        """Toggle plot size between half and full when resize handle is clicked."""
        if not ctx.triggered:
            raise PreventUpdate

        # Find which resize handle was clicked
        triggered_id = ctx.triggered_id
        if not triggered_id or not isinstance(triggered_id, dict):
            raise PreventUpdate

        plot_id = triggered_id.get("index")
        if not plot_id:
            raise PreventUpdate

        # Check if this was an actual click (n_clicks > 0)
        # This prevents firing on initial render or component recreation
        trigger_value = ctx.triggered[0].get("value")
        if not trigger_value or trigger_value == 0:
            raise PreventUpdate

        # Get current plot configs
        plot_configs = {**plot_state.get("plot_configs", {})}

        # Toggle the size for the clicked plot
        if plot_id in plot_configs:
            current_size = plot_configs[plot_id].get("size", "half")
            new_size = "full" if current_size == "half" else "half"

            plot_configs[plot_id] = {
                **plot_configs[plot_id],
                "size": new_size,
            }

        # Return updated state
        new_state = {
            "visible_plots": plot_state.get("visible_plots", []),
            "hidden_plots": plot_state.get("hidden_plots", []),
            "plot_configs": plot_configs,
            "config_version": plot_state.get("config_version"),
        }

        return new_state


def register_context_menu_callbacks(app: dash.Dash):
    """
    Register callbacks for right-click context menu on plots.

    Handles:
    - Showing context menu on right-click
    - Hiding context menu on any click
    - Context menu actions (Edit, Hide, Remove)

    Args:
        app: Dash application instance
    """

    # Clientside callback to handle right-click and show/hide context menu
    app.clientside_callback(
        """
        function(n_intervals) {
            // Add right-click event listeners to all plot containers
            const containers = document.querySelectorAll('.plot-container');
            const contextMenu = document.getElementById('plot-context-menu');

            if (!contextMenu) return window.dash_clientside.no_update;

            containers.forEach(container => {
                // Remove existing listeners to avoid duplicates
                container.oncontextmenu = function(e) {
                    e.preventDefault();

                    // Get plot ID from container ID
                    const plotId = container.id.replace('container-', '');

                    // Position and show context menu
                    contextMenu.style.left = e.pageX + 'px';
                    contextMenu.style.top = e.pageY + 'px';
                    contextMenu.style.display = 'block';

                    // Store plot ID
                    window.contextMenuPlotId = plotId;

                    return false;
                };
            });

            // Hide context menu on any left click
            document.onclick = function(e) {
                if (contextMenu && !contextMenu.contains(e.target)) {
                    contextMenu.style.display = 'none';
                }
            };

            return window.dash_clientside.no_update;
        }
        """,
        Output("plot-context-menu", "n_clicks", allow_duplicate=True),
        Input("plots-grid", "children"),
        prevent_initial_call=True,
    )

    # Context menu "Edit" action - opens edit modal
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            const plotId = window.contextMenuPlotId;
            if (!plotId) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            // Hide context menu
            const contextMenu = document.getElementById('plot-context-menu');
            if (contextMenu) contextMenu.style.display = 'none';

            // Trigger settings button click for this plot
            const settingsBtn = document.querySelector(`[id*='settings-plot-btn'][id*='${plotId}']`);
            if (settingsBtn) {
                settingsBtn.click();
            }

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        [
            Output("context-menu-plot-id", "data", allow_duplicate=True),
            Output("plot-context-menu", "style", allow_duplicate=True),
        ],
        Input("context-menu-edit", "n_clicks"),
        prevent_initial_call=True,
    )

    # Context menu "Hide" action
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            const plotId = window.contextMenuPlotId;
            if (!plotId) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            // Hide context menu
            const contextMenu = document.getElementById('plot-context-menu');
            if (contextMenu) contextMenu.style.display = 'none';

            // Trigger remove button click for this plot
            const removeBtn = document.querySelector(`[id*='remove-plot-btn'][id*='${plotId}']`);
            if (removeBtn) {
                removeBtn.click();
            }

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        [
            Output("context-menu-plot-id", "data", allow_duplicate=True),
            Output("plot-context-menu", "style", allow_duplicate=True),
        ],
        Input("context-menu-hide", "n_clicks"),
        prevent_initial_call=True,
    )

    # Context menu "Remove" action (for custom plots only)
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            const plotId = window.contextMenuPlotId;
            if (!plotId) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            // Hide context menu
            const contextMenu = document.getElementById('plot-context-menu');
            if (contextMenu) contextMenu.style.display = 'none';

            // Only allow removal of custom plots
            if (plotId.startsWith('custom-')) {
                // Trigger remove button click
                const removeBtn = document.querySelector(`[id*='remove-plot-btn'][id*='${plotId}']`);
                if (removeBtn) {
                    removeBtn.click();
                }
            } else {
                alert('Cannot remove default plots. Use "Hide" instead.');
            }

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        [
            Output("context-menu-plot-id", "data", allow_duplicate=True),
            Output("plot-context-menu", "style", allow_duplicate=True),
        ],
        Input("context-menu-remove", "n_clicks"),
        prevent_initial_call=True,
    )


def register_layout_theme_callbacks(app: dash.Dash):
    """Register callbacks to update header, sidebar, and main area when theme changes."""

    @app.callback(
        [
            Output("dashboard-header", "style"),
            Output("header-title", "style"),
            Output("dashboard-sidebar", "style"),
            Output("dashboard-main-area", "style"),
            Output("run-selector-container", "style"),
        ],
        [Input("theme-store", "data")],
    )
    def update_layout_theme(theme_data):
        """Update header, sidebar, and main area styles when theme changes."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        # Get theme-appropriate styles
        header_style = get_header_style(current_theme)
        sidebar_style = get_sidebar_style(current_theme)
        main_area_style = get_main_area_style(current_theme)

        # Header title color
        title_color = NVIDIA_WHITE if current_theme == PlotTheme.DARK else NVIDIA_DARK
        header_title_style = {
            "margin": "0",
            "font-size": "22px",
            "font-weight": "600",
            "letter-spacing": "-0.5px",
            "font-family": PLOT_FONT_FAMILY,
            "color": title_color,
        }

        # Run selector container style
        run_selector_style = {
            "max-height": "200px",
            "overflow-y": "auto",
            "padding": "8px",
            "background": colors["paper"],
            "border-radius": "6px",
            "margin-bottom": "12px",
        }

        return (
            header_style,
            header_title_style,
            sidebar_style,
            main_area_style,
            run_selector_style,
        )

    @app.callback(
        Output("app-root-container", "className"),
        [Input("theme-store", "data")],
    )
    def update_theme_class(theme_data):
        """Update root container theme class when theme changes."""
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        class_name = f"theme-{current_theme.value}"
        return class_name


def _is_single_stat_metric(metric) -> bool:
    """
    Check if metric only has 'avg' stat (no distribution stats like p50, std, etc.).

    Single-stat metrics are derived values (like throughput, count) where the aggregated
    "avg" is a calculated value (total/duration), not a statistical average of samples.

    Args:
        metric: MetricResult object or dict containing metric data

    Returns:
        True if metric only has 'avg' stat, False otherwise
    """
    distribution_stats = {
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "std",
        "min",
        "max",
    }

    # Check if any non-None distribution stat exists
    for stat in distribution_stats:
        if hasattr(metric, stat):
            val = getattr(metric, stat)
        elif isinstance(metric, dict):
            val = metric.get(stat)
        else:
            continue
        if val is not None:
            return False

    return True


def _get_stat_for_timeslice_metric(
    metric_display_name: str, run: "RunData", stat: str = "avg"
) -> tuple[float | None, str | None, float | None]:
    """
    Get stat value and std for a timeslice metric from aggregated stats.

    Args:
        metric_display_name: Display name of the metric (e.g., "Time to First Token")
        run: RunData object containing aggregated stats
        stat: Statistic to extract (e.g., "avg", "p25", "p90")

    Returns:
        Tuple of (stat_value, formatted_label, std_value) or (None, None, None) if not found
    """
    # Skip reference line for cumulative metrics (aggregated value is sum, not comparable)
    metric_lower = metric_display_name.lower()
    if any(pattern in metric_lower for pattern in CUMULATIVE_METRIC_PATTERNS):
        return None, None, None

    display_to_tag = {v: k for k, v in get_all_metric_display_names().items()}
    metric_tag = display_to_tag.get(metric_display_name)
    if metric_tag is None:
        return None, None, None

    metric = run.get_metric(metric_tag)
    if not metric:
        return None, None, None

    # Skip reference line for single-stat metrics (derived values like throughput, count)
    # These only have "avg" because they're calculated values (total/duration),
    # not per-request measurements with distributions
    if _is_single_stat_metric(metric):
        return None, None, None

    stat_value = (
        getattr(metric, stat, None) if hasattr(metric, stat) else metric.get(stat)
    )
    unit = metric.unit if hasattr(metric, "unit") else metric.get("unit", "")
    std = metric.std if hasattr(metric, "std") else metric.get("std")

    if stat_value is None:
        return None, None, None

    stat_display = STAT_LABELS.get(stat, stat)
    label = f"Run {stat_display}: {stat_value:.2f}"
    if unit:
        label += f" {unit}"

    return stat_value, label, std


def _generate_custom_single_run_plot(
    config: dict,
    run: RunData,
    plot_gen,
    theme: PlotTheme,
):
    """Generate custom plot from user config for single-run mode.

    Args:
        config: Plot configuration dict
        run: RunData object
        plot_gen: Plot generator instance
        theme: Current theme

    Returns:
        Plotly figure or None
    """

    plot_type = config["plot_type"]
    y_metric = config["y_metric"]
    title = config.get("title", "Custom Plot")

    try:
        if plot_type == "scatter":
            if run.requests is None or run.requests.empty:
                return _create_empty_figure(
                    "Scatter plot cannot be generated: no per-request data available.\n"
                    "Per-request data is generated during benchmark runs.",
                    theme,
                )
            # Check if requested column exists
            if y_metric not in run.requests.columns:
                return _create_empty_figure(
                    f"Column '{y_metric}' not available in per-request data.\n"
                    "This metric may not have been captured during the benchmark run.",
                    theme,
                )
            df, x_col = prepare_timeseries_dataframe(run.requests)
            return plot_gen.create_time_series_scatter(
                df=df, x_col=x_col, y_metric=y_metric, title=title
            )

        elif plot_type == "area":
            if run.requests is None or run.requests.empty:
                return _create_empty_figure(
                    "Area plot cannot be generated: no per-request data available.\n"
                    "Per-request data is generated during benchmark runs.",
                    theme,
                )
            df, x_col = prepare_timeseries_dataframe(run.requests)
            return plot_gen.create_time_series_area(
                df=df, x_col=x_col, y_metric=y_metric, title=title
            )

        elif plot_type == "timeslice":
            if run.timeslices is None or run.timeslices.empty:
                return _create_empty_figure(
                    "Timeslice plot cannot be generated: no timeslice data available.\n"
                    "Timeslice data requires running benchmarks with slice_duration configured.",
                    theme,
                )

            stat = config.get("stat") or "avg"

            try:
                # Prepare timeslice data using prepare_timeslice_metrics
                # y_metric is a display name (e.g., "Time to First Token")
                # Extract user-selected stat plus std for error bars (if different)
                stats_to_extract = [stat]
                if stat != "std":
                    stats_to_extract.append("std")
                plot_df, unit = prepare_timeslice_metrics(
                    run, y_metric, stats_to_extract
                )

                if plot_df.empty:
                    return _create_empty_figure(
                        f"No timeslice data for metric '{y_metric}'", theme
                    )

                y_label = f"{y_metric} ({unit})" if unit else y_metric

                # Extract run stat value and std for legend and outlier detection
                average_value, average_label, average_std = (
                    _get_stat_for_timeslice_metric(y_metric, run, stat)
                )

                # Create timeslice scatter plot
                return plot_gen.create_timeslice_scatter(
                    df=plot_df,
                    x_col="Timeslice",
                    y_col=stat,
                    metric_name=y_metric,
                    title=title,
                    y_label=y_label,
                    slice_duration=run.slice_duration,
                    unit=unit,
                    average_value=average_value,
                    average_label=average_label,
                    average_std=average_std,
                )
            except Exception as e:
                _logger.error(f"Error creating timeslice scatter plot: {e}")
                return _create_empty_figure(
                    f"Error preparing timeslice data: {str(e)}", theme
                )

        elif plot_type == "dual_axis":
            y2_metric = config.get("y2_metric")
            if not y2_metric:
                return _create_empty_figure(
                    "Secondary Y-axis metric not specified", theme
                )

            requests_cols = (
                set(run.requests.columns) if run.requests is not None else set()
            )
            gpu_cols = (
                set(run.gpu_telemetry.columns)
                if run.gpu_telemetry is not None
                else set()
            )

            y1_is_gpu = y_metric in gpu_cols and y_metric not in requests_cols
            y2_is_gpu = y2_metric in gpu_cols and y2_metric not in requests_cols

            if y1_is_gpu and run.gpu_telemetry is None:
                return _create_empty_figure(
                    f"Dual-axis plot cannot be generated: no GPU telemetry data for {y_metric}.\n"
                    "GPU telemetry requires DCGM to be configured during benchmark runs.",
                    theme,
                )
            if y2_is_gpu and run.gpu_telemetry is None:
                return _create_empty_figure(
                    f"Dual-axis plot cannot be generated: no GPU telemetry data for {y2_metric}.\n"
                    "GPU telemetry requires DCGM to be configured during benchmark runs.",
                    theme,
                )
            if not y1_is_gpu and (run.requests is None or run.requests.empty):
                return _create_empty_figure(
                    f"Dual-axis plot cannot be generated: no per-request data for {y_metric}.\n"
                    "Per-request data is generated during benchmark runs.",
                    theme,
                )
            if not y2_is_gpu and (run.requests is None or run.requests.empty):
                return _create_empty_figure(
                    f"Dual-axis plot cannot be generated: no per-request data for {y2_metric}.\n"
                    "Per-request data is generated during benchmark runs.",
                    theme,
                )

            if y1_is_gpu:
                df_primary = run.gpu_telemetry
                x_col_primary = "timestamp_s"
            else:
                df_primary, x_col_primary = prepare_timeseries_dataframe(run.requests)

            if y2_is_gpu:
                df_secondary = run.gpu_telemetry
                x_col_secondary = "timestamp_s"
            else:
                df_secondary, x_col_secondary = prepare_timeseries_dataframe(
                    run.requests
                )

            return plot_gen.create_dual_axis_plot(
                df_primary=df_primary,
                df_secondary=df_secondary,
                x_col_primary=x_col_primary,
                x_col_secondary=x_col_secondary,
                y1_metric=y_metric,
                y2_metric=y2_metric,
                title=title,
            )

        elif plot_type == "request_timeline":
            if run.requests is None or run.requests.empty:
                return _create_empty_figure(
                    "Request timeline plot cannot be generated: no per-request data available.\n"
                    "Per-request data is generated during benchmark runs.",
                    theme,
                )

            df = run.requests.copy()
            required_cols = [
                "request_start_ns",
                "request_end_ns",
                "time_to_first_token",
                y_metric,
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return _create_empty_figure(
                    f"Request timeline plot cannot be generated: missing columns {missing_cols}.\n"
                    "Request timing data may not have been captured during the benchmark.",
                    theme,
                )

            df = df.dropna(subset=required_cols)
            if df.empty:
                return _create_empty_figure(
                    "Request timeline plot cannot be generated: no valid data after removing NaN values.\n"
                    "The required columns contain null values for all requests.",
                    theme,
                )

            start_min = df["request_start_ns"].min()
            df["start_s"] = (df["request_start_ns"] - start_min) / 1e9
            df["end_s"] = (df["request_end_ns"] - start_min) / 1e9
            df["ttft_s"] = df["time_to_first_token"] / 1000.0
            df["ttft_end_s"] = df["start_s"] + df["ttft_s"]

            df["duration_s"] = df["end_s"] - df["start_s"]
            df = df[df["ttft_s"] <= df["duration_s"]]

            if df.empty:
                return _create_empty_figure(
                    "Request timeline plot cannot be generated: no valid data after filtering.\n"
                    "All requests were filtered out (TTFT > total duration).",
                    theme,
                )

            df["request_id"] = range(len(df))
            df["y_value"] = df[y_metric]

            plot_df = df[["request_id", "y_value", "start_s", "ttft_end_s", "end_s"]]

            return plot_gen.create_request_timeline(
                df=plot_df,
                y_metric=y_metric,
                title=title,
                x_label="Time (seconds)",
                y_label=y_metric,
            )

        return _create_empty_figure(f"Unsupported plot type: {plot_type}", theme)

    except Exception as e:
        _logger.error(f"Error generating custom single-run plot: {e!r}")
        return _create_empty_figure(f"Error: {e}", theme)


def _create_empty_figure(message: str, theme: PlotTheme):
    """Create empty figure with error message.

    Args:
        message: Error message to display
        theme: Current theme

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "gray"},
    )
    template = "plotly_dark" if theme == PlotTheme.DARK else "plotly_white"
    fig.update_layout(template=template)
    return fig


def _build_available_metrics_dict(plot_specs: list[PlotSpec]) -> dict:
    """
    Build available_metrics dict needed by handlers.

    This extracts all metrics from plot specs and creates the dict
    that handlers use to look up display names and units.

    Args:
        plot_specs: List of plot specifications

    Returns:
        Dictionary mapping metric names to their metadata
    """
    available_metrics = {}
    for spec in plot_specs:
        for metric in spec.metrics:
            if metric.name not in available_metrics:
                available_metrics[metric.name] = {
                    "display_name": get_metric_display_name(metric.name),
                    "unit": None,
                }
    return available_metrics


def _generate_singlerun_figure(
    plot_id: str,
    config: dict,
    run: RunData,
    plot_specs: list,
    theme: PlotTheme,
) -> go.Figure | None:
    """
    Generate a single-run plot figure for the specified theme.

    Args:
        plot_id: Plot identifier
        config: Plot configuration dictionary
        run: RunData object
        plot_specs: List of PlotSpec objects for default plots
        theme: Theme to use for plot generation

    Returns:
        Plotly figure or None if generation fails
    """
    plot_gen = _get_plot_generator(theme)
    is_default = config.get("is_default", True)

    try:
        if is_default:
            spec = next((s for s in plot_specs if s.name == plot_id), None)
            if not spec:
                return None

            HandlerClass = plugins.get_class(PluginType.PLOT, spec.plot_type)
            handler = HandlerClass(plot_generator=plot_gen, logger=None)
            available_metrics = _build_available_metrics_dict(plot_specs)
            fig = handler.create_plot(spec, run, available_metrics)

            config_title = config.get("title")
            if config_title and config_title != spec.title:
                fig.update_layout(title=config_title)

            return fig
        else:
            return _generate_custom_single_run_plot(config, run, plot_gen, theme)
    except Exception as e:
        _logger.error(f"Error generating single-run plot '{plot_id}': {e!r}")
        return _create_empty_figure(f"Error: {str(e)}", theme)


def _render_single_run_plots(
    plot_state: dict,
    theme_data: dict,
    theme: PlotTheme,
    runs: list[RunData],
    plot_specs: list,
    default_plot_order: list[str],
):
    """Render plots for single-run mode with caching.

    Args:
        plot_state: Plot state from store
        theme_data: Theme data from store
        theme: Default theme
        runs: List of RunData (should have exactly 1 element)
        plot_specs: List of PlotSpec objects
        default_plot_order: List of default plot names

    Returns:
        Tuple of (children list, warnings list)
    """
    run = runs[0]
    current_theme = _get_current_theme(theme_data, theme)
    other_theme = (
        PlotTheme.DARK if current_theme == PlotTheme.LIGHT else PlotTheme.LIGHT
    )

    visible_plots = plot_state.get("visible_plots", [])
    plot_configs = plot_state.get("plot_configs", {})

    # Initialize cache with run-specific hash for single-run mode
    cache = get_plot_cache()
    runs_hash = compute_runs_hash([run])

    children = []
    all_warnings = []

    # Get all plot IDs (render ALL plots, use CSS to hide/show)
    all_plot_ids = list(plot_configs.keys())

    for plot_id in all_plot_ids:
        config = plot_configs.get(plot_id)
        if not config:
            _logger.warning(f"No config found for plot_id '{plot_id}'. Skipping.")
            continue

        is_visible = plot_id in visible_plots
        size_class = config.get("size", "half")
        config_hash = compute_config_hash(config)

        cache_key = CacheKey(
            plot_id=plot_id,
            config_hash=config_hash,
            runs_hash=runs_hash,
            theme=current_theme,
        )

        fig = cache.get(cache_key)

        if fig is None:
            fig = _generate_singlerun_figure(
                plot_id, config, run, plot_specs, current_theme
            )

            if fig is not None:
                cache.set(cache_key, fig)

                # Also generate and cache for opposite theme
                other_fig = _generate_singlerun_figure(
                    plot_id, config, run, plot_specs, other_theme
                )
                if other_fig is not None:
                    other_key = CacheKey(
                        plot_id=plot_id,
                        config_hash=config_hash,
                        runs_hash=runs_hash,
                        theme=other_theme,
                    )
                    cache.set(other_key, other_fig)

        if fig is not None:
            container = create_plot_container_component(
                plot_id=plot_id,
                figure=fig,
                theme=current_theme,
                size_class=size_class,
                visible=is_visible,
            )
            children.append(container)
        else:
            _logger.warning(f"Plot '{plot_id}' generation returned None. Skipping.")

    # Add "+ Create Custom Plot" button at end
    children.append(
        html.Div(
            [
                html.Div("+", className="plot-add-icon"),
                html.Div("Create Custom Plot", className="plot-add-text"),
            ],
            id="add-singlerun-plot-slot",
            n_clicks=0,
            className="plot-add-slot",
        )
    )
    return children, all_warnings


def register_dynamic_grid_callback(
    app: dash.Dash,
    runs: list[RunData],
    mode: VisualizationMode,
    theme: PlotTheme,
    plot_config,
):
    """
    Register callback to dynamically rebuild plot grid based on visible plots.

    Uses plot specs from YAML config (same as PNG export) for default plots.

    Args:
        app: Dash application instance
        runs: List of RunData objects
        mode: Visualization mode
        theme: Plot theme
        plot_config: PlotConfig instance
    """
    # Capture theme in closure explicitly
    default_theme = theme

    try:
        # Get plot specs from config
        if mode == VisualizationMode.MULTI_RUN:
            plot_specs = plot_config.get_multi_run_plot_specs()
        else:
            plot_specs = plot_config.get_single_run_plot_specs()

        # Create default plot order
        default_plot_order = [spec.name for spec in plot_specs]

        # Define inputs based on mode - single-run doesn't need run-selector
        # NOTE: theme-store is now an Input to ensure callback fires on initial load
        # NOTE: mode-store is added to support dynamic mode switching (drill-down)
        if mode == VisualizationMode.SINGLE_RUN:
            grid_inputs = [
                Input("plot-state-store", "data"),
                Input("theme-store", "data"),
                Input("mode-store", "data"),
            ]
            grid_states = []
        else:
            grid_inputs = [
                Input("plot-state-store", "data"),
                Input("run-selector", "value"),
                Input("theme-store", "data"),
                Input("mode-store", "data"),
            ]
            grid_states = []
    except Exception:
        traceback.print_exc()

        # Don't return - register callback anyway with empty specs
        plot_specs = []
        default_plot_order = []
        # Define defaults for both grid_inputs and grid_states
        grid_inputs = [
            Input("plot-state-store", "data"),
            Input("theme-store", "data"),
            Input("mode-store", "data"),
        ]
        grid_states = []

    @app.callback(
        [
            Output("plots-grid", "children"),
            Output("plot-warnings-store", "data"),
        ],
        grid_inputs,
        grid_states,
        prevent_initial_call=False,
    )
    def update_grid_children(plot_state, *args):
        """Rebuild grid children based on visible plots including custom plots."""
        sys.stdout.flush()

        # Extract arguments based on registered mode
        # Args order:
        # - Single-run: (theme_data, mode_data)
        # - Multi-run: (selected_runs, theme_data, mode_data)
        try:
            if mode == VisualizationMode.SINGLE_RUN:
                # Single-run: (plot_state, theme_data, mode_data)
                if len(args) < 2:
                    theme_data = args[0] if args else None
                    mode_data = None
                else:
                    theme_data = args[0]
                    mode_data = args[1]
                selected_runs = None  # Will be set to [0] below
            else:
                # Multi-run: (plot_state, selected_runs, theme_data, mode_data)
                selected_runs = args[0]
                theme_data = args[1]
                mode_data = args[2] if len(args) > 2 else None
        except Exception:
            traceback.print_exc()
            raise

        # Determine actual mode from mode_data (supports drill-down mode switching)
        actual_mode_str = mode_data.get("mode", mode.value) if mode_data else mode.value
        if actual_mode_str == "single_run":
            actual_mode = VisualizationMode.SINGLE_RUN
        else:
            actual_mode = VisualizationMode.MULTI_RUN

        # Check if we need to use drill-down caches
        # Use drill-down cache whenever we're in single-run mode and cache exists
        use_drill_down_cache = (
            actual_mode == VisualizationMode.SINGLE_RUN
            and "current" in _drill_down_run_cache
        )

        if use_drill_down_cache:
            runs_to_use = [_drill_down_run_cache["current"]]
            specs_to_use = _drill_down_specs_cache.get("current", [])
            order_to_use = [spec.name for spec in specs_to_use]
        else:
            runs_to_use = runs
            specs_to_use = plot_specs
            order_to_use = default_plot_order

        # Handle single-run mode (either registered or from drill-down)
        if actual_mode == VisualizationMode.SINGLE_RUN:
            # For single-run mode, default to run 0 if no selection
            if not selected_runs:
                selected_runs = [0]
            children, all_warnings = _render_single_run_plots(
                plot_state=plot_state,
                theme_data=theme_data,
                theme=default_theme,
                runs=runs_to_use,
                plot_specs=specs_to_use,
                default_plot_order=order_to_use,
            )

            return children, all_warnings

        # Multi-run mode logic below
        if not selected_runs:
            raise PreventUpdate

        filtered_runs = [runs[i] for i in selected_runs]
        current_theme = _get_current_theme(theme_data, default_theme)
        other_theme = (
            PlotTheme.DARK if current_theme == PlotTheme.LIGHT else PlotTheme.LIGHT
        )

        # Extract visible plots and configs from state
        visible_plots = plot_state.get("visible_plots", [])
        plot_configs = plot_state.get("plot_configs", {})

        # Initialize cache and compute runs hash
        cache = get_plot_cache()
        runs_hash = compute_runs_hash(selected_runs)
        # cache_stats_before = cache.get_stats()

        # Default plot size
        size = 400

        children = []
        all_warnings = []

        # Render ALL plots with caching (use CSS to hide/show)
        all_plot_ids = list(plot_configs.keys())

        for plot_id in all_plot_ids:
            plot_config_dict = plot_configs.get(plot_id)
            if not plot_config_dict:
                _logger.warning(f"No config found for plot_id '{plot_id}'. Skipping.")
                continue

            is_visible = plot_id in visible_plots
            size_class = plot_config_dict.get("size", "half")
            config_hash = compute_config_hash(plot_config_dict)

            # Build cache key for current theme
            cache_key = CacheKey(
                plot_id=plot_id,
                config_hash=config_hash,
                runs_hash=runs_hash,
                theme=current_theme,
            )

            # Try cache lookup
            fig = cache.get(cache_key)

            if fig is None:
                # Generate figure for current theme
                fig, plot_warnings = _generate_multirun_figure(
                    filtered_runs, plot_config_dict, current_theme
                )

                if plot_warnings:
                    all_warnings.extend(plot_warnings)

                if fig is not None:
                    # Cache for current theme
                    cache.set(cache_key, fig)

                    # Also generate and cache for opposite theme (dual-theme caching)
                    other_fig, _ = _generate_multirun_figure(
                        filtered_runs, plot_config_dict, other_theme
                    )
                    if other_fig is not None:
                        other_key = CacheKey(
                            plot_id=plot_id,
                            config_hash=config_hash,
                            runs_hash=runs_hash,
                            theme=other_theme,
                        )
                        cache.set(other_key, other_fig)

            if fig is not None:
                children.append(
                    create_plot_container_component(
                        plot_id,
                        fig,
                        current_theme,
                        size=size,
                        size_class=size_class,
                        visible=is_visible,
                    )
                )
            else:
                _logger.warning(f"Plot '{plot_id}' generation returned None. Skipping.")

        # Add "+ Create Custom Plot (Multi-Run)" button at end
        children.append(
            html.Div(
                [
                    html.Div("+", className="plot-add-icon"),
                    html.Div(
                        "Create Custom Plot (Multi-Run)", className="plot-add-text"
                    ),
                ],
                id="add-multirun-plot-slot",
                n_clicks=0,
                className="plot-add-slot",
            )
        )

        return children, all_warnings

    # NOTE: Server-side theme callback removed - replaced by clientside callback
    # in register_theme_callback(). The clientside callback uses Plotly.relayout()
    # to update themes instantly without server roundtrip, avoiding JSON
    # serialization, network transfer, and full re-render of all figures.


def register_sidebar_sync_callback(app: dash.Dash, theme: PlotTheme):
    """
    Register callback to sync sidebar state on page load and theme changes.

    This callback ensures the sidebar's visual state matches the persisted Store state
    when the page loads or refreshes. Without this, users may need to click the toggle
    button twice to open/close the sidebar due to state mismatch.

    Args:
        app: Dash application instance
        theme: Default plot theme
    """

    # Capture theme in closure explicitly
    default_theme = theme

    @app.callback(
        [
            Output("sidebar-container", "style"),
            Output("sidebar-toggle-btn", "style"),
        ],
        [
            Input("sidebar-collapsed", "data"),
            Input("theme-store", "data"),
        ],
    )
    def sync_sidebar_state(is_collapsed, theme_data):
        """Sync sidebar visibility with Store state on page load and theme changes."""
        # Get current theme
        current_theme = _get_current_theme(theme_data, default_theme)

        # Get sidebar style based on collapsed state
        sidebar_style = get_sidebar_style(current_theme, collapsed=is_collapsed)

        # Calculate button position based on sidebar state
        # When sidebar is expanded (not collapsed), button should be at 301px (300px width + 1px border)
        # When sidebar is collapsed, button should be at screen edge (10px)
        button_left = "10px" if is_collapsed else "301px"

        # Button style with dynamic positioning
        button_style = {
            "position": "fixed",
            "top": "70px",
            "left": button_left,
            "width": "32px",
            "height": "32px",
            "background": "rgba(118, 185, 0, 0.15)",
            "color": "white",
            "border": f"2px solid {NVIDIA_GREEN}",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "16px",
            "zIndex": "2000",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.2)",
            "transition": "left 0.3s ease-in-out",
            "fontFamily": PLOT_FONT_FAMILY,
            "backdropFilter": "blur(4px)",
        }

        return sidebar_style, button_style


def register_sidebar_toggle_callback(app: dash.Dash, theme: PlotTheme):
    """
    Register callback for sidebar toggle button.

    This callback handles the interactive toggling of the sidebar when the user
    clicks the toggle button. It updates both the sidebar visibility and the
    button position.

    Args:
        app: Dash application instance
        theme: Default plot theme
    """

    @app.callback(
        Output("sidebar-collapsed", "data"),
        Input("sidebar-toggle-btn", "n_clicks"),
        State("sidebar-collapsed", "data"),
    )
    def toggle_sidebar(n_clicks, is_collapsed):
        """Toggle sidebar collapsed state in Store."""
        if not n_clicks or n_clicks == 0:
            raise PreventUpdate

        # Toggle the collapsed state
        new_state = not is_collapsed
        return new_state


def register_collapsible_sections_callback(app: dash.Dash):
    """
    Register callback for collapsible section toggles.

    This callback handles the toggling of collapsible sections (plot controls,
    layout, run selector) when the user clicks on section headers.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output({"type": "section-content", "id": dash.dependencies.ALL}, "style"),
        Output({"type": "section-arrow", "id": dash.dependencies.ALL}, "children"),
        Input({"type": "section-header", "id": dash.dependencies.ALL}, "n_clicks"),
        State({"type": "section-content", "id": dash.dependencies.ALL}, "style"),
        prevent_initial_call=True,
    )
    def toggle_collapsible_sections(n_clicks_list, current_styles):
        """Toggle visibility of collapsible sections."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        # Find which section was clicked
        triggered_id = ctx.triggered[0]["prop_id"]

        # Parse the triggered ID to get the index
        try:
            id_dict = json.loads(triggered_id.split(".")[0])
            clicked_index = None
            for i, input_id in enumerate(ctx.inputs_list[0]):
                if input_id["id"] == id_dict:
                    clicked_index = i
                    break

            if clicked_index is None:
                raise PreventUpdate

        except (json.JSONDecodeError, KeyError, IndexError):
            raise PreventUpdate from None

        # Toggle the clicked section
        new_styles = []
        new_arrows = []

        for i, style in enumerate(current_styles):
            if i == clicked_index:
                # Toggle this section
                is_visible = style.get("display", "block") == "block"
                new_styles.append({"display": "none" if is_visible else "block"})
                new_arrows.append("▶" if is_visible else "▼")
            else:
                # Keep other sections unchanged
                new_styles.append(style)
                is_visible = style.get("display", "block") == "block"
                new_arrows.append("▼" if is_visible else "▶")

        _logger.debug("Toggled section at index {clicked_index}")
        return new_styles, new_arrows


def register_run_count_badge_callback(app: dash.Dash):
    """
    Register callback for run count badge update.

    This callback updates the run count badge to show how many runs are currently
    selected in multi-run mode.

    Args:
        app: Dash application instance
    """

    @app.callback(
        Output("run-count-badge", "children"),
        Output("run-count-badge", "style"),
        [Input("run-selector", "value"), Input("theme-store", "data")],
    )
    def update_run_count(selected_runs, theme_data):
        """Update run count badge with current count."""
        count = len(selected_runs) if selected_runs else 0

        badge_text = f"{count} selected"

        badge_style = {
            "font-size": "9px",
            "font-weight": "500",
            "padding": "2px 8px",
            "background": "rgba(118, 185, 0, 0.15)",
            "border": f"1px solid {NVIDIA_GREEN}",
            "border-radius": "10px",
            "color": NVIDIA_GREEN,
            "font-family": PLOT_FONT_FAMILY,
        }

        return badge_text, badge_style


def register_drill_down_callbacks(
    app: dash.Dash,
    runs: list[RunData],
    run_dirs: list[Path],
    loader: DataLoader,
    plot_config: PlotConfig,
) -> None:
    """
    Register callbacks for drill-down from multi-run to single-run view.

    When a user clicks "View Single-Run Plots" in the config modal, this opens
    a new browser tab with the single-run dashboard for that specific run.

    Args:
        app: Dash application instance
        runs: List of RunData objects
        run_dirs: List of run directory paths for lazy loading
        loader: DataLoader instance for loading per-request data
        plot_config: PlotConfig instance for getting single-run plot specs
    """
    app.clientside_callback(
        """
        function(n_clicks, run_idx) {
            if (n_clicks && n_clicks > 0 && run_idx !== null && run_idx !== undefined) {
                window.open('/?run_idx=' + run_idx, '_blank');
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("btn-view-single-run", "n_clicks"),
        Input("btn-view-single-run", "n_clicks"),
        State("current-run-idx-store", "data"),
        prevent_initial_call=True,
    )

    # Cache for loaded single-run data
    single_run_cache: dict[int, RunData] = {}

    @app.callback(
        [
            Output("dashboard-sidebar", "children"),
            Output("dashboard-main-area", "children"),
            Output("plot-state-store", "data", allow_duplicate=True),
            Output("mode-store", "data", allow_duplicate=True),
        ],
        Input("url", "search"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def handle_url_routing(search, theme_data):
        """Handle URL routing for drill-down to single-run view.

        When navigating to /?run_idx=X, this rebuilds the entire dashboard
        in single-run mode with full sidebar functionality.
        """
        current_theme = _get_current_theme(theme_data, PlotTheme.LIGHT)
        colors = get_theme_colors(current_theme)

        if not search or "run_idx" not in search:
            raise PreventUpdate

        # Parse URL parameters
        params = parse_qs(search.lstrip("?"))
        run_idx_str = params.get("run_idx", [None])[0]

        if run_idx_str is None:
            raise PreventUpdate

        try:
            run_idx = int(run_idx_str)
        except ValueError:
            raise PreventUpdate from None

        if run_idx < 0 or run_idx >= len(runs):
            raise PreventUpdate

        # Get or load per-request data for this run
        if run_idx in single_run_cache:
            run = single_run_cache[run_idx]
        else:
            # Load with per-request data
            run_dir = run_dirs[run_idx]
            _logger.debug("Loading per-request data for run {run_idx} from {run_dir}")
            run = loader.load_run(run_dir, load_per_request_data=True)
            single_run_cache[run_idx] = run

        # Build single-run dashboard using DashboardBuilder
        builder = DashboardBuilder(
            runs=[run],
            mode=VisualizationMode.SINGLE_RUN,
            theme=current_theme,
            plot_config=plot_config,
        )

        # Populate module-level caches for drill-down mode
        # These are used by update_grid_children when mode switches to single_run
        _drill_down_run_cache["current"] = run
        _drill_down_specs_cache["current"] = plot_config.get_single_run_plot_specs()
        _drill_down_plot_config_cache["current"] = plot_config

        # Get sidebar children (strip the outer div wrapper)
        sidebar_div = builder._build_sidebar()
        sidebar_children = sidebar_div.children

        # Build main area with header (back link) + single-run plots
        model = run.metadata.model or "Unknown Model"
        concurrency = run.metadata.concurrency or "N/A"

        main_area_children = [
            # Header with back link
            html.Div(
                [
                    html.A(
                        "← Back to Multi-Run View",
                        href="/",
                        style={
                            "color": "#1976d2",
                            "textDecoration": "none",
                            "fontSize": "14px",
                            "marginRight": "20px",
                        },
                    ),
                    html.Span(
                        f"Single-Run Details: {model} - Concurrency {concurrency}",
                        style={
                            "fontSize": "16px",
                            "fontWeight": "600",
                            "color": colors["text"],
                        },
                    ),
                ],
                style={
                    "padding": "12px 16px 12px 16px",
                    "borderBottom": f"1px solid {colors['border']}",
                    "marginBottom": "12px",
                },
            ),
            # Single-run plots from builder (unpack list to flatten structure)
            *builder._build_single_run_tab(),
        ]

        # Get plot state for single-run mode
        plot_state = builder.build_single_run_plot_state()

        return (
            sidebar_children,
            main_area_children,
            plot_state,
            {"mode": "single_run"},
        )


def register_toast_notifications_callbacks(app: dash.Dash):
    """
    Register callbacks for toast notifications.

    This includes:
    1. Show warnings toast when plot-warnings-store updates
    2. Auto-dismiss toast after 5 seconds

    Args:
        app: Dash application instance
    """

    @app.callback(
        [
            Output("toast-container", "children"),
            Output("toast-timestamp-store", "data"),
            Output("toast-dismiss-interval", "disabled"),
        ],
        Input("plot-warnings-store", "data"),
        prevent_initial_call=True,
    )
    def show_warnings_toast(warnings):
        """Display warnings as Toast notifications in the browser."""
        if not warnings:
            return [], 0, True

        unique_warnings = list(dict.fromkeys(warnings))

        toast_message = html.Div(
            [
                html.H6("⚠️ Stat Fallbacks Applied", style={"margin": "0 0 8px 0"}),
                html.Div(
                    [
                        html.Div(
                            f"• {w}", style={"fontSize": "13px", "marginBottom": "4px"}
                        )
                        for w in unique_warnings[:5]
                    ]
                ),
                html.Div(
                    f"...and {len(unique_warnings) - 5} more"
                    if len(unique_warnings) > 5
                    else "",
                    style={
                        "fontSize": "11px",
                        "fontStyle": "italic",
                        "marginTop": "6px",
                    },
                )
                if len(unique_warnings) > 5
                else None,
            ]
        )

        toast_element = html.Div(
            toast_message,
            style={
                "background": "#fff3cd",
                "border": "1px solid #ffc107",
                "borderRadius": "6px",
                "padding": "16px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                "minWidth": "320px",
                "maxWidth": "400px",
                "animation": "slideIn 0.3s ease-out",
                "pointerEvents": "auto",
            },
        )

        _logger.debug("Showing toast with {len(unique_warnings)} warnings")
        return toast_element, time.time(), False

    @app.callback(
        [
            Output("toast-container", "children", allow_duplicate=True),
            Output("toast-dismiss-interval", "disabled", allow_duplicate=True),
        ],
        Input("toast-dismiss-interval", "n_intervals"),
        State("toast-timestamp-store", "data"),
        prevent_initial_call=True,
    )
    def auto_dismiss_toast(n_intervals, timestamp):
        """Auto-dismiss toast after 5 seconds."""
        if timestamp == 0:
            return [], True

        elapsed = time.time() - timestamp
        if elapsed >= 5:
            _logger.debug("Auto-dismissing toast after 5 seconds")
            return [], True

        raise PreventUpdate


def register_drag_drop_callbacks(app: dash.Dash):
    """Register drag-and-drop callbacks for plot reordering (placeholder)."""
    pass

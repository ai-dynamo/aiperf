# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dashboard layout builder for AIPerf visualization.

This module constructs the complete dashboard UI including header, sidebar,
tabs, and plot grids with all interactive controls.
"""

import time

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from aiperf.plot.config import PlotConfig
from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    METRIC_CATEGORY_RULES,
    NON_METRIC_KEYS,
    NVIDIA_DARK,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_WHITE,
    PLOT_FONT_FAMILY,
    STAT_LABELS,
    PlotTheme,
)
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.mode_detector import VisualizationMode
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import PlotType, get_plot_type_info
from aiperf.plot.core.swept_params import detect_swept_parameters
from aiperf.plot.dashboard.components import (
    create_button,
    create_collapsible_section,
    create_export_controls_card,
    create_global_stat_selector,
    create_label,
    create_run_selector_checklist,
    create_sidebar_toggle_button,
)
from aiperf.plot.dashboard.styling import (
    get_header_style,
    get_main_area_style,
    get_sidebar_style,
    get_theme_colors,
)
from aiperf.plot.dashboard.utils import (
    add_run_idx_to_figure,
    create_plot_container_component,
    get_single_run_metrics_with_stats,
    runs_to_dataframe,
)
from aiperf.plot.metric_names import (
    _format_server_metric_name,
    get_gpu_metrics,
    get_metric_display_name,
    get_metric_display_name_with_unit,
)
from aiperf.plot.utils import get_server_metrics_summary

CATEGORY_HEADERS = {
    "per_request": "── Per-Request Plots ──",
    "aggregated": "── Aggregated Plots ──",
    "combined": "── Combined Views ──",
    "comparison": "── Comparison Plots ──",
}


def _build_plot_type_options_with_headers(
    plot_types: list[PlotType],
) -> list[dict]:
    """
    Build dropdown options grouped by category with section headers.

    Args:
        plot_types: List of PlotType enum values to include

    Returns:
        List of dropdown option dicts with disabled headers between categories
    """
    by_category: dict[str, list[PlotType]] = {}
    for pt in plot_types:
        info = get_plot_type_info(pt)
        by_category.setdefault(info.category, []).append(pt)

    options = []
    category_order = ["per_request", "aggregated", "combined", "comparison"]
    for category in category_order:
        if category not in by_category:
            continue
        options.append(
            {
                "label": CATEGORY_HEADERS[category],
                "value": f"_header_{category}",
                "disabled": True,
            }
        )
        for pt in by_category[category]:
            info = get_plot_type_info(pt)
            options.append(
                {
                    "label": f"  {info.display_name}",
                    "value": pt.value,
                    "title": info.description,
                }
            )
    return options


# Multi-run plot types (all are comparison plots, no section headers needed)
MULTI_RUN_PLOT_TYPES = [
    {
        "label": get_plot_type_info(PlotType.PARETO).display_name,
        "value": PlotType.PARETO.value,
        "title": get_plot_type_info(PlotType.PARETO).description,
    },
    {
        "label": get_plot_type_info(PlotType.SCATTER_LINE).display_name,
        "value": PlotType.SCATTER_LINE.value,
        "title": get_plot_type_info(PlotType.SCATTER_LINE).description,
    },
    {
        "label": "Scatter Only",
        "value": "scatter",
        "title": "Data points without connecting lines",
    },
]

# Columns to exclude from y-axis metric options in single-run plots
EXCLUDED_METRIC_COLUMNS = [
    # Time/index columns
    "request_number",
    "timestamp",
    "timestamp_s",
    "request_start_ns",
    "request_ack_ns",
    "request_end_ns",
    "cancellation_time_ns",
    # Identifiers
    "session_num",
    "model",
    "concurrency",
    "x_request_id",
    "x_correlation_id",
    "conversation_id",
    "turn_index",
    "worker_id",
    "record_processor_id",
    # Text fields
    "error_msg",
    "input_text",
    "output_text",
    # Boolean/status flags
    "was_cancelled",
    "is_error",
    # Internal fields
    "benchmark_phase",
]


class DashboardBuilder:
    """
    Builder for dashboard layout components.

    Constructs the full dashboard UI including header, sidebar, and main content area
    with support for both multi-run and single-run visualization modes.

    Args:
        runs: List of RunData objects
        mode: Visualization mode
        theme: Plot theme
        plot_config: Plot configuration
    """

    def __init__(
        self,
        runs: list[RunData],
        mode: VisualizationMode,
        theme: PlotTheme,
        plot_config: PlotConfig,
    ):
        """Initialize the dashboard builder."""
        self.runs = runs
        self.mode = mode
        self.theme = theme
        self.plot_config = plot_config
        self.colors = get_theme_colors(theme)
        self.plot_generator = PlotGenerator(theme=theme)

    def _categorize_metric(self, metric_name: str) -> str:
        """
        Categorize metric for grouping in dropdown.

        Args:
            metric_name: Name of the metric

        Returns:
            Category name for grouping
        """
        metric_lower = metric_name.lower()
        for category, keywords in METRIC_CATEGORY_RULES.items():
            if keywords and any(kw in metric_lower for kw in keywords):
                return category
        return "Other Metrics"

    def _get_all_available_metrics(self) -> dict[str, dict]:
        """
        Discover all available metrics from loaded runs.

        Returns:
            Dict mapping metric_name → {'display': str, 'category': str, 'stats': list}
        """
        if not self.runs:
            return {}

        # Sample first run to find available metrics
        first_run = self.runs[0]
        metrics_info = {}

        # Iterate directly over aggregated dict - metrics are at top level
        for metric_name, metric_data in first_run.aggregated.items():
            # Skip non-metric keys (input_config, telemetry_data, etc.)
            if metric_name in NON_METRIC_KEYS:
                continue

            # Check if this is a metric (has "unit" field)
            if not isinstance(metric_data, dict) or "unit" not in metric_data:
                continue

            # Get available stats (all keys except "unit")
            available_stats = [k for k in metric_data if k != "unit"]

            # Get display name and append unit if available
            display_name = get_metric_display_name(metric_name)
            unit = metric_data.get("unit")
            if unit:
                display_name = f"{display_name} ({unit})"

            metrics_info[metric_name] = {
                "display": display_name,
                "category": self._categorize_metric(metric_name),
                "stats": available_stats,
                "unit": unit,
            }

        # Add server metrics aggregated (for multi-run comparison)
        if (
            hasattr(first_run, "server_metrics_aggregated")
            and first_run.server_metrics_aggregated
        ):
            for (
                metric_name,
                endpoint_data,
            ) in first_run.server_metrics_aggregated.items():
                # For multi-run, we use merged endpoint data (first endpoint)
                first_endpoint = next(iter(endpoint_data.values()))
                first_series = next(iter(first_endpoint.values()))

                metric_type = first_series.get("type", "")
                unit = first_series.get("unit", "")
                stats_obj = first_series.get("stats")

                # Determine available stats based on metric type
                available_stats = []
                if stats_obj:
                    if hasattr(stats_obj, "__dict__"):
                        # Pydantic model - get all non-None attributes
                        available_stats = [
                            k for k, v in stats_obj.__dict__.items() if v is not None
                        ]
                    elif isinstance(stats_obj, dict):
                        # Dict - get all keys
                        available_stats = list(stats_obj.keys())

                # Default stats based on metric type
                if metric_type == "COUNTER" and not available_stats:
                    available_stats = ["rate", "total"]
                elif metric_type in ["GAUGE", "HISTOGRAM"] and not available_stats:
                    available_stats = ["avg", "p50", "p90", "p95", "p99"]

                # Format display name
                display_name = _format_server_metric_name(metric_name)
                if unit:
                    display_name = f"{display_name} ({unit})"

                metrics_info[metric_name] = {
                    "display": display_name,
                    "category": "Server Metrics",
                    "stats": available_stats,
                    "unit": unit,
                }

        # Add metadata metrics
        metrics_info["concurrency"] = {
            "display": "Concurrency",
            "category": "Configuration",
            "stats": ["value"],
        }

        return metrics_info

    def _get_stat_options_ordered(self) -> list[dict]:
        """
        Get stat options in consistent sensible order for UI dropdowns.

        Returns:
            List of option dicts for dropdown
        """
        return [{"label": STAT_LABELS[stat], "value": stat} for stat in ALL_STAT_KEYS]

    def _get_server_metrics_options(self, run: "RunData") -> list[dict]:
        """
        Get server metrics options for dropdown menus.

        Builds dropdown options from server metrics with display names and
        series count information.

        Args:
            run: RunData object containing server_metrics DataFrame

        Returns:
            List of option dicts with label/value pairs for dropdown.
            Empty list if no server metrics available.
        """
        if run.server_metrics is None or run.server_metrics.empty:
            return []

        server_summary = get_server_metrics_summary(run)

        options = []
        for metric_name, info in sorted(server_summary.items()):
            # Build label with additional context
            label = f"{metric_name} ({info['display_name']})"

            # Add label combination count if > 1
            if info["label_combinations"] > 1:
                label += f" [{info['label_combinations']} series]"

            options.append({"label": label, "value": metric_name})

        return options

    def _flatten_config(self, config: dict, row: dict, prefix: str = "") -> None:
        """
        Flatten nested config dict into row dict.

        Args:
            config: Nested config dictionary
            row: Output dictionary to populate
            prefix: Current key prefix
        """
        if not isinstance(config, dict):
            return

        for key, value in config.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dict
                self._flatten_config(value, row, f"{full_key}.")
            elif isinstance(value, int | float | str | bool):
                # Add primitive value
                row[full_key] = value

    def _detect_swept_parameters(self) -> list[str]:
        """
        Detect which parameters vary across runs.

        Returns:
            List of parameter names that are swept (vary across runs)
        """
        # Build DataFrame with all config params from runs
        data = []
        for run in self.runs:
            config = run.aggregated.get("input_config", {})

            row = {
                "model": run.metadata.model,
                "concurrency": run.metadata.concurrency,
            }

            # Flatten nested config
            self._flatten_config(config, row)

            data.append(row)

        df = pd.DataFrame(data)
        return detect_swept_parameters(df)

    def build(self) -> html.Div:
        """
        Build complete dashboard layout.

        Returns:
            Dash HTML Div containing the full dashboard
        """
        return html.Div(
            [
                # Hidden stores for state management
                self._build_stores(),
                # Toast container for warnings
                html.Div(
                    id="toast-container",
                    style={
                        "position": "fixed",
                        "top": "80px",
                        "right": "20px",
                        "zIndex": "9999",
                        "pointerEvents": "none",
                    },
                ),
                # Config viewer modal
                self._build_config_modal(),
                # Custom plot creator modal (multi-run)
                self._build_custom_plot_modal()
                if self.mode == VisualizationMode.MULTI_RUN
                else html.Div(),
                # Custom plot creator modal (single-run) - always rendered for drill-down support
                self._build_single_run_custom_plot_modal(),
                # Edit plot modal (single-run) - always rendered for drill-down support
                self._build_single_run_edit_modal(),
                # Plot edit modal (multi-run)
                self._build_plot_edit_modal()
                if self.mode == VisualizationMode.MULTI_RUN
                else html.Div(),
                # Context menu for right-click on plots
                self._build_context_menu(),
                # Header
                self._build_header(),
                # Sidebar toggle button
                create_sidebar_toggle_button(self.theme),
                # Main content (sidebar + plot area)
                html.Div(
                    [
                        # Sidebar (collapsible, starts collapsed)
                        html.Div(
                            self._build_sidebar(),
                            id="sidebar-container",
                            style=get_sidebar_style(self.theme, collapsed=True),
                        ),
                        # Main plot area
                        self._build_main_area(),
                    ],
                    style={
                        "display": "flex",
                        "align-items": "flex-start",
                        "height": "calc(100vh - 70px)",
                    },
                ),
            ],
            id="app-root-container",
            className=f"theme-{self.theme.value}",
            style={
                "font-family": PLOT_FONT_FAMILY,
                "height": "100vh",
                "margin": "0",
                "background": self.colors["background"],
            },
        )

    def _get_default_plot_ids(self) -> list[str]:
        """
        Get default plot IDs from plot config.

        Returns:
            List of plot IDs from YAML config
        """
        if self.mode == VisualizationMode.MULTI_RUN:
            specs = self.plot_config.get_multi_run_plot_specs()
        else:
            specs = self.plot_config.get_single_run_plot_specs()

        return [spec.name for spec in specs]

    def _get_initial_plot_configs(self) -> dict[str, dict]:
        """
        Build initial plot configurations from YAML config.

        Returns:
            Dictionary mapping plot names to their configuration dicts
        """
        plot_configs = {}
        if self.mode == VisualizationMode.MULTI_RUN:
            plot_specs = self.plot_config.get_multi_run_plot_specs()

            for spec in plot_specs:
                x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
                y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

                if x_metric_spec and y_metric_spec:
                    plot_type_val = (
                        spec.plot_type.value if spec.plot_type else "scatter_line"
                    )
                    plot_configs[spec.name] = {
                        "mode": "multi_run",
                        "x_metric": x_metric_spec.name,
                        "x_stat": x_metric_spec.stat or "p50",
                        "y_metric": y_metric_spec.name,
                        "y_stat": y_metric_spec.stat or "avg",
                        "log_scale": "none",
                        "is_default": True,
                        "plot_type": plot_type_val,
                        "size": "half",  # All plots start at half-width for uniform sizing
                        "label_by": spec.label_by or "concurrency",
                        "group_by": spec.group_by or "model",
                        "title": spec.title or spec.name.replace("-", " ").title(),
                    }

        return plot_configs

    def build_single_run_plot_state(self) -> dict:
        """
        Build plot-state-store data for single-run mode.

        This method can be used when dynamically switching to single-run mode
        (e.g., during drill-down from multi-run view).

        Returns:
            Dictionary containing plot state data for single-run mode
        """
        plot_specs = self.plot_config.get_single_run_plot_specs()
        default_visible = [spec.name for spec in plot_specs]
        plot_configs = {}

        plot_type_map = {
            "timeslice": "timeslice",
            "scatter": "scatter",
            "area": "area",
            "dual_axis": "dual_axis",
            "scatter_with_percentiles": "scatter",
            "request_timeline": "scatter",
        }

        for spec in plot_specs:
            plot_type_val = plot_type_map.get(spec.plot_type.value, "scatter")

            if plot_type_val == "timeslice":
                x_axis = "Timeslice"
            else:
                x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
                x_axis = x_metric.name if x_metric else "request_number"

            y_metric = next((m for m in spec.metrics if m.axis == "y"), None)
            stored_title = spec.title or spec.name.replace("-", " ").title()

            plot_configs[spec.name] = {
                "is_default": True,
                "size": "half",
                "mode": "single_run",
                "plot_type": plot_type_val,
                "x_axis": x_axis,
                "y_metric": y_metric.name if y_metric else "",
                "y_metric_base": y_metric.name if y_metric else "",
                "stat": y_metric.stat if y_metric and y_metric.stat else "avg",
                "source": y_metric.source.value if y_metric else "requests",
                "title": stored_title,
            }

        config_version = int(time.time())
        slice_duration = (
            getattr(self.runs[0], "slice_duration", None) if self.runs else None
        )

        return {
            "visible_plots": default_visible,
            "hidden_plots": [],
            "plot_configs": plot_configs,
            "config_version": config_version,
            "slice_duration": slice_duration,
        }

    def _build_stores(self) -> html.Div:
        """Build hidden stores for state management with full plot configurations."""
        # Get default visible plots from config (not hardcoded!)
        default_visible = self._get_default_plot_ids()

        # Initialize plot_configs from YAML config
        plot_configs = {}
        if self.mode == VisualizationMode.MULTI_RUN:
            plot_specs = self.plot_config.get_multi_run_plot_specs()

            for spec in plot_specs:
                # Extract x and y metric specs
                x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
                y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

                if x_metric_spec and y_metric_spec:
                    plot_type_val = (
                        spec.plot_type.value if spec.plot_type else "scatter_line"
                    )
                    plot_configs[spec.name] = {
                        "mode": "multi_run",
                        "x_metric": x_metric_spec.name,
                        "x_stat": x_metric_spec.stat or "p50",
                        "y_metric": y_metric_spec.name,
                        "y_stat": y_metric_spec.stat or "avg",
                        "log_scale": "none",
                        "is_default": True,
                        "plot_type": plot_type_val,
                        "size": "half",  # All plots start at half-width for uniform sizing
                        "label_by": spec.label_by or "concurrency",
                        "group_by": spec.group_by or "model",
                        "title": spec.title or spec.name.replace("-", " ").title(),
                    }
        else:
            # Single-run mode: populate plot_configs for default plots
            plot_specs = self.plot_config.get_single_run_plot_specs()

            # Map YAML plot types to dashboard internal values
            plot_type_map = {
                "timeslice": "timeslice",
                "scatter": "scatter",
                "area": "area",
                "dual_axis": "dual_axis",
                "scatter_with_percentiles": "scatter",
                "request_timeline": "scatter",
            }

            for spec in plot_specs:
                plot_type_val = plot_type_map.get(spec.plot_type.value, "scatter")

                # Determine x_axis based on plot type
                if plot_type_val == "timeslice":
                    x_axis = "Timeslice"
                else:
                    x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
                    x_axis = x_metric.name if x_metric else "request_number"

                # Get y metric info
                y_metric = next((m for m in spec.metrics if m.axis == "y"), None)

                stored_title = spec.title or spec.name.replace("-", " ").title()
                plot_configs[spec.name] = {
                    "is_default": True,
                    "size": "half",
                    "mode": "single_run",
                    "plot_type": plot_type_val,
                    "x_axis": x_axis,
                    "y_metric": y_metric.name if y_metric else "",
                    "y_metric_base": y_metric.name if y_metric else "",
                    "stat": y_metric.stat if y_metric and y_metric.stat else "avg",
                    "source": y_metric.source.value if y_metric else "requests",
                    "title": stored_title,
                }

        # Generate version based on timestamp - forces reset on every restart
        # This ensures dashboard always loads from YAML, ignoring localStorage
        config_version = int(time.time())

        # Get slice_duration for single-run mode
        slice_duration = None
        if self.mode != VisualizationMode.MULTI_RUN and self.runs:
            slice_duration = getattr(self.runs[0], "slice_duration", None)

        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                dcc.Store(
                    id="current-run-idx-store",
                    storage_type="memory",
                    data=None,
                ),
                dcc.Store(
                    id="mode-store",
                    storage_type="session",
                    data={"mode": self.mode.value},
                ),
                dcc.Store(
                    id="plot-state-store",
                    storage_type="session",
                    data={
                        "visible_plots": default_visible,
                        "hidden_plots": [],
                        "plot_configs": plot_configs,
                        "config_version": config_version,
                        "slice_duration": slice_duration,
                    },
                ),
                dcc.Store(
                    id="config-version-store",
                    storage_type="memory",
                    data=config_version,
                ),
                dcc.Store(
                    id="theme-store",
                    storage_type="session",
                    data={"theme": self.theme.value},
                ),
                dcc.Store(id="sidebar-collapsed", storage_type="memory", data=True),
                dcc.Store(id="plot-warnings-store", storage_type="memory", data=[]),
                dcc.Store(id="toast-timestamp-store", storage_type="memory", data=0),
                dcc.Interval(
                    id="toast-dismiss-interval",
                    interval=5000,
                    n_intervals=0,
                    disabled=True,
                ),
            ]
        )

    def _build_header(self) -> html.Div:
        """Build dashboard header with branding and controls."""
        return html.Div(
            [
                html.H1(
                    "AIPerf Dashboard",
                    id="header-title",
                    style={
                        "margin": "0",
                        "font-size": "22px",
                        "font-weight": "600",
                        "letter-spacing": "-0.5px",
                        "font-family": PLOT_FONT_FAMILY,
                        "color": NVIDIA_WHITE
                        if self.theme == PlotTheme.DARK
                        else NVIDIA_DARK,
                    },
                ),
                html.Div(
                    [
                        dbc.Switch(
                            id="theme-toggle",
                            value=self.theme == PlotTheme.DARK,
                            label="Dark",
                            style={
                                "font-size": "12px",
                                "font-family": PLOT_FONT_FAMILY,
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "align-items": "center",
                        "margin-left": "auto",
                    },
                ),
            ],
            id="dashboard-header",
            style=get_header_style(self.theme),
        )

    def _build_sidebar(self) -> html.Div:
        """Build reorganized sidebar with collapsible controls and visual cards."""
        sections = []

        # Card style for visual separation
        card_style = {
            "background": self.colors["paper"],
            "border": f"1px solid {self.colors['border']}",
            "border-radius": "8px",
            "padding": "12px",
            "margin-bottom": "24px",
            "width": "260px",
            "box-sizing": "border-box",
            "display": "flex",
            "flex-direction": "column",
            "align-items": "stretch",
        }

        # 1. EXPORT CARD (NEW - at the top)
        sections.append(
            html.Div(
                [create_export_controls_card(self.theme)],
                id="sidebar-export-card",
                style=card_style,
            )
        )

        # 2. Layout & Plots - wrapped in card
        sections.append(
            html.Div(
                [self._build_layout_and_plots_section()],
                id="sidebar-layout-card",
                style=card_style,
            )
        )

        # 3. Run Selector (collapsed, at bottom) - wrapped in card
        sections.append(
            html.Div(
                [self._build_run_selector_section()],
                id="sidebar-run-selector-card",
                style={
                    **card_style,
                    "margin-bottom": "0",  # Last item, no bottom margin
                },
            )
        )

        return html.Div(sections, id="dashboard-sidebar")

    def _build_run_selector_section(self) -> html.Div:
        """Build collapsible run selector with count display."""
        run_options = []
        all_run_indices = []

        for idx, run in enumerate(self.runs):
            label = f"{run.metadata.model or 'Unknown'}"
            if run.metadata.concurrency:
                label += f" - C{run.metadata.concurrency}"
            run_options.append({"label": label, "value": idx})
            all_run_indices.append(idx)

        # Get group_by from first plot config
        first_plot_configs = self._get_initial_plot_configs()
        first_config = next(iter(first_plot_configs.values()), {})
        group_by = first_config.get("group_by", "model")

        # Calculate initial count
        selected_count = len(all_run_indices)

        # Custom header with badge
        header = html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "RUNS",
                            style={
                                "font-size": "11px",
                                "font-weight": "600",
                                "text-transform": "uppercase",
                                "letter-spacing": "0.8px",
                                "color": NVIDIA_GRAY,
                                "font-family": PLOT_FONT_FAMILY,
                                "margin-right": "8px",
                            },
                        ),
                        html.Span(
                            f"{selected_count} selected",
                            id="run-count-badge",
                            style={
                                "font-size": "9px",
                                "font-weight": "500",
                                "padding": "2px 8px",
                                "background": "rgba(118, 185, 0, 0.15)",
                                "border": f"1px solid {NVIDIA_GREEN}",
                                "border-radius": "10px",
                                "color": NVIDIA_GREEN,
                                "font-family": PLOT_FONT_FAMILY,
                            },
                        ),
                    ],
                    style={
                        "flex": "1",
                        "display": "flex",
                        "align-items": "center",
                    },
                ),
                html.Span(
                    "▼",
                    id={"type": "section-arrow", "id": "run-selector-section"},
                    style={
                        "font-size": "10px",
                        "color": NVIDIA_GRAY,
                        "transition": "transform 0.2s",
                    },
                ),
            ],
            id={"type": "section-header", "id": "run-selector-section"},
            n_clicks=0,
            style={
                "display": "flex",
                "align-items": "center",
                "cursor": "pointer",
                "padding-bottom": "6px",
                "margin-bottom": "12px",
            },
        )

        # Content section
        content = html.Div(
            [
                html.Div(
                    [
                        create_run_selector_checklist(
                            run_options,
                            all_run_indices,
                            self.theme,
                            runs=self.runs,
                            group_by=group_by,
                        )
                    ],
                    id="run-selector-container",
                    className="run-selector-container",
                    style={
                        "max-height": "200px",
                        "overflow-y": "auto",
                        "padding": "8px",
                        "background": self.colors["paper"],
                        "border-radius": "6px",
                        "margin-bottom": "12px",
                    },
                )
            ],
            id={"type": "section-content", "id": "run-selector-section"},
            style={"display": "block"},
        )

        return html.Div([header, content])

    def _build_theme_controls(self) -> html.Div:
        """Build theme toggle control."""
        return html.Div(
            [
                create_label("Toggle Light/Dark Mode", self.theme),
                html.Div(
                    [
                        dbc.Switch(
                            id="theme-toggle",
                            value=self.theme == PlotTheme.DARK,
                            label="Dark Mode",
                            style={"font-size": "12px"},
                        )
                    ],
                    style={"margin-bottom": "12px"},
                ),
            ]
        )

    def _build_export_controls(self) -> html.Div:
        """Build export control buttons."""
        return html.Div(
            [
                dcc.Download(id="download-png-bundle"),
                create_button(
                    "btn-export-png",
                    "Export All Plots (PNG)",
                    self.theme,
                    variant="outline",
                ),
            ]
        )

    def _build_global_stat_selector(self) -> html.Div:
        """Build global stat selector with quick apply button."""
        return create_global_stat_selector(self.theme)

    def _build_layout_and_plots_section(self) -> html.Div:
        """Build combined layout and hidden plots section (collapsible)."""

        # Combine layout controls and hidden plots
        content = [
            # Hidden plots section
            html.Div(
                [
                    create_label("Hidden Plots", self.theme),
                    html.Div(
                        id="hidden-plots-list",
                        children=[
                            html.Div(
                                "No hidden plots",
                                style={
                                    "font-size": "11px",
                                    "color": NVIDIA_GRAY,
                                    "font-style": "italic",
                                    "font-family": PLOT_FONT_FAMILY,
                                },
                            )
                        ],
                    ),
                ]
            ),
            html.Div(style={"height": "16px"}),  # Spacer
            create_button(
                "btn-reset-layout", "Reset to Defaults", self.theme, variant="secondary"
            ),
        ]

        return create_collapsible_section(
            section_id="layout-and-plots",
            title="Plot Layout",
            children=content,
            theme=self.theme,
            initially_open=True,
        )

    def _build_main_area(self) -> html.Div:
        """Build main content area with tabs."""
        if self.mode == VisualizationMode.MULTI_RUN:
            content = self._build_multi_run_tab()
        else:
            content = self._build_single_run_tab()

        return html.Div(
            content, id="dashboard-main-area", style=get_main_area_style(self.theme)
        )

    def _build_multi_run_tab(self) -> list:
        """Build multi-run comparison tab with grid of plots from config."""
        # Get plot specs from config
        plot_specs = self.plot_config.get_multi_run_plot_specs()

        # Generate initial plots from config specs
        initial_plot_containers = []
        for spec in plot_specs:
            # Extract metrics
            x_metric_spec = next((m for m in spec.metrics if m.axis == "x"), None)
            y_metric_spec = next((m for m in spec.metrics if m.axis == "y"), None)

            if not x_metric_spec or not y_metric_spec:
                continue

            # Use default stats from spec
            x_stat = x_metric_spec.stat or "p50"
            y_stat = y_metric_spec.stat or "avg"

            # Create DataFrame
            result = runs_to_dataframe(
                self.runs, x_metric_spec.name, x_stat, y_metric_spec.name, y_stat
            )
            df = result["df"]

            if df.empty:
                continue

            # Generate plot
            title = (
                spec.title
                or f"{y_metric_spec.name.replace('_', ' ').title()} vs {x_metric_spec.name.replace('_', ' ').title()}"
            )
            label_by = spec.label_by or "concurrency"
            group_by = spec.group_by or "model"

            # Extract experiment_types mapping for color assignment
            experiment_types = None
            if "experiment_type" in df.columns and group_by:
                group_col = group_by[0] if isinstance(group_by, list) else group_by
                if group_col in df.columns:
                    experiment_types = {
                        g: df[df[group_col] == g]["experiment_type"].iloc[0]
                        for g in df[group_col].unique()
                    }

            if spec.plot_type.value == "pareto":
                fig = self.plot_generator.create_pareto_plot(
                    df,
                    x_metric_spec.name,
                    y_metric_spec.name,
                    label_by=label_by,
                    group_by=group_by,
                    title=title,
                    experiment_types=experiment_types,
                )
            elif spec.plot_type.value in ["scatter_line", "scatter"]:
                fig = self.plot_generator.create_scatter_line_plot(
                    df,
                    x_metric_spec.name,
                    y_metric_spec.name,
                    label_by=label_by,
                    group_by=group_by
                    if spec.plot_type.value == "scatter_line"
                    else None,
                    title=title,
                    experiment_types=experiment_types,
                )
            else:
                continue

            # Add run_idx for click functionality
            fig = add_run_idx_to_figure(fig, df)

            # Create container
            initial_plot_containers.append(self._create_plot_container(spec.name, fig))

        return [
            html.Div(
                initial_plot_containers
                + [
                    # Add Plot button
                    html.Div(
                        [
                            html.Div("+", className="plot-add-icon"),
                            html.Div(
                                "Create Custom Plot (Multi-Run)",
                                className="plot-add-text",
                            ),
                        ],
                        id="add-multirun-plot-slot",
                        n_clicks=0,
                        className="plot-add-slot",
                    ),
                ],
                id="plots-grid",
                style={
                    "display": "grid",
                    "grid-template-columns": "1fr 1fr",
                    "gap": "12px",
                    "padding": "0 16px 20px 55px",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "max-height": "calc(100vh - 70px)",
                    "width": "100%",
                    "box-sizing": "border-box",
                },
            ),
        ]

    def _build_single_run_tab(self) -> list:
        """Build single-run analysis tab with dynamic plot rendering."""
        run = self.runs[0]

        return [
            html.Div(
                f"Analyzing run: {run.metadata.run_name}",
                style={
                    "font-size": "12px",
                    "color": self.colors["text"],
                    "margin": "12px 16px 12px 55px",
                    "font-family": PLOT_FONT_FAMILY,
                    "font-weight": "500",
                },
            ),
            html.Div(
                [],
                id="plots-grid",
                style={
                    "display": "grid",
                    "grid-template-columns": "1fr 1fr",
                    "gap": "12px",
                    "padding": "0 16px 20px 55px",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "max-height": "calc(100vh - 70px)",
                    "width": "100%",
                    "box-sizing": "border-box",
                },
            ),
        ]

    def _create_plot_container(
        self, plot_id: str, figure, resizable: bool = True
    ) -> html.Div:
        """
        Create a plot container with optional resize handles.

        Args:
            plot_id: Unique ID for the plot
            figure: Plotly figure object
            resizable: Whether the plot should be resizable

        Returns:
            Dash HTML Div containing the plot
        """
        # All plots default to half-width for uniform initial sizing
        size_class = "half"

        # Use shared function from utils.py (same as callbacks.py uses)
        return create_plot_container_component(
            plot_id=plot_id,
            figure=figure,
            theme=self.theme,
            resizable=resizable,
            size=400,
            size_class=size_class,
        )

    def _build_hidden_plots_section(self) -> html.Div:
        """
        Build section showing hidden plots with restore buttons.

        Returns:
            Dash HTML Div with hidden plots list
        """
        return html.Div(
            [
                html.Div(
                    "Recently Deleted Plots",
                    style={
                        "font-size": "11px",
                        "font-weight": "600",
                        "text-transform": "uppercase",
                        "letter-spacing": "0.8px",
                        "color": NVIDIA_GRAY,
                        "margin-bottom": "8px",
                        "font-family": PLOT_FONT_FAMILY,
                    },
                ),
                html.Div(
                    id="hidden-plots-list",
                    children=[
                        html.Div(
                            "No deleted plots",
                            style={
                                "font-size": "11px",
                                "color": NVIDIA_GRAY,
                                "font-style": "italic",
                                "font-family": PLOT_FONT_FAMILY,
                            },
                        )
                    ],
                ),
            ],
            style={"margin-bottom": "20px"},
        )

    def _build_config_modal(self) -> dbc.Modal:
        """
        Build modal for displaying run configuration.

        Returns:
            Dash Bootstrap Modal component
        """
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    html.Div(
                        [
                            html.Span(
                                "Run Configuration",
                                style={"fontWeight": "500", "fontSize": "18px"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-chart-line me-2"),
                                            "View Single-Run Plots",
                                        ],
                                        id="btn-view-single-run",
                                        n_clicks=0,
                                        size="sm",
                                        style={
                                            "background": "#76b900",
                                            "border": "none",
                                            "color": "white",
                                            "fontWeight": "500",
                                            "marginRight": "12px",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"}
                                if self.mode == VisualizationMode.MULTI_RUN
                                else {"display": "none"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "width": "100%",
                        },
                    ),
                    id="config-modal-header-container",
                    style={
                        "background-color": self.colors["paper"],
                        "color": self.colors["text"],
                        "border-bottom": f"1px solid {self.colors['border']}",
                    },
                ),
                dbc.ModalBody(
                    [
                        html.Div(
                            id="config-modal-header",
                            style={
                                "font-size": "16px",
                                "font-weight": "600",
                                "color": self.colors["text"],
                                "margin-bottom": "16px",
                                "padding-bottom": "8px",
                                "border-bottom": f"1px solid {self.colors['border']}",
                            },
                        ),
                        html.Div(
                            id="config-modal-summary",
                            style={
                                "margin-bottom": "16px",
                            },
                        ),
                        html.Div(
                            "Full Configuration",
                            id="config-modal-yaml-label",
                            style={
                                "font-size": "12px",
                                "font-weight": "600",
                                "color": self.colors["text"],
                                "margin-bottom": "8px",
                                "font-family": PLOT_FONT_FAMILY,
                            },
                        ),
                        html.Pre(
                            id="config-modal-yaml",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "padding": "12px",
                                "border-radius": "4px",
                                "overflow-x": "auto",
                                "font-family": "monospace",
                                "font-size": "11px",
                                "line-height": "1.5",
                                "border": f"1px solid {self.colors['border']}",
                                "max-height": "400px",
                                "overflow-y": "auto",
                            },
                        ),
                    ],
                    id="config-modal-body-container",
                    style={
                        "background-color": self.colors["background"],
                        "color": self.colors["text"],
                    },
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Close",
                            id="btn-close-config-modal",
                            className="ml-auto",
                            style={
                                "background": NVIDIA_GREEN,
                                "border": "none",
                                "color": "white",
                            },
                        ),
                    ],
                    id="config-modal-footer-container",
                    style={
                        "background-color": self.colors["paper"],
                        "border-top": f"1px solid {self.colors['border']}",
                    },
                ),
            ],
            id="config-modal",
            size="lg",
            is_open=False,
            backdrop=False,
            className=f"config-modal-{self.theme.value}",
            style={"color": self.colors["text"]},
        )

    def _build_custom_plot_modal(self) -> dbc.Modal:
        """
        Build modal for creating custom plots with all available metrics.

        Returns:
            Dash Bootstrap Modal component with form
        """
        # Discover all available metrics from data
        all_metrics = self._get_all_available_metrics()

        # Build metric options grouped by category (order from METRIC_CATEGORY_RULES)
        metric_options = []
        for category in METRIC_CATEGORY_RULES:
            category_metrics = [
                {"label": info["display"], "value": name}
                for name, info in all_metrics.items()
                if info["category"] == category
            ]
            metric_options.extend(category_metrics)

        # Stat options for custom plots (consistent ordering)
        stat_options = self._get_stat_options_ordered()

        # Detect swept parameters for group_by options
        swept_params = self._detect_swept_parameters()

        # Base options that are always available
        base_options = ["model", "concurrency"]
        metadata_options = [
            {"label": "Endpoint Type", "value": "endpoint_type"},
            {"label": "Request Count", "value": "request_count"},
            {"label": "Duration (seconds)", "value": "duration_seconds"},
        ]

        # Build group_by options
        group_by_options = [
            {"label": "Model", "value": "model"},
            {"label": "Concurrency", "value": "concurrency"},
        ]

        # Add experiment group if experimental classification is enabled
        exp_class_config = self.plot_config.get_experiment_classification_config()
        if exp_class_config is not None:
            group_by_options.append(
                {"label": "Experiment Group", "value": "experiment_group"}
            )

        group_by_options.extend(metadata_options)

        # Add swept parameters (exclude already listed options)
        for param in swept_params:
            if param not in base_options + [
                "endpoint_type",
                "request_count",
                "duration_seconds",
                "experiment_group",
            ]:
                display_name = param.replace("_", " ").replace(".", " ").title()
                group_by_options.append({"label": display_name, "value": param})

        group_by_options.append({"label": "None (Single Series)", "value": "none"})

        # Label by options
        label_by_options = [
            {"label": "Concurrency", "value": "concurrency"},
            {"label": "Model", "value": "model"},
            {"label": "Run Name", "value": "run_name"},
        ]
        label_by_options.extend(metadata_options)

        # Add swept parameters to label_by as well
        for param in swept_params:
            if param not in base_options + [
                "endpoint_type",
                "request_count",
                "duration_seconds",
                "run_name",
            ]:
                display_name = param.replace("_", " ").replace(".", " ").title()
                label_by_options.append({"label": display_name, "value": param})

        label_by_options.append({"label": "None", "value": "none"})

        return dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Create Custom Plot (Multi-Run)"),
                    id="custom-plot-modal-header-container",
                    style={
                        "background-color": self.colors["paper"],
                        "color": self.colors["text"],
                        "border-bottom": f"1px solid {self.colors['border']}",
                    },
                ),
                dbc.ModalBody(
                    [
                        create_label("Plot Type", self.theme),
                        dcc.Dropdown(
                            id="custom-plot-type",
                            options=MULTI_RUN_PLOT_TYPES,
                            placeholder="Select plot type",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        create_label("X-Axis Metric", self.theme),
                        dcc.Dropdown(
                            id="custom-x-metric",
                            options=metric_options,
                            placeholder="Select metric",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        create_label("X-Axis Statistic", self.theme),
                        dcc.Dropdown(
                            id="custom-x-stat",
                            options=stat_options,
                            placeholder="Select statistic",
                            style={"font-size": "12px"},
                        ),
                        html.Div(
                            id="custom-x-stat-warning",
                            style={
                                "font-size": "11px",
                                "color": "#ff9800",
                                "min-height": "14px",
                                "margin-bottom": "16px",
                            },
                        ),
                        create_label("Y-Axis Metric", self.theme),
                        dcc.Dropdown(
                            id="custom-y-metric",
                            options=metric_options,
                            placeholder="Select metric",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        create_label("Y-Axis Statistic", self.theme),
                        dcc.Dropdown(
                            id="custom-y-stat",
                            options=stat_options,
                            placeholder="Select statistic",
                            style={"font-size": "12px"},
                        ),
                        html.Div(
                            id="custom-y-stat-warning",
                            style={
                                "font-size": "11px",
                                "color": "#ff9800",
                                "min-height": "14px",
                                "margin-bottom": "16px",
                            },
                        ),
                        create_label("Label Points By", self.theme),
                        dcc.Dropdown(
                            id="custom-label-by",
                            options=label_by_options,
                            placeholder="Select option",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        create_label("Group/Color By", self.theme),
                        dcc.Dropdown(
                            id="custom-group-by",
                            options=group_by_options,
                            placeholder="Select option",
                            style={"font-size": "12px", "margin-bottom": "16px"},
                        ),
                        html.Details(
                            [
                                html.Summary(
                                    "More Options",
                                    style={
                                        "cursor": "pointer",
                                        "font-weight": "600",
                                        "font-size": "12px",
                                        "color": self.colors["text"],
                                        "margin-bottom": "8px",
                                    },
                                ),
                                html.Div(
                                    [
                                        create_label("Title", self.theme),
                                        dcc.Input(
                                            id="custom-plot-title",
                                            type="text",
                                            placeholder="Custom title (leave blank for auto)",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                            },
                                        ),
                                        create_label("X-Axis Label", self.theme),
                                        dcc.Input(
                                            id="custom-x-label",
                                            type="text",
                                            placeholder="X-axis label",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("Y-Axis Label", self.theme),
                                        dcc.Input(
                                            id="custom-y-label",
                                            type="text",
                                            placeholder="Y-axis label",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "8px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("X-Axis Log Scale", self.theme),
                                        dcc.Dropdown(
                                            id="custom-x-log-switch",
                                            options=[
                                                {"label": "Off", "value": False},
                                                {"label": "On", "value": True},
                                            ],
                                            value=False,
                                            clearable=False,
                                            style={
                                                "font-size": "12px",
                                                "margin-bottom": "8px",
                                            },
                                        ),
                                        create_label("X-Axis Autoscale", self.theme),
                                        dcc.Dropdown(
                                            id="custom-x-autoscale-switch",
                                            options=[
                                                {"label": "Off", "value": False},
                                                {"label": "On", "value": True},
                                            ],
                                            value=False,
                                            clearable=False,
                                            style={
                                                "font-size": "12px",
                                                "margin-bottom": "12px",
                                            },
                                        ),
                                        create_label("Y-Axis Log Scale", self.theme),
                                        dcc.Dropdown(
                                            id="custom-y-log-switch",
                                            options=[
                                                {"label": "Off", "value": False},
                                                {"label": "On", "value": True},
                                            ],
                                            value=False,
                                            clearable=False,
                                            style={
                                                "font-size": "12px",
                                                "margin-bottom": "8px",
                                            },
                                        ),
                                        create_label("Y-Axis Autoscale", self.theme),
                                        dcc.Dropdown(
                                            id="custom-y-autoscale-switch",
                                            options=[
                                                {"label": "Off", "value": False},
                                                {"label": "On", "value": True},
                                            ],
                                            value=False,
                                            clearable=False,
                                            style={"font-size": "12px"},
                                        ),
                                    ],
                                    style={"padding": "12px 0"},
                                ),
                            ],
                            open=False,
                        ),
                    ],
                    id="custom-plot-modal-body-container",
                    style={
                        "background-color": self.colors["background"],
                        "color": self.colors["text"],
                    },
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Create",
                            id="btn-create-custom-plot",
                            style={
                                "background": NVIDIA_GREEN,
                                "border": "none",
                                "margin-right": "8px",
                                "color": "white",
                            },
                        ),
                        dbc.Button(
                            "Cancel",
                            id="btn-cancel-custom-plot",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {self.colors['border']}",
                            },
                        ),
                    ],
                    id="custom-plot-modal-footer-container",
                    style={
                        "background-color": self.colors["paper"],
                        "border-top": f"1px solid {self.colors['border']}",
                    },
                ),
            ],
            id="custom-plot-modal",
            size="md",
            is_open=False,
            className=f"theme-{self.theme.value}",
            style={"color": self.colors["text"]},
        )

    def _build_single_run_custom_plot_modal(self) -> dbc.Modal:
        """
        Build modal for creating custom single-run plots.

        Returns:
            Dash Bootstrap Modal component with form for single-run plot creation
        """
        run = self.runs[0]

        # Build plot type options based on available data with section headers
        available_types = [
            PlotType.SCATTER,
            PlotType.SCATTER_WITH_PERCENTILES,
            PlotType.REQUEST_TIMELINE,
        ]
        if run.timeslices is not None and not run.timeslices.empty:
            available_types.append(PlotType.TIMESLICE)
        available_types.append(PlotType.AREA)
        if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
            available_types.append(PlotType.DUAL_AXIS)

        plot_type_options = _build_plot_type_options_with_headers(available_types)

        request_metrics = []
        metric_stats: dict[str, list[str]] = {}
        if run.requests is not None and not run.requests.empty:
            request_metrics, metric_stats = get_single_run_metrics_with_stats(
                list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
            )

        # GPU metrics (for dual-axis plots and y-axis options)
        # Filter to only include plottable GPU metrics (excludes gpu_uuid, etc.)
        plottable_gpu_metrics = set(get_gpu_metrics())
        gpu_metrics = []
        if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
            gpu_metrics = [
                {"label": get_metric_display_name_with_unit(col), "value": col}
                for col in run.gpu_telemetry.columns
                if col in plottable_gpu_metrics
            ]

        # Build y-axis metric options (request + GPU if available)
        y_metric_options = request_metrics.copy()
        if gpu_metrics:
            y_metric_options.append(
                {
                    "label": "── GPU Metrics ──",
                    "value": "_gpu_divider",
                    "disabled": True,
                }
            )
            y_metric_options.extend(gpu_metrics)

        # Stat options for histogram (kept for backward compatibility)
        stat_options = self._get_stat_options_ordered()

        # Y-axis stat options (default to Average only)
        y_stat_options = [{"label": "Average", "value": "avg"}]

        return dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Create Custom Plot (Single-Run)"),
                    id="single-run-custom-plot-modal-header",
                    style={
                        "background-color": self.colors["paper"],
                        "color": self.colors["text"],
                        "border-bottom": f"1px solid {self.colors['border']}",
                    },
                ),
                dbc.ModalBody(
                    [
                        dcc.Store(
                            id="single-run-metric-stats-store", data=metric_stats
                        ),
                        dcc.Store(
                            id="single-run-request-metrics-store", data=y_metric_options
                        ),
                        create_label("Plot Type", self.theme),
                        dcc.Dropdown(
                            id="single-run-plot-type",
                            options=plot_type_options,
                            placeholder="Select plot type",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        html.Div(
                            [
                                create_label("X-Axis", self.theme),
                                dcc.Dropdown(
                                    id="single-run-x-axis",
                                    options=[
                                        {
                                            "label": "Request Number",
                                            "value": "request_number",
                                        },
                                        {
                                            "label": "Timestamp (s)",
                                            "value": "timestamp_s",
                                        },
                                    ],
                                    placeholder="Select X-axis",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                            ],
                            id="single-run-x-axis-container",
                        ),
                        create_label("Y-Axis Metric", self.theme),
                        dcc.Dropdown(
                            id="single-run-y-metric",
                            options=y_metric_options,
                            placeholder="Select metric",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        html.Div(
                            [
                                create_label("Y-Axis Statistic", self.theme),
                                dcc.Dropdown(
                                    id="single-run-y-stat",
                                    options=y_stat_options,
                                    placeholder="Select statistic",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                            ],
                            id="single-run-y-stat-container",
                        ),
                        html.Div(
                            [
                                create_label("Secondary Y-Axis Metric", self.theme),
                                dcc.Dropdown(
                                    id="single-run-y2-metric",
                                    options=y_metric_options,
                                    placeholder="Select secondary metric",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                            ],
                            id="single-run-y2-container",
                            style={"display": "none"},
                        ),
                        html.Details(
                            [
                                html.Summary(
                                    "More Options",
                                    style={
                                        "cursor": "pointer",
                                        "font-weight": "600",
                                        "font-size": "12px",
                                        "color": self.colors["text"],
                                        "margin-bottom": "8px",
                                    },
                                ),
                                html.Div(
                                    [
                                        create_label("Title", self.theme),
                                        dcc.Input(
                                            id="single-run-plot-title",
                                            type="text",
                                            placeholder="Custom title (leave blank for auto)",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("X-Axis Label", self.theme),
                                        dcc.Input(
                                            id="single-run-x-label",
                                            type="text",
                                            placeholder="Custom label (leave blank for auto)",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("Y-Axis Label", self.theme),
                                        dcc.Input(
                                            id="single-run-y-label",
                                            type="text",
                                            placeholder="Custom label (leave blank for auto)",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                create_label(
                                                    "Timeslice Statistic", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="single-run-stat",
                                                    options=stat_options,
                                                    placeholder="Select statistic",
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "12px",
                                                    },
                                                ),
                                            ],
                                            id="single-run-stat-container",
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            [
                                                create_label(
                                                    "Secondary Y-Axis Label", self.theme
                                                ),
                                                dcc.Input(
                                                    id="single-run-y2-label",
                                                    type="text",
                                                    placeholder="Custom label (leave blank for auto)",
                                                    style={
                                                        "font-size": "12px",
                                                        "width": "100%",
                                                        "background-color": self.colors[
                                                            "paper"
                                                        ],
                                                        "color": self.colors["text"],
                                                        "border": f"1px solid {self.colors['border']}",
                                                        "padding": "6px 8px",
                                                        "border-radius": "4px",
                                                    },
                                                ),
                                            ],
                                            id="single-run-y2-label-container",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"padding": "12px 0"},
                                ),
                            ],
                            open=False,
                        ),
                    ],
                    id="single-run-custom-plot-modal-body",
                    style={
                        "background-color": self.colors["background"],
                        "color": self.colors["text"],
                    },
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Create",
                            id="btn-create-single-run-custom-plot",
                            style={
                                "background": NVIDIA_GREEN,
                                "border": "none",
                                "margin-right": "8px",
                                "color": "white",
                            },
                        ),
                        dbc.Button(
                            "Cancel",
                            id="btn-cancel-single-run-custom-plot",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {self.colors['border']}",
                            },
                        ),
                    ],
                    id="single-run-custom-plot-modal-footer",
                    style={
                        "background-color": self.colors["paper"],
                        "border-top": f"1px solid {self.colors['border']}",
                    },
                ),
            ],
            id="single-run-custom-plot-modal",
            size="md",
            is_open=False,
            className=f"theme-{self.theme.value}",
            style={"color": self.colors["text"]},
        )

    def _build_single_run_edit_modal(self) -> dbc.Modal:
        """
        Build modal for editing existing single-run plots.

        Returns:
            Dash Bootstrap Modal component with form pre-populated for single-run plot editing
        """
        run = self.runs[0]

        # Build plot type options based on available data with section headers
        available_types = [
            PlotType.SCATTER,
            PlotType.SCATTER_WITH_PERCENTILES,
            PlotType.REQUEST_TIMELINE,
        ]
        if run.timeslices is not None and not run.timeslices.empty:
            available_types.append(PlotType.TIMESLICE)
        available_types.append(PlotType.AREA)
        if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
            available_types.append(PlotType.DUAL_AXIS)

        plot_type_options = _build_plot_type_options_with_headers(available_types)

        request_metrics = []
        metric_stats: dict[str, list[str]] = {}
        if run.requests is not None and not run.requests.empty:
            request_metrics, metric_stats = get_single_run_metrics_with_stats(
                list(run.requests.columns), EXCLUDED_METRIC_COLUMNS
            )

        # GPU metrics (for dual-axis plots and y-axis options)
        # Filter to only include plottable GPU metrics (excludes gpu_uuid, etc.)
        plottable_gpu_metrics = set(get_gpu_metrics())
        gpu_metrics = []
        if run.gpu_telemetry is not None and not run.gpu_telemetry.empty:
            gpu_metrics = [
                {"label": get_metric_display_name_with_unit(col), "value": col}
                for col in run.gpu_telemetry.columns
                if col in plottable_gpu_metrics
            ]

        # Build y-axis metric options (request + GPU if available)
        y_metric_options = request_metrics.copy()
        if gpu_metrics:
            y_metric_options.append(
                {
                    "label": "── GPU Metrics ──",
                    "value": "_gpu_divider",
                    "disabled": True,
                }
            )
            y_metric_options.extend(gpu_metrics)

        # Stat options for histogram (kept for backward compatibility)
        stat_options = self._get_stat_options_ordered()

        # Y-axis stat options (default to Average only)
        y_stat_options = [{"label": "Average", "value": "avg"}]

        return dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Edit Plot Configuration (Single-Run)"),
                    id="edit-sr-plot-modal-header",
                    style={
                        "background-color": self.colors["paper"],
                        "color": self.colors["text"],
                        "border-bottom": f"1px solid {self.colors['border']}",
                    },
                ),
                dbc.ModalBody(
                    [
                        dcc.Store(id="edit-sr-plot-id-store", data=None),
                        dcc.Store(id="edit-sr-metric-stats-store", data=metric_stats),
                        dcc.Store(
                            id="edit-sr-request-metrics-store", data=y_metric_options
                        ),
                        dcc.Store(id="edit-sr-original-y-metric-store", data=None),
                        create_label("Plot Type", self.theme),
                        dcc.Dropdown(
                            id="edit-sr-plot-type",
                            options=plot_type_options,
                            value="scatter",
                            clearable=False,
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        html.Div(
                            [
                                create_label("X-Axis", self.theme),
                                dcc.Dropdown(
                                    id="edit-sr-x-axis",
                                    options=[
                                        {
                                            "label": "Request Number",
                                            "value": "request_number",
                                        },
                                        {
                                            "label": "Timestamp (s)",
                                            "value": "timestamp_s",
                                        },
                                    ],
                                    value="request_number",
                                    clearable=False,
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                            ],
                            id="edit-sr-x-axis-container",
                        ),
                        create_label("Y-Axis Metric", self.theme),
                        dcc.Dropdown(
                            id="edit-sr-y-metric",
                            options=y_metric_options,
                            placeholder="Select metric",
                            style={"font-size": "12px", "margin-bottom": "12px"},
                        ),
                        html.Div(
                            [
                                create_label("Y-Axis Statistic", self.theme),
                                dcc.Dropdown(
                                    id="edit-sr-y-stat",
                                    options=y_stat_options,
                                    placeholder="Select stat",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                            ],
                            id="edit-sr-y-stat-container",
                        ),
                        html.Details(
                            [
                                html.Summary(
                                    "More Options",
                                    style={
                                        "cursor": "pointer",
                                        "font-weight": "600",
                                        "font-size": "12px",
                                        "color": self.colors["text"],
                                        "margin-bottom": "8px",
                                    },
                                ),
                                html.Div(
                                    [
                                        create_label("Title", self.theme),
                                        dcc.Input(
                                            id="edit-sr-plot-title",
                                            type="text",
                                            placeholder="Custom title (leave blank for auto)",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("X-Axis Label", self.theme),
                                        dcc.Input(
                                            id="edit-sr-x-label",
                                            type="text",
                                            placeholder="X-axis label",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        create_label("Y-Axis Label", self.theme),
                                        dcc.Input(
                                            id="edit-sr-y-label",
                                            type="text",
                                            placeholder="Y-axis label",
                                            style={
                                                "font-size": "12px",
                                                "width": "100%",
                                                "margin-bottom": "12px",
                                                "background-color": self.colors[
                                                    "paper"
                                                ],
                                                "color": self.colors["text"],
                                                "border": f"1px solid {self.colors['border']}",
                                                "padding": "6px 8px",
                                                "border-radius": "4px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                create_label(
                                                    "Timeslice Statistic", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-sr-stat",
                                                    options=stat_options,
                                                    value="avg",
                                                    clearable=False,
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "12px",
                                                    },
                                                ),
                                            ],
                                            id="edit-sr-stat-container",
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            [
                                                create_label(
                                                    "Secondary Metric", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-sr-y2-metric",
                                                    options=y_metric_options,
                                                    placeholder="Select secondary metric",
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "8px",
                                                    },
                                                ),
                                                create_label(
                                                    "Secondary Y-Axis Label", self.theme
                                                ),
                                                dcc.Input(
                                                    id="edit-sr-y2-label",
                                                    type="text",
                                                    placeholder="Secondary Y-axis label",
                                                    style={
                                                        "font-size": "12px",
                                                        "width": "100%",
                                                        "background-color": self.colors[
                                                            "paper"
                                                        ],
                                                        "color": self.colors["text"],
                                                        "border": f"1px solid {self.colors['border']}",
                                                        "padding": "6px 8px",
                                                        "border-radius": "4px",
                                                    },
                                                ),
                                            ],
                                            id="edit-sr-y2-container",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    style={"padding": "12px 0"},
                                ),
                            ],
                            open=False,
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="edit-sr-plot-size",
                                    options=[
                                        {"label": "Small (1 column)", "value": "half"},
                                        {"label": "Large (2 columns)", "value": "full"},
                                    ],
                                    value="half",
                                    clearable=False,
                                ),
                            ],
                            style={"display": "none"},
                        ),
                    ],
                    id="edit-sr-plot-modal-body",
                    style={
                        "background-color": self.colors["background"],
                        "color": self.colors["text"],
                    },
                ),
                dbc.ModalFooter(
                    [
                        # Hide button on the left
                        dbc.Button(
                            "Hide Plot",
                            id="btn-sr-hide-plot",
                            n_clicks=0,
                            size="sm",
                            style={
                                "background": "#d32f2f",
                                "color": "white",
                                "border": "none",
                                "margin-right": "auto",
                                "padding": "4px 12px",
                            },
                        ),
                        # Action buttons on the right
                        dbc.Button(
                            "Update",
                            id="btn-sr-update-plot",
                            size="sm",
                            style={
                                "background": NVIDIA_GREEN,
                                "border": "none",
                                "color": "white",
                                "padding": "4px 12px",
                            },
                        ),
                        dbc.Button(
                            "Save As New",
                            id="btn-sr-save-as-new",
                            size="sm",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {NVIDIA_GREEN}",
                                "padding": "4px 12px",
                            },
                        ),
                        dbc.Button(
                            "Cancel",
                            id="btn-sr-cancel-edit",
                            size="sm",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {self.colors['border']}",
                                "padding": "4px 12px",
                            },
                        ),
                    ],
                    id="edit-sr-plot-modal-footer",
                    style={
                        "background-color": self.colors["paper"],
                        "border-top": f"1px solid {self.colors['border']}",
                        "display": "flex",
                        "flex-wrap": "nowrap",
                        "justify-content": "space-between",
                        "align-items": "center",
                        "gap": "6px",
                        "padding": "10px 12px",
                    },
                ),
            ],
            id="edit-single-run-plot-modal",
            size="md",
            is_open=False,
            className=f"theme-{self.theme.value}",
            style={"color": self.colors["text"]},
        )

    def _build_plot_edit_modal(self) -> dbc.Modal:
        """
        Build modal for editing existing multi-run plot configuration.

        Returns:
            Dash Bootstrap Modal component with form pre-populated with multi-run plot config
        """
        # Discover all available metrics from data
        all_metrics = self._get_all_available_metrics()

        # Build metric options grouped by category (order from METRIC_CATEGORY_RULES)
        metric_options = []
        for category in METRIC_CATEGORY_RULES:
            category_metrics = [
                {"label": info["display"], "value": name}
                for name, info in all_metrics.items()
                if info["category"] == category
            ]
            metric_options.extend(category_metrics)

        # Stat options
        stat_options = self._get_stat_options_ordered()

        return dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Edit Plot Configuration (Multi-Run)"),
                    id="edit-plot-modal-header-container",
                    style={
                        "background-color": self.colors["paper"],
                        "color": self.colors["text"],
                        "border-bottom": f"1px solid {self.colors['border']}",
                    },
                ),
                dbc.ModalBody(
                    [
                        dcc.Store(id="edit-plot-id-store", data=None),
                        dcc.Store(id="edit-original-x-metric-store", data=None),
                        dcc.Store(id="edit-original-y-metric-store", data=None),
                        html.Div(
                            [
                                create_label("X-Axis Metric", self.theme),
                                dcc.Dropdown(
                                    id="edit-x-metric",
                                    options=metric_options,
                                    placeholder="Select metric",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "12px",
                                    },
                                ),
                                create_label("X-Axis Statistic", self.theme),
                                dcc.Dropdown(
                                    id="edit-x-stat",
                                    options=stat_options,
                                    value="p50",
                                    placeholder="Select stat",
                                    style={"font-size": "12px"},
                                ),
                                html.Div(
                                    id="edit-x-stat-warning",
                                    style={
                                        "font-size": "11px",
                                        "color": "#ff9800",
                                        "min-height": "14px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                                create_label("Y-Axis Metric", self.theme),
                                dcc.Dropdown(
                                    id="edit-y-metric",
                                    options=metric_options,
                                    placeholder="Select metric",
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "12px",
                                    },
                                ),
                                create_label("Y-Axis Statistic", self.theme),
                                dcc.Dropdown(
                                    id="edit-y-stat",
                                    options=stat_options,
                                    placeholder="Select stat",
                                    style={"font-size": "12px"},
                                ),
                                html.Div(
                                    id="edit-y-stat-warning",
                                    style={
                                        "font-size": "11px",
                                        "color": "#ff9800",
                                        "min-height": "14px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                                create_label("Plot Type", self.theme),
                                dcc.Dropdown(
                                    id="edit-plot-type",
                                    options=MULTI_RUN_PLOT_TYPES,
                                    clearable=False,
                                    style={
                                        "font-size": "12px",
                                        "margin-bottom": "16px",
                                    },
                                ),
                                html.Details(
                                    [
                                        html.Summary(
                                            "More Options",
                                            style={
                                                "cursor": "pointer",
                                                "font-weight": "600",
                                                "font-size": "12px",
                                                "color": self.colors["text"],
                                                "margin-bottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                create_label("Title", self.theme),
                                                dcc.Input(
                                                    id="edit-plot-title",
                                                    type="text",
                                                    placeholder="Custom title (leave blank for auto)",
                                                    style={
                                                        "font-size": "12px",
                                                        "width": "100%",
                                                        "margin-bottom": "12px",
                                                    },
                                                ),
                                                create_label(
                                                    "X-Axis Label", self.theme
                                                ),
                                                dcc.Input(
                                                    id="edit-x-label",
                                                    type="text",
                                                    placeholder="X-axis label",
                                                    style={
                                                        "font-size": "12px",
                                                        "width": "100%",
                                                        "margin-bottom": "8px",
                                                        "background-color": self.colors[
                                                            "paper"
                                                        ],
                                                        "color": self.colors["text"],
                                                        "border": f"1px solid {self.colors['border']}",
                                                        "padding": "6px 8px",
                                                        "border-radius": "4px",
                                                    },
                                                ),
                                                create_label(
                                                    "X-Axis Log Scale", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-x-log-switch",
                                                    options=[
                                                        {
                                                            "label": "Off",
                                                            "value": False,
                                                        },
                                                        {"label": "On", "value": True},
                                                    ],
                                                    value=False,
                                                    clearable=False,
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "8px",
                                                    },
                                                ),
                                                create_label(
                                                    "X-Axis Autoscale", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-x-autoscale-switch",
                                                    options=[
                                                        {
                                                            "label": "Off",
                                                            "value": False,
                                                        },
                                                        {"label": "On", "value": True},
                                                    ],
                                                    value=False,
                                                    clearable=False,
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "12px",
                                                    },
                                                ),
                                                create_label(
                                                    "Y-Axis Label", self.theme
                                                ),
                                                dcc.Input(
                                                    id="edit-y-label",
                                                    type="text",
                                                    placeholder="Y-axis label",
                                                    style={
                                                        "font-size": "12px",
                                                        "width": "100%",
                                                        "margin-bottom": "8px",
                                                        "background-color": self.colors[
                                                            "paper"
                                                        ],
                                                        "color": self.colors["text"],
                                                        "border": f"1px solid {self.colors['border']}",
                                                        "padding": "6px 8px",
                                                        "border-radius": "4px",
                                                    },
                                                ),
                                                create_label(
                                                    "Y-Axis Log Scale", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-y-log-switch",
                                                    options=[
                                                        {
                                                            "label": "Off",
                                                            "value": False,
                                                        },
                                                        {"label": "On", "value": True},
                                                    ],
                                                    value=False,
                                                    clearable=False,
                                                    style={
                                                        "font-size": "12px",
                                                        "margin-bottom": "8px",
                                                    },
                                                ),
                                                create_label(
                                                    "Y-Axis Autoscale", self.theme
                                                ),
                                                dcc.Dropdown(
                                                    id="edit-y-autoscale-switch",
                                                    options=[
                                                        {
                                                            "label": "Off",
                                                            "value": False,
                                                        },
                                                        {"label": "On", "value": True},
                                                    ],
                                                    value=False,
                                                    clearable=False,
                                                    style={"font-size": "12px"},
                                                ),
                                            ],
                                            style={"padding": "12px 0"},
                                        ),
                                    ],
                                    open=False,
                                ),
                            ],
                            id="edit-multi-run-fields",
                            style={"display": "block"},
                        ),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="edit-plot-size",
                                    options=[
                                        {"label": "Small (1 column)", "value": "half"},
                                        {"label": "Large (2 columns)", "value": "full"},
                                    ],
                                    value="half",
                                    clearable=False,
                                ),
                            ],
                            style={"display": "none"},
                        ),
                    ],
                    id="edit-plot-modal-body-container",
                    style={
                        "background-color": self.colors["background"],
                        "color": self.colors["text"],
                    },
                ),
                dbc.ModalFooter(
                    [
                        # Hide button on the left
                        dbc.Button(
                            "Hide Plot",
                            id="btn-hide-plot-from-modal",
                            n_clicks=0,
                            size="sm",
                            style={
                                "background": "#d32f2f",
                                "color": "white",
                                "border": "none",
                                "margin-right": "auto",
                                "padding": "4px 12px",
                            },
                        ),
                        # Action buttons on the right
                        dbc.Button(
                            "Update",
                            id="btn-update-plot",
                            size="sm",
                            style={
                                "background": NVIDIA_GREEN,
                                "border": "none",
                                "color": "white",
                                "padding": "4px 12px",
                            },
                        ),
                        dbc.Button(
                            "Save As New",
                            id="btn-save-as-new-plot",
                            size="sm",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {NVIDIA_GREEN}",
                                "padding": "4px 12px",
                            },
                        ),
                        dbc.Button(
                            "Cancel",
                            id="btn-cancel-edit-plot",
                            size="sm",
                            style={
                                "background": self.colors["paper"],
                                "color": self.colors["text"],
                                "border": f"1px solid {self.colors['border']}",
                                "padding": "4px 12px",
                            },
                        ),
                    ],
                    id="edit-plot-modal-footer-container",
                    style={
                        "background-color": self.colors["paper"],
                        "border-top": f"1px solid {self.colors['border']}",
                        "display": "flex",
                        "flex-wrap": "nowrap",
                        "justify-content": "space-between",
                        "align-items": "center",
                        "gap": "6px",
                        "padding": "10px 12px",
                    },
                ),
            ],
            id="edit-plot-modal",
            size="md",
            is_open=False,
            className=f"theme-{self.theme.value}",
            style={"color": self.colors["text"]},
        )

    def _build_context_menu(self) -> html.Div:
        """
        Build right-click context menu for plots.

        Returns:
            Hidden context menu component that appears on right-click
        """
        return html.Div(
            [
                # Hidden store for plot_id that was right-clicked
                dcc.Store(id="context-menu-plot-id", data=None),
                # Context menu container
                html.Div(
                    [
                        html.Div(
                            "⚙ Edit Plot Settings",
                            id="context-menu-edit",
                            className="context-menu-item",
                        ),
                        html.Div(
                            "👁 Hide Plot",
                            id="context-menu-hide",
                            className="context-menu-item",
                        ),
                        html.Div(
                            "🗑 Remove Plot",
                            id="context-menu-remove",
                            className="context-menu-item context-menu-item-danger",
                        ),
                    ],
                    id="plot-context-menu",
                    className="plot-context-menu",
                    style={
                        "display": "none",
                        "position": "fixed",
                        "z-index": "9999",
                        "background": self.colors["paper"],
                        "border": f"1px solid {self.colors['border']}",
                        "border-radius": "6px",
                        "box-shadow": "0 4px 12px rgba(0, 0, 0, 0.15)",
                        "padding": "4px 0",
                        "min-width": "180px",
                    },
                ),
            ]
        )

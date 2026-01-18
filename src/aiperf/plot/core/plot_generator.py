# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot generation module for AIPerf visualization.

This module provides the PlotGenerator class which creates Plotly Figure objects
with NVIDIA brand styling for various plot types including pareto curves, scatter
plots, line charts, and time series.
"""

import logging

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from aiperf.common.enums import PlotMetricDirection
from aiperf.common.enums.metric_enums import MetricFlags
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    DARK_THEME_COLORS,
    DERIVED_METRIC_DIRECTIONS,
    LIGHT_THEME_COLORS,
    NVIDIA_GOLD,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    OUTLIER_RED,
    PLOT_FONT_FAMILY,
    PlotTheme,
)
from aiperf.plot.core.plot_specs import Style
from aiperf.plot.metric_names import get_gpu_metric_unit, get_metric_display_name


def get_nvidia_color_scheme(
    n_colors: int,
    palette_name: str = "bright",
    use_brand_colors: bool = True,
) -> list[str]:
    """
    Generate color scheme with optional NVIDIA brand colors and seaborn palette.

    For dark theme: Uses NVIDIA green and gold with "bright" palette for vibrant contrast.
    For light theme: Uses "deep" palette for professional, subdued colors without brand prefix.

    Args:
        n_colors: Number of colors needed
        palette_name: Seaborn palette name ("bright" or "deep")
        use_brand_colors: If True, prefix with NVIDIA_GREEN and NVIDIA_GOLD

    Returns:
        List of hex color strings
    """
    if use_brand_colors:
        custom_colors = [NVIDIA_GREEN, NVIDIA_GOLD]

        if n_colors <= len(custom_colors):
            return custom_colors[:n_colors]

        additional_needed = n_colors - len(custom_colors)
        palette = sns.color_palette(palette_name, additional_needed)
        additional = [mcolors.to_hex(color) for color in palette]
        return custom_colors + additional
    else:
        palette = sns.color_palette(palette_name, n_colors)
        return [mcolors.to_hex(color) for color in palette]


def detect_directional_outliers(
    values: np.ndarray | pd.Series,
    metric_name: str,
    run_average: float | None = None,
    run_std: float | None = None,
    slice_stds: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """
    Detect "bad" performance outliers using run_std + slice_std threshold.

    High values are considered bad for latency-related metrics (TTFT, ITL, latency),
    while low values are considered bad for throughput metrics. Points are marked
    as outliers if they exceed run_average ± (run_std + slice_std).

    Args:
        values: Array of metric values to analyze (point values, not including error bars)
        metric_name: Name of the metric (used to determine direction)
        run_average: Average value across the entire run
        run_std: Standard deviation across the entire run
        slice_stds: Array of standard deviations for each timeslice (error bar values)

    Returns:
        Boolean array where True indicates an outlier point
    """
    if len(values) == 0:
        return np.array([], dtype=bool)

    if run_average is None or run_std is None:
        return np.zeros(len(values), dtype=bool)

    if slice_stds is None or len(slice_stds) != len(values):
        slice_stds = np.zeros(len(values))

    upper_bounds = run_average + run_std + slice_stds
    lower_bounds = run_average - run_std - slice_stds

    metric_lower = metric_name.lower()
    if "throughput" in metric_lower:
        return values < lower_bounds
    else:
        return values > upper_bounds


class PlotGenerator:
    """Generate Plotly figures for AIPerf profiling data with NVIDIA branding.

    This class provides generic, reusable plot functions that can visualize any
    metric combination. Plots can use either light mode (default) or dark mode
    styling for professional presentations.

    Args:
        theme: Theme to use for plots (LIGHT or DARK). Defaults to LIGHT.
    """

    def __init__(self, theme: PlotTheme = PlotTheme.LIGHT, color_pool_size: int = 10):
        """Initialize PlotGenerator with specified theme.

        Args:
            theme: Theme to use for plots (LIGHT or DARK). Defaults to LIGHT.
            color_pool_size: Number of colors to pre-generate for group assignments.
                Defaults to 10, which is the standard perceptual limit for
                distinguishing colors in visualizations (based on seaborn palettes).
                Colors cycle via modulo when groups exceed this limit. Future
                versions will auto-detect from swept parameters.
        """
        self.theme = theme
        self.colors = (
            LIGHT_THEME_COLORS if theme == PlotTheme.LIGHT else DARK_THEME_COLORS
        )
        self._group_color_registry: dict[str, str] = {}
        self._color_pool: list[str] = self._generate_color_pool(color_pool_size)
        self._next_color_index: int = 0
        self._shown_warnings: set[str] = set()

    def reset_color_registry(self) -> None:
        """Reset color registry to ensure consistent colors across export sessions."""
        self._group_color_registry = {}
        self._next_color_index = 0

    def _generate_color_pool(self, pool_size: int) -> list[str]:
        """Generate master color pool for consistent group coloring.

        Pre-generates a palette to assign to groups consistently across all
        plots in a session. Dark theme uses NVIDIA brand colors with bright
        palette, light theme uses deep palette.

        Seaborn palettes provide up to 10 perceptually distinct colors.
        Groups beyond this limit will cycle through the palette via modulo.

        Args:
            pool_size: Number of colors to generate (typically 10 based on
                seaborn's perceptual limit)

        Returns:
            List of hex color strings for the master color pool
        """
        if self.theme == PlotTheme.DARK:
            return get_nvidia_color_scheme(
                pool_size,
                palette_name="bright",
                use_brand_colors=True,
            )
        else:
            return get_nvidia_color_scheme(
                pool_size,
                palette_name="deep",
                use_brand_colors=False,
            )

    def _get_palette_colors(self, n_colors: int = 1) -> list[str]:
        """Get N colors from the master color pool.

        Returns the first N colors from the pre-generated pool. All colors come
        from the same master palette used for group assignments, ensuring visual
        consistency across all plot types.

        Args:
            n_colors: Number of colors needed

        Returns:
            List of hex color strings sliced from the master pool
        """
        return self._color_pool[:n_colors]

    def _get_base_layout(
        self,
        title: str,
        x_label: str,
        y_label: str,
        hovermode: str | None = None,
        autoscale: str = "none",
    ) -> dict:
        """
        Get base layout configuration with NVIDIA branding.

        Provides consistent styling (fonts, colors, margins, grid) that can be
        applied to all plot types. This is the single source of truth for
        NVIDIA brand styling.

        Args:
            title: Plot title text
            x_label: X-axis label text
            y_label: Y-axis label text
            hovermode: Optional hover mode (e.g., "x unified")
            autoscale: Which axes to autoscale ("none", "x", "y", "both")

        Returns:
            Dictionary of layout configuration ready for fig.update_layout()
        """
        template = "plotly_dark" if self.theme == PlotTheme.DARK else "plotly_white"

        layout = {
            "title": {
                "text": title,
                "font": {
                    "size": 18,
                    "family": PLOT_FONT_FAMILY,
                    "weight": "bold",
                    "color": self.colors["text"],
                },
            },
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "template": template,
            "font": {
                "size": 10,
                "family": PLOT_FONT_FAMILY,
                "color": self.colors["text"],
            },
            "height": 400,
            "autosize": True,
            "margin": {"l": 60, "r": 150, "t": 70, "b": 80},
            "plot_bgcolor": self.colors["background"],
            "paper_bgcolor": self.colors["paper"],
            "xaxis": {
                "gridcolor": self.colors["grid"],
                "showline": True,
                "linecolor": self.colors["border"],
                "color": self.colors["text"],
                "rangemode": "normal" if autoscale in ("x", "both") else "tozero",
            },
            "yaxis": {
                "gridcolor": self.colors["grid"],
                "showline": True,
                "linecolor": self.colors["border"],
                "color": self.colors["text"],
                "rangemode": "normal" if autoscale in ("y", "both") else "tozero",
            },
            "legend": {
                "font": {
                    "size": 11,
                    "family": PLOT_FONT_FAMILY,
                    "color": self.colors["text"],
                },
                "bgcolor": f"rgba({int(self.colors['paper'][1:3], 16)}, {int(self.colors['paper'][3:5], 16)}, {int(self.colors['paper'][5:7], 16)}, 0.8)",
                "bordercolor": self.colors["border"],
                "borderwidth": 1,
                "x": 1.02,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
            },
        }

        if hovermode:
            layout["hovermode"] = hovermode

        return layout

    def _prepare_groups(
        self,
        df: pd.DataFrame,
        group_by: str | None,
        experiment_types: dict[str, str] | None = None,
        group_display_names: dict[str, str] | None = None,
    ) -> tuple[list[str | None], dict[str, str], dict[str, str]]:
        """
        Prepare group list and color mapping for multi-series plots.

        Supports two modes:
        1. Experiment groups coloring: When experiment_types provided, uses NVIDIA brand colors
           (baselines=grey, treatments=green) with custom legend ordering.
        2. Other coloring: Uses distinct seaborn colors for each group.

        Args:
            df: DataFrame containing the data
            group_by: Column name to group by (e.g., "model", "concurrency"), or None for no grouping
            experiment_types: Optional mapping of group_name -> "baseline"|"treatment".
                If provided, uses NVIDIA brand colors (grey for baselines, green for treatments).
                If None, uses distinct seaborn colors. Raises ValueError if any group has an
                experiment_type other than "baseline" or "treatment".
            group_display_names: Optional mapping of group_name -> display_name for legends

        Returns:
            Tuple of (groups, group_colors, group_display_names) where:
            - groups: Sorted list of group values (baselines first, then treatments),
                or [None] if no grouping
            - group_colors: Dict mapping group values to color hex codes
            - group_display_names: Dict mapping group values to display names (or empty dict)
        """
        logger = logging.getLogger(__name__)

        if not group_by or group_by not in df.columns:
            logger.info(f"No grouping applied (group_by={group_by})")
            return [None], {}, {}

        groups = sorted(df[group_by].unique())
        logger.info(
            f"Preparing groups with group_by='{group_by}': found {len(groups)} unique values: {groups}"
        )

        # Experiment groups coloring: Use grey for baselines, green for first treatment, and distinct seaborn colors for remaining treatments
        if experiment_types:
            baselines = [g for g in groups if experiment_types.get(g) == "baseline"]
            treatments = [g for g in groups if experiment_types.get(g) == "treatment"]

            # Validate that all groups have valid experiment_types
            unknown_groups = [
                g
                for g in groups
                if experiment_types.get(g) not in ("baseline", "treatment")
            ]
            if unknown_groups:
                invalid_mappings = {g: experiment_types.get(g) for g in unknown_groups}
                raise ValueError(
                    f"Invalid experiment_type for groups: {invalid_mappings}. "
                    f"Expected 'baseline' or 'treatment'."
                )

            baselines = sorted(baselines)
            treatments = sorted(treatments)

            ordered_groups = baselines + treatments

            group_colors = {}

            for group in baselines:
                group_colors[group] = NVIDIA_GRAY

            if len(treatments) > 0:
                group_colors[treatments[0]] = NVIDIA_GREEN

            if len(treatments) > 1:
                seaborn_colors = sns.color_palette(
                    "bright", n_colors=len(treatments) - 1
                ).as_hex()
                for i, group in enumerate(treatments[1:]):
                    group_colors[group] = seaborn_colors[i]

            logger.info(
                f"Applied semantic coloring: {len(baselines)} baselines, {len(treatments)} treatments"
            )
            logger.info(f"  Baselines: {baselines}")
            logger.info(f"  Treatments: {treatments}")
            logger.info(f"  Color assignments: {group_colors}")

            self._validate_line_count(len(ordered_groups))

            display_names = group_display_names or {}

            return ordered_groups, group_colors, display_names

        # Other coloring: Use distinct seaborn colors for each group
        for group in groups:
            if group not in self._group_color_registry:
                color_index = self._next_color_index % len(self._color_pool)
                self._group_color_registry[group] = self._color_pool[color_index]
                self._next_color_index += 1

        group_colors = {group: self._group_color_registry[group] for group in groups}
        return groups, group_colors, (group_display_names or {})

    def _validate_line_count(self, n_traces: int) -> None:
        """Warn if more than 4 lines/traces in a single plot (once per session)."""
        if n_traces > 4:
            warning_key = f"too_many_traces_{n_traces}"
            if warning_key not in self._shown_warnings:
                self._shown_warnings.add(warning_key)
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Plot contains {n_traces} traces, which exceeds the recommended "
                    f"maximum of 4 for clarity."
                )

    def _get_metric_direction(self, metric_tag: str) -> PlotMetricDirection | str:
        """
        Get direction indicator for metric.

        Checks MetricRegistry first, then falls back to derived metrics registry.
        Handles stat suffixes like _avg, _p50, _p99, etc.

        Args:
            metric_tag: Metric tag name (e.g., "request_latency", "output_token_throughput_per_gpu")

        Returns:
            PlotMetricDirection.HIGHER if higher is better (LARGER_IS_BETTER or derived metric marked as True)
            PlotMetricDirection.LOWER if lower is better (not LARGER_IS_BETTER or derived metric marked as False)
            "" if metric not found in either registry
        """
        # Strip stat suffixes to get base metric name
        stat_suffixes = tuple(f"_{key}" for key in ALL_STAT_KEYS)
        base_metric = metric_tag
        for suffix in stat_suffixes:
            if metric_tag.endswith(suffix):
                base_metric = metric_tag[: -len(suffix)]
                break

        # Try both the original metric_tag and the base_metric
        for tag in [metric_tag, base_metric]:
            try:
                metric_class = MetricRegistry.get_class(tag)
                if metric_class.has_flags(MetricFlags.LARGER_IS_BETTER):
                    return PlotMetricDirection.HIGHER
                return PlotMetricDirection.LOWER
            except Exception:
                pass

            if tag in DERIVED_METRIC_DIRECTIONS:
                return (
                    PlotMetricDirection.HIGHER
                    if DERIVED_METRIC_DIRECTIONS[tag]
                    else PlotMetricDirection.LOWER
                )

        logger = logging.getLogger(__name__)
        logger.debug(f"Could not determine direction for metric: {metric_tag}")
        return ""

    def _compute_pareto_frontier(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_direction: PlotMetricDirection,
        y_direction: PlotMetricDirection,
    ) -> np.ndarray:
        """
        Compute Pareto frontier using O(n log n) sweep algorithm.

        The algorithm leverages the fact that after sorting by x-coordinate, we can
        scan once (left-to-right or right-to-left depending on metric directions)
        and track the best y-value seen so far to determine Pareto optimality.

        Args:
            x_values: X-axis metric values (must already be sorted ascending)
            y_values: Y-axis metric values (corresponding to x_values)
            x_direction: Whether higher or lower x is better
            y_direction: Whether higher or lower y is better

        Returns:
            Boolean array where True indicates point is on Pareto frontier
        """
        n = len(x_values)

        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.array([True], dtype=bool)

        is_pareto = np.zeros(n, dtype=bool)

        # Use non-strict comparisons (>= and <=) so identical points are all on the frontier.
        # When points have the same coordinates, none dominates any other.
        if x_direction == PlotMetricDirection.LOWER:
            if y_direction == PlotMetricDirection.HIGHER:
                best_y = float("-inf")
                for i in range(n):
                    if y_values[i] >= best_y:
                        is_pareto[i] = True
                        best_y = y_values[i]
            else:
                best_y = float("inf")
                for i in range(n):
                    if y_values[i] <= best_y:
                        is_pareto[i] = True
                        best_y = y_values[i]
        else:
            if y_direction == PlotMetricDirection.HIGHER:
                best_y = float("-inf")
                for i in range(n - 1, -1, -1):
                    if y_values[i] >= best_y:
                        is_pareto[i] = True
                        best_y = y_values[i]
            else:
                best_y = float("inf")
                for i in range(n - 1, -1, -1):
                    if y_values[i] <= best_y:
                        is_pareto[i] = True
                        best_y = y_values[i]

        return is_pareto

    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """Find Pareto-efficient points where we want to maximize both dimensions.

        A point is Pareto-efficient if no other point dominates it.
        A point dominates another if it is >= in all dimensions and > in at least one.

        Args:
            costs: Array of shape (n_points, 2) with [x, y] values to maximize

        Returns:
            Boolean array marking Pareto-efficient (non-dominated) points
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                other_points = np.arange(n_points) != i
                dominated = np.all(costs[other_points] >= costs[i], axis=1) & np.any(
                    costs[other_points] > costs[i], axis=1
                )
                if np.any(dominated):
                    is_efficient[i] = False

        return is_efficient

    def create_pareto_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        experiment_types: dict[str, str] | None = None,
        group_display_names: dict[str, str] | None = None,
    ) -> go.Figure:
        """Create a Pareto curve plot showing trade-offs between two metrics.

        The Pareto frontier is calculated automatically, highlighting optimal
        configurations where improving one metric doesn't worsen the other.

        Args:
            df: DataFrame containing the metrics
            x_metric: Column name for x-axis metric (e.g., "latency")
            y_metric: Column name for y-axis metric (e.g., "throughput")
            label_by: Column to use for point labels (default: "concurrency")
            group_by: Column to group data by for multi-series (default: "model")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with pareto curve and data points
        """
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = (
            title
            or f"Pareto Curve: {get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )
        x_label = x_label or get_metric_display_name(x_metric)
        y_label = y_label or get_metric_display_name(y_metric)

        # Use default label_by if None provided
        if label_by is None:
            label_by = "concurrency"

        groups, group_colors, display_names = self._prepare_groups(
            df_sorted, group_by, experiment_types, group_display_names
        )

        for group in groups:
            if group is None:
                group_data = df_sorted
                group_color = self._get_palette_colors(1)[0]
                group_name = "Data"
            else:
                # df_sorted is already sorted by x_metric, filtering preserves order
                group_data = df_sorted[df_sorted[group_by] == group]
                group_color = group_colors[group]
                # Use display name if available, otherwise use group ID
                # Convert to string to ensure compatibility with Plotly (handles numpy types)
                group_name = str(display_names.get(group, group))

            # Calculate Pareto frontier for this group based on metric directions
            x_dir = self._get_metric_direction(x_metric)
            y_dir = self._get_metric_direction(y_metric)

            if not x_dir or not y_dir:
                missing = []
                if not x_dir:
                    missing.append(f"x-axis metric '{x_metric}'")
                if not y_dir:
                    missing.append(f"y-axis metric '{y_metric}'")

                raise ValueError(
                    f"Cannot determine optimization direction for {' and '.join(missing)}. "
                    f"Metrics must be registered in MetricRegistry with LARGER_IS_BETTER flag "
                    f"or defined in DERIVED_METRIC_DIRECTIONS. Add the metric(s) to ensure "
                    f"correct Pareto frontier calculation."
                )

            # Sort by x, then by y (best y first) to handle ties in x correctly.
            # For ties in x, only the point with best y can be on the frontier.
            y_ascending = y_dir == PlotMetricDirection.LOWER
            group_data = group_data.sort_values(
                [x_metric, y_metric], ascending=[True, y_ascending]
            )
            x_values = group_data[x_metric].values
            y_values = group_data[y_metric].values
            is_pareto = self._compute_pareto_frontier(x_values, y_values, x_dir, y_dir)

            df_pareto = group_data[is_pareto].sort_values(x_metric)

            if not df_pareto.empty:
                # Shadow for Pareto frontier line (only connects optimal points)
                fig.add_trace(
                    go.Scatter(
                        x=df_pareto[x_metric],
                        y=df_pareto[y_metric],
                        mode="lines",
                        line=dict(width=8, color="rgba(255, 255, 255, 0.1)"),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=group_name,
                    )
                )

                # Main Pareto frontier line (only connects optimal points)
                fig.add_trace(
                    go.Scatter(
                        x=df_pareto[x_metric],
                        y=df_pareto[y_metric],
                        mode="lines",
                        line=dict(width=3, color=group_color),
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=group_name,
                    )
                )

            # Prepare labels and hover text
            labels = [str(val) for val in group_data[label_by]]
            hovertexts = [
                f"<b>{group_name} - {label}</b><br>{x_label}: {x:.1f}<br>{y_label}: {y:.1f}<br><i>💡 Click for full config</i>"
                for label, x, y in zip(
                    labels, group_data[x_metric], group_data[y_metric], strict=False
                )
            ]

            # Shadow layer for markers
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="markers",
                    marker=dict(
                        size=14,
                        symbol="circle",
                        color="rgba(255, 255, 255, 0.15)",
                        line=dict(width=0),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=group_name,
                )
            )

            # Main markers
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        symbol="circle",
                        color=group_color,
                        line=dict(width=0),
                    ),
                    text=labels,
                    textposition="top center",
                    textfont=dict(
                        size=10,
                        color=self.colors["text"],
                        family=PLOT_FONT_FAMILY,
                        weight="bold",
                    ),
                    hovertemplate="%{customdata.text}<extra></extra>",
                    customdata=hovertexts,
                    name=group_name,
                    showlegend=(group is not None),
                    legendgroup=group_name,
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)

        return fig

    def create_scatter_line_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        experiment_types: dict[str, str] | None = None,
        group_display_names: dict[str, str] | None = None,
        mode: str = "lines+markers",
    ) -> go.Figure:
        """Create a scatter plot with or without connecting lines.

        Args:
            df: DataFrame containing the metrics
            x_metric: Column name for x-axis metric
            y_metric: Column name for y-axis metric
            label_by: Column to use for point labels (default: "concurrency")
            group_by: Column to group data by for multi-series (default: "model")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-label label (auto-generated if None)
            mode: Plot mode - "lines+markers" or "markers" (default: "lines+markers")

        Returns:
            Plotly Figure object with scatter plot
        """
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = (
            title
            or f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )
        x_label = x_label or get_metric_display_name(x_metric)
        y_label = y_label or get_metric_display_name(y_metric)

        # Prepare groups and colors
        groups, group_colors, display_names = self._prepare_groups(
            df_sorted, group_by, experiment_types, group_display_names
        )

        for group in groups:
            if group is None:
                group_data = df_sorted
                group_color = self._get_palette_colors(1)[0]
                group_name = "Data"
            else:
                # df_sorted is already sorted by x_metric, filtering preserves order
                group_data = df_sorted[df_sorted[group_by] == group]
                group_color = group_colors[group]
                # Use display name if available, otherwise use group ID
                # Convert to string to ensure compatibility with Plotly (handles numpy types)
                group_name = str(display_names.get(group, group))

            # Determine shadow and main modes based on mode parameter
            shadow_mode = mode
            main_mode = f"{mode}+text" if "text" not in mode else mode

            # Shadow layer
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode=shadow_mode,
                    marker=dict(
                        size=14,
                        color="rgba(255, 255, 255, 0.12)",
                        symbol="circle",
                        line=dict(width=0),
                    ),
                    line=dict(width=8, color="rgba(255, 255, 255, 0.08)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Main trace
            labels = [str(val) for val in group_data[label_by]]
            fig.add_trace(
                go.Scatter(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    mode=main_mode,
                    marker=dict(
                        size=9,
                        color=group_color,
                        symbol="circle",
                        line=dict(width=0),
                    ),
                    line=dict(width=3, color=group_color),
                    text=labels,
                    textposition="top center",
                    textfont=dict(
                        size=9, color=self.colors["text"], family=PLOT_FONT_FAMILY
                    ),
                    hovertemplate=f"<b>{group_name} - %{{text}}</b><br>{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br><i>💡 Click for full config</i><extra></extra>",
                    name=group_name,
                    showlegend=(group is not None),
                    legendgroup=group_name,
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)

        return fig

    def create_multi_run_bar_chart(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        group_by: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a multi-run bar chart with NVIDIA styling.

        Args:
            df: DataFrame containing the metrics
            x_metric: Column name for x-axis metric
            y_metric: Column name for y-axis metric
            group_by: Column to group data by (default: None)
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with bar chart
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = (
            title
            or f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )
        x_label = x_label or get_metric_display_name(x_metric)
        y_label = y_label or get_metric_display_name(y_metric)

        # Prepare groups and colors
        groups, group_colors, display_names = self._prepare_groups(df, group_by)

        for group in groups:
            if group is None:
                group_data = df
                group_color = self._get_palette_colors(1)[0]
                group_name = "Data"
            else:
                group_data = df[df[group_by] == group]
                group_color = group_colors[group]
                # Convert to string to ensure compatibility with Plotly (handles numpy types)
                group_name = str(display_names.get(group, group))

            r, g, b = mcolors.to_rgb(group_color)
            fillcolor = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.7)"

            # Create bar trace with transparent fill and colored border
            marker_config = dict(
                color=fillcolor,
                line=dict(color=group_color, width=2),
            )

            hover_template = (
                f"{x_label}: %{{x}}<br>"
                f"{y_label}: %{{y:.2f}}<br>"
                f"Group: {group_name}<extra></extra>"
            )

            fig.add_trace(
                go.Bar(
                    x=group_data[x_metric],
                    y=group_data[y_metric],
                    name=group_name,
                    marker=marker_config,
                    hovertemplate=hover_template,
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        layout["bargap"] = 0.15
        layout["bargroupgap"] = 0.1
        fig.update_layout(layout)

        return fig

    def create_time_series_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a time series scatter plot.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "request_number" or "timestamp")
            y_metric: Column name for y-axis metric
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with time series scatter plot
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = title or f"{get_metric_display_name(y_metric)} Over Time"
        x_label = x_label or get_metric_display_name(x_col)
        y_label = y_label or get_metric_display_name(y_metric)

        # Main scatter points
        primary_color = self._get_palette_colors(1)[0]
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="markers",
                marker=dict(size=4, opacity=0.95, color=primary_color),
                name=y_label,
                showlegend=True,
                hovertemplate=f"{x_label} %{{x}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        # Apply NVIDIA branding layout with unified hover
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_time_series_area(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Create a time series area plot with filled region.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "timestamp")
            y_metric: Column name for y-axis metric
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with filled area plot
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = title or f"{get_metric_display_name(y_metric)} Over Time"
        x_label = x_label or get_metric_display_name(x_col)
        y_label = y_label or get_metric_display_name(y_metric)

        # Main trace with fill
        primary_color = self._get_palette_colors(1)[0]
        # Extract RGB from hex for fillcolor
        r, g, b = mcolors.to_rgb(primary_color)
        fillcolor = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.2)"

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="lines",
                line=dict(width=2, color=primary_color, shape="hv"),
                fill="tozeroy",
                fillcolor=fillcolor,
                name=y_label,
                showlegend=True,
                hovertemplate=f"{x_label}: %{{x:.0f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label)
        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_time_series_histogram(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        slice_duration: float | None = None,
        warning_text: str | None = None,
        average_value: float | None = None,
        average_label: str | None = None,
        average_std: float | None = None,
    ) -> go.Figure:
        """Create a time series histogram/bar chart.

        Args:
            df: DataFrame containing the time series data
            x_col: Column name for x-axis (e.g., "Timeslice")
            y_col: Column name for y-axis values (e.g., "avg", "p50", "p90")
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)
            slice_duration: Duration of each slice in seconds (for time-based x-axis)
            warning_text: Optional warning text to display at bottom of plot
            average_value: Optional average value across whole run to display as horizontal line
            average_label: Optional label for the average line
            average_std: Optional standard deviation to show as error band around average line

        Returns:
            Plotly Figure object with bar chart
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = title or f"{get_metric_display_name(y_col)} Over Time"
        x_label = x_label or (
            "Timeslice (s)" if slice_duration else get_metric_display_name(x_col)
        )
        y_label = y_label or get_metric_display_name(y_col)

        # Get primary color from theme-specific palette
        primary_color = self._get_palette_colors(1)[0]
        r, g, b = mcolors.to_rgb(primary_color)

        # Prepare x-axis values and bar configuration
        if slice_duration is not None:
            # Use continuous time scale
            slice_indices = df[x_col].values
            # X-values are the center of each slice (bars are centered on x-value in plotly)
            x_values = slice_indices * slice_duration + slice_duration / 2
            # Bar width equals slice duration for continuous coverage
            bar_width = slice_duration

            # Prepare hover data with time ranges and slice indices
            slice_start_times = slice_indices * slice_duration
            time_ranges = [
                f"{int(start)}s-{int(start + slice_duration)}s"
                for start in slice_start_times
            ]
            hover_template = (
                f"Time: %{{customdata[0]}}<br>"
                f"Slice: %{{customdata[1]}}<br>"
                f"{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = list(zip(time_ranges, slice_indices.astype(int), strict=False))

            # Transparent bars with borders
            marker_config = dict(
                color=f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.7)",
                line=dict(color=primary_color, width=2),
            )
        else:
            # Fallback for non-time-based data
            x_values = df[x_col]
            bar_width = None
            hover_template = (
                f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = None
            marker_config = dict(
                color=primary_color,
                line=dict(color=primary_color, width=0),
            )

        # Create bar chart with error bars if std is available
        error_y_config = None
        if "std" in df.columns:
            error_y_config = dict(
                type="data",
                array=df["std"],
                visible=True,
                color=primary_color,
                thickness=2,
                width=6,
            )

        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df[y_col],
                width=bar_width,
                marker=marker_config,
                error_y=error_y_config,
                showlegend=False,
                hovertemplate=hover_template,
                customdata=customdata,
            )
        )

        # Add horizontal average line if provided
        if average_value is not None:
            if slice_duration is not None:
                x_range = [0, (df[x_col].max() + 1) * slice_duration]
            else:
                x_range = [df[x_col].min() - 0.5, df[x_col].max() + 0.5]

            # Add shaded region for ±1 std if provided
            if average_std is not None:
                upper_bound = average_value + average_std
                lower_bound = average_value - average_std

                # Add filled area for std band
                fig.add_trace(
                    go.Scatter(
                        x=x_range + x_range[::-1],
                        y=[upper_bound, upper_bound, lower_bound, lower_bound],
                        fill="toself",
                        fillcolor="rgba(255, 184, 28, 0.2)",  # NVIDIA gold with 20% opacity
                        line=dict(width=0),
                        showlegend=True,
                        name="±1 Std Dev",
                        hovertemplate=f"±1 Std Dev: {lower_bound:.2f} - {upper_bound:.2f}<extra></extra>",
                    )
                )

            # Add average line on top of std band
            # Use secondary color from palette for average line
            palette_colors = self._get_palette_colors(2)
            avg_line_color = (
                palette_colors[1] if len(palette_colors) > 1 else palette_colors[0]
            )

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[average_value, average_value],
                    mode="lines",
                    line=dict(color=avg_line_color, width=3),
                    name=average_label or "Run Average",
                    showlegend=True,
                    hovertemplate=f"{average_label or 'Run Average'}<extra></extra>",
                )
            )

        # Apply NVIDIA branding layout
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")

        # Configure x-axis for continuous time
        layout["bargap"] = 0
        layout["bargroupgap"] = 0
        if slice_duration is not None:
            # Primary x-axis: Time values at boundaries
            slice_indices = df[x_col].values
            max_slice = slice_indices.max()
            layout["xaxis"]["dtick"] = slice_duration
            layout["xaxis"]["tick0"] = 0
            layout["xaxis"]["range"] = [0, (max_slice + 1) * slice_duration]

        if warning_text:
            if "annotations" not in layout:
                layout["annotations"] = []

            layout["margin"]["b"] = 140

            warning_annotation = dict(
                x=0.5,
                y=-0.10,
                xref="paper",
                yref="paper",
                text=warning_text,
                showarrow=False,
                font=dict(
                    size=11, family=PLOT_FONT_FAMILY, color=self.colors["secondary"]
                ),
                bgcolor=f"rgba({int(self.colors['secondary'][1:3], 16)}, {int(self.colors['secondary'][3:5], 16)}, {int(self.colors['secondary'][5:7], 16)}, 0.1)",
                bordercolor=self.colors["secondary"],
                borderwidth=2,
                borderpad=8,
                xanchor="center",
                yanchor="top",
            )
            layout["annotations"] = list(layout.get("annotations", [])) + [
                warning_annotation
            ]

        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_timeslice_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        metric_name: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        slice_duration: float | None = None,
        warning_text: str | None = None,
        average_value: float | None = None,
        average_label: str | None = None,
        average_std: float | None = None,
        unit: str = "",
    ) -> go.Figure:
        """Create a timeslice scatter plot with outlier highlighting.

        Designed specifically for timeslice data with low data-to-ink ratio:
        - Scatter points instead of bars
        - Seaborn deep palette colors for normal points
        - Red highlighting for bad outliers outside the run average ± std band
        - Minimal grid and axes styling
        - Error bars and average line overlay preserved
        - Time range labels (e.g., "0-10", "10-20") on diagonal

        Args:
            df: DataFrame containing the timeslice data
            x_col: Column name for x-axis (e.g., "Timeslice")
            y_col: Column name for y-axis values (e.g., "avg", "p50", "p90")
            metric_name: Name of the metric for outlier detection
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)
            slice_duration: Duration of each slice in seconds (for time-based x-axis)
            warning_text: Optional warning text to display at bottom of plot
            average_value: Run average value (center of gold band) for outlier detection
            average_label: Optional label for the average line
            average_std: Run standard deviation (width of gold band) for outlier detection
            unit: Unit of measurement for the metric (e.g., "ms", "tokens/s")

        Returns:
            Plotly Figure object with timeslice scatter plot
        """
        fig = go.Figure()

        title = title or f"{get_metric_display_name(y_col)} Over Time"
        x_label = x_label or (
            "Timeslice (s)" if slice_duration else get_metric_display_name(x_col)
        )
        y_label = y_label or get_metric_display_name(y_col)

        # Get primary color from theme-specific palette for normal points
        primary_color = self._get_palette_colors(1)[0]

        if slice_duration is not None:
            slice_indices = df[x_col].values
            x_values = slice_indices * slice_duration + slice_duration / 2

            slice_start_times = slice_indices * slice_duration
            time_ranges = [
                f"{int(start)}s-{int(start + slice_duration)}s"
                for start in slice_start_times
            ]
            hover_template = (
                f"Time: %{{customdata[0]}}<br>"
                f"Slice: %{{customdata[1]}}<br>"
                f"{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = list(zip(time_ranges, slice_indices.astype(int), strict=False))
        else:
            x_values = df[x_col].values
            hover_template = (
                f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            )
            customdata = None

        y_values = df[y_col].values
        slice_stds = df["std"].values if "std" in df.columns else None
        outlier_mask = detect_directional_outliers(
            y_values,
            metric_name,
            run_average=average_value,
            run_std=average_std,
            slice_stds=slice_stds,
        )

        normal_mask = ~outlier_mask
        normal_x = x_values[normal_mask]
        normal_y = y_values[normal_mask]
        outlier_x = x_values[outlier_mask]
        outlier_y = y_values[outlier_mask]

        error_y_normal = None
        error_y_outlier = None
        if "std" in df.columns:
            std_values = df["std"].values
            if np.any(normal_mask):
                error_y_normal = dict(
                    type="data",
                    array=std_values[normal_mask],
                    visible=True,
                    color=primary_color,
                    thickness=1.5,
                    width=4,
                )
            if np.any(outlier_mask):
                error_y_outlier = dict(
                    type="data",
                    array=std_values[outlier_mask],
                    visible=True,
                    color=OUTLIER_RED,
                    thickness=1.5,
                    width=4,
                )

        if average_value is not None:
            if slice_duration is not None:
                x_max = (df[x_col].max() + 1) * slice_duration
                x_range = [0, x_max]
            else:
                x_range = [df[x_col].min() - 0.5, df[x_col].max() + 0.5]

            if average_std is not None:
                upper_bound = average_value + average_std
                lower_bound = average_value - average_std

                std_label = f"Run Std: {average_std:.2f}"
                if unit:
                    std_label = f"{std_label} {unit}"

                band_color = (
                    "rgba(232, 232, 232, 0.3)"
                    if self.theme == PlotTheme.LIGHT
                    else "rgba(255, 184, 28, 0.15)"
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_range + x_range[::-1],
                        y=[upper_bound, upper_bound, lower_bound, lower_bound],
                        mode="lines",
                        fill="toself",
                        fillcolor=band_color,
                        line=dict(width=0),
                        showlegend=True,
                        legendrank=3,
                        name=std_label,
                        hovertemplate=f"±1 Std Dev: {lower_bound:.2f} - {upper_bound:.2f}<extra></extra>",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[average_value, average_value],
                    mode="lines",
                    line=dict(color="#555555", width=2),
                    name=average_label or "Run Average",
                    showlegend=True,
                    legendrank=4,
                    hovertemplate=f"{average_label or 'Run Average'}<extra></extra>",
                )
            )

        if np.any(normal_mask):
            normal_customdata = (
                [customdata[i] for i in range(len(customdata)) if normal_mask[i]]
                if customdata is not None
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=normal_x,
                    y=normal_y,
                    mode="markers",
                    marker=dict(
                        color=primary_color,
                        size=6,
                        line=dict(width=0),
                    ),
                    error_y=error_y_normal,
                    name="Timeslice Average",
                    showlegend=True,
                    legendrank=1,
                    hovertemplate=hover_template,
                    customdata=normal_customdata,
                )
            )

        if np.any(outlier_mask):
            outlier_customdata = (
                [customdata[i] for i in range(len(customdata)) if outlier_mask[i]]
                if customdata is not None
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=outlier_x,
                    y=outlier_y,
                    mode="markers",
                    marker=dict(
                        color=OUTLIER_RED,
                        size=6,
                        symbol="diamond",
                        line=dict(width=0),
                    ),
                    error_y=error_y_outlier,
                    name="Outliers",
                    showlegend=True,
                    legendrank=5,
                    hovertemplate=hover_template,
                    customdata=outlier_customdata,
                )
            )

        if "std" in df.columns and (np.any(normal_mask) or np.any(outlier_mask)):
            fig.add_trace(
                go.Scatter(
                    x=[-999999, -999999],
                    y=[0, 1],
                    mode="lines",
                    line=dict(
                        color=primary_color,
                        width=3,
                    ),
                    name="±1 Timeslice Std",
                    showlegend=True,
                    legendrank=2,
                    hoverinfo="skip",
                )
            )

        layout = self._get_base_layout(title, x_label, y_label, hovermode="closest")

        layout["xaxis"]["showgrid"] = True
        layout["xaxis"]["gridwidth"] = 0.5
        layout["xaxis"]["gridcolor"] = (
            "rgba(200, 200, 200, 0.2)"
            if self.theme == PlotTheme.LIGHT
            else "rgba(100, 100, 100, 0.2)"
        )

        layout["yaxis"]["showgrid"] = True
        layout["yaxis"]["gridwidth"] = 0.5
        layout["yaxis"]["gridcolor"] = (
            "rgba(200, 200, 200, 0.2)"
            if self.theme == PlotTheme.LIGHT
            else "rgba(100, 100, 100, 0.2)"
        )

        layout["xaxis"]["linewidth"] = 1
        layout["yaxis"]["linewidth"] = 1
        layout["yaxis"]["rangemode"] = "tozero"

        if slice_duration is not None:
            slice_indices = df[x_col].values
            max_slice = slice_indices.max()

            # Configure custom ticks with range labels at center positions
            layout["xaxis"]["tickmode"] = "array"
            tick_positions = [
                i * slice_duration + slice_duration / 2
                for i in range(int(max_slice) + 1)
            ]
            tick_labels = [
                f"{int(i * slice_duration)}-{int((i + 1) * slice_duration)}"
                for i in range(int(max_slice) + 1)
            ]
            layout["xaxis"]["tickvals"] = tick_positions
            layout["xaxis"]["ticktext"] = tick_labels
            layout["xaxis"]["tickangle"] = -45

            # Increase bottom margin to accommodate diagonal labels
            if "margin" not in layout:
                layout["margin"] = {}
            layout["margin"]["b"] = 100

            x_max = (max_slice + 1) * slice_duration
            layout["xaxis"]["range"] = [0, x_max]

        if warning_text:
            if "annotations" not in layout:
                layout["annotations"] = []

            # Use pixel-based yshift for precise positioning below x-axis
            has_diagonal_labels = slice_duration is not None
            # Shift below x-axis: account for tick labels + axis title
            yshift_pixels = -85 if has_diagonal_labels else -50
            layout["margin"]["b"] = 140 if has_diagonal_labels else 100

            warning_annotation = dict(
                x=0.5,
                y=0,
                xref="paper",
                yref="paper",
                yshift=yshift_pixels,
                text=warning_text,
                showarrow=False,
                font=dict(
                    size=11, family=PLOT_FONT_FAMILY, color=self.colors["secondary"]
                ),
                bgcolor=f"rgba({int(self.colors['secondary'][1:3], 16)}, {int(self.colors['secondary'][3:5], 16)}, {int(self.colors['secondary'][5:7], 16)}, 0.1)",
                bordercolor=self.colors["secondary"],
                borderwidth=2,
                borderpad=8,
                xanchor="center",
                yanchor="top",
            )
            layout["annotations"] = list(layout.get("annotations", [])) + [
                warning_annotation
            ]

        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_dual_axis_plot(
        self,
        df_primary: pd.DataFrame,
        df_secondary: pd.DataFrame,
        x_col_primary: str,
        x_col_secondary: str,
        y1_metric: str,
        y2_metric: str,
        primary_style: Style | None = None,
        secondary_style: Style | None = None,
        active_count_col: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y1_label: str | None = None,
        y2_label: str | None = None,
    ) -> go.Figure:
        """
        Create a dual Y-axis plot with independent data sources and configurable visualization styles.

        This generic method supports plotting any two metrics on separate Y-axes with
        independent data sources and full control over visualization styles (line modes,
        shapes, fill patterns, widths, and marker properties).

        Examples:
            - Throughput + GPU utilization (step function + filled area)
            - Price + volume (lines + bars)
            - Temperature + pressure (smooth splines + lines)

        Args:
            df_primary: DataFrame for primary metric (left Y-axis)
            df_secondary: DataFrame for secondary metric (right Y-axis)
            x_col_primary: Column name for x-axis in primary DataFrame
            x_col_secondary: Column name for x-axis in secondary DataFrame
            y1_metric: Column name for primary y-axis (left)
            y2_metric: Column name for secondary y-axis (right)
            primary_style: Style configuration for primary trace
            secondary_style: Style configuration for secondary trace
            active_count_col: Optional column name in df_primary for supplementary data (shown in tooltip)
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated from x_col_primary if None)
            y1_label: Primary Y-axis label (auto-generated from y1_metric if None)
            y2_label: Secondary Y-axis label (auto-generated from y2_metric if None)

        Returns:
            Plotly Figure object with dual Y-axes
        """
        fig = go.Figure()

        # Provide default styles if not specified
        if primary_style is None:
            primary_style = Style(mode="lines", line_shape=None, fill=None)
        if secondary_style is None:
            secondary_style = Style(mode="lines", line_shape=None, fill=None)

        title = (
            title
            or f"{get_metric_display_name(y1_metric)} with {get_metric_display_name(y2_metric)}"
        )
        x_label = x_label or "Time (s)"
        y1_label = y1_label or get_metric_display_name(y1_metric)
        y2_label = y2_label or get_metric_display_name(y2_metric)

        primary_hover = f"{x_label}: %{{x:.1f}}s<br>{y1_label}: %{{y:.1f}}"
        if active_count_col and active_count_col in df_primary.columns:
            primary_hover += "<br>Active Requests: %{customdata}"

        primary_hover += "<extra></extra>"

        customdata = df_primary[active_count_col] if active_count_col else None

        # Get palette colors for primary and secondary traces
        palette_colors = self._get_palette_colors(2)
        primary_color = palette_colors[0]
        secondary_color = (
            palette_colors[1] if len(palette_colors) > 1 else palette_colors[0]
        )

        # Extract RGB from colors for fillcolor
        primary_rgb = mcolors.to_rgb(primary_color)
        secondary_rgb = mcolors.to_rgb(secondary_color)

        # Build primary trace configuration
        primary_trace_config = {
            "x": df_primary[x_col_primary],
            "y": df_primary[y1_metric],
            "mode": primary_style.mode,
            "line": dict(width=primary_style.line_width, color=primary_color),
            "name": y1_label,
            "yaxis": "y",
            "customdata": customdata,
            "hovertemplate": primary_hover,
        }

        # Apply line shape if specified
        if primary_style.line_shape:
            primary_trace_config["line"]["shape"] = primary_style.line_shape

        # Apply fill if specified
        if primary_style.fill:
            primary_trace_config["fill"] = primary_style.fill
            primary_trace_config["fillcolor"] = (
                f"rgba({int(primary_rgb[0] * 255)}, {int(primary_rgb[1] * 255)}, {int(primary_rgb[2] * 255)}, {primary_style.fill_opacity})"
            )

        fig.add_trace(go.Scatter(**primary_trace_config))

        # Build secondary trace configuration
        secondary_trace_config = {
            "x": df_secondary[x_col_secondary],
            "y": df_secondary[y2_metric],
            "mode": secondary_style.mode,
            "line": dict(width=secondary_style.line_width, color=secondary_color),
            "name": y2_label,
            "yaxis": "y2",
            "hovertemplate": f"{x_label}: %{{x:.1f}}s<br>{y2_label}: %{{y:.1f}}<extra></extra>",
        }

        # Apply line shape if specified
        if secondary_style.line_shape:
            secondary_trace_config["line"]["shape"] = secondary_style.line_shape

        # Apply fill if specified
        if secondary_style.fill:
            secondary_trace_config["fill"] = secondary_style.fill
            secondary_trace_config["fillcolor"] = (
                f"rgba({int(secondary_rgb[0] * 255)}, {int(secondary_rgb[1] * 255)}, {int(secondary_rgb[2] * 255)}, {secondary_style.fill_opacity})"
            )

        fig.add_trace(go.Scatter(**secondary_trace_config))

        layout = self._get_base_layout(title, x_label, y1_label, hovermode="x unified")

        # Check if both metrics are percentage-based for aligned Y-axes
        # Uses unit lookup or heuristic detection for custom metrics
        def is_percentage_metric(metric_name: str) -> bool:
            unit = get_gpu_metric_unit(metric_name)
            if unit == "%":
                return True
            # Heuristic: metrics with "utilization" in the name are percentages
            return "utilization" in metric_name.lower()

        y1_is_pct = is_percentage_metric(y1_metric)
        y2_is_pct = is_percentage_metric(y2_metric)
        both_percentage = y1_is_pct and y2_is_pct

        if both_percentage:
            layout["yaxis"]["range"] = [0, 100]

        layout["yaxis2"] = {
            "title": y2_label,
            "overlaying": "y",
            "side": "right",
            "gridcolor": self.colors["grid"],
            "showline": True,
            "linecolor": self.colors["border"],
            "color": self.colors["text"],
            "rangemode": "tozero",
        }

        if both_percentage:
            layout["yaxis2"]["range"] = [0, 100]

        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_latency_scatter_with_percentiles(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        percentile_cols: list[str],
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """
        Create a scatter plot with rolling percentile overlays for latency analysis.

        Displays individual request latencies as scatter points with overlaid percentile
        lines to provide statistical context. This visualization is ideal for identifying
        tail latency, temporal patterns, and debugging anomalies.

        Args:
            df: DataFrame containing the time series data with percentile columns
            x_col: Column name for x-axis (e.g., "timestamp")
            y_metric: Column name for y-axis metric (e.g., "request_latency")
            percentile_cols: List of column names for percentile lines (e.g., ["p50", "p95", "p99"])
            title: Plot title (auto-generated if None)
            x_label: X-axis label (auto-generated if None)
            y_label: Y-axis label (auto-generated if None)

        Returns:
            Plotly Figure object with scatter points and percentile lines
        """
        fig = go.Figure()

        # Auto-generate labels if not provided
        title = (
            title or f"{get_metric_display_name(y_metric)} Over Time with Percentiles"
        )
        x_label = x_label or get_metric_display_name(x_col)
        y_label = y_label or get_metric_display_name(y_metric)

        # Get theme-specific color palette for percentile lines
        n_percentiles = len(percentile_cols)
        percentile_colors = self._get_palette_colors(n_percentiles)

        # Individual request scatter points (semi-transparent)
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_metric],
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.4,
                    color=self.colors["secondary"],
                    line=dict(width=0),
                ),
                name="Individual Requests",
                hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
            )
        )

        # Add percentile lines with NVIDIA color palette
        for idx, percentile_col in enumerate(percentile_cols):
            if percentile_col not in df.columns:
                continue

            # Extract percentile number from column name (e.g., "p95" -> "p95")
            percentile_display = percentile_col.upper()
            color = percentile_colors[idx % len(percentile_colors)]

            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[percentile_col],
                    mode="lines",
                    line=dict(width=2.5, color=color),
                    name=percentile_display,
                    hovertemplate=f"{x_label}: %{{x:.2f}}<br>{percentile_display}: %{{y:.2f}}<extra></extra>",
                )
            )

        # Apply NVIDIA branding layout with unified hover
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_request_timeline(
        self,
        df: pd.DataFrame,
        y_metric: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """
        Create request timeline with prefill and decode phases.

        Each request is shown as a horizontal line at its Y-metric value,
        split into two colored phases: prefill (green) and decode (blue).

        Args:
            df: DataFrame with [request_id, y_value, start_s, ttft_end_s, end_s]
            y_metric: Metric name for labels
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        title = title or f"Request Timeline: {get_metric_display_name(y_metric)}"
        x_label = x_label or "Time (seconds)"
        y_label = y_label or get_metric_display_name(y_metric)

        ttft_color = NVIDIA_GREEN
        palette = self._get_palette_colors(2)
        generation_color = (
            palette[1] if len(palette) > 1 else palette[0] if palette else NVIDIA_GOLD
        )

        ttft_legend_added = False
        generation_legend_added = False

        df_sorted = df.sort_values("y_value", ascending=True)

        for _, row in df_sorted.iterrows():
            request_id = row["request_id"]
            y_val = row["y_value"]
            start_s = row["start_s"]
            ttft_end_s = row["ttft_end_s"]
            end_s = row["end_s"]

            ttft_duration = ttft_end_s - start_s
            fig.add_trace(
                go.Scatter(
                    x=[start_s, ttft_end_s],
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(width=2, color=ttft_color),
                    name="Prefill Phase",
                    legendgroup="ttft",
                    showlegend=not ttft_legend_added,
                    hovertemplate=(
                        f"Request {request_id}<br>"
                        f"Prefill Phase<br>"
                        f"Start: {start_s:.2f}s<br>"
                        f"End: {ttft_end_s:.2f}s<br>"
                        f"Duration: {ttft_duration:.2f}s<br>"
                        f"{y_label}: {y_val:.2f}<extra></extra>"
                    ),
                )
            )
            ttft_legend_added = True

            generation_duration = end_s - ttft_end_s
            if generation_duration > 0.001:
                fig.add_trace(
                    go.Scatter(
                        x=[ttft_end_s, end_s],
                        y=[y_val, y_val],
                        mode="lines",
                        line=dict(width=2, color=generation_color),
                        name="Decode Phase",
                        legendgroup="generation",
                        showlegend=not generation_legend_added,
                        hovertemplate=(
                            f"Request {request_id}<br>"
                            f"Decode Phase<br>"
                            f"Start: {ttft_end_s:.2f}s<br>"
                            f"End: {end_s:.2f}s<br>"
                            f"Duration: {generation_duration:.2f}s<br>"
                            f"{y_label}: {y_val:.2f}<extra></extra>"
                        ),
                    )
                )
                generation_legend_added = True

        layout = self._get_base_layout(title, x_label, y_label, hovermode="closest")
        layout["yaxis"]["rangemode"] = "normal"
        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )

        return fig

    def create_percentile_bands(
        self,
        df: pd.DataFrame,
        x_col: str,
        percentile_cols: list[str],
        lower_col: str | None,
        metric_name: str,
        metric_type: str,
        title: str,
        x_label: str,
        y_label: str,
        unit: str,
    ) -> go.Figure:
        """Create percentile bands plot with p50 line and p95/p99 shaded bands.

        Visualizes uncertainty and variance over time with median line and
        percentile confidence bands. Perfect for SLA monitoring and stability analysis.

        Args:
            df: DataFrame with timestamp and percentile columns
            x_col: Column name for x-axis (usually timestamp_s)
            percentile_cols: List of percentile column names (e.g., ["p50", "p95", "p99"])
            lower_col: Optional lower percentile column (e.g., "p05" for gauges)
            metric_name: Metric name for legend
            metric_type: Metric type (HISTOGRAM, GAUGE, COUNTER)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            unit: Metric unit

        Returns:
            Plotly Figure with percentile bands
        """
        fig = go.Figure()

        x = df[x_col]

        # Add bands from highest to lowest (for proper stacking)
        # p99 band (outermost - lightest)
        if "p99" in percentile_cols and "p99" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["p99"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate=f"Time: %{{x:.2f}}s<br>p99: %{{y:.3f}} {unit}<extra></extra>",
                )
            )

        # p95 band (middle - medium shade)
        if "p95" in percentile_cols and "p95" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["p95"],
                    fill="tonexty" if "p99" in df.columns else None,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(68, 138, 255, 0.2)",  # Light blue
                    name="p95-p99 band" if "p99" in df.columns else "p95",
                    hovertemplate=f"Time: %{{x:.2f}}s<br>p95: %{{y:.3f}} {unit}<extra></extra>",
                )
            )

        # p50 median line (darkest - solid line)
        if "p50" in percentile_cols and "p50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["p50"],
                    fill="tonexty" if "p95" in df.columns else None,
                    mode="lines",
                    line=dict(color="rgb(31, 119, 180)", width=2.5),
                    fillcolor="rgba(68, 138, 255, 0.3)"
                    if "p95" in df.columns
                    else None,
                    name="p50 (median)" if "p95" in df.columns else "p50-p95 band",
                    hovertemplate=f"Time: %{{x:.2f}}s<br>p50: %{{y:.3f}} {unit}<extra></extra>",
                )
            )

        # Lower band (for gauges with min values)
        if lower_col and lower_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[lower_col],
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(68, 138, 255, 0.2)",
                    name="p05-p50 band",
                    hovertemplate=f"Time: %{{x:.2f}}s<br>min: %{{y:.3f}} {unit}<extra></extra>",
                )
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
            showlegend=True,
            height=600,
        )

        # Add annotation explaining the bands
        band_type = "percentiles" if metric_type == "HISTOGRAM" else "min/avg/max"
        fig.add_annotation(
            text=f"Shaded bands show {band_type} range over time",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        return fig

    def create_bucket_histogram(
        self,
        buckets: dict[str, int],
        metric_name: str,
        title: str,
        x_label: str,
        y_label: str,
        unit: str,
    ) -> go.Figure:
        """Create Prometheus histogram bucket distribution bar chart.

        Visualizes the actual bucket boundaries (le values) and observation counts
        from a Prometheus histogram metric. Perfect for understanding distribution
        shape and validating percentile estimates.

        Args:
            buckets: Dict mapping bucket upper bounds (le) to counts
            metric_name: Metric name for annotations
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            unit: Metric unit

        Returns:
            Plotly Figure with bucket distribution bar chart
        """
        # Sort buckets by upper bound (handle +Inf specially)
        sorted_buckets = []
        for le, count in buckets.items():
            if le == "+Inf":
                sort_key = float("inf")
            else:
                try:
                    sort_key = float(le)
                except ValueError:
                    sort_key = float("inf")  # Fallback
            sorted_buckets.append((le, sort_key, count))

        sorted_buckets.sort(key=lambda x: x[1])

        # Extract labels and counts
        bucket_labels = [b[0] for b in sorted_buckets]
        bucket_counts = [b[2] for b in sorted_buckets]

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=bucket_labels,
                y=bucket_counts,
                name="Observations",
                marker=dict(color="rgb(68, 138, 255)"),
                hovertemplate="Bucket ≤ %{x}<br>Count: %{y:,}<extra></extra>",
            )
        )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            showlegend=False,
            height=600,
            xaxis=dict(
                type="category",  # Treat bucket boundaries as categories
                categoryorder="array",
                categoryarray=bucket_labels,  # Already sorted
            ),
        )

        # Add annotation explaining bucket format
        total_count = sum(bucket_counts)
        fig.add_annotation(
            text=f"Total observations: {total_count:,} | Each bar shows count in bucket ≤ upper bound",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        return fig

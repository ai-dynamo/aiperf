# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiperf.common.enums import MetricFlags, PlotMetricDirection
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


def _is_percentage_metric(metric_name: str) -> bool:
    """Return True when a metric is expressed in percent (unit == '%' or has 'utilization')."""
    if get_gpu_metric_unit(metric_name) == "%":
        return True
    return "utilization" in metric_name.lower()


def _sort_prometheus_buckets(
    buckets: dict[str, int],
) -> tuple[list[str], list[int]]:
    """Sort Prometheus histogram buckets by numeric ``le`` upper bound ('+Inf' last)."""
    sorted_entries: list[tuple[str, float, int]] = []
    for le, count in buckets.items():
        if le == "+Inf":
            sort_key = float("inf")
        else:
            try:
                sort_key = float(le)
            except ValueError:
                sort_key = float("inf")
        sorted_entries.append((le, sort_key, count))
    sorted_entries.sort(key=lambda entry: entry[1])
    return [b[0] for b in sorted_entries], [b[2] for b in sorted_entries]


def _sweep_pareto(
    y_values: np.ndarray,
    y_direction: "PlotMetricDirection",
    *,
    reverse: bool,
) -> np.ndarray:
    """Single-pass Pareto sweep over y_values (forward when reverse=False)."""
    n = len(y_values)
    is_pareto = np.zeros(n, dtype=bool)
    indices = range(n - 1, -1, -1) if reverse else range(n)
    if y_direction == PlotMetricDirection.HIGHER:
        best = float("-inf")
        for i in indices:
            if y_values[i] >= best:
                is_pareto[i] = True
                best = y_values[i]
    else:
        best = float("inf")
        for i in indices:
            if y_values[i] <= best:
                is_pareto[i] = True
                best = y_values[i]
    return is_pareto


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
    *,
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

    def _build_axis_layout(self, autoscale_active: bool) -> dict:
        """NVIDIA-themed axis layout dict."""
        return {
            "gridcolor": self.colors["grid"],
            "showline": True,
            "linecolor": self.colors["border"],
            "color": self.colors["text"],
            "rangemode": "normal" if autoscale_active else "tozero",
        }

    def _build_legend_layout(self) -> dict:
        """NVIDIA-themed legend layout dict."""
        paper = self.colors["paper"]
        return {
            "font": {
                "size": 11,
                "family": PLOT_FONT_FAMILY,
                "color": self.colors["text"],
            },
            "bgcolor": (
                f"rgba({int(paper[1:3], 16)}, {int(paper[3:5], 16)}, "
                f"{int(paper[5:7], 16)}, 0.8)"
            ),
            "bordercolor": self.colors["border"],
            "borderwidth": 1,
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
        }

    def _get_base_layout(
        self,
        title: str,
        x_label: str,
        y_label: str,
        *,
        hovermode: str | None = None,
        autoscale: str = "none",
    ) -> dict:
        """Base layout with NVIDIA branding (fonts, colors, margins, grid).

        ``autoscale`` selects which axes use rangemode ``normal`` vs ``tozero``.
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
            "xaxis": self._build_axis_layout(autoscale in ("x", "both")),
            "yaxis": self._build_axis_layout(autoscale in ("y", "both")),
            "legend": self._build_legend_layout(),
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
        """Group list + color map for multi-series plots.

        If ``experiment_types`` is given, uses grey for baselines and green for the
        first treatment (seaborn for remaining treatments); otherwise assigns
        distinct palette colors from the instance color pool.
        """
        logger = logging.getLogger(__name__)

        if not group_by or group_by not in df.columns:
            logger.info(f"No grouping applied (group_by={group_by})")
            return [None], {}, {}

        groups = sorted(df[group_by].unique())
        logger.info(
            f"Preparing groups with group_by='{group_by}': found {len(groups)} unique values: {groups}"
        )

        if experiment_types:
            return self._prepare_experiment_groups(
                groups, experiment_types, group_display_names, logger
            )

        for group in groups:
            if group not in self._group_color_registry:
                color_index = self._next_color_index % len(self._color_pool)
                self._group_color_registry[group] = self._color_pool[color_index]
                self._next_color_index += 1

        group_colors = {group: self._group_color_registry[group] for group in groups}
        return groups, group_colors, (group_display_names or {})

    def _prepare_experiment_groups(
        self,
        groups: list,
        experiment_types: dict[str, str],
        group_display_names: dict[str, str] | None,
        logger: logging.Logger,
    ) -> tuple[list[str | None], dict[str, str], dict[str, str]]:
        """Baseline/treatment coloring: grey baselines, green first treatment, seaborn rest."""
        baselines = [g for g in groups if experiment_types.get(g) == "baseline"]
        treatments = [g for g in groups if experiment_types.get(g) == "treatment"]

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

        group_colors: dict[str, str] = {g: NVIDIA_GRAY for g in baselines}
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
        return ordered_groups, group_colors, group_display_names or {}

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
            except Exception:  # noqa: BLE001, S110 - registry lookup can surface plugin loader errors beyond MetricTypeError; fall through to derived-registry lookup on any failure
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
        """Pareto frontier via single-sweep O(n log n) after sort by x-coordinate.

        Uses non-strict comparisons so identical points are all on the frontier.
        """
        n = len(x_values)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.array([True], dtype=bool)

        if x_direction == PlotMetricDirection.LOWER:
            return _sweep_pareto(y_values, y_direction, reverse=False)
        return _sweep_pareto(y_values, y_direction, reverse=True)

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
        *,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        experiment_types: dict[str, str] | None = None,
        group_display_names: dict[str, str] | None = None,
    ) -> go.Figure:
        """Pareto curve plot: trade-off between two metrics with auto-computed frontier."""
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        title = (
            title
            or f"Pareto Curve: {get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )
        x_label = x_label or get_metric_display_name(x_metric)
        y_label = y_label or get_metric_display_name(y_metric)
        if label_by is None:
            label_by = "concurrency"

        groups, group_colors, display_names = self._prepare_groups(
            df_sorted, group_by, experiment_types, group_display_names
        )
        x_dir = self._get_metric_direction(x_metric)
        y_dir = self._get_metric_direction(y_metric)
        self._require_metric_directions(x_metric, y_metric, x_dir, y_dir)

        for group in groups:
            self._add_pareto_group(
                fig,
                df_sorted=df_sorted,
                group=group,
                group_by=group_by,
                group_colors=group_colors,
                display_names=display_names,
                x_metric=x_metric,
                y_metric=y_metric,
                label_by=label_by,
                x_dir=x_dir,
                y_dir=y_dir,
                x_label=x_label,
                y_label=y_label,
            )

        fig.update_layout(self._get_base_layout(title, x_label, y_label))
        return fig

    def _require_metric_directions(
        self,
        x_metric: str,
        y_metric: str,
        x_dir: PlotMetricDirection | str,
        y_dir: PlotMetricDirection | str,
    ) -> None:
        """Raise ValueError listing any metrics missing a direction registration."""
        if x_dir and y_dir:
            return
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

    def _resolve_group_slice(
        self,
        df_sorted: pd.DataFrame,
        group: str | None,
        *,
        group_by: str | None,
        group_colors: dict[str, str],
        display_names: dict[str, str],
    ) -> tuple[pd.DataFrame, str, str]:
        """Slice dataframe and pick color + display name for one group (None == no grouping)."""
        if group is None:
            return df_sorted, self._get_palette_colors(1)[0], "Data"
        return (
            df_sorted[df_sorted[group_by] == group],
            group_colors[group],
            str(display_names.get(group, group)),
        )

    def _add_pareto_group(
        self,
        fig: go.Figure,
        *,
        df_sorted: pd.DataFrame,
        group: str | None,
        group_by: str | None,
        group_colors: dict[str, str],
        display_names: dict[str, str],
        x_metric: str,
        y_metric: str,
        label_by: str,
        x_dir: PlotMetricDirection,
        y_dir: PlotMetricDirection,
        x_label: str,
        y_label: str,
    ) -> None:
        """Render one group's frontier line + markers (shadow + main)."""
        group_data, group_color, group_name = self._resolve_group_slice(
            df_sorted,
            group,
            group_by=group_by,
            group_colors=group_colors,
            display_names=display_names,
        )
        y_ascending = y_dir == PlotMetricDirection.LOWER
        group_data = group_data.sort_values(
            [x_metric, y_metric], ascending=[True, y_ascending]
        )
        is_pareto = self._compute_pareto_frontier(
            group_data[x_metric].values, group_data[y_metric].values, x_dir, y_dir
        )
        df_pareto = group_data[is_pareto].sort_values(x_metric)

        if not df_pareto.empty:
            self._add_pareto_frontier_lines(
                fig,
                df_pareto,
                x_metric=x_metric,
                y_metric=y_metric,
                group_color=group_color,
                group_name=group_name,
            )

        labels = [str(val) for val in group_data[label_by]]
        hovertexts = [
            f"<b>{group_name} - {label}</b><br>{x_label}: {x:.1f}<br>{y_label}: {y:.1f}<br><i>💡 Click for full config</i>"
            for label, x, y in zip(
                labels, group_data[x_metric], group_data[y_metric], strict=False
            )
        ]
        self._add_pareto_markers(
            fig,
            group_data=group_data,
            x_metric=x_metric,
            y_metric=y_metric,
            labels=labels,
            hovertexts=hovertexts,
            group=group,
            group_color=group_color,
            group_name=group_name,
        )

    def _add_pareto_frontier_lines(
        self,
        fig: go.Figure,
        df_pareto: pd.DataFrame,
        *,
        x_metric: str,
        y_metric: str,
        group_color: str,
        group_name: str,
    ) -> None:
        """Shadow + main frontier line traces."""
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

    def _add_pareto_markers(
        self,
        fig: go.Figure,
        *,
        group_data: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        labels: list[str],
        hovertexts: list[str],
        group: str | None,
        group_color: str,
        group_name: str,
    ) -> None:
        """Shadow + main marker traces (labels + hover) for one Pareto group."""
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

    def create_scatter_line_plot(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        *,
        label_by: str = "concurrency",
        group_by: str | None = "model",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        experiment_types: dict[str, str] | None = None,
        group_display_names: dict[str, str] | None = None,
        mode: str = "lines+markers",
    ) -> go.Figure:
        """Scatter plot with optional connecting lines ('lines+markers' by default)."""
        df_sorted = df.sort_values(x_metric)
        fig = go.Figure()

        title = (
            title
            or f"{get_metric_display_name(y_metric)} vs {get_metric_display_name(x_metric)}"
        )
        x_label = x_label or get_metric_display_name(x_metric)
        y_label = y_label or get_metric_display_name(y_metric)

        groups, group_colors, display_names = self._prepare_groups(
            df_sorted, group_by, experiment_types, group_display_names
        )

        for group in groups:
            group_data, group_color, group_name = self._resolve_group_slice(
                df_sorted,
                group,
                group_by=group_by,
                group_colors=group_colors,
                display_names=display_names,
            )
            self._add_scatter_line_group(
                fig,
                group_data=group_data,
                group=group,
                group_color=group_color,
                group_name=group_name,
                x_metric=x_metric,
                y_metric=y_metric,
                label_by=label_by,
                mode=mode,
                x_label=x_label,
                y_label=y_label,
            )

        fig.update_layout(self._get_base_layout(title, x_label, y_label))
        return fig

    def _add_scatter_line_group(
        self,
        fig: go.Figure,
        *,
        group_data: pd.DataFrame,
        group: str | None,
        group_color: str,
        group_name: str,
        x_metric: str,
        y_metric: str,
        label_by: str,
        mode: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """Shadow + main scatter/line traces for a single group."""
        shadow_mode = mode
        main_mode = f"{mode}+text" if "text" not in mode else mode

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

    def create_multi_run_bar_chart(
        self,
        df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        *,
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
        layout["bargap"], layout["bargroupgap"] = 0.15, 0.1
        fig.update_layout(layout)

        return fig

    def create_time_series_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        *,
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
        *,
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
        *,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        slice_duration: float | None = None,
        warning_text: str | None = None,
        average_value: float | None = None,
        average_label: str | None = None,
        average_std: float | None = None,
    ) -> go.Figure:
        """Time-series histogram/bar chart with optional run-average overlay and warning banner."""
        fig = go.Figure()
        title = title or f"{get_metric_display_name(y_col)} Over Time"
        x_label = x_label or (
            "Timeslice (s)" if slice_duration else get_metric_display_name(x_col)
        )
        y_label = y_label or get_metric_display_name(y_col)

        self._add_histogram_bar(
            fig,
            df=df,
            x_col=x_col,
            y_col=y_col,
            x_label=x_label,
            y_label=y_label,
            slice_duration=slice_duration,
        )

        if average_value is not None:
            self._add_run_average_overlay(
                fig,
                df=df,
                x_col=x_col,
                slice_duration=slice_duration,
                average_value=average_value,
                average_std=average_std,
                average_label=average_label,
            )

        fig.update_layout(
            self._build_histogram_layout(
                title,
                x_label,
                y_label,
                df=df,
                x_col=x_col,
                slice_duration=slice_duration,
                warning_text=warning_text,
            )
        )
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )
        return fig

    def _add_histogram_bar(
        self,
        fig: go.Figure,
        *,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        slice_duration: float | None,
    ) -> None:
        """Add the main Bar trace (including error_y if 'std' column present)."""
        primary_color = self._get_palette_colors(1)[0]
        x_values, bar_width, hover_template, customdata, marker_config = (
            self._build_histogram_bar_config(df, x_col, y_label, slice_duration)
        )
        if slice_duration is None:
            hover_template = f"{x_label}: {hover_template}"
        error_y_config = (
            dict(
                type="data",
                array=df["std"],
                visible=True,
                color=primary_color,
                thickness=2,
                width=6,
            )
            if "std" in df.columns
            else None
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

    def _build_histogram_bar_config(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_label: str,
        slice_duration: float | None,
    ) -> tuple:
        """Return (x_values, bar_width, hover_template, customdata, marker_config)."""
        primary_color = self._get_palette_colors(1)[0]
        r, g, b = mcolors.to_rgb(primary_color)

        if slice_duration is None:
            marker_config = dict(
                color=primary_color, line=dict(color=primary_color, width=0)
            )
            hover = f"%{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            return df[x_col], None, hover, None, marker_config

        slice_indices = df[x_col].values
        x_values = slice_indices * slice_duration + slice_duration / 2
        slice_start_times = slice_indices * slice_duration
        time_ranges = [
            f"{int(start)}s-{int(start + slice_duration)}s"
            for start in slice_start_times
        ]
        hover = (
            f"Time: %{{customdata[0]}}<br>"
            f"Slice: %{{customdata[1]}}<br>"
            f"{y_label}: %{{y:.2f}}<extra></extra>"
        )
        customdata = list(zip(time_ranges, slice_indices.astype(int), strict=False))
        marker_config = dict(
            color=f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.7)",
            line=dict(color=primary_color, width=2),
        )
        return x_values, slice_duration, hover, customdata, marker_config

    def _add_run_average_overlay(
        self,
        fig: go.Figure,
        *,
        df: pd.DataFrame,
        x_col: str,
        slice_duration: float | None,
        average_value: float,
        average_std: float | None,
        average_label: str | None,
    ) -> None:
        """Horizontal run-average line plus optional ±1 std filled band."""
        if slice_duration is not None:
            x_range = [0, (df[x_col].max() + 1) * slice_duration]
        else:
            x_range = [df[x_col].min() - 0.5, df[x_col].max() + 0.5]

        if average_std is not None:
            upper_bound = average_value + average_std
            lower_bound = average_value - average_std
            fig.add_trace(
                go.Scatter(
                    x=x_range + x_range[::-1],
                    y=[upper_bound, upper_bound, lower_bound, lower_bound],
                    fill="toself",
                    fillcolor="rgba(255, 184, 28, 0.2)",
                    line=dict(width=0),
                    showlegend=True,
                    name="±1 Std Dev",
                    hovertemplate=f"±1 Std Dev: {lower_bound:.2f} - {upper_bound:.2f}<extra></extra>",
                )
            )

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

    def _build_warning_annotation(
        self,
        warning_text: str,
        *,
        y: float = -0.10,
        yshift: int | None = None,
    ) -> dict:
        """Bottom-of-plot secondary-color warning banner annotation dict."""
        secondary = self.colors["secondary"]
        annotation = {
            "x": 0.5,
            "y": y,
            "xref": "paper",
            "yref": "paper",
            "text": warning_text,
            "showarrow": False,
            "font": {
                "size": 11,
                "family": PLOT_FONT_FAMILY,
                "color": secondary,
            },
            "bgcolor": (
                f"rgba({int(secondary[1:3], 16)}, {int(secondary[3:5], 16)}, "
                f"{int(secondary[5:7], 16)}, 0.1)"
            ),
            "bordercolor": secondary,
            "borderwidth": 2,
            "borderpad": 8,
            "xanchor": "center",
            "yanchor": "top",
        }
        if yshift is not None:
            annotation["yshift"] = yshift
        return annotation

    def _build_histogram_layout(
        self,
        title: str,
        x_label: str,
        y_label: str,
        *,
        df: pd.DataFrame,
        x_col: str,
        slice_duration: float | None,
        warning_text: str | None,
    ) -> dict:
        """Compose layout (zero gaps, slice ticks, warning banner) for histograms."""
        layout = self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        layout["bargap"] = 0
        layout["bargroupgap"] = 0
        if slice_duration is not None:
            slice_indices = df[x_col].values
            max_slice = slice_indices.max()
            layout["xaxis"]["dtick"] = slice_duration
            layout["xaxis"]["tick0"] = 0
            layout["xaxis"]["range"] = [0, (max_slice + 1) * slice_duration]
        if warning_text:
            layout["margin"]["b"] = 140
            layout["annotations"] = list(layout.get("annotations", [])) + [
                self._build_warning_annotation(warning_text)
            ]
        return layout

    def create_timeslice_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        metric_name: str,
        *,
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
        """Timeslice scatter with outlier highlighting (bad points in red).

        Outliers are detected via :func:`detect_directional_outliers` against the
        run average ± std band.
        """
        fig = go.Figure()
        ctx = self._prepare_timeslice_context(
            df=df,
            x_col=x_col,
            y_col=y_col,
            metric_name=metric_name,
            title=title,
            x_label=x_label,
            y_label=y_label,
            slice_duration=slice_duration,
            average_value=average_value,
            average_std=average_std,
        )

        if average_value is not None:
            self._add_timeslice_average_overlay(
                fig,
                df=df,
                x_col=x_col,
                slice_duration=slice_duration,
                average_value=average_value,
                average_std=average_std,
                average_label=average_label,
                unit=unit,
            )
        self._add_timeslice_point_traces(
            fig,
            x_values=ctx["x_values"],
            y_values=ctx["y_values"],
            normal_mask=ctx["normal_mask"],
            outlier_mask=ctx["outlier_mask"],
            primary_color=ctx["primary_color"],
            error_y_normal=ctx["error_y_normal"],
            error_y_outlier=ctx["error_y_outlier"],
            hover_template=ctx["hover_template"],
            customdata=ctx["customdata"],
        )
        if "std" in df.columns and (
            np.any(ctx["normal_mask"]) or np.any(ctx["outlier_mask"])
        ):
            self._add_timeslice_std_legend_proxy(fig, ctx["primary_color"])

        fig.update_layout(
            self._build_timeslice_layout(
                ctx["title"],
                ctx["x_label"],
                ctx["y_label"],
                df=df,
                x_col=x_col,
                slice_duration=slice_duration,
                warning_text=warning_text,
            )
        )
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )
        return fig

    def _prepare_timeslice_context(
        self,
        *,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        metric_name: str,
        title: str | None,
        x_label: str | None,
        y_label: str | None,
        slice_duration: float | None,
        average_value: float | None,
        average_std: float | None,
    ) -> dict:
        """Build the shared context dict (labels, colors, masks, hover) for a timeslice plot."""
        title = title or f"{get_metric_display_name(y_col)} Over Time"
        x_label = x_label or (
            "Timeslice (s)" if slice_duration else get_metric_display_name(x_col)
        )
        y_label = y_label or get_metric_display_name(y_col)
        primary_color = self._get_palette_colors(1)[0]

        x_values, hover_template, customdata = self._timeslice_x_and_hover(
            df,
            x_col,
            y_label=y_label,
            x_label=x_label,
            slice_duration=slice_duration,
        )
        y_values = df[y_col].values
        normal_mask, outlier_mask, error_y_normal, error_y_outlier = (
            self._compute_timeslice_outliers(
                df,
                y_values,
                metric_name,
                average_value=average_value,
                average_std=average_std,
                primary_color=primary_color,
            )
        )
        return {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "primary_color": primary_color,
            "x_values": x_values,
            "y_values": y_values,
            "hover_template": hover_template,
            "customdata": customdata,
            "normal_mask": normal_mask,
            "outlier_mask": outlier_mask,
            "error_y_normal": error_y_normal,
            "error_y_outlier": error_y_outlier,
        }

    def _compute_timeslice_outliers(
        self,
        df: pd.DataFrame,
        y_values: np.ndarray,
        metric_name: str,
        *,
        average_value: float | None,
        average_std: float | None,
        primary_color: str,
    ) -> tuple[np.ndarray, np.ndarray, dict | None, dict | None]:
        """Compute outlier mask + normal/outlier error-bar configs for timeslice scatter."""
        outlier_mask = detect_directional_outliers(
            y_values,
            metric_name,
            run_average=average_value,
            run_std=average_std,
            slice_stds=df["std"].values if "std" in df.columns else None,
        )
        normal_mask = ~outlier_mask
        error_y_normal, error_y_outlier = self._timeslice_error_bars(
            df, normal_mask, outlier_mask, primary_color
        )
        return normal_mask, outlier_mask, error_y_normal, error_y_outlier

    def _timeslice_x_and_hover(
        self,
        df: pd.DataFrame,
        x_col: str,
        *,
        y_label: str,
        x_label: str,
        slice_duration: float | None,
    ) -> tuple[np.ndarray, str, list | None]:
        """Return (x_values, hover_template, customdata) for a timeslice scatter."""
        if slice_duration is None:
            hover = f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            return df[x_col].values, hover, None

        slice_indices = df[x_col].values
        x_values = slice_indices * slice_duration + slice_duration / 2
        slice_start_times = slice_indices * slice_duration
        time_ranges = [
            f"{int(start)}s-{int(start + slice_duration)}s"
            for start in slice_start_times
        ]
        hover = (
            f"Time: %{{customdata[0]}}<br>"
            f"Slice: %{{customdata[1]}}<br>"
            f"{y_label}: %{{y:.2f}}<extra></extra>"
        )
        customdata = list(zip(time_ranges, slice_indices.astype(int), strict=False))
        return x_values, hover, customdata

    def _timeslice_error_bars(
        self,
        df: pd.DataFrame,
        normal_mask: np.ndarray,
        outlier_mask: np.ndarray,
        primary_color: str,
    ) -> tuple[dict | None, dict | None]:
        """Return (error_y_normal, error_y_outlier) configs for timeslice scatter."""
        if "std" not in df.columns:
            return None, None
        std_values = df["std"].values
        error_y_normal = (
            dict(
                type="data",
                array=std_values[normal_mask],
                visible=True,
                color=primary_color,
                thickness=1.5,
                width=4,
            )
            if np.any(normal_mask)
            else None
        )
        error_y_outlier = (
            dict(
                type="data",
                array=std_values[outlier_mask],
                visible=True,
                color=OUTLIER_RED,
                thickness=1.5,
                width=4,
            )
            if np.any(outlier_mask)
            else None
        )
        return error_y_normal, error_y_outlier

    def _add_timeslice_average_overlay(
        self,
        fig: go.Figure,
        *,
        df: pd.DataFrame,
        x_col: str,
        slice_duration: float | None,
        average_value: float,
        average_std: float | None,
        average_label: str | None,
        unit: str,
    ) -> None:
        """Gold ±1 std band plus a run-average line trace."""
        if slice_duration is not None:
            x_range = [0, (df[x_col].max() + 1) * slice_duration]
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

    def _add_timeslice_point_traces(
        self,
        fig: go.Figure,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        normal_mask: np.ndarray,
        outlier_mask: np.ndarray,
        primary_color: str,
        error_y_normal: dict | None,
        error_y_outlier: dict | None,
        hover_template: str,
        customdata: list | None,
    ) -> None:
        """Normal-point and outlier-point scatter traces."""
        if np.any(normal_mask):
            normal_customdata = (
                [customdata[i] for i in range(len(customdata)) if normal_mask[i]]
                if customdata is not None
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values[normal_mask],
                    y=y_values[normal_mask],
                    mode="markers",
                    marker=dict(color=primary_color, size=6, line=dict(width=0)),
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
                    x=x_values[outlier_mask],
                    y=y_values[outlier_mask],
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

    def _add_timeslice_std_legend_proxy(
        self, fig: go.Figure, primary_color: str
    ) -> None:
        """Off-canvas legend-only trace representing ±1 timeslice std error bars."""
        fig.add_trace(
            go.Scatter(
                x=[-999999, -999999],
                y=[0, 1],
                mode="lines",
                line=dict(color=primary_color, width=3),
                name="±1 Timeslice Std",
                showlegend=True,
                legendrank=2,
                hoverinfo="skip",
            )
        )

    def _build_timeslice_layout(
        self,
        title: str,
        x_label: str,
        y_label: str,
        *,
        df: pd.DataFrame,
        x_col: str,
        slice_duration: float | None,
        warning_text: str | None,
    ) -> dict:
        """Timeslice-specific layout: grid, diagonal ticks, bottom warning banner."""
        layout = self._get_base_layout(title, x_label, y_label, hovermode="closest")
        grid_color = (
            "rgba(200, 200, 200, 0.2)"
            if self.theme == PlotTheme.LIGHT
            else "rgba(100, 100, 100, 0.2)"
        )
        for axis in ("xaxis", "yaxis"):
            layout[axis]["showgrid"] = True
            layout[axis]["gridwidth"] = 0.5
            layout[axis]["gridcolor"] = grid_color
            layout[axis]["linewidth"] = 1
        layout["yaxis"]["rangemode"] = "tozero"

        if slice_duration is not None:
            self._apply_timeslice_tick_config(layout, df, x_col, slice_duration)
        if warning_text:
            self._apply_timeslice_warning(layout, warning_text, slice_duration)
        return layout

    def _apply_timeslice_tick_config(
        self,
        layout: dict,
        df: pd.DataFrame,
        x_col: str,
        slice_duration: float,
    ) -> None:
        """Diagonal time-range ticks on the x-axis."""
        slice_indices = df[x_col].values
        max_slice = slice_indices.max()
        layout["xaxis"]["tickmode"] = "array"
        tick_positions = [
            i * slice_duration + slice_duration / 2 for i in range(int(max_slice) + 1)
        ]
        tick_labels = [
            f"{int(i * slice_duration)}-{int((i + 1) * slice_duration)}"
            for i in range(int(max_slice) + 1)
        ]
        layout["xaxis"]["tickvals"] = tick_positions
        layout["xaxis"]["ticktext"] = tick_labels
        layout["xaxis"]["tickangle"] = -45
        if "margin" not in layout:
            layout["margin"] = {}
        layout["margin"]["b"] = 100
        layout["xaxis"]["range"] = [0, (max_slice + 1) * slice_duration]

    def _apply_timeslice_warning(
        self,
        layout: dict,
        warning_text: str,
        slice_duration: float | None,
    ) -> None:
        """Append a warning-banner annotation sized to clear diagonal tick labels."""
        if "annotations" not in layout:
            layout["annotations"] = []
        has_diagonal_labels = slice_duration is not None
        yshift_pixels = -85 if has_diagonal_labels else -50
        layout["margin"]["b"] = 140 if has_diagonal_labels else 100
        layout["annotations"] = list(layout.get("annotations", [])) + [
            self._build_warning_annotation(warning_text, y=0, yshift=yshift_pixels)
        ]

    def create_dual_axis_plot(
        self,
        df_primary: pd.DataFrame,
        df_secondary: pd.DataFrame,
        x_col_primary: str,
        x_col_secondary: str,
        y1_metric: str,
        y2_metric: str,
        *,
        primary_style: Style | None = None,
        secondary_style: Style | None = None,
        active_count_col: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y1_label: str | None = None,
        y2_label: str | None = None,
    ) -> go.Figure:
        """Dual Y-axis plot with independent data sources + per-series :class:`Style` (0-100 when both percentages)."""
        default_style = Style(mode="lines", line_shape=None, fill=None)
        primary_style = primary_style or default_style
        secondary_style = secondary_style or default_style
        title, x_label, y1_label, y2_label = self._resolve_dual_axis_labels(
            title,
            x_label,
            y1_label,
            y2_label,
            y1_metric=y1_metric,
            y2_metric=y2_metric,
        )
        primary_hover = self._build_dual_axis_primary_hover(
            x_label, y1_label, df_primary, active_count_col
        )
        customdata = df_primary[active_count_col] if active_count_col else None
        palette = self._get_palette_colors(2)
        primary_color = palette[0]
        secondary_color = palette[1] if len(palette) > 1 else palette[0]

        fig = go.Figure()
        self._add_dual_axis_trace(
            fig,
            df=df_primary,
            x_col=x_col_primary,
            y_col=y1_metric,
            style=primary_style,
            color=primary_color,
            name=y1_label,
            yaxis="y",
            hovertemplate=primary_hover,
            customdata=customdata,
        )
        self._add_dual_axis_trace(
            fig,
            df=df_secondary,
            x_col=x_col_secondary,
            y_col=y2_metric,
            style=secondary_style,
            color=secondary_color,
            name=y2_label,
            yaxis="y2",
            hovertemplate=f"{x_label}: %{{x:.1f}}s<br>{y2_label}: %{{y:.1f}}<extra></extra>",
            customdata=None,
        )
        fig.update_layout(
            self._build_dual_axis_layout(
                title,
                x_label,
                y1_label,
                y2_label=y2_label,
                y1_metric=y1_metric,
                y2_metric=y2_metric,
            )
        )
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )
        return fig

    def _resolve_dual_axis_labels(
        self,
        title: str | None,
        x_label: str | None,
        y1_label: str | None,
        y2_label: str | None,
        *,
        y1_metric: str,
        y2_metric: str,
    ) -> tuple[str, str, str, str]:
        """Fill in auto-generated title + axis labels for a dual-axis plot."""
        title = (
            title
            or f"{get_metric_display_name(y1_metric)} with {get_metric_display_name(y2_metric)}"
        )
        x_label = x_label or "Time (s)"
        y1_label = y1_label or get_metric_display_name(y1_metric)
        y2_label = y2_label or get_metric_display_name(y2_metric)
        return title, x_label, y1_label, y2_label

    def _build_dual_axis_primary_hover(
        self,
        x_label: str,
        y1_label: str,
        df_primary: pd.DataFrame,
        active_count_col: str | None,
    ) -> str:
        """Primary hover template with optional Active Requests line."""
        hover = f"{x_label}: %{{x:.1f}}s<br>{y1_label}: %{{y:.1f}}"
        if active_count_col and active_count_col in df_primary.columns:
            hover += "<br>Active Requests: %{customdata}"
        return hover + "<extra></extra>"

    def _add_dual_axis_trace(
        self,
        fig: go.Figure,
        *,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        style: Style,
        color: str,
        name: str,
        yaxis: str,
        hovertemplate: str,
        customdata: pd.Series | None,
    ) -> None:
        """Build + add one Scatter trace honoring :class:`Style` (line_shape / fill)."""
        config: dict = {
            "x": df[x_col],
            "y": df[y_col],
            "mode": style.mode,
            "line": dict(width=style.line_width, color=color),
            "name": name,
            "yaxis": yaxis,
            "customdata": customdata,
            "hovertemplate": hovertemplate,
        }
        if style.line_shape:
            config["line"]["shape"] = style.line_shape
        if style.fill:
            r, g, b = mcolors.to_rgb(color)
            config["fill"] = style.fill
            config["fillcolor"] = (
                f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {style.fill_opacity})"
            )
        fig.add_trace(go.Scatter(**config))

    def _build_dual_axis_layout(
        self,
        title: str,
        x_label: str,
        y1_label: str,
        *,
        y2_label: str,
        y1_metric: str,
        y2_metric: str,
    ) -> dict:
        """Base layout plus right-side yaxis2 for a dual-axis plot."""
        layout = self._get_base_layout(title, x_label, y1_label, hovermode="x unified")
        both_percentage = _is_percentage_metric(y1_metric) and _is_percentage_metric(
            y2_metric
        )
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
        return layout

    def create_latency_scatter_with_percentiles(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_metric: str,
        percentile_cols: list[str],
        *,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Scatter of individual request latencies with overlaid rolling-percentile lines."""
        fig = go.Figure()

        title = (
            title or f"{get_metric_display_name(y_metric)} Over Time with Percentiles"
        )
        x_label = x_label or get_metric_display_name(x_col)
        y_label = y_label or get_metric_display_name(y_metric)

        percentile_colors = self._get_palette_colors(len(percentile_cols))

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

        for idx, percentile_col in enumerate(percentile_cols):
            if percentile_col not in df.columns:
                continue
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

        fig.update_layout(
            self._get_base_layout(title, x_label, y_label, hovermode="x unified")
        )
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )
        return fig

    def create_request_timeline(
        self,
        df: pd.DataFrame,
        y_metric: str,
        *,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> go.Figure:
        """Request timeline: horizontal lines per request split into prefill + decode phases.

        Expects ``df`` with columns ``[request_id, y_value, start_s, ttft_end_s, end_s]``.
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
            self._add_request_timeline_row(
                fig,
                row,
                y_label=y_label,
                ttft_color=ttft_color,
                generation_color=generation_color,
                ttft_legend_added=ttft_legend_added,
                generation_legend_added=generation_legend_added,
            )
            ttft_legend_added = True
            if (row["end_s"] - row["ttft_end_s"]) > 0.001:
                generation_legend_added = True

        layout = self._get_base_layout(title, x_label, y_label, hovermode="closest")
        layout["yaxis"]["rangemode"] = "normal"
        fig.update_layout(layout)
        fig.update_layout(
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(r=20),
        )
        return fig

    def _add_request_timeline_row(
        self,
        fig: go.Figure,
        row: pd.Series,
        *,
        y_label: str,
        ttft_color: str,
        generation_color: str,
        ttft_legend_added: bool,
        generation_legend_added: bool,
    ) -> None:
        """Prefill + optional decode segment traces for one request row."""
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

    def create_percentile_bands(
        self,
        df: pd.DataFrame,
        x_col: str,
        *,
        percentile_cols: list[str],
        lower_col: str | None,
        metric_name: str,
        metric_type: str,
        title: str,
        x_label: str,
        y_label: str,
        unit: str,
    ) -> go.Figure:
        """Percentile bands plot: p50 line with p95/p99 shaded bands (optional lower band)."""
        fig = go.Figure()
        x = df[x_col]

        self._add_percentile_band_traces(
            fig, df, x, percentile_cols=percentile_cols, lower_col=lower_col, unit=unit
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
            showlegend=True,
            height=600,
        )
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

    def _add_percentile_band_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        x: pd.Series,
        *,
        percentile_cols: list[str],
        lower_col: str | None,
        unit: str,
    ) -> None:
        """Add p99/p95/p50 and optional lower-band traces (outermost first for stacking)."""
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
        if "p95" in percentile_cols and "p95" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df["p95"],
                    fill="tonexty" if "p99" in df.columns else None,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(68, 138, 255, 0.2)",
                    name="p95-p99 band" if "p99" in df.columns else "p95",
                    hovertemplate=f"Time: %{{x:.2f}}s<br>p95: %{{y:.3f}} {unit}<extra></extra>",
                )
            )
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

    def create_bucket_histogram(
        self,
        buckets: dict[str, int],
        metric_name: str,
        *,
        title: str,
        x_label: str,
        y_label: str,
        unit: str,
    ) -> go.Figure:
        """Prometheus histogram bucket distribution: one bar per ``le`` upper bound."""
        bucket_labels, bucket_counts = _sort_prometheus_buckets(buckets)

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
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            showlegend=False,
            height=600,
            xaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=bucket_labels,
            ),
        )

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

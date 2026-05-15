# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib renderer for latency-throughput uncertainty plots."""

import math

import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_GREEN,
    PlotTheme,
)
from aiperf.plot.models.uncertainty import LatencyThroughputUncertaintyData

_EIGENVALUE_FLOOR = 1e-12


def _ellipse_params_from_covariance(
    point_cov_xy: float,
    x_half_width: float,
    y_half_width: float,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Compute Ellipse width, height, and angle from covariance.

    No chi-square scaling is applied because the input half-widths are already
    confidence-scaled (t_crit * SE) by the handler. This matches the Plotly
    renderer's _build_ellipse_trace and _ellipse_params_axis_aligned.

    Args:
        point_cov_xy: Sample covariance between x and y metrics.
        x_half_width: Pre-scaled CI half-width for x axis.
        y_half_width: Pre-scaled CI half-width for y axis.
        confidence_level: Retained for API compatibility (unused).

    Returns:
        (width, height, angle_degrees) for matplotlib.patches.Ellipse.
    """
    var_x = x_half_width**2
    var_y = y_half_width**2
    cov = np.array([[var_x, point_cov_xy], [point_cov_xy, var_y]])

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, _EIGENVALUE_FLOOR)

    width = 2.0 * math.sqrt(float(eigenvalues[1]))
    height = 2.0 * math.sqrt(float(eigenvalues[0]))
    angle_rad = math.atan2(float(eigenvectors[1, 1]), float(eigenvectors[0, 1]))
    angle_deg = math.degrees(angle_rad)

    return width, height, angle_deg


def _ellipse_params_axis_aligned(
    x_half_width: float,
    y_half_width: float,
) -> tuple[float, float, float]:
    """Compute Ellipse width, height for axis-aligned case.

    No chi-square scaling is applied because the input half-widths are already
    confidence-scaled (t_crit * SE) by the handler. Both this function and
    _ellipse_params_from_covariance use the same contract: pre-scaled inputs,
    no additional scaling.

    Returns:
        (width, height, 0.0) — full diameters, zero rotation.
    """
    return 2.0 * x_half_width, 2.0 * y_half_width, 0.0


def _apply_theme(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    colors: dict[str, str],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Apply NVIDIA brand styling to figure and axes."""
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["paper"])
    ax.set_title(title, color=colors["text"], fontsize=14)
    ax.set_xlabel(x_label, color=colors["text"])
    ax.set_ylabel(y_label, color=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["border"])
    ax.grid(True, color=colors["grid"], alpha=0.3)


def _add_mean_line_and_errorbars(
    ax: plt.Axes,
    sorted_points: list,
    *,
    color: str = NVIDIA_GREEN,
    label: str = "Mean",
) -> None:
    """Add mean-point line with markers and asymmetric crosshair error bars."""
    x_vals = [p.x_mean for p in sorted_points]
    y_vals = [p.y_mean for p in sorted_points]

    linestyle = "-" if len(sorted_points) > 1 else "None"
    ax.plot(
        x_vals,
        y_vals,
        color=color,
        marker="o",
        markersize=6,
        linestyle=linestyle,
        linewidth=2,
        label=label,
        zorder=3,
    )
    ax.errorbar(
        x_vals,
        y_vals,
        xerr=[
            [p.x_mean - p.x_ci_low for p in sorted_points],
            [p.x_ci_high - p.x_mean for p in sorted_points],
        ],
        yerr=[
            [p.y_mean - p.y_ci_low for p in sorted_points],
            [p.y_ci_high - p.y_mean for p in sorted_points],
        ],
        fmt="none",
        ecolor=color,
        elinewidth=1,
        capsize=3,
        alpha=0.7,
        zorder=2,
    )


def _add_ellipses(
    ax: plt.Axes,
    sorted_points: list,
    confidence_level: float,
    *,
    color: str = NVIDIA_GREEN,
) -> None:
    """Add confidence ellipse patches for each point."""
    level_pct = int(confidence_level * 100)
    ellipse_added = False
    low_n_added = False

    for point in sorted_points:
        x_half = (point.x_ci_high - point.x_ci_low) / 2.0
        y_half = (point.y_ci_high - point.y_ci_low) / 2.0

        if point.cov_xy is not None and point.cov_xy != 0:
            width, height, angle = _ellipse_params_from_covariance(
                point_cov_xy=point.cov_xy,
                x_half_width=x_half,
                y_half_width=y_half,
                confidence_level=confidence_level,
            )
        else:
            width, height, angle = _ellipse_params_axis_aligned(
                x_half_width=x_half,
                y_half_width=y_half,
            )

        is_low_n = point.n_runs is not None and point.n_runs < 3
        if is_low_n and not low_n_added:
            label = "Low sample (n < 3)"
            low_n_added = True
        elif not is_low_n and not ellipse_added:
            label = f"{level_pct}% Confidence Region"
            ellipse_added = True
        else:
            label = None

        ax.add_patch(
            matplotlib.patches.Ellipse(
                xy=(point.x_mean, point.y_mean),
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=0.08 if is_low_n else 0.15,
                linewidth=1,
                linestyle="--" if is_low_n else "-",
                label=label,
                zorder=1,
            )
        )


def render_matplotlib_uncertainty(
    data: LatencyThroughputUncertaintyData,
    theme: PlotTheme = PlotTheme.LIGHT,
) -> matplotlib.figure.Figure:
    """Render uncertainty plot using Matplotlib.

    Args:
        data: Shared data contract with benchmark points and metadata.
        theme: Plot theme for NVIDIA brand styling.

    Returns:
        Matplotlib Figure object.
    """
    colors = LIGHT_THEME_COLORS if theme == PlotTheme.LIGHT else DARK_THEME_COLORS
    fig, ax = plt.subplots(figsize=(10, 6))

    title = data.title or "Latency vs Throughput (Joint Uncertainty)"
    x_label = data.x_label or "Latency"
    y_label = data.y_label or "Throughput"
    _apply_theme(fig, ax, colors, title=title, x_label=x_label, y_label=y_label)

    all_series = data.get_series()
    if not all_series:
        return fig

    # Generate distinct colors for each series
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    series_colors = [
        f"#{int(cmap(i)[0] * 255):02x}{int(cmap(i)[1] * 255):02x}{int(cmap(i)[2] * 255):02x}"
        if len(all_series) > 1
        else NVIDIA_GREEN
        for i in range(len(all_series))
    ]

    for s, s_color in zip(all_series, series_colors, strict=False):
        sorted_points = sorted(s.points, key=lambda p: p.x_mean)
        if not sorted_points:
            continue
        _add_mean_line_and_errorbars(ax, sorted_points, color=s_color, label=s.name)
        _add_ellipses(ax, sorted_points, data.confidence_level, color=s_color)

        for point in sorted_points:
            if point.label is not None:
                ax.annotate(
                    point.label,
                    (point.x_mean, point.y_mean),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=colors["text"],
                    fontsize=8,
                )

    ax.legend(
        facecolor=colors["paper"], edgecolor=colors["border"], labelcolor=colors["text"]
    )
    fig.tight_layout()
    return fig

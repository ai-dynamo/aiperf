# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot container construction helpers for the dashboard."""

from dash import dcc, html

from aiperf.plot.constants import PlotTheme
from aiperf.plot.dashboard.styling import get_theme_colors


def _build_container_style(
    theme: PlotTheme, *, size: int, size_class: str, visible: bool
) -> dict:
    """Compute the outer Div style for a plot container."""
    colors = get_theme_colors(theme)
    # Half-width (50%): base height, Full-width (100%): 2x height to keep aspect ratio.
    calculated_height = size * 2 if size_class == "full" else size
    return {
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


def _build_control_buttons(plot_id: str, *, resizable: bool) -> list:
    """Build settings, hide, and (optional) resize controls for a plot container."""
    buttons = [
        html.Button(
            "⚙",
            id={"type": "settings-plot-btn", "index": plot_id},
            className="plot-settings-btn",
            title="Edit plot settings",
        ),
        html.Button(
            "👁",
            id={"type": "hide-plot-btn-direct", "index": plot_id},
            className="plot-hide-btn",
            title="Hide plot",
        ),
    ]
    if resizable:
        buttons.append(
            html.Button(
                "⇲",
                id={"type": "resize-handle", "index": plot_id},
                className="resize-handle",
                title="Click to toggle size",
                style={"background": "none", "border": "none", "padding": "4px"},
            )
        )
    return buttons


def _build_graph(plot_id: str, figure) -> dcc.Graph:
    """Build the dcc.Graph with standard dashboard config."""
    return dcc.Graph(
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
    )


def create_plot_container_component(
    plot_id: str,
    figure,
    theme: PlotTheme,
    *,
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
    return html.Div(
        [
            *_build_control_buttons(plot_id, resizable=resizable),
            _build_graph(plot_id, figure),
        ],
        id={"type": "plot-container", "index": plot_id},
        className=f"plot-container size-{size_class}",
        style=_build_container_style(
            theme, size=size, size_class=size_class, visible=visible
        ),
    )

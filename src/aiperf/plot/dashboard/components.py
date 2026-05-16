# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reusable UI components for the AIPerf dashboard.

This module provides factory functions for creating common dashboard elements
like dropdowns, buttons, labels, and sections with consistent styling.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    PLOT_FONT_FAMILY,
    PlotTheme,
)
from aiperf.plot.dashboard.run_selector import create_run_selector_checklist
from aiperf.plot.dashboard.styling import (
    get_button_style,
    get_label_style,
    get_section_header_style,
    get_theme_colors,
)

__all__ = [
    "create_button",
    "create_collapsible_section",
    "create_export_controls_card",
    "create_fieldset",
    "create_global_stat_selector",
    "create_label",
    "create_log_scale_dropdown",
    "create_autoscale_dropdown",
    "create_run_selector_checklist",
    "create_section_header",
    "create_sidebar_toggle_button",
    "create_stat_selector_dropdown",
    "create_switch_toggle",
    "create_two_column_row",
]


def create_section_header(text: str, theme: PlotTheme) -> html.Div:
    """
    Create a section header with consistent styling.

    Args:
        text: Header text (will be uppercase)
        theme: Plot theme

    Returns:
        Dash HTML Div component
    """
    return html.Div(text, style=get_section_header_style(theme))


def create_label(text: str, theme: PlotTheme, id: str | None = None) -> html.Label:
    """
    Create a form label with consistent styling.

    Args:
        text: Label text
        theme: Plot theme
        id: Optional component ID

    Returns:
        Dash HTML Label component
    """
    kwargs = {"style": get_label_style(theme)}
    if id is not None:
        kwargs["id"] = id
    return html.Label(text, **kwargs)


def create_fieldset(title: str, children: list, theme: PlotTheme) -> html.Fieldset:
    """
    Create a styled fieldset with legend for visual grouping.

    Args:
        title: Fieldset legend title
        children: Content components to display inside the fieldset
        theme: Plot theme for styling

    Returns:
        Dash HTML Fieldset component with styled legend
    """
    colors = get_theme_colors(theme)
    return html.Fieldset(
        [
            html.Legend(
                title,
                style={
                    "font-size": "11px",
                    "font-weight": "600",
                    "color": colors["text"],
                    "padding": "0 8px",
                },
            ),
            html.Div(children, style={"padding": "8px 12px"}),
        ],
        style={
            "border": f"1px solid {colors['border']}",
            "border-radius": "4px",
            "margin-bottom": "12px",
            "padding": "0",
        },
    )


def create_two_column_row(left: list, right: list, gap: str = "16px") -> html.Div:
    """
    Create a flex row with two equal columns.

    Args:
        left: Components for the left column
        right: Components for the right column
        gap: Gap between columns (CSS value)

    Returns:
        Dash HTML Div with flex layout
    """
    return html.Div(
        [
            html.Div(left, style={"flex": "1", "min-width": "0"}),
            html.Div(right, style={"flex": "1", "min-width": "0"}),
        ],
        style={
            "display": "flex",
            "gap": gap,
            "align-items": "flex-start",
        },
    )


def create_switch_toggle(
    component_id: str,
    label: str,
    theme: PlotTheme,
    default: bool = False,
) -> html.Div:
    """
    Create a labeled switch toggle.

    Args:
        component_id: Unique ID for the switch component
        label: Text label to display next to the switch
        theme: Plot theme for styling
        default: Default value (True=on, False=off)

    Returns:
        Dash HTML Div containing a styled switch toggle
    """
    colors = get_theme_colors(theme)
    return html.Div(
        [
            dbc.Switch(
                id=component_id,
                value=default,
                label=label,
                style={
                    "font-size": "11px",
                    "font-family": PLOT_FONT_FAMILY,
                },
                className=f"theme-{theme.value}",
            ),
        ],
        style={
            "display": "flex",
            "align-items": "center",
            "color": colors["text"],
        },
    )


def create_stat_selector_dropdown(
    metric_name: str, default_stat: str = "p50", theme: PlotTheme = PlotTheme.DARK
) -> html.Div:
    """
    Create a dropdown for selecting metric statistics.

    Args:
        metric_name: Name of the metric (used for ID)
        default_stat: Default statistic to show
        theme: Plot theme

    Returns:
        Dash HTML Div containing label and dropdown
    """
    # Format label from metric name
    label_text = metric_name.replace("_", " ").title()

    # Create options with nice labels
    stat_labels = {
        "p50": "p50 (Median)",
        "p90": "p90",
        "p95": "p95",
        "p99": "p99",
        "avg": "Average",
        "min": "Minimum",
        "max": "Maximum",
        "std": "Std Dev",
    }

    # Filter to relevant stats (percentiles + avg/min/max)
    relevant_stats = [
        s for s in ALL_STAT_KEYS if s.startswith("p") or s in ["avg", "min", "max"]
    ]
    options = [{"label": stat_labels.get(s, s), "value": s} for s in relevant_stats]

    return html.Div(
        [
            create_label(label_text, theme),
            dcc.Dropdown(
                id={"type": "metric-stat-selector", "metric": metric_name},
                options=options,
                value=default_stat,
                clearable=False,
                style={
                    "font-size": "11px",
                    "font-family": PLOT_FONT_FAMILY,
                    "margin-bottom": "10px",
                },
                className="dark-dropdown" if theme == PlotTheme.DARK else "",
            ),
        ],
        style={"margin-bottom": "12px"},
    )


def create_log_scale_dropdown(plot_id: str, theme: PlotTheme) -> html.Div:
    """
    Create a dropdown for log scale selection.

    Args:
        plot_id: ID of the plot (e.g., 'pareto')
        theme: Plot theme

    Returns:
        Dash HTML Div containing label and dropdown
    """
    return html.Div(
        [
            create_label(plot_id.replace("-", " ").replace("_", " ").title(), theme),
            dcc.Dropdown(
                id=f"{plot_id}-log-scale",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "X-axis", "value": "x"},
                    {"label": "Y-axis", "value": "y"},
                    {"label": "Both", "value": "both"},
                ],
                value="none",
                clearable=False,
                style={
                    "font-size": "11px",
                    "font-family": PLOT_FONT_FAMILY,
                    "margin-bottom": "10px",
                },
                className="dark-dropdown" if theme == PlotTheme.DARK else "",
            ),
        ]
    )


def create_autoscale_dropdown(plot_id: str, theme: PlotTheme) -> html.Div:
    """
    Create a dropdown for controlling which axes to autoscale.

    Args:
        plot_id: ID of the plot (e.g., 'custom', 'edit')
        theme: Plot theme

    Returns:
        Dash HTML Div containing label and dropdown
    """
    return html.Div(
        [
            create_label("Autoscale", theme),
            dcc.Dropdown(
                id=f"{plot_id}-autoscale",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "X-axis", "value": "x"},
                    {"label": "Y-axis", "value": "y"},
                    {"label": "Both", "value": "both"},
                ],
                value="none",
                clearable=False,
                style={
                    "font-size": "11px",
                    "font-family": PLOT_FONT_FAMILY,
                    "margin-bottom": "10px",
                },
                className="dark-dropdown" if theme == PlotTheme.DARK else "",
            ),
        ]
    )


def create_collapsible_section(
    section_id: str,
    title: str,
    children: list,
    theme: PlotTheme,
    *,
    initially_open: bool = False,
) -> html.Div:
    """
    Create a collapsible section with header and content.

    Args:
        section_id: Unique ID for the section
        title: Section title
        children: Content to show when expanded
        theme: Plot theme
        initially_open: Whether section is open by default

    Returns:
        Dash HTML Div containing header and collapsible content
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        title,
                        style={
                            "flex": "1",
                            "font-size": "11px",
                            "font-weight": "600",
                            "text-transform": "uppercase",
                            "letter-spacing": "0.8px",
                            "color": NVIDIA_GRAY,
                            "font-family": PLOT_FONT_FAMILY,
                        },
                    ),
                    html.Span(
                        "▼" if initially_open else "▶",
                        id={"type": "section-arrow", "id": section_id},
                        style={
                            "font-size": "10px",
                            "color": NVIDIA_GRAY,
                            "transition": "transform 0.2s",
                        },
                    ),
                ],
                id={"type": "section-header", "id": section_id},
                n_clicks=0,
                style={
                    "display": "flex",
                    "align-items": "center",
                    "cursor": "pointer",
                    "padding-bottom": "6px",
                    "margin-bottom": "12px",
                },
            ),
            html.Div(
                children,
                id={"type": "section-content", "id": section_id},
                style={"display": "block" if initially_open else "none"},
            ),
        ]
    )


def create_button(
    button_id: str,
    text: str,
    theme: PlotTheme,
    *,
    variant: str = "primary",
    n_clicks: int = 0,
) -> html.Button:
    """
    Create a styled button.

    Args:
        button_id: Unique ID for the button
        text: Button text
        theme: Plot theme
        variant: Button style variant ('primary', 'secondary', 'outline')
        n_clicks: Initial click count

    Returns:
        Dash HTML Button component
    """
    return html.Button(
        text, id=button_id, n_clicks=n_clicks, style=get_button_style(theme, variant)
    )


def create_sidebar_toggle_button(theme: PlotTheme) -> html.Button:
    """
    Create a button to toggle sidebar visibility.

    Args:
        theme: Plot theme

    Returns:
        Dash HTML Button component with hamburger icon
    """
    return html.Button(
        "☰",
        id="sidebar-toggle-btn",
        n_clicks=0,
        style={
            "position": "fixed",
            "top": "70px",
            "left": "10px",
            "width": "40px",
            "height": "40px",
            "background": "rgba(118, 185, 0, 0.15)",
            "color": "white",
            "border": f"2px solid {NVIDIA_GREEN}",
            "border-radius": "6px",
            "cursor": "pointer",
            "font-size": "20px",
            "z-index": "2000",
            "display": "flex",
            "align-items": "center",
            "justify-content": "center",
            "box-shadow": "0 2px 8px rgba(0,0,0,0.2)",
            "transition": "all 0.2s",
            "font-family": PLOT_FONT_FAMILY,
            "backdrop-filter": "blur(4px)",
        },
    )


# Re-exported here so callers importing from ``components`` continue to work
# after these panels were split into ``sidebar_panels`` to keep this module
# under the file-size limit.
from aiperf.plot.dashboard.sidebar_panels import (  # noqa: E402
    create_export_controls_card,
    create_global_stat_selector,
)

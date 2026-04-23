# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sidebar panel components (global stat selector, export controls card).

Separated from ``components.py`` to keep each UI module small.
"""

from dash import dcc, html

from aiperf.plot.constants import PLOT_FONT_FAMILY, PlotTheme
from aiperf.plot.dashboard.components import (
    create_button,
    create_collapsible_section,
    create_label,
)


def create_global_stat_selector(theme: PlotTheme) -> html.Div:
    """
    Create a global stat selector to apply one stat to all metrics.

    Args:
        theme: Plot theme

    Returns:
        Dash HTML Div with dropdown and apply button
    """
    return html.Div(
        [
            create_label("Quick Apply Stats to All Metrics", theme),
            html.Div(
                [
                    dcc.Dropdown(
                        id="global-stat-selector",
                        options=[
                            {"label": "p50 (Median)", "value": "p50"},
                            {"label": "p90", "value": "p90"},
                            {"label": "p95", "value": "p95"},
                            {"label": "p99", "value": "p99"},
                            {"label": "Average", "value": "avg"},
                            {"label": "Min", "value": "min"},
                            {"label": "Max", "value": "max"},
                        ],
                        value="p50",
                        clearable=False,
                        style={
                            "font-size": "11px",
                            "margin-bottom": "8px",
                            "font-family": PLOT_FONT_FAMILY,
                        },
                    ),
                    create_button(
                        "btn-apply-global-stat",
                        "Apply to All Metrics",
                        theme,
                        variant="secondary",
                    ),
                ],
            ),
        ],
        style={"margin-bottom": "20px"},
    )


def create_export_controls_card(theme: PlotTheme) -> html.Div:
    """
    Create export controls card with format and size selection.

    Provides a collapsible card containing:
    - Format selection: PNG (static) or HTML (interactive)
    - Size selection: Small/Medium/Large (PNG only)
    - Export button to trigger download
    - Hidden download component (outside collapsible section for proper functionality)

    Args:
        theme: Plot theme for consistent styling

    Returns:
        Dash HTML Div containing download component and collapsible export controls
    """
    collapsible_content = [
        create_label("Export Format", theme),
        dcc.Dropdown(
            id="export-format-selector",
            options=[
                {"label": "PNG", "value": "png"},
                {"label": "HTML", "value": "html"},
            ],
            value="png",
            clearable=False,
            style={
                "margin-bottom": "16px",
                "font-size": "12px",
            },
        ),
        create_label("Export Size (PNG only)", theme, id="export-size-label"),
        dcc.Dropdown(
            id="export-size-selector",
            options=[
                {"label": "Small (800×400)", "value": "small"},
                {"label": "Medium (1600×800)", "value": "medium"},
                {"label": "Large (2400×1200)", "value": "large"},
            ],
            value="medium",
            clearable=False,
            style={
                "margin-bottom": "16px",
                "font-size": "12px",
            },
        ),
        create_button(
            "btn-export-png",
            "Export Visible Plots",
            theme,
            variant="secondary",
        ),
    ]

    return html.Div(
        [
            dcc.Download(id="download-png-bundle"),
            create_collapsible_section(
                section_id="export-controls",
                title="EXPORT",
                children=collapsible_content,
                theme=theme,
                initially_open=True,
            ),
        ]
    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run-selector checklist component for the AIPerf dashboard sidebar.

Separated from ``components.py`` to keep individual UI component modules small.
"""

from dash import dcc, html

from aiperf.plot.constants import (
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    PLOT_FONT_FAMILY,
    PlotTheme,
)


def _group_runs_by_field(
    runs: list, group_by: str, selected_indices: list[int]
) -> dict[str, list[dict]]:
    """Bucket runs by the given metadata field, preserving selection state."""
    groups: dict[str, list[dict]] = {}
    for idx, run in enumerate(runs):
        group_value = getattr(run.metadata, group_by, None) or "Unknown"
        groups.setdefault(group_value, [])

        label = f"{run.metadata.model or 'Unknown'}"
        if run.metadata.concurrency:
            label += f" - C{run.metadata.concurrency}"

        groups[group_value].append(
            {"label": label, "value": idx, "selected": idx in selected_indices}
        )
    return groups


def _build_group_block(
    group_name: str, group_runs: list[dict], text_color: str
) -> html.Div:
    """Build a single collapsible group with its nested run checklist."""
    all_selected = all(r["selected"] for r in group_runs)
    header = html.Div(
        [
            dcc.Checklist(
                id={"type": "group-selector", "index": group_name},
                options=[{"label": f"  {group_name}", "value": group_name}],
                value=[group_name] if all_selected else [],
                style={
                    "font-size": "12px",
                    "font-family": PLOT_FONT_FAMILY,
                    "font-weight": "600",
                    "display": "inline-block",
                },
                labelStyle={"color": NVIDIA_GREEN},
            ),
            html.Span(
                "▶",
                id={"type": "run-group-arrow", "index": group_name},
                style={
                    "font-size": "10px",
                    "color": text_color,
                    "margin-left": "4px",
                    "cursor": "pointer",
                    "transition": "transform 0.2s",
                },
            ),
        ],
        id={"type": "run-group-header", "index": group_name},
        n_clicks=0,
        style={
            "cursor": "pointer",
            "display": "flex",
            "align-items": "center",
        },
    )
    content = html.Div(
        [
            dcc.Checklist(
                id={"type": "run-selector-nested", "index": group_name},
                options=[
                    {"label": r["label"], "value": r["value"]} for r in group_runs
                ],
                value=[r["value"] for r in group_runs if r["selected"]],
                style={
                    "font-size": "11px",
                    "font-family": PLOT_FONT_FAMILY,
                    "margin-left": "16px",
                },
                labelStyle={
                    "display": "block",
                    "margin": "2px 0",
                    "color": text_color,
                },
            ),
        ],
        id={"type": "run-group-content", "index": group_name},
        style={"display": "none"},
    )
    return html.Div([header, content], style={"margin-bottom": "4px"})


def _build_grouped_checklist(
    run_options: list[dict],
    selected_indices: list[int],
    runs: list,
    *,
    group_by: str,
    text_color: str,
) -> html.Div:
    """Render nested checklists grouped by the given metadata field."""
    groups = _group_runs_by_field(runs, group_by, selected_indices)
    nested_items = [
        _build_group_block(name, groups[name], text_color) for name in sorted(groups)
    ]
    return html.Div(
        [
            # Hidden checklist that holds the aggregated value for compatibility
            # with existing callbacks.
            dcc.Checklist(
                id="run-selector",
                options=run_options,
                value=selected_indices,
                style={"display": "none"},
            ),
            *nested_items,
        ],
        id="run-selector-wrapper",
    )


def create_run_selector_checklist(
    run_options: list[dict],
    selected_indices: list[int],
    theme: PlotTheme,
    *,
    runs: list = None,
    group_by: str = None,
) -> html.Div:
    """
    Create a nested checklist for selecting runs to display.

    Args:
        run_options: List of dicts with 'label' and 'value' keys
        selected_indices: List of selected run indices
        theme: Plot theme
        runs: Optional list of RunData objects for grouping
        group_by: Optional field to group runs by (e.g., "model", "experiment_group")

    Returns:
        Dash HTML Div containing nested checklist
    """
    text_color = NVIDIA_GRAY if theme == PlotTheme.LIGHT else "#E0E0E0"

    if runs and group_by:
        return _build_grouped_checklist(
            run_options,
            selected_indices,
            runs,
            group_by=group_by,
            text_color=text_color,
        )

    return dcc.Checklist(
        id="run-selector",
        options=run_options,
        value=selected_indices,
        style={"font-size": "12px", "font-family": PLOT_FONT_FAMILY},
        labelStyle={"display": "block", "margin": "4px 0", "color": text_color},
    )

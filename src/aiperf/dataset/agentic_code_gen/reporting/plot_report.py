# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plotly HTML report rendering for Agentic Code datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from aiperf.dataset.agentic_code_gen.reporting.templates import render_template
from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn
from aiperf.plot.constants import NVIDIA_GREEN

_HISTOGRAM_PLOTS: list[tuple[str, str]] = [
    ("total_isl", "ISL Distribution"),
    ("total_osl", "OSL Distribution"),
    ("initial_context", "Initial Context (Turn 0 Input Length)"),
    ("new_tokens_per_turn", "New Tokens Per Turn"),
    ("generation_length", "Generation Length"),
    ("inter_turn_delay_s", "Inter-Turn Delay (s)"),
    ("turns_per_session", "Turns Per Session"),
    ("hash_id_block_count", "Hash ID Blocks Per Turn"),
    ("request_latency_s", "Estimated Request Latency (s)"),
    ("session_duration_min", "Estimated Session Duration (min)"),
]

_CACHE_HISTOGRAM_PLOTS: list[tuple[str, str]] = [
    ("prefix_length", "Prefix Length (tokens)"),
    ("unique_prompt_length", "Unique Prompt Length (tokens)"),
    ("prefix_ratio", "Prefix Ratio"),
    ("sequential_cache_hit_rate", "Sequential Cache Hit Rate"),
]


def render_plot_report(
    metrics: dict[str, np.ndarray],
    sessions: dict[str, list[ParsedTurn]],
    output_dir: Path,
) -> Path:
    figures: list[tuple[go.Figure, str]] = []

    for key, title in _HISTOGRAM_PLOTS:
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        figures.append((_histogram(arr, title, key), ""))

    for key, title in _CACHE_HISTOGRAM_PLOTS:
        arr = metrics.get(key)
        if arr is None or len(arr) == 0:
            continue
        figures.append((_histogram(arr, title, key), ""))

    sample_ids = list(sessions.keys())[:10]
    if sample_ids:
        figures.append((_context_growth_figure(sessions, sample_ids), "span2"))

    if sample_ids and "per_session_cache_hit_rate" in metrics:
        figures.append((_cache_evolution_figure(sessions, sample_ids), "span2"))

    seq_hr = metrics.get("sequential_cache_hit_rate")
    total_isl = metrics.get("total_isl")
    if seq_hr is not None and total_isl is not None and len(seq_hr) > 0:
        figures.append((_cache_hit_scatter(total_isl, seq_hr), "span2"))

    cards_html: list[str] = []
    for idx, (fig, css_class) in enumerate(figures):
        inner_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=False,
            div_id=f"plot-{idx}",
        )
        cls = f"card {css_class}".strip()
        cards_html.append(f'<div class="{cls}">{inner_html}</div>')

    html = render_template(
        "report.html",
        PLOTLY_JS_CDN="https://cdn.plot.ly/plotly-2.35.2.min.js",
        CARDS_HTML="".join(cards_html),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path


def _histogram(arr: np.ndarray, title: str, key: str) -> go.Figure:
    return go.Figure(
        data=[go.Histogram(x=arr, marker_color=NVIDIA_GREEN, showlegend=False)],
        layout=go.Layout(
            title=title,
            xaxis_title=key.replace("_", " ").title(),
            yaxis_title="Count",
            template="plotly_white",
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
        ),
    )


def _context_growth_figure(
    sessions: dict[str, list[ParsedTurn]], sample_ids: list[str]
) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title="Context Growth (Sample Sessions)",
            xaxis_title="Turn Index",
            yaxis_title="Input Length (tokens)",
            template="plotly_white",
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
        )
    )
    for sid in sample_ids:
        turns = sessions[sid]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(turns))),
                y=[t.input_length for t in turns],
                mode="lines",
                name=sid[:8],
            )
        )
    return fig


def _cache_evolution_figure(
    sessions: dict[str, list[ParsedTurn]], sample_ids: list[str]
) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title="Per-Session Cache Hit Rate Evolution",
            xaxis_title="Turn Index",
            yaxis_title="Cache Hit Rate",
            template="plotly_white",
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
        )
    )
    for sid in sample_ids:
        session_seen: set[int] = set()
        rates: list[float] = []
        for turn in sessions[sid]:
            if turn.hash_ids:
                first_unseen = len(turn.hash_ids)
                for idx, hid in enumerate(turn.hash_ids):
                    if hid not in session_seen:
                        first_unseen = idx
                        break
                rates.append(first_unseen / len(turn.hash_ids))
                session_seen.update(turn.hash_ids)
            else:
                rates.append(0.0)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sessions[sid]))),
                y=rates,
                mode="lines",
                name=sid[:8],
            )
        )
    return fig


def _cache_hit_scatter(total_isl: np.ndarray, seq_hr: np.ndarray) -> go.Figure:
    return go.Figure(
        data=[
            go.Scattergl(
                x=total_isl,
                y=seq_hr,
                mode="markers",
                marker=dict(color=NVIDIA_GREEN, size=3, opacity=0.5),
                showlegend=False,
            )
        ],
        layout=go.Layout(
            title="Cache Hit Rate vs Input Length",
            xaxis_title="Input Length (tokens)",
            yaxis_title="Sequential Cache Hit Rate",
            template="plotly_white",
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
        ),
    )

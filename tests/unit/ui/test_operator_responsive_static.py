# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static responsive/layout invariant checks for the operator UI."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_STYLE_CSS = _UI_ROOT / "style.css"
_COMPONENTS = _UI_ROOT / "components"
_PAGES = _UI_ROOT / "pages"


_CSS_RULE_RE = re.compile(r"(?P<selector>[^{}@]+)\{(?P<body>[^{}]*)\}", re.MULTILINE)
_MEDIA_RULE_RE = re.compile(
    r"@media\s*\((?P<query>[^)]*)\)\s*\{(?P<body>.*?)\n\}",
    re.DOTALL,
)


def _source(path: Path) -> str:
    return path.read_text()


def _css() -> str:
    return _source(_STYLE_CSS)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _css_without_comments(src: str) -> str:
    return re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)


def _selector_matches(raw_selector: str, wanted: str) -> bool:
    return any(_normalized(part) == wanted for part in raw_selector.split(","))


def _rule_bodies(selector: str, *, css: str | None = None) -> list[str]:
    src = _css_without_comments(_css() if css is None else css)
    return [
        match.group("body")
        for match in _CSS_RULE_RE.finditer(src)
        if _selector_matches(match.group("selector"), selector)
    ]


def _assert_rule_contains(
    selector: str, fragments: set[str], *, css: str | None = None
) -> None:
    bodies = _rule_bodies(selector, css=css)
    assert bodies, selector
    compact_bodies = [body.replace("\n", " ") for body in bodies]
    missing = {
        fragment
        for fragment in fragments
        if not any(fragment in body for body in compact_bodies)
    }
    assert missing == set()


def _assert_source_references(path: Path, fragments: set[str]) -> None:
    src = _source(path)
    missing = {fragment for fragment in fragments if fragment not in src}
    assert missing == set()


def _media_block(query_fragment: str) -> str:
    matches = [
        match.group("body")
        for match in _MEDIA_RULE_RE.finditer(_css())
        if query_fragment in match.group("query")
    ]
    assert matches, query_fragment
    return "\n".join(matches)


def test_detail_split_collapses_from_two_columns_to_one_column() -> None:
    _assert_source_references(
        _PAGES / "job-detail.js",
        {'class="detail-split"', "<${ChartWrapper}", "<${DiagnosticsPanel}"},
    )
    _assert_rule_contains(
        ".detail-split",
        {"display: grid", "grid-template-columns: 1fr 1fr", "gap: var(--space-6)"},
    )
    _assert_rule_contains(
        ".detail-split",
        {"grid-template-columns: 1fr"},
        css=_media_block("max-width: 900px"),
    )


def test_job_and_sweep_tables_share_scrollable_sticky_table_contract() -> None:
    _assert_source_references(
        _COMPONENTS / "job-table.js",
        {
            'class="job-table-wrapper"',
            'class="job-table"',
            'class="job-table-th"',
            'class="job-table-row"',
            'class="job-table-td job-table-name"',
        },
    )
    _assert_source_references(
        _PAGES / "jobs.js",
        {"import { JobTable }", "<${JobTable}"},
    )
    _assert_source_references(
        _PAGES / "sweeps.js",
        {
            'class="job-table-wrapper"',
            'class="job-table" data-testid="sweep-table"',
            'class="job-table-th"',
            'class="job-table-row"',
            'class="job-table-td job-table-name"',
        },
    )
    _assert_rule_contains(
        ".job-table-wrapper",
        {
            "overflow-x: auto",
            "overflow-y: auto",
            "max-height: calc(100vh - 240px)",
        },
    )
    _assert_rule_contains(".job-table", {"width: 100%", "border-collapse: collapse"})
    _assert_rule_contains(
        ".job-table thead th",
        {"position: sticky", "top: 0", "z-index: 2"},
    )
    _assert_rule_contains(".job-table-td", {"white-space: nowrap"})


def test_diagnostics_drawer_has_viewport_clamp_and_mobile_full_width_rules() -> None:
    _assert_source_references(
        _COMPONENTS / "diagnostics-panel.js",
        {'class="diag-tabs"', "diag-tab--active", 'class="diag-tab-count"'},
    )
    _assert_rule_contains(
        ".diagnostics-drawer",
        {"width: 420px", "max-width: 100vw", "position: fixed"},
    )
    _assert_rule_contains(
        ".diagnostics-drawer__body",
        {"flex: 1", "overflow-y: auto"},
    )
    _assert_rule_contains(
        ".diagnostics-drawer",
        {"width: 100vw"},
        css=_media_block("max-width: 600px"),
    )


def test_log_strip_is_app_level_and_height_bounded() -> None:
    _assert_source_references(
        _UI_ROOT / "app.js",
        {"import { LogStrip }", "<${LogStrip}"},
    )
    _assert_source_references(
        _COMPONENTS / "log-strip.js",
        {
            'data-testid="log-strip"',
            "log-strip--collapsed",
            'class="log-strip-head"',
            'class="log-strip-body"',
            "log-strip-cat--",
        },
    )
    _assert_rule_contains(
        ".log-strip",
        {
            "height: var(--log-strip-height)",
            "display: flex",
            "flex-direction: column",
            "flex: 0 0 auto",
        },
    )
    _assert_rule_contains(".log-strip--collapsed", {"height: auto"})
    _assert_rule_contains(
        ".log-strip--collapsed .log-strip-body",
        {"max-height: calc(11px * 1.6 * 5 + 12px)"},
    )
    _assert_rule_contains(
        ".log-strip-body",
        {"overflow-y: auto", "font-family: var(--font-mono)"},
    )


def test_chart_cards_keep_responsive_canvas_and_grid_wrappers() -> None:
    _assert_source_references(
        _COMPONENTS / "chart-wrapper.js",
        {
            'class="chart-container"',
            "responsive: true",
            "maintainAspectRatio: false",
        },
    )
    _assert_source_references(
        _PAGES / "dashboard.js",
        {
            'class="metrics-row"',
            'class="card" style="margin-bottom: var(--space-6)"',
            "<${ChartWrapper}",
        },
    )
    _assert_source_references(
        _PAGES / "job-detail.js",
        {'class="detail-split"', "<${ChartWrapper}", 'class="card"'},
    )
    _assert_rule_contains(
        ".chart-container",
        {"position: relative", "width: 100%", "min-height: 200px"},
    )
    _assert_rule_contains(
        ".chart-container canvas",
        {"width: 100% !important", "height: 100% !important"},
    )
    _assert_rule_contains(
        ".metrics-row",
        {"display: grid", "grid-template-columns: repeat(5, 1fr)"},
    )


def test_navigation_uses_responsive_flex_groups_and_app_content_scrolls() -> None:
    _assert_source_references(
        _COMPONENTS / "top-nav.js",
        {
            'class="topbar"',
            'class="topbar-left"',
            'class="nav"',
            "nav-sep",
            "nav-tab",
            'class="topbar-right"',
        },
    )
    _assert_source_references(
        _UI_ROOT / "app.js",
        {'class="app"', 'class="content"', "<${TopNav}"},
    )
    _assert_rule_contains(
        ".topbar",
        {"display: flex", "justify-content: space-between", "flex-shrink: 0"},
    )
    _assert_rule_contains(".topbar-left", {"display: flex", "align-items: center"})
    _assert_rule_contains(".nav", {"display: flex", "flex-direction: row"})
    _assert_rule_contains(".content", {"overflow-y: auto", "padding: 16px"})
    _assert_rule_contains(".main", {"min-width: 0", "overflow: hidden"})

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static edge-case tests for operator UI app shell navigation.

The app-shell pieces keep their route tables and active-link helpers private to
browser-only Preact modules, so these tests intentionally inspect source text
instead of importing components through a DOM renderer. They guard cross-file
navigation contracts that otherwise drift silently: app routes, top-nav tabs,
breadcrumb labels, and command-palette page search entries.
"""

from __future__ import annotations

from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
APP_PATH = UI_ROOT / "app.js"
TOP_NAV_PATH = UI_ROOT / "components" / "top-nav.js"
BREADCRUMB_PATH = UI_ROOT / "components" / "breadcrumb.js"
COMMAND_PALETTE_PATH = UI_ROOT / "components" / "command-palette.js"


def _source(path: Path) -> str:
    return path.read_text()


def test_launch_route_is_represented_in_all_app_shell_navigation_surfaces() -> None:
    """The Launch page should not be reachable only from the top nav."""
    app_source = _source(APP_PATH)
    top_nav_source = _source(TOP_NAV_PATH)
    breadcrumb_source = _source(BREADCRUMB_PATH)
    command_palette_source = _source(COMMAND_PALETTE_PATH)

    assert "currentRoute === '/launch'" in app_source
    assert "{ path: '/launch', label: 'Launch' }" in top_nav_source
    assert "'/launch': 'Launch'" in breadcrumb_source
    assert "{ label: 'Launch', path: '/launch' }" in command_palette_source


def test_top_nav_active_state_matches_path_segment_boundaries() -> None:
    """Section tabs should not be active for unrelated prefix routes."""
    source = _source(TOP_NAV_PATH)

    assert "currentRoute.startsWith(itemPath)" not in source
    assert "currentRoute === itemPath" in source
    assert "currentRoute.startsWith(itemPath + '/')" in source


def test_command_palette_search_covers_every_internal_top_nav_destination() -> None:
    """Ctrl+K page search should include all internal top-nav destinations."""
    top_nav_source = _source(TOP_NAV_PATH)
    command_palette_source = _source(COMMAND_PALETTE_PATH)

    internal_paths = [
        "/",
        "/jobs",
        "/sweeps",
        "/launch",
        "/leaderboard",
        "/compare",
        "/history",
    ]
    for path in internal_paths:
        if f"path: '{path}'" in top_nav_source:
            assert f"path: '{path}'" in command_palette_source

    assert "/dashboard/" not in command_palette_source

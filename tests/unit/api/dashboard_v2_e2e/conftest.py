# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for dashboard-v2 e2e tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit.api.dashboard_v2_e2e.harness import (
    DashboardHarness,
    dashboard_harness_for_browser,
    playwright_ready,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser


_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = playwright_ready()

# The test modules in this package import ``playwright`` at module scope, so
# they cannot even be collected without it installed. Playwright is an opt-in
# extra (``uv pip install playwright && uv run playwright install chromium``);
# these tests also carry the ``e2e`` marker and are deselected by default.
collect_ignore_glob = ["test_*.py"] if not _PLAYWRIGHT_AVAILABLE else []


@pytest.fixture
def _browser() -> Iterator[Browser]:
    # sync_playwright keeps a running event loop while open; scope it to one
    # test so later pytest-asyncio tests on the same xdist worker can run.
    if not _PLAYWRIGHT_AVAILABLE:
        pytest.skip(_PLAYWRIGHT_REASON)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            try:
                browser.close()
            except RuntimeError as exc:
                if "no running event loop" not in str(exc):
                    raise


@pytest.fixture
def dashboard(_browser: Browser) -> Iterator[DashboardHarness]:
    """Fresh Playwright page plus dashboard-v2 server helpers for each test."""
    yield from dashboard_harness_for_browser(_browser)


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    """Mark this package's tests as e2e and skip when Playwright is unavailable."""
    skip_marker = pytest.mark.skipif(
        not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON
    )
    e2e_marker = pytest.mark.e2e
    for item in items:
        if Path(str(item.fspath)).is_relative_to(Path(__file__).parent):
            item.add_marker(skip_marker)
            item.add_marker(e2e_marker)

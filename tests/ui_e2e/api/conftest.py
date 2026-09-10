# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for dashboard-v2 e2e tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from tests.ui_e2e.api.harness import (
    DashboardHarness,
    dashboard_harness_for_browser,
)
from tests.ui_e2e.conftest import PLAYWRIGHT_AVAILABLE as _PLAYWRIGHT_AVAILABLE
from tests.ui_e2e.conftest import PLAYWRIGHT_REASON as _PLAYWRIGHT_REASON

if TYPE_CHECKING:
    from playwright.sync_api import Browser


# The test modules in this package import ``playwright`` at module scope, so
# they cannot even be collected without it installed. Playwright is an opt-in
# extra (``uv pip install playwright && uv run playwright install chromium``);
# these tests also carry the ``e2e`` marker (applied tree-wide by
# ``tests/ui_e2e/conftest.py``) and are deselected by default.
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

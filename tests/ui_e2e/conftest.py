# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared collection hooks for every browser-based UI e2e suite.

``tests/ui_e2e/`` holds the operator dashboard suite (``operator/``), the
dashboard-v2 suite (``api/``), and the dashboard render suite (``dashboard/``).
All need a real Chromium: probe it once here instead of once per subpackage,
and mark every test under this tree ``ui_e2e`` (deselected by default, see
the ``ui_e2e`` marker in ``pyproject.toml``) plus skip it when Chromium can't
launch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest


def playwright_ready() -> tuple[bool, str]:
    """Return ``(available, reason)``. ``reason`` is the pytest skip reason on miss."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return (
            False,
            "playwright not installed (`uv pip install playwright pytest-playwright`)",
        )
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001 - skip reason should surface raw launch detail
        return (
            False,
            f"Chromium not launchable: {exc!s}. Run `uv run playwright install chromium`.",
        )
    return True, ""


PLAYWRIGHT_AVAILABLE, PLAYWRIGHT_REASON = playwright_ready()


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    """Mark every test under ``tests/ui_e2e`` as ui_e2e and skip if Chromium is unavailable."""
    skip_marker = pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason=PLAYWRIGHT_REASON)
    ui_e2e_marker = pytest.mark.ui_e2e
    for item in items:
        if Path(str(item.fspath)).is_relative_to(Path(__file__).parent):
            item.add_marker(skip_marker)
            item.add_marker(ui_e2e_marker)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for cross-page navigation (top-rail, command palette, deep links).

Covers top-rail button clicks, Ctrl+K command-palette keybinding, fuzzy-filtered
job navigation via the palette, and direct hash-URL deep links into the single-run
workbench. The SPA uses hash routing (``window.location.hash``); ``app.js``
routes ``/``, ``/archive``, ``/compare``, ``/log``, ``/launch``, and
``/run/:ns/:name`` (plus legacy aliases like ``/jobs`` → archive and
``/jobs/:ns/:name`` → run detail). Unknown routes fall through to the Home view
rather than rendering a dedicated Not-Found page.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import HomePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_nav_click_archive_from_home(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking the Archive rail button navigates the SPA to ``/archive``.

    Starts on Home, clicks ``rail-archive``, waits for the hash URL to
    reflect ``#/archive``, and asserts the archive page root is visible.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await page.get_by_test_id("rail-archive").click()
    await page.wait_for_url("**/#/archive")
    await expect(page.get_by_test_id("page-archive")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_opens_on_ctrl_k(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Pressing Ctrl+K opens the command palette modal.

    ``app.js`` binds both ``ctrlKey`` and ``metaKey`` + ``k`` to toggle the
    palette; headless Chromium on Linux dispatches ``Control+k``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_search_navigates_to_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Typing a job name + Enter in the palette navigates to that run detail.

    ``command-palette.js`` builds items from ``PAGES`` + ``jobs.value`` and
    fuzzy-matches on ``label`` or ``sub``. Typing the full job name narrows
    to one row; Enter fires ``navigate(/run/<ns>/<name>)``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()
    await page.get_by_test_id("command-palette-input").fill("aiperf-llama3-c128")
    await page.get_by_test_id("command-palette-input").press("Enter")
    await page.wait_for_url("**aiperf-llama3-c128**")
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_deep_link_loads_job_detail_directly(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Navigating directly to ``/#/jobs/<ns>/<name>`` renders the run workbench.

    Validates the hash router's ``matchRoute('/jobs/:ns/:name', ...)``
    legacy alias on a cold page load, not via in-app navigation.
    """
    url = f"{live_operator_app.base_url}/#/jobs/aiperf-bench/aiperf-llama3-c128"
    await page.goto(url)
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_unknown_route_falls_back_to_home(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Unknown hash routes resolve to Home (no dedicated Not-Found page).

    ``resolveView`` in ``app.js`` only explicitly matches the known routes
    and ``return { kind: 'home' }`` for anything else — the WORKBENCH
    rewrite intentionally dropped the old Not-Found stub in favour of
    always showing the operator something useful.
    """
    await page.goto(f"{live_operator_app.base_url}/#/does-not-exist")
    await expect(page.get_by_test_id("page-home")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_top_rail_buttons_present(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
) -> None:
    """The three primary top-rail actions (Launch / Archive / Compare) render."""
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    for testid in ("rail-launch", "rail-archive", "rail-compare"):
        await expect(page.get_by_test_id(testid)).to_be_visible()

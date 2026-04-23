# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for cross-page navigation (top-nav, command palette, deep links).

Covers top-nav clicks, Ctrl+K command-palette keybinding, fuzzy-filtered job
navigation via the palette, direct hash-URL deep links into job detail, and
the "Not Found" stub for unknown routes. The SPA uses hash routing
(``window.location.hash``); ``app.js`` routes ``/``, ``/jobs``,
``/jobs/:ns/:name``, ``/leaderboard``, ``/compare``, ``/history``, and falls
through to a Not Found stub for anything else.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import DashboardPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_nav_click_jobs_from_dashboard(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking the Jobs nav tab navigates the SPA to the jobs page.

    Starts on the dashboard, clicks ``nav-link-jobs``, waits for the hash
    URL to reflect ``/#/jobs``, and asserts the jobs page root is visible.
    """
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await page.get_by_test_id("nav-link-jobs").click()
    await page.wait_for_url("**/#/jobs")
    await expect(page.get_by_test_id("page-jobs")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_opens_on_ctrl_k(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Pressing Ctrl+K opens the command palette modal.

    ``app.js`` binds both ``ctrlKey`` and ``metaKey`` + ``k`` to toggle the
    palette; headless Chromium on Linux dispatches ``Control+k``.
    """
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_search_navigates_to_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Typing a job name + Enter in the palette navigates to that job detail.

    ``command-palette.js`` builds items from ``PAGES`` + ``jobs.value`` and
    fuzzy-matches on ``label`` or ``sub``. Typing the full job name narrows
    to one row; Enter fires ``navigate(/jobs/<ns>/<name>)``.
    """
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
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
    """Navigating directly to ``/#/jobs/<ns>/<name>`` renders job detail.

    Validates the hash router's ``matchRoute('/jobs/:ns/:name', ...)``
    parameter extraction on a cold page load, not via in-app navigation.
    """
    url = f"{live_operator_app.base_url}/#/jobs/aiperf-bench/aiperf-llama3-c128"
    await page.goto(url)
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_unknown_route_shows_not_found(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Unknown hash routes render the ``Not Found`` stub in ``app.js``."""
    await page.goto(f"{live_operator_app.base_url}/#/does-not-exist")
    await expect(page.get_by_text("Not Found")).to_be_visible()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI dashboard page.

Covers the five KPI cards rendered by ``dashboard.js`` (``Running``,
``Completed``, ``Peak Throughput``, ``Best TTFT``, ``Token Throughput``),
the empty-results-dir render, and the persistent top-nav / breadcrumb
chrome.

The Running / Completed KPIs are driven by the Kubernetes API (via
``fake_k8s_client``'s canned AIPerfJob list), not by the on-disk results
tree. The throughput / TTFT / token KPIs are driven by the leaderboard +
per-job summary fetches against ``results_dir``.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._builders import build_empty
from ._pages import DashboardPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_loads_with_seeded_data(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Dashboard root URL renders the ``page-dashboard`` root element."""
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_test_id("page-dashboard")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_shows_completed_kpi(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``kpi-completed`` shows the count of Succeeded+Completed k8s jobs.

    The ``fake_k8s_client`` golden jobs.json has 3 Succeeded jobs
    (``aiperf-llama3-c128``, ``aiperf-llama3-c256``, ``mistral-7b-run1``),
    so the Completed KPI renders literal "3".
    """
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("completed")
    await expect(kpi).to_be_visible()
    await expect(kpi).to_contain_text("3")


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_shows_running_kpi(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``kpi-running`` shows the count of Running/Initializing/Pending jobs.

    The golden jobs.json has exactly one Running job (``live-run``); no
    Initializing or Pending entries, so the Running KPI renders "1".
    """
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("running")
    await expect(kpi).to_be_visible()
    await expect(kpi).to_contain_text("1")


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_empty_state(
    live_operator_app, fake_k8s_client, page
) -> None:
    """Empty ``results_dir`` still renders the dashboard without crashing.

    Running / Completed KPIs remain populated from the k8s fixture; the
    three result-driven KPIs (Peak Throughput, Best TTFT, Token
    Throughput) render the em-dash placeholder ``---`` because no summary
    files exist on disk.
    """
    build_empty(live_operator_app.results_dir)
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_test_id("page-dashboard")).to_be_visible()
    # All five KPI cards still render (UI does not hide them when empty).
    for label in (
        "running",
        "completed",
        "peak-throughput",
        "best-ttft",
        "token-throughput",
    ):
        await expect(dash.kpi(label)).to_be_visible()
    # Result-driven KPIs show the em-dash fallback.
    await expect(dash.kpi("peak-throughput")).to_contain_text("---")


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_renders_top_nav_and_breadcrumb(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Top-nav and breadcrumb chrome are both visible on the dashboard."""
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_test_id("top-nav")).to_be_visible()
    await expect(page.get_by_test_id("breadcrumb")).to_be_visible()

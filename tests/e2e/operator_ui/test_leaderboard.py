# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the Analysis (former Leaderboard) view.

The WORKBENCH rewrite replaced the dedicated leaderboard page with a
combined Analysis view at ``/compare`` / ``/analysis`` / ``/leaderboard``
(all three aliases resolve to the same view). The Pareto chart + cluster
grouping display all jobs that have ``profile_export_aiperf.json``
artifacts; the former ranked-table and metric-selector widgets are gone.

These tests exercise what survived: the page mounts at
``/leaderboard`` (via alias), renders a chart canvas, and the legacy
alias continues to serve the Analysis view.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import AnalysisPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_leaderboard_alias_resolves_to_analysis(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The legacy ``/leaderboard`` URL renders the Analysis view.

    ``resolveView`` in ``app.js`` maps ``/leaderboard`` → kind ``analysis``
    so older bookmarks keep working. We cold-load the legacy URL and
    assert the new root ``page-leaderboard`` (Analysis view) is visible.
    """
    await page.goto(f"{live_operator_app.base_url}/#/leaderboard")
    await expect(page.get_by_test_id("page-leaderboard")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_renders_chart_canvas(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The Analysis view renders at least one ``<canvas>`` element.

    With the golden fixture's four results-carrying jobs, the Pareto
    chart has real data. We don't inspect the canvas contents (Playwright
    sees ``<canvas>`` as opaque); we just confirm it's present and
    visible after ``networkidle``.
    """
    analysis = AnalysisPage(page, live_operator_app.base_url)
    await analysis.goto()
    await page.wait_for_load_state("networkidle")
    await expect(page.locator("canvas").first).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_row_click_not_applicable(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The Analysis view has no ranked-table rows to click through.

    The former leaderboard row→detail click path is gone; the Pareto
    chart uses canvas-drawn points (not DOM rows) and per-cluster
    toggles control visibility rather than navigation. Skip until an
    equivalent click-through affordance is added upstream.
    """
    pytest.skip(
        "Analysis view replaced the leaderboard's ranked-table rows with "
        "canvas-drawn Pareto points; no DOM row click-through exists. "
        "See src/aiperf/operator/ui/views/analysis.js."
    )

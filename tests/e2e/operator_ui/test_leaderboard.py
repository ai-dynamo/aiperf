# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the Analysis (former Leaderboard) view.

The WORKBENCH rewrite replaced the dedicated leaderboard page with a
combined Analysis view at ``/compare`` / ``/analysis``. The Pareto
chart + cluster grouping display all jobs that have
``profile_export_aiperf.json`` artifacts; the former ranked-table and
metric-selector widgets are gone. The legacy ``/leaderboard`` alias was
retired in Task 11.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import AnalysisPage

pytestmark = [pytest.mark.e2e]


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

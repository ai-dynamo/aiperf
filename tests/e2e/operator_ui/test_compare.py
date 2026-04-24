# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E smoke tests for the Analysis (``/compare``) view.

The WORKBENCH rewrite merged Compare + Leaderboard into a single
``/compare`` (aka ``/analysis``) view implemented by
``src/aiperf/operator/ui/views/analysis.js``. The former compare page's
checkbox-selector + side-by-side Metric Comparison table was replaced
by a Pareto chart + cluster-group overlay; there is no ``compare-select``
element or ``Compare (N)`` button in the new UI, so the prior
selection-driven tests have no direct analog. These smoke tests cover
what survived: the page mounts, the Pareto chart canvas renders, and
axis-switch buttons are clickable.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import AnalysisPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_page_loads(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/compare`` renders the ``page-leaderboard`` root + a chart canvas.

    The Analysis view always renders a Pareto chart; with the golden
    fixture's 4 jobs that have ``profile_export_aiperf.json`` artifacts,
    the chart has real data. We don't inspect the canvas contents (the
    ``<canvas>`` is an opaque bitmap in Playwright's tree); we just
    assert it's present and visible after ``networkidle``.
    """
    analysis = AnalysisPage(page, live_operator_app.base_url)
    await analysis.goto()
    await page.wait_for_load_state("networkidle")
    await expect(page.locator("canvas").first).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_axis_switch_does_not_crash(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Switching Pareto axes is clickable and doesn't trip the error gate.

    Analysis exposes a row of axis-pair buttons (``r/s × p99``,
    ``r/s × ttft``, etc.) inside ``.v-analysis-axes``. Clicking through
    them should not throw any ``pageerror`` or ``console.error``; the
    ``page`` fixture's console gate validates that on teardown.
    """
    analysis = AnalysisPage(page, live_operator_app.base_url)
    await analysis.goto()
    await page.wait_for_load_state("networkidle")

    # The axis-pair row lives inside `.v-analysis-axes`; click through each
    # visible button in sequence. We don't assert specific axis labels — the
    # concrete set is owned by `analysis.js` and can evolve — just that all
    # of them are clickable without throwing.
    buttons = page.locator(".v-analysis-axes button")
    count = await buttons.count()
    assert count >= 1, f"expected >=1 axis-pair button, got {count}"
    for i in range(count):
        await buttons.nth(i).click()
    # Settle the 300ms Chart.js update animation before teardown.
    await page.wait_for_timeout(600)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI compare page.

Covers the side-by-side comparison rendered by ``compare.js`` against the
``/api/v1/analytics/compare`` endpoint. The selector is a scrollable
container (``data-testid="compare-select"``) of per-job ``<label>`` +
``<input type=checkbox>`` rows — not a ``<select multiple>``. A user must
check 2+ jobs and then click the "Compare" button to fire the backend call
and render the metric table + bar chart.

Note: unlike the leaderboard, the compare page has no metric selector. The
bar chart renders the full default metric set with one dataset per job.
Test 3 therefore asserts that the ``<canvas>`` renders after a successful
compare — the closest observable analog to "metric selector redraws
charts" for a page with no metric selector. See the test docstring for
why a selection-change redraw test is intentionally out of scope.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import ComparePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_compare_page_loads_with_selector(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The compare page renders its multi-select container on load.

    The ``compare-select`` element holds one ``<label>`` per job from the
    ``/api/v1/results`` response; with the golden fixture that's the four
    completed-or-failed jobs (``aiperf-llama3-c128``, ``aiperf-llama3-c256``,
    ``mistral-7b-run1``, ``failed-run``).
    """
    cp = ComparePage(page, live_operator_app.base_url)
    await cp.goto()
    selector = page.get_by_test_id("compare-select")
    await expect(selector).to_be_visible()
    # Fixture tree has 4 jobs with results; all should render as checkbox rows.
    checkboxes = selector.locator("input[type=checkbox]")
    await expect(checkboxes).to_have_count(4)


@pytest.mark.asyncio(loop_scope="session")
async def test_compare_two_jobs_renders_side_by_side(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Selecting two jobs and clicking Compare renders the side-by-side view.

    After the API call succeeds the page shows a Metric Comparison table
    with one column per selected job plus a canvas-backed bar chart. We
    assert both job_ids appear in the rendered output (as column headers /
    chart-legend chips) as evidence the compare pivot fired correctly.
    """
    cp = ComparePage(page, live_operator_app.base_url)
    await cp.goto()
    selector = page.get_by_test_id("compare-select")
    checkboxes = selector.locator("input[type=checkbox]")
    await expect(checkboxes).to_have_count(4)

    # Check the first two jobs. The list is ordered by _scan_job_dirs,
    # which sorts by namespace then job_id — so the first two are both in
    # the ``aiperf-bench`` namespace: c128 then c256. We don't pin the
    # exact identities here; we just assert that whatever two jobIds the
    # first two rows reference, both appear in the compare output.
    await checkboxes.nth(0).check()
    await checkboxes.nth(1).check()

    # Grab the two job_id text nodes from the selected rows for later
    # assertions. The selected row's label contains a mono-font div with
    # the bare job_id.
    label_0 = selector.locator("label").nth(0)
    label_1 = selector.locator("label").nth(1)
    job_id_0 = (await label_0.inner_text()).splitlines()[0].strip()
    job_id_1 = (await label_1.inner_text()).splitlines()[0].strip()
    assert job_id_0 and job_id_1 and job_id_0 != job_id_1

    # Click the "Compare (2)" button to trigger the /analytics/compare call.
    compare_btn = page.get_by_role("button", name="Compare (2)")
    await expect(compare_btn).to_be_enabled()
    await compare_btn.click()

    # Metric Comparison table should appear with a column per job.
    table = page.locator("table")
    await expect(table).to_be_visible()
    await expect(table).to_contain_text(job_id_0)
    await expect(table).to_contain_text(job_id_1)


@pytest.mark.asyncio(loop_scope="session")
async def test_compare_renders_bar_chart_canvas(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """After a successful compare, the bar chart canvas is rendered.

    The plan's "metric selector redraws charts" assertion was written
    around the leaderboard pattern; the compare page has NO metric
    selector — the bar chart always renders the default metric set with
    one dataset per job. The nearest observable analog here is that
    ``<canvas>`` is visible after the compare pivot succeeds.

    We deliberately do NOT re-trigger a compare with a changed selection
    in this test: Chart.js has a race between the 300ms bar-mount
    animation and ``chart.update()`` when the dataset cardinality
    changes, which surfaces as ``Cannot read properties of null (reading
    'x')`` and trips the ``conftest`` page-error gate. Verifying redraw
    would require either modifying ``chart-wrapper.js`` to disable
    animations on update (UI change, out of scope per task 12) or a
    page-error filter exception (overly permissive). Document the gap
    rather than paper over it.
    """
    cp = ComparePage(page, live_operator_app.base_url)
    await cp.goto()
    selector = page.get_by_test_id("compare-select")
    checkboxes = selector.locator("input[type=checkbox]")
    await expect(checkboxes).to_have_count(4)

    await checkboxes.nth(0).check()
    await checkboxes.nth(1).check()
    await page.get_by_role("button", name="Compare (2)").click()

    # Bar chart canvas appears inside the "Visual Comparison" card once
    # /analytics/compare returns and ``chartData`` is non-null.
    canvas = page.locator("canvas")
    await expect(canvas).to_be_visible()
    # Let the 300ms Chart.js mount animation settle before teardown so
    # its own destroy() path doesn't race a running animation frame.
    await page.wait_for_timeout(600)

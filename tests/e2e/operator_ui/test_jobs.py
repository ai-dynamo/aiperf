# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI jobs page.

Covers table rendering from the live k8s source (no merge with results —
``jobs.js`` polls ``api.listJobs()`` only), phase rendering for the
``Running`` job, the search input's namespace-matching behaviour (the
page has no dedicated namespace selector), column header sort toggling,
and row-click navigation to the job detail route.

The ``fake_k8s_client`` golden ``jobs.json`` seeds five AIPerfJobs:
``aiperf-bench/{aiperf-llama3-c128, aiperf-llama3-c256, live-run}`` and
``ml-lab/{mistral-7b-run1, failed-run}``. All five surface on ``/jobs``
regardless of phase; only ``live-run`` has no on-disk results, which
does not affect the jobs table (that merge happens on the dashboard KPIs
and leaderboard, not here).
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import JobsPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_table_renders_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/jobs`` renders the ``job-table`` tbody with one row per k8s job.

    ``jobs.js`` sources rows from ``api.listJobs()`` only — no merge with
    on-disk results — so all five golden AIPerfJobs (including the
    Running ``live-run`` and the Failed ``failed-run``) appear.
    """
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await expect(page.get_by_test_id("job-table")).to_be_visible()
    await expect(jobs_page.rows()).to_have_count(5)


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_row_shows_live_status_for_running_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The row for ``aiperf-bench/live-run`` shows the ``Running`` phase badge.

    ``job-table.js``'s ``renderPhase`` writes ``job.phase`` verbatim into a
    ``.phase-badge`` span, and the golden CR has ``status.phase: Running``.
    """
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    row = jobs_page.row("aiperf-bench", "live-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("Running")


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_search_by_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Typing a namespace into the search input narrows to matching rows only.

    ``jobs.js`` filters by ``j.name.includes(q) || j.namespace.includes(q)``.
    No golden job name contains ``ml-lab`` as a substring, so typing it
    leaves exactly the two ``ml-lab`` rows (``mistral-7b-run1`` and
    ``failed-run``).
    """
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await expect(jobs_page.rows()).to_have_count(5)
    await jobs_page.set_namespace_filter("ml-lab")
    ml_lab_rows = page.get_by_test_id("job-table").locator(
        "[data-testid^='job-row-ml-lab-']"
    )
    await expect(ml_lab_rows).to_have_count(2)
    await expect(jobs_page.rows()).to_have_count(2)


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_sort_by_column(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking the ``name`` column header sorts the table by job name.

    The default sort is ``age desc``; clicking ``col-header-name`` sets
    ``sortKey='name', sortDir=1`` (ascending). The five golden jobs sort
    alphabetically with ``aiperf-llama3-c128`` first.
    """
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await expect(jobs_page.rows()).to_have_count(5)
    await jobs_page.click_column_header("name")
    first_row = jobs_page.rows().first
    await expect(first_row).to_have_attribute(
        "data-testid", "job-row-aiperf-bench-aiperf-llama3-c128"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_row_click_navigates_to_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a row navigates to ``/jobs/<ns>/<name>`` and renders the detail page."""
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await jobs_page.row("aiperf-bench", "aiperf-llama3-c128").click()
    await page.wait_for_url("**/jobs/aiperf-bench/aiperf-llama3-c128")
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()

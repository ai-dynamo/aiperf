# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the Archive view (``/archive``).

Covers the row list sourced from ``api.listJobs()``, the running-phase
label on the live CR, the shared search input's substring match on
name / namespace / model, the sort dropdown, and row-click navigation
into the single-run workbench. Per ``src/aiperf/operator/ui/views/archive.js``,
Archive is bucketed by namespace with one row per AIPerfJob (no mergewith on-disk results — the merge happens only on the run detail page).

The ``fake_k8s_client`` golden ``jobs.json`` seeds five AIPerfJobs:
``aiperf-bench/{aiperf-llama3-c128, aiperf-llama3-c256, live-run}`` and
``ml-lab/{mistral-7b-run1, failed-run}``. All five surface on
``/archive`` regardless of phase because the default bucket is "ALL".
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import ArchivePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_lists_all_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/archive`` renders one ``arch-row-*`` per AIPerfJob in the golden data.

    Archive sources rows from ``api.listJobs()`` only — no merge with
    on-disk results — so all five golden CRs (including the Running
    ``live-run`` and the Failed ``failed-run``) appear.
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(5)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_row_shows_running_phase_for_live_run(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The row for ``aiperf-bench/live-run`` shows the ``RUNNING`` phase text.

    Each row renders ``job.phase.toUpperCase()`` in the phase cell, and
    the golden CR has ``status.phase: Running``.
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    row = archive.row("aiperf-bench", "live-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("RUNNING")


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_search_by_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Typing a namespace into the search input narrows to matching rows only.

    The search input substring-matches name, namespace, or model. No
    golden job name contains ``ml-lab`` as a substring, so typing it
    leaves exactly the two ``ml-lab`` rows (``mistral-7b-run1`` and
    ``failed-run``).
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(5)
    await archive.search("ml-lab")
    ml_lab_rows = page.locator("[data-testid^='arch-row-ml-lab-']")
    await expect(ml_lab_rows).to_have_count(2)
    await expect(archive.rows()).to_have_count(2)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_sort_selector_applies(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Choosing a different sort option updates the ``archive-sort`` value.

    The sort dropdown (``archive-sort``) carries five options: ``newest``,
    ``oldest``, ``rps``, ``p99``, ``dur``. The UI's controlled-select
    writes the chosen ``key`` back into the ``value=${sort}`` binding, so
    asserting the new value is visible is the deterministic signal that
    the dropdown is functional. Row ordering after the change depends on
    per-job start / throughput / latency data that's not uniformly
    populated on the golden CRs, so we deliberately don't pin an order.
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(5)
    sort_select = page.get_by_test_id("archive-sort")
    await expect(sort_select).to_have_value("newest")
    await archive.set_sort("oldest")
    await expect(sort_select).to_have_value("oldest")
    # Rows should still be rendered — only the ordering changed.
    await expect(archive.rows()).to_have_count(5)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_row_click_navigates_to_run_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a row navigates to ``/run/<ns>/<name>`` and renders the detail page."""
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await archive.row("aiperf-bench", "aiperf-llama3-c128").click()
    await page.wait_for_url("**/run/aiperf-bench/aiperf-llama3-c128")
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()

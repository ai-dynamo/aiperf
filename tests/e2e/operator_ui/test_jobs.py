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

from ._builders import build_empty
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


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_bucket_tabs_filter_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """LIVE / PASSED / FAULT tabs narrow the row list to matching phases.

    Golden fixture has 5 CRs: 3 Succeeded (PASSED), 1 Failed (FAULT),
    1 Running (LIVE). ALL is the default and shows all 5. Scope the tab
    lookup to ``.v-archive-tabs`` so it doesn't collide with the
    same-labelled LIVE/PASSED/FAULT tabs in the bottom ``log-strip``
    component.
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(5)
    tablist = page.locator(".v-archive-tabs")

    await tablist.get_by_role("tab", name="LIVE").click()
    await expect(archive.rows()).to_have_count(1)
    await expect(archive.row("aiperf-bench", "live-run")).to_be_visible()

    await tablist.get_by_role("tab", name="PASSED").click()
    await expect(archive.rows()).to_have_count(3)

    await tablist.get_by_role("tab", name="FAULT").click()
    await expect(archive.rows()).to_have_count(1)
    await expect(archive.row("ml-lab", "failed-run")).to_be_visible()

    await tablist.get_by_role("tab", name="ALL").click()
    await expect(archive.rows()).to_have_count(5)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_search_regex_chars_are_literal(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Regex-like search strings are treated as literal substrings.

    The search input in ``archive.js`` uses ``String.includes``, not a
    regex — so ``.*`` should match zero rows (no golden job name
    contains those two characters). A regression that accidentally
    passed the input to ``new RegExp`` would match everything. Zero
    matches is the specific observable that disambiguates the two
    behaviours.
    """
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(5)
    await archive.search(".*")
    await expect(archive.rows()).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_empty_bucket_shows_empty_state(
    live_operator_app, fake_k8s_client, page
) -> None:
    """A bucket with zero matches renders the ``arch-empty`` section.

    Archive's row list is the union of live CRs AND on-disk archived
    runs from the results tree — wipe both sources down to one Running
    CR, switch to the PASSED tab, and assert the empty-state section
    surfaces. If either source leaked rows in, the assertion catches it.
    """
    fake_k8s_client.jobs_raw = [
        j for j in fake_k8s_client.jobs_raw if j["metadata"]["name"] == "live-run"
    ]
    build_empty(live_operator_app.results_dir)
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    tablist = page.locator(".v-archive-tabs")
    await tablist.get_by_role("tab", name="PASSED").click()
    await expect(archive.rows()).to_have_count(0)
    await expect(page.get_by_test_id("arch-empty")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_entirely_empty_shows_empty_state(
    live_operator_app, fake_k8s_client, page
) -> None:
    """Archive with zero CRs and zero on-disk runs renders ``arch-empty``.

    Archive sources the union of live CRs + on-disk PVC directories.
    Zero rows requires clearing both — the ``jobs_raw`` list on the
    fake and the results-dir tree via ``build_empty``.
    """
    fake_k8s_client.jobs_raw = []
    build_empty(live_operator_app.results_dir)
    archive = ArchivePage(page, live_operator_app.base_url)
    await archive.goto()
    await expect(archive.rows()).to_have_count(0)
    await expect(page.get_by_test_id("arch-empty")).to_be_visible()

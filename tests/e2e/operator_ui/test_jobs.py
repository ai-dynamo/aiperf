# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the namespace-scoped Archive view (``/ns/:ns/archive``).

Covers the row list sourced from ``api.listJobs()`` filtered to the
current namespace, the running-phase label on the live CR, the search
input's substring match on name / model within the namespace, the
sort dropdown, and row-click navigation into the single-run workbench.

The ``fake_k8s_client`` golden ``jobs.json`` seeds five AIPerfJobs:
``aiperf-bench/{aiperf-llama3-c128, aiperf-llama3-c256, live-run}`` and
``ml-lab/{mistral-7b-run1, failed-run}``. Three rows surface on
``/ns/aiperf-bench/archive``; two on ``/ns/ml-lab/archive``.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._builders import build_empty
from ._pages import ArchivePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_lists_namespace_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/ns/aiperf-bench/archive`` renders one row per AIPerfJob in that namespace."""
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(3)

    archive_ml = ArchivePage(page, live_operator_app.base_url, namespace="ml-lab")
    await archive_ml.goto()
    await expect(archive_ml.rows()).to_have_count(2)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_row_shows_running_phase_for_live_run(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The row for ``aiperf-bench/live-run`` shows the ``Running`` phase chip."""
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    row = archive.row("live-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("Running")


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_search_filters_within_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Search narrows rows within the current namespace by name/model substring.

    ``aiperf-bench`` has three jobs; searching ``c128`` matches only one.
    """
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(3)
    await archive.search().fill("c128")
    await expect(archive.rows()).to_have_count(1)
    await expect(archive.row("aiperf-llama3-c128")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_sort_selector_applies(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Choosing a different sort option updates the ``arch-sort`` value.

    The sort dropdown carries five options: ``newest``, ``oldest``,
    ``rps``, ``p99``, ``dur``. The controlled-select writes the chosen
    ``key`` back into the ``value=${sort}`` binding, so asserting the new
    value is visible is the deterministic signal that the dropdown is
    functional.
    """
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(3)
    sort_select = page.get_by_test_id("arch-sort")
    await expect(sort_select).to_have_value("newest")
    await archive.set_sort("oldest")
    await expect(sort_select).to_have_value("oldest")
    await expect(archive.rows()).to_have_count(3)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_row_click_navigates_to_run_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a row navigates to ``/run/<ns>/<name>`` and renders the detail page."""
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await archive.row("aiperf-llama3-c128").click()
    await page.wait_for_url("**/run/aiperf-bench/aiperf-llama3-c128")
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_bucket_tabs_filter_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Live / Passed / Failed tabs narrow the row list to matching phases.

    The ``aiperf-bench`` namespace has 3 jobs: 2 Succeeded (Passed),
    1 Running (Live). Scope the tab lookup to ``.arch-tabs`` so it
    doesn't collide with same-labelled tabs in the log-strip.
    """
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(3)
    tablist = page.locator(".arch-tabs")

    await tablist.get_by_role("tab", name="Live").click()
    await expect(archive.rows()).to_have_count(1)
    await expect(archive.row("live-run")).to_be_visible()

    await tablist.get_by_role("tab", name="Passed").click()
    await expect(archive.rows()).to_have_count(2)

    await tablist.get_by_role("tab", name="All").click()
    await expect(archive.rows()).to_have_count(3)

    archive_ml = ArchivePage(page, live_operator_app.base_url, namespace="ml-lab")
    await archive_ml.goto()
    tablist_ml = page.locator(".arch-tabs")
    await tablist_ml.get_by_role("tab", name="Failed").click()
    await expect(archive_ml.rows()).to_have_count(1)
    await expect(archive_ml.row("failed-run")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_search_regex_chars_are_literal(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Regex-like search strings are treated as literal substrings.

    The search input uses ``String.includes``, not a regex — so ``.*``
    should match zero rows (no golden job name contains those two
    characters). A regression that passed the input to ``new RegExp``
    would match everything. Zero matches is the specific observable
    that disambiguates the two behaviours.
    """
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(3)
    await archive.search().fill(".*")
    await expect(archive.rows()).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_empty_bucket_shows_empty_state(
    live_operator_app, fake_k8s_client, page
) -> None:
    """A bucket with zero matches in the namespace renders ``arch-empty``."""
    fake_k8s_client.jobs_raw = [
        j for j in fake_k8s_client.jobs_raw if j["metadata"]["name"] == "live-run"
    ]
    build_empty(live_operator_app.results_dir)
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    tablist = page.locator(".arch-tabs")
    await tablist.get_by_role("tab", name="Passed").click()
    await expect(archive.rows()).to_have_count(0)
    await expect(page.get_by_test_id("arch-empty")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_entirely_empty_shows_empty_state(
    live_operator_app, fake_k8s_client, page
) -> None:
    """Archive with zero CRs in the namespace renders ``arch-empty``."""
    fake_k8s_client.jobs_raw = []
    build_empty(live_operator_app.results_dir)
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows()).to_have_count(0)
    await expect(page.get_by_test_id("arch-empty")).to_be_visible()

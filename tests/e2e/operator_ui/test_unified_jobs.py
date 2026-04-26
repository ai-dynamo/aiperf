# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the unified cluster+PVC jobs view.

Archived jobs (PVC dir but no CR) should appear on the Archive page and
be navigable into a run-detail view; they also participate in the
Home-view Passed count alongside live-Succeeded CRs.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import ArchivePage, HomePage, JobDetailPage
from .conftest import write_archived_job

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_page_lists_archived_job(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived PVC-only job appears as an ``arch-row-*`` on ``/ns/<ns>/archive``."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    archive = ArchivePage(page, live_operator_app.base_url, namespace="ml-lab")
    await archive.goto()
    row = archive.row("ghost-run")
    await expect(row).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_run_detail_archived_loads(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived run detail mounts ``page-job-detail`` without the Cancel button.

    The new run view has no dedicated "archived" banner test-id; the
    WORKBENCH rewrite dropped the explicit banner copy in favour of
    hiding lifecycle actions (``run-cancel``) whose CR is gone. The
    observable contract for archived-only runs is therefore: the detail
    view mounts, and the Cancel button is absent.
    """
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "ghost-run")
    await detail.goto()
    await expect(page.get_by_test_id("run-cancel")).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_run_detail_archived_shows_results(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived run detail lists the on-disk result artifacts.

    The rewritten run view no longer renders per-KPI tiles; archived
    results surface via the ``run-results`` section (one row per file in
    the PVC directory). Assert the section is visible and the seeded
    ``profile_export_aiperf.json`` is listed.
    """
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "ghost-run")
    await detail.goto()
    results = page.get_by_test_id("run-results")
    await expect(results).to_be_visible()
    await expect(results).to_contain_text("profile_export_aiperf.json")


@pytest.mark.asyncio(loop_scope="session")
async def test_home_passed_count_includes_archived(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Home's Passed cell counts archived-Succeeded jobs alongside live ones.

    Seeded golden CRs: 3 Succeeded, 1 Failed, 1 Running. Adding one
    archived-only run should take the Passed total to 4.
    """
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    passed = home.summary_cell("Passed")
    await expect(passed).to_contain_text("4")

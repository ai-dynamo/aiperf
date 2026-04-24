# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the unified cluster+PVC jobs view.

Archived jobs (PVC dir but no CR) should appear on the Jobs page, render a
job-detail page with a banner and no Cancel/Pods, and be tallied in the
dashboard Completed count.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import DashboardPage, JobDetailPage, JobsPage
from .conftest import write_archived_job

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_page_lists_archived_job(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived PVC-only job appears in the Jobs table with an 'archived' badge."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    row = jobs_page.row("ml-lab", "ghost-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("archived")


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_archived_shows_banner_hides_cancel(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived job detail renders the banner and omits Cancel and Pods."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "ghost-run")
    await detail.goto()
    await expect(
        page.get_by_text("Kubernetes resource has been deleted", exact=False)
    ).to_be_visible()
    await expect(page.get_by_test_id("job-detail-cancel")).to_have_count(0)
    await expect(page.get_by_test_id("job-detail-pods")).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_archived_shows_metrics(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Archived detail page surfaces KPIs synthesized from profile_export_aiperf.json."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "ghost-run")
    await detail.goto()
    throughput = page.get_by_test_id("kpi-throughput")
    await expect(throughput).to_be_visible()
    await expect(throughput).to_contain_text("55.5")
    latency = page.get_by_test_id("kpi-latency-p99")
    await expect(latency).to_be_visible()
    await expect(latency).to_contain_text("421")


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_completed_count_includes_archived(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
):
    """Dashboard Completed tile counts archived-Succeeded jobs."""
    # Seeded CR jobs: 3 Succeeded, 1 Failed, 1 Running. Add one archived-only.
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("completed")
    await expect(kpi).to_contain_text("4")  # 3 live + 1 archived

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E coverage for the per-namespace overview at ``/ns/:ns``."""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import NamespaceOverviewPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_overview_renders_only_jobs_in_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    p = NamespaceOverviewPage(
        page=page, base_url=live_operator_app.base_url, namespace="aiperf-bench"
    )
    await p.goto()
    rows = page.locator("[data-testid^='no-row-']")
    await expect(rows.first).to_be_visible()
    count = await rows.count()
    assert count > 0
    for i in range(count):
        testid = await rows.nth(i).get_attribute("data-testid")
        assert testid.startswith("no-row-aiperf-bench-"), (
            f"row {testid!r} not in target namespace"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_overview_empty_namespace_renders_launch_cta(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    p = NamespaceOverviewPage(
        page=page, base_url=live_operator_app.base_url, namespace="empty-ns"
    )
    await p.goto()
    await expect(page.get_by_test_id("no-empty")).to_be_visible()
    await expect(page.get_by_test_id("no-empty-launch-cta")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_overview_empty_launch_cta_navigates(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    p = NamespaceOverviewPage(
        page=page, base_url=live_operator_app.base_url, namespace="empty-ns"
    )
    await p.goto()
    await page.get_by_test_id("no-empty-launch-cta").click()
    await page.wait_for_url(lambda u: u.rstrip("/").endswith("#/ns/empty-ns/launch"))

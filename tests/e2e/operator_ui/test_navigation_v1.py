# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E navigation tests for the ui-v1 SPA at /v1/.

Currently asserts the conditional 'Plots ↗' top-nav entry that
gets gated by /api/v1/config/features.dashboard_enabled.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_plots_link_hidden_when_dashboard_disabled(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """With dashboard_enabled=false (default in tests), the Plots ↗ link is absent.

    ``seeded_results_dir`` and ``fake_k8s_client`` are pulled in even though
    this test only inspects the top nav: the SPA's default landing page
    immediately fetches ``/api/v1/jobs`` etc., and without the fake k8s
    fixture those return 503, which the page-fixture error gate fails on.
    """
    await page.goto(live_operator_app.base_url + "/v1/")
    await expect(page.get_by_test_id("top-nav")).to_be_visible()
    assert await page.locator('[data-testid="nav-link-plots"]').count() == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_plots_link_appears_when_features_route_returns_enabled(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Override /api/v1/config/features via Playwright route handler — the SPA fetches it at boot."""
    await page.route(
        "**/api/v1/config/features",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"dashboard_enabled": true}',
        ),
    )
    await page.goto(live_operator_app.base_url + "/v1/")
    await expect(page.get_by_test_id("top-nav")).to_be_visible()
    link = page.locator('[data-testid="nav-link-plots"]')
    await expect(link).to_have_count(1)
    assert await link.get_attribute("target") == "_blank"
    assert await link.get_attribute("href") == "/dashboard/"

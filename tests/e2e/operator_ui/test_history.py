# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the Log view (``/log`` / ``/history``).

The WORKBENCH rewrite replaced the former metric-chart history page with
a simple day-grouped run log (``src/aiperf/operator/ui/views/log.js``).
There's no chart canvas, no metric selector, and no card title to
inspect — just a per-day section with one clickable row per run. The
``/api/v1/analytics/history`` API endpoint still exists and is
exercised by the data-points test below.

Golden summary metrics (``profile_export_aiperf.json``):

================================ =================== ===============
Job                              request_throughput  request_latency
                                 avg                 avg
================================ =================== ===============
aiperf-bench/aiperf-llama3-c128  42.1                300.0 ms
aiperf-bench/aiperf-llama3-c256  78.4                410.0 ms
ml-lab/mistral-7b-run1           28.9                340.0 ms
ml-lab/failed-run                0.0                 0.0   ms
================================ =================== ===============
"""

from __future__ import annotations

import httpx
import pytest
from playwright.async_api import expect

from ._pages import LogPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_log_page_renders_run_entries(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The Log view lists one clickable ``.v-log-row`` per live CR.

    ``log.js`` sources entries from ``jobs.value`` (the CR poll), not from
    the results tree. With the golden fixture's five AIPerfJobs, the
    rendered list has five rows.
    """
    lp = LogPage(page, live_operator_app.base_url)
    await lp.goto()
    rows = page.locator(".v-log-row")
    await expect(rows.first).to_be_visible()
    await expect(rows).to_have_count(5)


@pytest.mark.asyncio(loop_scope="session")
async def test_log_row_click_navigates_to_run(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a Log row navigates to ``/run/<ns>/<name>``.

    Each row is a ``<button>`` whose ``onclick`` calls ``navigate('/run/…')``.
    Click any row and assert the detail view mounts.
    """
    lp = LogPage(page, live_operator_app.base_url)
    await lp.goto()
    await page.locator(".v-log-row").first.click()
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_history_api_returns_data_points(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/api/v1/analytics/history`` returns an entry per completed job.

    The UI no longer consumes this endpoint (the new Log view lists CRs,
    not chart data), but it still exists as part of the backend API. The
    golden fixture has four job directories with ``profile_export_aiperf.json``
    (c128, c256, mistral-7b-run1, failed-run). The history SQL filters
    only ``IS NOT NULL`` — failed-run's 0.0 is a non-null value — so the
    response contains all four, well above the >=3 threshold.

    We hit the API directly via ``httpx.AsyncClient(trust_env=False)`` so
    any ambient ``HTTP(S)_PROXY`` env vars don't route localhost through
    a proxy.
    """
    async with httpx.AsyncClient(trust_env=False) as client:
        resp = await client.get(
            f"{live_operator_app.base_url}/api/v1/analytics/history",
            params={"metric": "request_throughput", "stat": "avg"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["metric"] == "request_throughput"
    entries = body["entries"]
    assert len(entries) >= 3, (
        f"expected >=3 history entries for request_throughput in the golden "
        f"fixture, got {len(entries)}: {entries}"
    )

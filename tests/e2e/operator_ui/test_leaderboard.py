# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI leaderboard page.

Covers the ranked table rendered by ``leaderboard.js`` against the
``/api/v1/analytics/leaderboard`` endpoint. The page reads from the
on-disk results tree (via ``ResultsDB``); ``fake_k8s_client`` is
requested for fixture parity with the other pages but does not drive
any leaderboard behaviour.

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

The leaderboard SQL filters only ``IS NOT NULL``, not ``> 0``, so
``failed-run`` is included and wins the ascending-latency sort (its
0.0 ms is the minimum). Tests pin this actual behaviour rather than
assuming failed-run is filtered out.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import LeaderboardPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_leaderboard_ranks_by_throughput(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The default (``request_throughput`` desc) ranking puts c256 on top.

    ``aiperf-llama3-c256`` has ``request_throughput.avg = 78.4``, the
    highest in the golden results tree, so it renders as rank 1 in the
    ``All Results`` table.
    """
    lb = LeaderboardPage(page, live_operator_app.base_url)
    await lb.goto()
    # Wait for the ranked table to render (no spinner left).
    table = page.get_by_test_id("page-leaderboard").locator("table")
    await expect(table).to_be_visible()
    first_row = table.locator("tbody tr").first
    await expect(first_row).to_contain_text("aiperf-llama3-c256")


@pytest.mark.asyncio(loop_scope="session")
async def test_leaderboard_metric_selector_changes_order(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Switching metric to ``request_latency`` re-sorts the ranked table.

    The leaderboard API defaults to ``order=desc``, so under
    ``request_latency`` the ranking is c256=410, mistral=340, c128=300,
    failed=0. c256 happens to be rank 1 under both metrics (highest
    throughput AND highest latency), but rank 2 flips: under throughput
    it is c128 (42.1 > 28.9), and under latency it is mistral (340 > 300).
    This test pins that rank-2 flip as the observable effect of the
    metric change.
    """
    lb = LeaderboardPage(page, live_operator_app.base_url)
    await lb.goto()
    table = page.get_by_test_id("page-leaderboard").locator("table")
    await expect(table).to_be_visible()
    # Precondition: under default throughput metric, c256 is rank 1.
    first_row = table.locator("tbody tr").first
    await expect(first_row).to_contain_text("aiperf-llama3-c256")

    # The ``metric-selector`` test-id wraps two <select> elements; the
    # first is the metric picker (id=metric-select), the second is the
    # stat picker. Use the explicit id to avoid ambiguity.
    await page.locator("#metric-select").select_option("request_latency")

    # After the switch, c256 stays rank 1 (highest latency AND highest
    # throughput), but rank 2 flips: under throughput it was c128
    # (42.1 > 28.9); under latency it is mistral (340 ms > 300 ms). Pin
    # the flip as evidence that the selector re-queried the API.
    await expect(table.locator("tbody tr").nth(1)).to_contain_text("mistral-7b-run1")


@pytest.mark.asyncio(loop_scope="session")
async def test_leaderboard_row_click_opens_job_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a ranked row navigates to ``/jobs/<ns>/<name>``.

    ``leaderboard.js`` renders each ``<tr>`` with only a ``key`` prop —
    no ``onclick``, no anchor, no ``data-testid`` — so rows are not
    interactive. Skip until the UI adds row navigation (the plan's
    "click-through to detail" acceptance is pending a UI change).
    """
    pytest.skip(
        "leaderboard.js rows have no click handler or anchor; row-click "
        "navigation is not yet implemented (see leaderboard.js L198-224)."
    )

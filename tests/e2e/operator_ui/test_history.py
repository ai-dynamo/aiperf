# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI history page.

Covers the metric-over-time line chart rendered by ``history.js`` against
the ``/api/v1/analytics/history`` endpoint. The page embeds the shared
``MetricSelector`` component (id=``metric-select``), so switching metric
fires a re-fetch and the chart re-renders. The page reads from the on-disk
results tree; ``fake_k8s_client`` is requested for fixture parity with the
other pages but does not drive history behaviour.

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

The history endpoint filters ``IS NOT NULL`` — not ``> 0`` — so
``failed-run`` is included and the API returns all four entries for
``request_throughput`` / ``avg``.
"""

from __future__ import annotations

import httpx
import pytest
from playwright.async_api import expect

from ._pages import HistoryPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_history_chart_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The history page renders a canvas-backed line chart on load.

    With the golden fixture (4 jobs with summary metrics) the default
    ``request_throughput`` / ``avg`` query returns a non-empty entry list,
    so the ``<canvas>`` inside the "over time" card is rendered.
    """
    hp = HistoryPage(page, live_operator_app.base_url)
    await hp.goto()
    canvas = page.locator("canvas")
    await expect(canvas).to_be_visible()
    # Let the Chart.js mount animation settle before teardown so its
    # destroy() path doesn't race a running animation frame (same pattern
    # as test_compare.py::test_compare_renders_bar_chart_canvas).
    await page.wait_for_timeout(600)


@pytest.mark.asyncio(loop_scope="session")
async def test_history_metric_selector_switches_series(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Switching the metric selector re-renders the chart.

    ``history.js`` embeds the shared ``MetricSelector`` (id=``metric-select``)
    and its ``useEffect`` refetches whenever ``selected.metric`` changes.
    We assert both that the canvas stays visible after the switch and that
    the chart card title reflects the new metric name, as evidence the
    re-render actually fired.
    """
    hp = HistoryPage(page, live_operator_app.base_url)
    await hp.goto()

    canvas = page.locator("canvas")
    await expect(canvas).to_be_visible()

    # The MetricSelector wraps two <select>s; the first (id=metric-select)
    # is the metric picker. Switch to time_to_first_token and confirm the
    # chart card title updates — proving the useEffect re-ran.
    await page.locator("#metric-select").select_option("time_to_first_token")

    card_title = page.locator(".card-title").first
    await expect(card_title).to_contain_text("time_to_first_token")
    await expect(canvas).to_be_visible()

    # Let the Chart.js update animation settle before teardown.
    await page.wait_for_timeout(600)


@pytest.mark.asyncio(loop_scope="session")
async def test_history_shows_data_points(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/api/v1/analytics/history`` returns an entry per completed job.

    The golden fixture has four job directories with ``profile_export_aiperf.json``
    (c128, c256, mistral-7b-run1, failed-run). The history SQL filters only
    ``IS NOT NULL`` — failed-run's 0.0 is a non-null value — so the response
    contains all four, well above the >=3 threshold this test pins.

    We hit the API directly via ``httpx.AsyncClient(trust_env=False)`` so
    any ambient ``HTTP(S)_PROXY`` env vars don't route localhost through
    a proxy (same defensive pattern as ``test_smoke.py``).
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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E robustness tests: every route loads without server errors or JS crashes.

Visits every SPA route (``/``, ``/jobs``, ``/leaderboard``, ``/compare``,
``/history``) and then rapidly cycles through them without waiting, as a
smoke check that nothing throws on mount/unmount. The ``page`` fixture's
console-error gate fails the test if any ``pageerror`` or ``console.error``
fires during the run.
"""

from __future__ import annotations

import pytest
from playwright.async_api import Response, expect

pytestmark = [pytest.mark.e2e]

ROUTES = ["/", "/jobs", "/leaderboard", "/compare", "/history"]


def _hash_url(base_url: str, route: str) -> str:
    """Build an SPA hash URL: ``/`` -> bare index; others -> ``/#<route>``."""
    return f"{base_url}/" if route in ("", "/") else f"{base_url}/#{route}"


@pytest.mark.asyncio(loop_scope="session")
async def test_all_routes_return_ok_fetches(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Every SPA route loads without the server returning any >=500 response.

    Attaches a ``response`` listener that captures any same-origin response
    whose status is >=500, visits each route, and waits for ``networkidle``
    so the page's initial data fetches (leaderboard, results, version,
    jobs) complete before moving on. Chart-heavy pages (/compare, /history)
    need an extra settle for Chart.js mount animations.
    """
    bad_responses: list[tuple[str, int]] = []

    def _on_response(resp: Response) -> None:
        if resp.url.startswith(live_operator_app.base_url) and resp.status >= 500:
            bad_responses.append((resp.url, resp.status))

    page.on("response", _on_response)

    try:
        for route in ROUTES:
            await page.goto(_hash_url(live_operator_app.base_url, route))
            await page.wait_for_load_state("networkidle")
            # Let Chart.js mount animations settle on chart-heavy pages so
            # their teardown path doesn't race a running animation frame.
            if route in ("/compare", "/history", "/leaderboard"):
                await page.wait_for_timeout(600)
    finally:
        page.remove_listener("response", _on_response)

    assert bad_responses == [], (
        "Server returned >=500 responses for same-origin requests:\n"
        + "\n".join(f"  - {status} {url}" for url, status in bad_responses)
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_rapid_route_changes_do_not_crash(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Rapidly cycling through routes without waiting doesn't crash the SPA.

    Iterates ``ROUTES`` three times back-to-back without waiting between
    navigations. The ``page`` fixture's console-error gate validates that
    no ``pageerror`` or ``console.error`` fires across the rapid cycle. A
    final ``top-nav`` visibility assertion confirms the app is still
    rendered after the churn.
    """
    # Start from a settled page so the first cycle isn't racing initial load.
    await page.goto(live_operator_app.base_url + "/")
    await expect(page.get_by_test_id("page-dashboard")).to_be_visible()

    for _ in range(3):
        for route in ROUTES:
            await page.goto(_hash_url(live_operator_app.base_url, route))

    # Give in-flight fetches and Chart.js mount animations a chance to
    # settle before teardown so no late error trips the console gate.
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(600)
    await expect(page.get_by_test_id("top-nav")).to_be_visible()

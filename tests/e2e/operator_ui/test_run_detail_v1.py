# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke e2e for the rewritten ui-v1 job-detail page (mounted at ``/v1``).

The production UI at ``/`` is covered by ``test_run_detail.py``. The
ui-v1 SPA at ``/v1`` is hash-routed and exposes the new component
layout introduced by the Phase-3 rewrite — KpiRail (18 tiles), three
canonical strips (phase / records / pods), LiveChartsPanel, and
DiagnosticsPanel. These tests pin the contract for that layout.

Routing note: ui-v1 uses ``window.location.hash`` for navigation
(``src/aiperf/operator/ui-v1/lib/router.js``), so the address always
takes the form ``/v1/#/jobs/<ns>/<name>``. The DiagnosticsPanel reads
its initial active tab from ``new URL(window.location.href)
.searchParams.get('diag')``, which only sees query params **before**
the hash — hence the deep-link form ``/v1/?diag=conditions#/jobs/...``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import expect

pytestmark = [pytest.mark.e2e]


def _seed_pods_for_job(fake_k8s_client: Any, job_id: str, count: int) -> None:
    """Attach ``count`` Running pods to ``job_id`` on the fake k8s fixture.

    The golden ``pods.json`` only attaches pods to ``live-run``; the rewrite
    smoke uses ``aiperf-llama3-c128``, so seed a small fleet here. The plan
    requires enriching the fixture rather than weakening the ``>= 1``
    heatmap assertion.
    """
    pods = [
        {
            "metadata": {
                "name": f"{job_id}-pod-{i}",
                "namespace": "aiperf-bench",
                "labels": {"aiperf.nvidia.com/job-id": job_id},
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 0}],
            },
        }
        for i in range(count)
    ]
    fake_k8s_client.set_pods(job_id, pods)


def _seed_status_phases(fake_k8s_client: Any, namespace: str, name: str) -> None:
    """Stamp ``status.phases`` and ``status.results.latency_histogram`` on the CR.

    ``pages/job-detail.js`` builds ``phaseStripData`` from ``status.phases``
    and shells the strips behind ``phaseStripData.length > 0`` /
    ``recordTotal > 0``. The golden CR is bare (``status = {phase, conditions}``
    only), so without this seed both ``strip-phase`` and ``strip-records``
    are skipped at render time.

    The histogram piggybacks here because ``LiveChartsPanel`` returns
    ``null`` when neither a live throughput series nor a histogram is
    available — and a Succeeded run with no live feed has neither unless
    we plant the histogram on the CR's ``status.results``.
    """
    for raw in fake_k8s_client.jobs_raw:
        meta = raw["metadata"]
        if meta["name"] == name and meta["namespace"] == namespace:
            status = raw.setdefault("status", {})
            status["phases"] = {
                "benchmark": {
                    "requestsCompleted": 64,
                    "requestsTotal": 64,
                    "requestsProgressPercent": 100,
                    "recordsSuccess": 60,
                    "recordsError": 4,
                    "recordsProgressPercent": 100,
                    "sendingComplete": True,
                }
            }
            status["results"] = {
                "latency_histogram": [
                    {"le": 0.1, "count": 12},
                    {"le": 0.3, "count": 47},
                    {"le": 1.0, "count": 5},
                ]
            }
            return
    raise AssertionError(f"no CR {namespace}/{name} in fake client")


def _drop_persisted_epochs(results_dir: Path, namespace: str, name: str) -> None:
    """Remove ``<name>/<epoch>/`` subdirs + ``latest.txt`` so ``/epochs`` is empty.

    ``pages/job-detail.js`` auto-redirects ``/v1/#/jobs/<ns>/<name>`` to
    ``/runs/<latestPersistedEpoch>`` whenever a persisted epoch is found
    on disk. After that redirect, ``find_any_job(epoch=...)`` switches to
    the archived branch (``_get_job_impl`` returns ``pods=[]`` and a
    synthesized status without ``runEpoch``), which substitutes a "pods/
    events not retained" callout for the strip-pods bar. Wiping the
    persisted-epoch trail keeps the page on the no-epoch URL, where
    ``viewingCurrentRun`` is ``true`` against the live CR and the strip
    mounts with the seeded pod roster.
    """
    job_root = results_dir / namespace / name
    if not job_root.is_dir():
        return
    for child in job_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_job_detail_renders_kpi_rail_and_strips(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The new ui-v1 job-detail page renders the KPI rail and three strips.

    Pins the headline layout contract: an 18-tile KpiRail plus the three
    canonical strips (phase / records / pods), with both the live-charts
    and diagnostics panels mounted in the two-column row below.
    """
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    _drop_persisted_epochs(seeded_results_dir, namespace, name)
    _seed_status_phases(fake_k8s_client, namespace, name)
    _seed_pods_for_job(fake_k8s_client, name, count=2)
    await page.goto(f"{live_operator_app.base_url}/v1/#/jobs/{namespace}/{name}")
    # KPI rail and 18 tiles.
    await expect(page.get_by_test_id("kpi-rail")).to_be_visible(timeout=10_000)
    tiles = await page.locator("[data-tile-id]").count()
    assert tiles == 18, f"expected 18 KPI tiles, got {tiles}"
    # Three strips.
    await expect(page.get_by_test_id("strip-phase")).to_be_visible()
    await expect(page.get_by_test_id("strip-records")).to_be_visible()
    await expect(page.get_by_test_id("strip-pods")).to_be_visible()
    # Live charts panel + diagnostics panel.
    await expect(page.get_by_test_id("panel-live-charts")).to_be_visible()
    await expect(page.get_by_test_id("panel-diagnostics")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_diagnostics_tab_deep_link(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Visiting with ``?diag=conditions`` activates the Conditions tab.

    ``DiagnosticsPanel.readTabFromUrl`` reads ``searchParams.get('diag')``
    via ``new URL(window.location.href)``, which only sees query params
    placed **before** the hash. A URL like
    ``/v1/?diag=conditions#/jobs/<ns>/<name>`` therefore activates the
    Conditions tab on first mount.
    """
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    _drop_persisted_epochs(seeded_results_dir, namespace, name)
    _seed_status_phases(fake_k8s_client, namespace, name)
    await page.goto(
        f"{live_operator_app.base_url}/v1/?diag=conditions#/jobs/{namespace}/{name}"
    )
    await expect(page.get_by_test_id("panel-diagnostics")).to_be_visible(timeout=10_000)
    conditions_tab = page.locator('[data-tab-id="conditions"]')
    await expect(conditions_tab).to_have_class(
        re.compile(r"\bdiag-tab--active\b"), timeout=10_000
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_pods_strip_renders_heatmap(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The pods strip renders a tile per pod via ``PodHeatmap``.

    Seeds two pods on ``aiperf-llama3-c128`` (the golden ``pods.json``
    only attaches pods to ``live-run``) and asserts the heatmap renders
    one ``.pod-heatmap-tile`` per pod.
    """
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    _drop_persisted_epochs(seeded_results_dir, namespace, name)
    _seed_status_phases(fake_k8s_client, namespace, name)
    _seed_pods_for_job(fake_k8s_client, name, count=2)
    await page.goto(f"{live_operator_app.base_url}/v1/#/jobs/{namespace}/{name}")
    await expect(page.get_by_test_id("strip-pods")).to_be_visible(timeout=10_000)
    heatmap = page.locator(".pod-heatmap-tile")
    assert await heatmap.count() >= 1


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_pods_strip_click_switches_diagnostics_tab(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking PodsStrip writes ``?diag=pods`` AND switches the active tab.

    ``PodsStrip.onExpand`` writes ``?diag=pods`` to the URL and dispatches
    a synthetic ``popstate`` event. ``DiagnosticsPanel`` must listen for
    that event and switch its active tab — without the listener the panel
    only reads the URL on first mount and the click is a no-op.
    """
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    _drop_persisted_epochs(seeded_results_dir, namespace, name)
    _seed_status_phases(fake_k8s_client, namespace, name)
    _seed_pods_for_job(fake_k8s_client, name, count=2)
    await page.goto(f"{live_operator_app.base_url}/v1/#/jobs/{namespace}/{name}")
    await expect(page.get_by_test_id("strip-pods")).to_be_visible(timeout=10_000)
    # Click the pods strip bar.
    await page.locator('[data-testid="strip-pods"] .strip-bar').click()
    # The pods tab should now be active.
    pods_tab = page.locator('[data-tab-id="pods"]')
    await expect(pods_tab).to_have_class(
        re.compile(r"\bdiag-tab--active\b"), timeout=5_000
    )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.skip(reason="Run manually (comment the skip) to refresh docs/media/images/api-dashboard-v2.png")
async def test_capture_dashboard_screenshot_v1(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Screenshot capture for ``docs/media/images/api-dashboard-v2.png``.

    Skipped by default to keep the e2e suite hermetic. Remove the
    ``@pytest.mark.skip`` decorator locally (or invoke with ``-k`` and
    ``--runxfail``-style overrides) to overwrite the canonical
    dashboard screenshot in place. Per ``feedback_dashboard_screenshots_in_docs.md``,
    the file path is canonical and must not be variant-dated.
    """
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    _drop_persisted_epochs(seeded_results_dir, namespace, name)
    _seed_status_phases(fake_k8s_client, namespace, name)
    _seed_pods_for_job(fake_k8s_client, name, count=8)
    await page.set_viewport_size({"width": 1400, "height": 900})
    await page.goto(f"{live_operator_app.base_url}/v1/#/jobs/{namespace}/{name}")
    await page.wait_for_selector('[data-testid="kpi-rail"]')
    # Brief settle for chart canvases + WS reconnect attempts.
    await page.wait_for_timeout(500)
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "docs" / "media" / "images" / "api-dashboard-v2.png"
    await page.screenshot(path=str(out), full_page=False)

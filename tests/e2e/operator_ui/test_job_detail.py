# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the single-run workbench (``/run/:ns/:name``).

The WORKBENCH rewrite replaced the former Flight-Deck detail page
(KpiCard + HeroStrip + GpuTelemetryCard + ReliabilityTile + Sparkline
components) with ``src/aiperf/operator/ui/views/run.js``, a single dense
view with these sections:

- ``run-identity`` — headline + model + namespace
- ``run-conditions`` — condition badges
- ``run-pods`` — pod roster
- ``run-gpu`` — GPU metrics (hidden when absent)
- ``run-sparks`` — live sparklines
- ``run-results`` — downloadable bundle
- ``run-logs`` — live log pane
- ``run-cancel`` / ``run-relaunch`` — lifecycle actions

The per-KPI tiles, hero strip, SLO chip, reliability tile, and
``metric-val``/``metric-sub``/``sparkline svg`` DOM contracts the
previous tests asserted on no longer exist. Tests here cover what
survived: section visibility, conditions, pods, cancel path, GPU
hidden, and the sparkline section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import expect

from ._pages import JobDetailPage

pytestmark = [pytest.mark.e2e]


def _set_job_summary(
    fake_k8s_client: Any, namespace: str, name: str, summary: dict[str, Any]
) -> None:
    """Mutate the canned CR so ``status.summary`` drives KPI/chart values.

    ``_raw_status`` in the fake k8s client returns ``raw.get("status", {})``
    verbatim, which the UI reads via ``api.getJob`` -> ``status.summary``.
    """
    for raw in fake_k8s_client.jobs_raw:
        meta = raw["metadata"]
        if meta["name"] == name and meta["namespace"] == namespace:
            raw.setdefault("status", {})["summary"] = summary
            return
    raise AssertionError(f"no CR {namespace}/{name} in fake client")


def _ensure_empty_results_dir(base: Path, namespace: str, name: str) -> None:
    """Create a minimal job results dir so auxiliary endpoints return 200.

    The run view fires auxiliary fetches to ``/api/v1/results/<ns>/<name>``
    and ``/api/v1/config/<ns>/<name>`` on mount. For CRs that are live in k8s
    but have no on-disk results (e.g. a Running job), both routes 404 — which
    surfaces as ``console.error`` entries that the ``page`` fixture treats as
    teardown failures.

    We create the directory so the results route returns an empty file list,
    and drop a minimal ``job_spec.json`` so the config route returns 200 via
    its ``spec_file.exists()`` branch.
    """
    import orjson

    d = base / namespace / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "job_spec.json").write_bytes(orjson.dumps({"benchmark": {}}))


@pytest.mark.asyncio(loop_scope="session")
async def test_run_identity_section_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-identity`` renders with model + endpoint for a completed run.

    The identity section shows MODEL / ENDPOINT / COMPARABLE RUN rows;
    the run name itself lives in the top-rail breadcrumb, not inside this
    section. Assert on the model label that is definitively inside the
    identity region — ``llama3-8b`` is seeded into the golden CR's
    ``spec.model`` and is rendered verbatim under the MODEL header.
    """
    _set_job_summary(
        fake_k8s_client,
        "aiperf-bench",
        "aiperf-llama3-c128",
        {"throughput_rps": 42.1, "ttft_avg_ms": 150.0, "latency_p99_ms": 300.0},
    )
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    identity = page.get_by_test_id("run-identity")
    await expect(identity).to_be_visible()
    await expect(identity).to_contain_text("MODEL")
    await expect(identity).to_contain_text("llama3-8b")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_conditions_section_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-conditions`` renders condition entries from ``status.conditions``.

    The golden CR carries ``{type: Ready, status: True}`` so the Conditions
    section mounts and the ``Ready`` type label is visible.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    conditions = page.get_by_test_id("run-conditions")
    await expect(conditions).to_be_visible()
    await expect(conditions).to_contain_text("Ready")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_sparks_section_renders_for_running_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-sparks`` renders the live-metric sparkline row for a Running CR.

    ``run.js`` only mounts ``run-sparks`` when ``bucket === 'live'`` (see
    L1229), so we target the ``live-run`` CR which the golden fixture
    carries in phase ``Running``.
    """
    _ensure_empty_results_dir(seeded_results_dir, "aiperf-bench", "live-run")
    _set_job_summary(
        fake_k8s_client,
        "aiperf-bench",
        "live-run",
        {"throughput_rps": 42.1},
    )
    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    sparks = page.get_by_test_id("run-sparks")
    await expect(sparks).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_run_pods_section_renders_for_running_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-pods`` renders when the CR has associated pods.

    The fake k8s fixture attaches ``live-run-controller-0`` to the
    ``live-run`` CR via the ``aiperf.nvidia.com/job-id=live-run`` label
    selector; ``run.js`` reads ``data.job.pods`` and renders the Pods
    section.
    """
    _ensure_empty_results_dir(seeded_results_dir, "aiperf-bench", "live-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    pods = page.get_by_test_id("run-pods")
    await expect(pods).to_be_visible()
    await expect(pods).to_contain_text("live-run-controller-0")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_cancel_button_calls_api(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking ``run-cancel`` issues POST /jobs/.../cancel → fake recorder.

    ``handleCancel`` pops a native ``confirm()`` dialog; Playwright's
    ``page.on("dialog", ...)`` auto-accepts so the ``api.cancelJob`` path
    runs, which hits the monkeypatched ``_cancel`` in the fake client and
    appends ``(namespace, name)`` to ``fake_k8s_client.cancelled``.

    The button only renders while the run is live, so we target the
    ``live-run`` CR (golden phase = ``Running``).
    """
    _ensure_empty_results_dir(seeded_results_dir, "aiperf-bench", "live-run")
    page.on("dialog", lambda d: d.accept())
    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    await detail.cancel()
    # ``api.cancelJob`` is async (POST → monkeypatched ``_cancel`` → append);
    # poll the recorder rather than relying on a fixed sleep.
    deadline = 5.0
    step = 0.1
    waited = 0.0
    while waited < deadline:
        if ("aiperf-bench", "live-run") in fake_k8s_client.cancelled:
            break
        await page.wait_for_timeout(int(step * 1000))
        waited += step
    assert ("aiperf-bench", "live-run") in fake_k8s_client.cancelled


@pytest.mark.asyncio(loop_scope="session")
async def test_run_gpu_section_hidden_without_data(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
) -> None:
    """``run-gpu`` stays hidden when no GPU metrics are present.

    ``run.js``'s GPU section only mounts when ``gpuMetrics`` (from
    ``status.metrics`` / ``status.liveMetrics``) is non-empty. The golden
    CR carries neither, so the section should not be in the DOM.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_test_id("run-gpu")).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_run_results_section_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-results`` renders for a Succeeded CR with on-disk artifacts.

    The seeded results tree for ``aiperf-llama3-c128`` carries
    ``profile_export_aiperf.json`` etc., so the results section lists
    downloadable rows via the ``run-results-row-*`` test-ids. We just
    confirm the section root is visible; the exact file list is owned by
    the backend.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_test_id("run-results")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_run_events_section_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-events`` renders unconditionally (even as an empty state).

    ``run.js`` always mounts the events section — LOADING during fetch,
    empty-copy once the empty response comes back from the stubbed
    events endpoint. Confirm it's visible on the detail page mount.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_test_id("run-events")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_run_fault_callout_renders_for_failed_cr(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-fault-callout`` appears for a Failed CR with a False condition.

    The callout only renders when ``bucket === 'fault'`` AND there is a
    ``status=False`` condition (or a Failed pod). The golden Failed CR
    (``ml-lab/failed-run``) doesn't carry a False condition by default,
    so seed one into the fake before navigating.
    """
    for raw in fake_k8s_client.jobs_raw:
        if raw["metadata"]["name"] == "failed-run":
            status = raw.setdefault("status", {})
            status["phase"] = "Failed"
            status["conditions"] = [
                {
                    "type": "Ready",
                    "status": "False",
                    "message": "benchmark worker exited with non-zero code",
                }
            ]
            break
    else:  # pragma: no cover - defensive; the golden CR always exists
        raise AssertionError("failed-run CR missing from fake fixture")
    _ensure_empty_results_dir(seeded_results_dir, "ml-lab", "failed-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "failed-run")
    await detail.goto()
    callout = page.get_by_test_id("run-fault-callout")
    await expect(callout).to_be_visible()
    await expect(callout).to_contain_text("non-zero code")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_relaunch_button_submits_new_manifest(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``run-relaunch`` stores a prefill and navigates to /launch.

    The relaunch button only renders when the run's config carries a
    non-empty ``spec``; the seeded ``aiperf-llama3-c128`` run has
    ``job_spec.json`` written via the golden fixture. Clicking it writes
    the current YAML to ``sessionStorage['aiperf.launch.prefill']`` and
    navigates to ``/launch``, where the prefill notice surfaces via
    ``launch-prefill-notice``.
    """
    import orjson

    # Ensure the config endpoint returns a non-empty spec so RelaunchButton renders.
    spec_path = (
        seeded_results_dir / "aiperf-bench" / "aiperf-llama3-c128" / "job_spec.json"
    )
    spec_path.write_bytes(
        orjson.dumps({"benchmark": {"models": [{"name": "llama3-8b"}]}})
    )

    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    relaunch = page.get_by_test_id("run-relaunch")
    await expect(relaunch).to_be_visible()
    await relaunch.click()
    await page.wait_for_url("**/#/launch")
    # Prefill notice should surface the source run name.
    notice = page.get_by_test_id("launch-prefill-notice")
    await expect(notice).to_be_visible()
    await expect(notice).to_contain_text("aiperf-llama3-c128")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_view_breadcrumb_shows_run_name(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Top-rail ``breadcrumb`` on a run view contains the run name.

    ``breadcrumbFor`` in ``top-rail.js`` builds a two-segment trail for
    run views: the parent link + the run name in the emphasised slot.
    The name is the only test-identifiable part, so assert on that.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    crumbs = page.get_by_test_id("breadcrumb")
    await expect(crumbs).to_be_visible()
    await expect(crumbs).to_contain_text("aiperf-llama3-c128")


@pytest.mark.asyncio(loop_scope="session")
async def test_run_view_for_nonexistent_run_does_not_crash(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Hash-navigating to a run that doesn't exist renders without JS errors.

    ``_get_job_impl`` returns 404 when neither a CR nor a PVC dir is
    found; the UI's top-level fetch has ``.catch(() => null)`` so the
    view must still mount the ``page-job-detail`` shell. The actual
    404 is surfaced as a browser-level ``console.error``, which the
    page fixture now tolerates for 4xx (see conftest ``_on_console``).
    We pin both: the shell renders, and no ``pageerror`` fires.
    """
    await page.goto(f"{live_operator_app.base_url}/#/run/no-such-ns/no-such-name")
    # The view mounts even with no backing data — the shell is
    # lifecycle-independent of the data fetch.
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()

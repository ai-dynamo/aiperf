# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI job-detail page.

Covers: KPI metric rendering from ``status.summary``, condition badges
from ``status.conditions``, the live throughput chart ``<canvas>`` element,
the pods panel for a Running job, and the cancel-button handler path
(``api.cancelJob`` -> fake k8s client ``cancelled`` recorder).

KPI metric values come from ``api.getJob(ns, name)`` -> ``status.summary``
(or ``status.liveSummary``), not from the on-disk ``profile_export_aiperf.json``
tree. Tests therefore prime ``fake_k8s_client.jobs_raw[*].status.summary``
to drive the UI values — the committed golden CR fixtures only carry
``phase`` + ``conditions`` and are otherwise minimal.
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
    """Create a minimal job results dir so the auxiliary endpoints return 200.

    The job-detail page fires auxiliary fetches to ``/api/v1/results/<ns>/<name>``
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
async def test_job_detail_renders_metrics(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``kpi-throughput`` renders ``status.summary.throughput_rps`` via ``fmtThroughput``.

    ``fmtThroughput(42.1)`` -> ``fmtNumber(42.1, 1)`` -> ``"42.1"``.
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
    kpi = page.get_by_test_id("kpi-throughput")
    await expect(kpi).to_be_visible()
    await expect(kpi).to_contain_text("42.1")


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_renders_conditions(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Condition badges render from ``status.conditions``.

    The ``Conditions`` component maps known condition types via
    ``CONDITION_LABELS`` and falls back to the raw ``cond.type`` string.
    The golden CR has ``{type: Ready, status: True}`` (no entry in the
    label map), so the badge label is the raw ``"Ready"``.
    """
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    conditions = page.locator('[aria-label="Conditions"]')
    await expect(conditions).to_be_visible()
    await expect(conditions).to_contain_text("Ready")


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_renders_chart_canvas(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The live throughput chart creates a ``<canvas>`` once ``chartData`` is set.

    ``chartData`` starts ``null`` (rendering a "Waiting for data..." placeholder);
    the first poll tick that sees ``summary.throughput_rps`` populates points
    and the ``ChartWrapper`` mounts the canvas. Priming the summary ensures the
    canvas appears deterministically on the first poll.
    """
    _set_job_summary(
        fake_k8s_client,
        "aiperf-bench",
        "aiperf-llama3-c128",
        {"throughput_rps": 42.1},
    )
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.locator("canvas").first).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_shows_pods_for_running_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``job-detail-pods`` renders when the CR has associated pods.

    The fake k8s fixture attaches ``live-run-controller-0`` to the
    ``live-run`` CR via the ``aiperf.nvidia.com/job-id=live-run`` label
    selector; the detail page reads ``data.job.pods`` (populated by the
    jobs router's ``_pod_summary``) and renders the ``PodsBar`` card.
    """
    _ensure_empty_results_dir(seeded_results_dir, "aiperf-bench", "live-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    pods = page.get_by_test_id("job-detail-pods")
    await expect(pods).to_be_visible()
    # PodsBar truncates names > 20 chars in the visible label (keeping the
    # suffix behind a "..."), but the full name is preserved in the ``title``
    # attribute — assert against that rather than the truncated text.
    await expect(pods.locator('[title="live-run-controller-0"]').first).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_cancel_button_calls_api(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking ``job-detail-cancel`` issues POST /jobs/.../cancel -> fake recorder.

    ``handleCancel`` pops a native ``confirm()`` dialog ("Cancel job <name>?");
    Playwright's ``page.on("dialog", ...)`` auto-accepts so the ``api.cancelJob``
    path runs, which hits the monkeypatched ``_cancel`` and appends
    ``(namespace, name)`` to ``fake_k8s_client.cancelled``.

    The button only renders when ``phase.toLowerCase() === 'running'``, so we
    target the ``live-run`` CR (golden phase = ``Running``).
    """
    _ensure_empty_results_dir(seeded_results_dir, "aiperf-bench", "live-run")
    page.on("dialog", lambda d: d.accept())
    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    await detail.cancel()
    # ``api.cancelJob`` is async (POST -> monkeypatched ``_cancel`` -> append);
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

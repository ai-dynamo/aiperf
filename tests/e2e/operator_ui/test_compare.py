# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E smoke tests for the Analysis (``/compare``) view.

The WORKBENCH rewrite merged Compare + Leaderboard into a single
``/compare`` (aka ``/analysis``) view implemented by
``src/aiperf/operator/ui/views/analysis.js``. The former compare page's
checkbox-selector + side-by-side Metric Comparison table was replaced
by a Pareto chart + cluster-group overlay; there is no ``compare-select``
element or ``Compare (N)`` button in the new UI, so the prior
selection-driven tests have no direct analog. These smoke tests cover
what survived: the page mounts, the Pareto chart canvas renders, and
axis-switch buttons are clickable.

The run-diff view at ``/compare/<ns>/<name>/<a>/<b>`` — added alongside
the run-history dropdown — is a separate feature covered at the bottom
of the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.async_api import expect

from ._pages import AnalysisPage
from .test_job_detail import _seed_two_epoch_runs

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_page_loads(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/compare`` renders the ``page-leaderboard`` root + a chart canvas.

    The Analysis view always renders a Pareto chart; with the golden
    fixture's 4 jobs that have ``profile_export_aiperf.json`` artifacts,
    the chart has real data. We don't inspect the canvas contents (the
    ``<canvas>`` is an opaque bitmap in Playwright's tree); we just
    assert it's present and visible after ``networkidle``.
    """
    analysis = AnalysisPage(page, live_operator_app.base_url)
    await analysis.goto()
    await page.wait_for_load_state("networkidle")
    await expect(page.locator("canvas").first).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_analysis_axis_switch_does_not_crash(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Switching Pareto axes is clickable and doesn't trip the error gate.

    Analysis exposes a row of axis-pair buttons (``r/s × p99``,
    ``r/s × ttft``, etc.) inside ``.v-analysis-axes``. Clicking through
    them should not throw any ``pageerror`` or ``console.error``; the
    ``page`` fixture's console gate validates that on teardown.
    """
    analysis = AnalysisPage(page, live_operator_app.base_url)
    await analysis.goto()
    await page.wait_for_load_state("networkidle")

    # The axis-pair row lives inside `.v-analysis-axes`; click through each
    # visible button in sequence. We don't assert specific axis labels — the
    # concrete set is owned by `analysis.js` and can evolve — just that all
    # of them are clickable without throwing.
    buttons = page.locator(".v-analysis-axes button")
    count = await buttons.count()
    assert count >= 1, f"expected >=1 axis-pair button, got {count}"
    for i in range(count):
        await buttons.nth(i).click()
    # Settle the 300ms Chart.js update animation before teardown.
    await page.wait_for_timeout(600)


# ─────────────────── run-diff compare view ────────────────────


def _write_summary(path: Path, throughput_avg: float, latency_p99: float) -> None:
    """Fabricate a minimal ``profile_export_aiperf.json`` with the nine metrics
    the compare view reads. Only ``throughput`` + ``latency_p99`` vary across
    the two runs — enough to force a non-zero Δ in the rendered table.
    """
    import orjson

    path.write_bytes(
        orjson.dumps(
            {
                "request_throughput": {"unit": "requests/sec", "avg": throughput_avg},
                "request_latency": {
                    "unit": "ms",
                    "avg": 300.0,
                    "p50": 285.0,
                    "p99": latency_p99,
                },
                "time_to_first_token": {
                    "unit": "ms",
                    "avg": 150.0,
                    "p50": 142.5,
                    "p99": 195.0,
                },
                "inter_token_latency": {"unit": "ms", "avg": 25.0},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 5000.0},
            }
        )
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_run_diff_compare_view_renders_two_column_table(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``#/compare/<ns>/<name>/<a>/<b>`` mounts a 2-data-column table.

    Seeds two epoch-keyed run dirs for ``aiperf-bench/aiperf-llama3-c128`` via
    the same helper the run-history dropdown test uses, then overwrites each
    run's ``profile_export_aiperf.json`` with different throughput + p99 so
    the Δ column has something to colour. Navigates directly to the compare
    URL (the run-diff view must not require the run view to be traversed
    first) and asserts the table root + both data-column headers render.
    """
    older, latest = _seed_two_epoch_runs(
        seeded_results_dir, "aiperf-bench", "aiperf-llama3-c128"
    )
    run_root = seeded_results_dir / "aiperf-bench" / "aiperf-llama3-c128"
    _write_summary(run_root / older / "profile_export_aiperf.json", 40.0, 380.0)
    _write_summary(run_root / latest / "profile_export_aiperf.json", 50.0, 320.0)

    await page.goto(
        f"{live_operator_app.base_url}/#/compare/aiperf-bench/aiperf-llama3-c128/{older}/{latest}"
    )
    await expect(page.get_by_test_id("page-compare")).to_be_visible()
    table = page.get_by_test_id("compare-table")
    await expect(table).to_be_visible()
    # Two data columns (Run A, Run B) plus the Metric + Δ labels.
    await expect(page.get_by_test_id("compare-col-a")).to_be_visible()
    await expect(page.get_by_test_id("compare-col-b")).to_be_visible()
    # One row per metric (9 metrics — see METRICS in compare.js).
    await expect(table.locator("tbody tr")).to_have_count(9)

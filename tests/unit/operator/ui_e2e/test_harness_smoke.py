# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests proving the operator-UI Playwright harness works.

These cover the happy paths so harness regressions show up here, separate
from the adversarial files that intentionally break things.
"""

from __future__ import annotations


def test_smoke_dashboard_renders(harness):
    """Bare dashboard route loads without 'Operator API unreachable'."""
    page = harness.goto_dashboard()
    body = page.locator("body").inner_text()
    assert "AIPerf Operator" in body, body[:500]
    harness.assert_no_unreachable_banner()


def test_smoke_jobs_list_renders_empty(harness):
    """Jobs list with no seeded data renders an empty state, not an error."""
    page = harness.goto_jobs_list()
    body = page.locator("body").inner_text()
    assert "Jobs" in body
    harness.assert_no_unreachable_banner()


def test_smoke_archived_run_renders(harness):
    """Seeded successful run shows up at /jobs/<ns>/<name>/runs/<epoch>."""
    from tests.unit.operator.ui_e2e.conftest import good_summary

    harness.seed_run(
        name="happy-job",
        epoch="1714069323",
        summary=good_summary(throughput_rps=42.5, model="m"),
        is_latest=True,
    )
    page = harness.goto_job_detail(harness.ns, "happy-job", epoch="1714069323")
    harness.assert_no_unreachable_banner()
    body = page.locator("body").inner_text()
    assert "happy-job" in body
    assert "1714069323" in body


def test_smoke_no_summary_epoch_renders_archived_stub(harness):
    """Regression: epoch dir with no summary must NOT 404 the run-detail page."""
    harness.seed_run(name="partial-job", epoch="1714069323", summary=None)
    page = harness.goto_job_detail(harness.ns, "partial-job", epoch="1714069323")
    harness.assert_no_unreachable_banner()
    body = page.locator("body").inner_text()
    assert "partial-job" in body
    assert "Unknown" in body  # archived stub phase


def test_smoke_api_get_works(harness):
    """``harness.api_get`` round-trips against the live uvicorn."""
    status, body = harness.api_get(
        f"/api/v1/jobs/{harness.ns}/does-not-exist?epoch=1714069323"
    )
    assert status == 404, (status, body[:200])

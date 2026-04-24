# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the operator web UI Home view.

Covers the ``/`` route — the WORKBENCH shell's Home view
(``src/aiperf/operator/ui/views/home.js``). Home replaces the older
dashboard: instead of five KPI cards + a hero strip, it renders one
compact five-cell summary strip (Running / Passed / Fault / NS /
optional GPUs) plus a dense, per-namespace list of runs. The Running /
Passed / Fault counts are driven by the Kubernetes API (via
``fake_k8s_client``'s canned AIPerfJob list).
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._builders import build_empty
from ._pages import HomePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_home_loads_with_seeded_data(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Home root URL renders the ``page-home`` root element."""
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await expect(page.get_by_test_id("page-home")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_home_summary_shows_passed_count(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The Passed cell counts Succeeded/Completed jobs from the k8s fixture.

    The ``fake_k8s_client`` golden ``jobs.json`` has 3 Succeeded CRs
    (``aiperf-llama3-c128``, ``aiperf-llama3-c256``, ``mistral-7b-run1``),
    so the Passed cell renders literal ``3``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    passed = home.summary_cell("Passed")
    await expect(passed).to_be_visible()
    await expect(passed).to_contain_text("3")


@pytest.mark.asyncio(loop_scope="session")
async def test_home_summary_shows_running_count(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The Running cell counts Running/Initializing/Pending jobs.

    The golden ``jobs.json`` has exactly one Running CR (``live-run``);
    no Initializing or Pending entries, so the Running cell renders ``1``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    running = home.summary_cell("Running")
    await expect(running).to_be_visible()
    await expect(running).to_contain_text("1")


@pytest.mark.asyncio(loop_scope="session")
async def test_home_empty_results_dir_still_renders(
    live_operator_app, fake_k8s_client, page
) -> None:
    """Empty ``results_dir`` still renders Home without crashing.

    Running / Passed / Fault cells remain populated from the k8s fixture
    (which is independent of the on-disk results tree).
    """
    build_empty(live_operator_app.results_dir)
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await expect(page.get_by_test_id("page-home")).to_be_visible()
    await expect(home.summary()).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_home_renders_top_nav_and_breadcrumb(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Top-nav chrome is visible on Home.

    The breadcrumb nav is only rendered when the view actually carries a
    crumb trail (Run / Launch / Archive); Home intentionally omits it
    because ``/`` is the root, so we only assert ``top-nav`` here.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    await expect(page.get_by_test_id("top-nav")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_home_lists_live_run_row(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Home renders a clickable row for the single running CR in the golden data.

    The row's test-id is derived from namespace + name
    (``hm-row-aiperf-bench-live-run``) and carries the run name text.
    """
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    row = home.row("aiperf-bench", "live-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("live-run")


@pytest.mark.asyncio(loop_scope="session")
async def test_home_empty_state_shows_launch_cta(
    live_operator_app,
    fake_k8s_client,
    page,
) -> None:
    """With no CRs and an empty results tree, Home shows the launch CTA.

    After the 2s scanning animation settles, Home renders the
    ``home-launch-cta`` button (deep-linked at ``/launch``) in place of
    the run list.
    """
    fake_k8s_client.jobs_raw = []
    build_empty(live_operator_app.results_dir)
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    cta = page.get_by_test_id("home-launch-cta")
    await expect(cta).to_be_visible(timeout=7000)


@pytest.mark.asyncio(loop_scope="session")
async def test_home_groups_by_namespace_with_live_first(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Namespace with a Running run renders before namespaces without one.

    ``groupByNamespace`` in ``home.js`` orders namespace groups by:
    (1) any live run first, (2) any fault next, (3) alphabetical
    fallback. The golden fixture has ``aiperf-bench`` (carrying the
    Running ``live-run``) and ``ml-lab`` (only Succeeded + Failed).
    Alphabetical order would put ``aiperf-bench`` first anyway, so
    flip one CR — rename ``aiperf-bench``'s namespace to ``z-lab`` at
    the API level so alphabetical + live-priority disagree, and the
    live-priority rule is the only thing that can explain the observed
    ordering.
    """
    for raw in fake_k8s_client.jobs_raw:
        if raw["metadata"]["namespace"] == "aiperf-bench":
            raw["metadata"]["namespace"] = "z-lab"
    home = HomePage(page, live_operator_app.base_url)
    await home.goto()
    # Wait for the list variant of Home to mount (not the empty/scanning
    # state) — ``hm-summary`` is only rendered when there are runs.
    await expect(home.summary()).to_be_visible()
    # The first rendered row should be in ``z-lab`` (live namespace) even
    # though ``ml-lab`` comes first alphabetically.
    first_row = home.rows().first
    testid = await first_row.get_attribute("data-testid")
    assert testid is not None and testid.startswith("hm-row-z-lab-"), (
        f"expected first row to be in the live namespace (z-lab), got {testid!r}"
    )

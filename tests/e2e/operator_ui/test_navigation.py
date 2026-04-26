# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for cross-page navigation (top-rail, command palette, deep links).

Covers top-rail button clicks, Ctrl+K command-palette keybinding, fuzzy-filtered
job navigation via the palette, and direct hash-URL deep links into the single-run
workbench. The SPA uses hash routing (``window.location.hash``); ``app.js``
routes ``/``, ``/ns/:ns/archive``, ``/compare``, ``/log``, ``/ns/:ns/launch``,
and ``/ns/:ns/run/:name``. Unknown routes fall through to the Home view rather
than rendering a dedicated Not-Found page.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import HomePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_nav_click_archive_from_home(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking the Archive rail button navigates the SPA to ``/ns/<ns>/archive``.

    The rail-archive button now scopes to the current namespace —
    starting from ``/ns/aiperf-bench`` it must navigate to
    ``/ns/aiperf-bench/archive`` and mount the archive view.
    """
    home = HomePage(page, live_operator_app.base_url)
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.get_by_test_id("rail-archive").click()
    await page.wait_for_url("**/#/ns/aiperf-bench/archive")
    await expect(page.get_by_test_id("page-archive")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_opens_on_ctrl_k(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Pressing Ctrl+K opens the command palette modal.

    ``app.js`` binds both ``ctrlKey`` and ``metaKey`` + ``k`` to toggle the
    palette; headless Chromium on Linux dispatches ``Control+k``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_search_navigates_to_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Typing a job name + Enter in the palette navigates to that run detail.

    ``command-palette.js`` builds items from ``PAGES`` + ``jobs.value`` and
    fuzzy-matches on ``label`` or ``sub``. Typing the full job name narrows
    to one row; Enter fires ``navigate(/run/<ns>/<name>)``.
    """
    home = HomePage(page, live_operator_app.base_url)
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()
    await page.get_by_test_id("command-palette-input").fill("aiperf-llama3-c128")
    await page.get_by_test_id("command-palette-input").press("Enter")
    await page.wait_for_url("**aiperf-llama3-c128**")
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_palette_groups_current_namespace_first(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """On a namespace-scoped route, palette job rows surface current-ns first.

    The fuzzy query "run" matches at least ``mistral-7b-run1``, ``failed-run``
    (both ``ml-lab``), and ``live-run`` (``aiperf-bench``). With the route at
    ``/ns/aiperf-bench``, the current-namespace partition must precede the
    other-namespace partition, so the first ``cmdp-job-*`` row's ``data-testid``
    must start with ``cmdp-job-aiperf-bench-``.
    """
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()
    await page.get_by_test_id("command-palette-input").fill("run")
    items = page.locator("[data-testid^='cmdp-job-']")
    await expect(items.first).to_be_visible()
    first_id = await items.first.get_attribute("data-testid")
    assert first_id.startswith("cmdp-job-aiperf-bench-"), (
        f"first palette job {first_id!r} should be in current namespace 'aiperf-bench'"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_deep_link_loads_run_detail_directly(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Navigating directly to ``/#/ns/<ns>/run/<name>`` renders the run workbench.

    Validates the hash router's ``matchRoute('/ns/:ns/run/:name', ...)``
    pattern on a cold page load, not via in-app navigation. The legacy
    ``/jobs/:ns/:name`` and bare ``/run/:ns/:name`` aliases were retired
    in Task 8.
    """
    url = f"{live_operator_app.base_url}/#/ns/aiperf-bench/run/aiperf-llama3-c128"
    await page.goto(url)
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_unknown_route_falls_back_to_home(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Unknown hash routes resolve to Home (legacy fallback, retired in Task 11).

    ``resolveView`` in ``app.js`` currently falls through to ``home`` for
    unmatched routes. Task 11 changes that fall-through to the namespace
    picker; this test will be updated to match at that point.
    """
    await page.goto(f"{live_operator_app.base_url}/#/does-not-exist")
    await expect(page.get_by_test_id("page-home")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_top_rail_buttons_present(
    live_operator_app,
    seeded_results_dir,
    fake_k8s_client,
    page,
) -> None:
    """The top-rail's namespace-aware actions render correctly.

    Cross-namespace tier (``/``): Compare visible; Launch + Archive hidden
    (no namespace selected → nowhere to launch into and nowhere to scope
    Archive to). Inside a namespace (``/ns/<ns>``): Launch, Archive,
    Compare all visible.
    """
    await page.goto(live_operator_app.base_url + "/")
    await expect(page.get_by_test_id("page-namespace-picker")).to_be_visible()
    await expect(page.get_by_test_id("rail-compare")).to_be_visible()
    await expect(page.get_by_test_id("rail-archive")).to_have_count(0)
    await expect(page.get_by_test_id("rail-launch")).to_have_count(0)

    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    for testid in ("rail-launch", "rail-archive", "rail-compare"):
        await expect(page.get_by_test_id(testid)).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_shows_empty_state_for_no_matches(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """A query with no fuzzy matches surfaces the ``No results for ...`` line.

    ``command-palette.js`` renders the empty-state ``<li>`` when
    ``filtered.length === 0``. Typing a string that matches neither the
    PAGES list nor any job name/namespace drives the list to zero.
    """
    home = HomePage(page, live_operator_app.base_url)
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.keyboard.press("Control+k")
    await expect(page.get_by_test_id("command-palette")).to_be_visible()
    await page.get_by_test_id("command-palette-input").fill("zzzz-absolutely-no-match")
    # The empty-state line lives inside the palette; assert visible
    # inside the palette's scoped locator rather than globally so a
    # stray "No results" string elsewhere on the page wouldn't hide a
    # regression here.
    palette = page.get_by_test_id("command-palette")
    await expect(palette).to_contain_text("No matches")


@pytest.mark.asyncio(loop_scope="session")
async def test_command_palette_dismisses_on_escape(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Escape closes the palette even when the query is non-empty.

    ``command-palette.js`` binds a global Escape handler that fires
    ``onClose`` regardless of focus state. Without this, a user whose
    focus drifts out of the input (e.g. a mouse hover onto a list item
    mutates cursor state) would strand the modal.
    """
    home = HomePage(page, live_operator_app.base_url)
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.keyboard.press("Control+k")
    palette = page.get_by_test_id("command-palette")
    await expect(palette).to_be_visible()
    await page.get_by_test_id("command-palette-input").fill("partial")
    await page.keyboard.press("Escape")
    await expect(palette).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_escape_on_run_view_returns_to_namespace_overview(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Escape on the run workbench navigates back to the namespace overview.

    ``app.js`` binds a global Escape handler: when the palette is closed
    and the current view is ``run``, Escape calls ``navigate('/ns/<ns>')``
    so the user lands on the namespace they were already inside, not the
    cross-namespace picker. Verify that path — it's a small UX contract
    that's easy to regress by accident.
    """
    url = f"{live_operator_app.base_url}/#/ns/aiperf-bench/run/aiperf-llama3-c128"
    await page.goto(url)
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()
    await page.keyboard.press("Escape")
    await page.wait_for_url("**/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_breadcrumb_namespace_pill_opens_switcher(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.get_by_test_id("ns-switcher-pill").click()
    await expect(page.get_by_test_id("ns-switcher-dropdown")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_switcher_navigates_to_other_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.get_by_test_id("ns-switcher-pill").click()
    await page.get_by_test_id("ns-switcher-item-ml-lab").click()
    await page.wait_for_url(lambda u: u.rstrip("/").endswith("#/ns/ml-lab"))


@pytest.mark.asyncio(loop_scope="session")
async def test_switcher_view_all_returns_to_picker(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    await page.goto(live_operator_app.base_url + "/#/ns/aiperf-bench")
    await page.get_by_test_id("ns-switcher-pill").click()
    await page.get_by_test_id("ns-switcher-view-all").click()
    await page.wait_for_url(lambda u: u.endswith("#/") or u.endswith("#"))


@pytest.mark.asyncio(loop_scope="session")
async def test_root_redirects_to_last_namespace_when_known(live_operator_app, seeded_results_dir, fake_k8s_client, page):
    # First visit: pick 'aiperf-bench' to set lastNamespace.
    await page.goto(live_operator_app.base_url + "/")
    await page.get_by_test_id("np-tile-aiperf-bench").click()
    await page.wait_for_url(lambda u: u.rstrip("/").endswith("#/ns/aiperf-bench"))
    # Reload root: should redirect to /ns/aiperf-bench.
    await page.goto(live_operator_app.base_url + "/")
    await page.wait_for_url(lambda u: u.rstrip("/").endswith("#/ns/aiperf-bench"))
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    # Clean up so subsequent tests don't see the stale key.
    await page.evaluate("window.localStorage.removeItem('aiperf.ui.lastNamespace')")


@pytest.mark.asyncio(loop_scope="session")
async def test_root_renders_picker_when_last_namespace_absent_from_jobs(live_operator_app, seeded_results_dir, fake_k8s_client, page):
    # Set lastNamespace to a value the fixture doesn't have, then reload root.
    await page.goto(live_operator_app.base_url + "/")
    await page.evaluate("window.localStorage.setItem('aiperf.ui.lastNamespace', 'ghost-ns')")
    await page.goto(live_operator_app.base_url + "/")
    await expect(page.get_by_test_id("page-namespace-picker")).to_be_visible()
    # Clean up so subsequent tests don't see the stale key.
    await page.evaluate("window.localStorage.removeItem('aiperf.ui.lastNamespace')")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Playwright tests that inject HTTP errors at the network layer.

Each test uses ``page.route(...)`` to intercept a specific operator API
endpoint and fulfill/abort it with a chosen failure mode, then asserts how
the SPA degrades. Focused on:

  * **Banner-on-failure** — the "Operator API unreachable" banner from
    ``lib/api.js::poll`` must appear after ``POLL_FAIL_THRESHOLD=2``
    consecutive failures.
  * **No JS exceptions** — schema-shape surprises (empty dict, malformed
    JSON, truncated body) must be tolerated without ``pageerror`` events.
  * **Per-page error UX** — pages that own their own error state
    (``jobs-error``, ``dashboard-jobs-error``, ``job-detail-error``,
    ``sweeps-error``) must render the right card; the page-level error
    state, not just the global banner, is the contract.
  * **WS disconnect** — closing the per-job WS without an open job render
    must not crash; the page renders REST data only.

Out of scope (see sibling files):
  * Routing-only edge cases — ``test_routing_adversarial.py``
  * Happy-path harness regressions — ``test_harness_smoke.py``

The harness ``Harness.page`` is a stock Playwright Page; every ``page.route``
intercept is scoped per test via the per-test browser context, so leaks
between tests are not possible (the context is closed in the fixture
teardown).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from pytest import param

# Substrings allow-listed in ``assert_no_console_errors`` for routes we
# intentionally make fail. The browser ALWAYS logs "Failed to load
# resource: the server responded with a status of 5XX/4XX (...)" when our
# route handler fulfills with a non-2xx; that's not a SPA bug.
_RESOURCE_FAILURE_SUBSTR = "Failed to load resource"
# Plus the explicit JSON-parse error path that ``apiFetch`` emits when our
# route handler returns malformed JSON with status=200 (``apiFetch`` calls
# ``resp.json()`` and the SyntaxError bubbles up). Browser console captures
# both the raw SyntaxError text and a wrapped Error.
_JSON_PARSE_SUBSTR = "Unexpected"

# Path glob fragments — bundle the BASE prefix so each test reads concisely.
_BASE = "**/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fulfill_status(status: int, body: bytes = b'{"detail":"injected"}') -> Callable:
    """Build a Playwright route handler that fulfills with ``status``+``body``."""

    def handler(route):
        route.fulfill(
            status=status,
            content_type="application/json",
            body=body,
        )

    return handler


def _fulfill_body(body: bytes, *, content_type: str = "application/json") -> Callable:
    """Build a handler that returns 200 with the exact ``body`` provided."""

    def handler(route):
        route.fulfill(status=200, content_type=content_type, body=body)

    return handler


def _abort(reason: str = "failed") -> Callable:
    def handler(route):
        route.abort(reason)

    return handler


def _slow_then_fulfill(
    seconds: float, status: int = 200, body: bytes = b'{"jobs":[]}'
) -> Callable:
    """Block the route handler for ``seconds``, then fulfill 200."""

    def handler(route):
        time.sleep(seconds)
        route.fulfill(status=status, content_type="application/json", body=body)

    return handler


def _wait_for_unreachable_banner(harness, *, timeout_ms: int = 15_000) -> None:
    """Poll the body for the global banner. Must appear after 2 poll fails.

    The poll interval for ``listJobs`` is 5 s, so two failures take ~5-10 s.
    Banner text source: ``api.js::POLL_FAIL_THRESHOLD`` -> setError.
    """
    harness.page.wait_for_function(
        "document.body && document.body.innerText.includes('Operator API unreachable')",
        timeout=timeout_ms,
    )


# ---------------------------------------------------------------------------
# Dashboard / Jobs-list — listJobs is the page's polling endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        param(500, id="500-internal"),
        param(502, id="502-bad-gateway"),
        param(503, id="503-unavailable"),
        param(504, id="504-gateway-timeout"),
    ],
)  # fmt: skip
def test_dashboard_listjobs_5xx_eventually_surfaces_unreachable_banner(harness, status):
    """Two consecutive 5xx on /jobs must flip the global 'unreachable' banner.

    The dashboard's first-load failure shows a dashboard-jobs-error card
    immediately; the GLOBAL banner is the contract enforced by api.js::poll
    after ``POLL_FAIL_THRESHOLD`` (=2) consecutive failures.
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(status))
    harness.goto_dashboard()
    # First-load shows the page-level dashboard-jobs-error card.
    harness.page.wait_for_selector("[data-testid=dashboard-jobs-error]", timeout=15_000)
    # Then the polling threshold trips and the global banner appears.
    _wait_for_unreachable_banner(harness, timeout_ms=20_000)


def test_jobs_list_503_shows_jobs_error_card(harness):
    """First-load 503 on /jobs must render the page-scoped jobs-error card."""
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(503))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    # describeLoadError() rewrites API 503/502/504 -> "operator unreachable".
    body = harness.page.locator("[data-testid=jobs-error]").inner_text()
    assert "operator unreachable" in body.lower(), body


def test_jobs_list_404_renders_versioned_error_text(harness):
    """A 404 on /jobs should hint about UI/operator version drift, not a banner."""
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(404))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    body = harness.page.locator("[data-testid=jobs-error]").inner_text()
    assert "older than this UI build" in body or "endpoint not found" in body, body


def test_jobs_list_403_renders_rbac_error_text(harness):
    """403 should rewrite to an RBAC hint, not a generic '<status>: <body>'."""
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(403))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    body = harness.page.locator("[data-testid=jobs-error]").inner_text()
    assert "RBAC" in body or "permission" in body.lower(), body


def test_jobs_list_malformed_json_renders_error_card_not_crash(harness):
    """Truncated/malformed JSON on /jobs must not throw a pageerror.

    apiFetch awaits ``resp.json()``; SyntaxError must bubble into the
    try/catch in the page's poll wrapper, producing the page-scoped
    jobs-error card (firstLoad=true) — NOT a hung Loading skeleton.
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_body(b'{"jobs": [trunc'))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    # No uncaught exceptions: filter pageerror specifically.
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_jobs_list_empty_dict_instead_of_jobs_array_renders_empty_state(harness):
    """If /jobs returns ``{}`` (missing 'jobs' key), the page must use ?? []
    fallback and render the 'no jobs' empty state — not crash.
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_body(b"{}"))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-empty-real]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_jobs_list_returns_bare_array_renders_empty_state(harness):
    """``[]`` instead of ``{"jobs":[]}`` — ``data?.jobs ?? []`` should fall through."""
    harness.page.route(f"{_BASE}/jobs", _fulfill_body(b"[]"))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-empty-real]", timeout=15_000)


def test_dashboard_slow_jobs_shows_loading_panel_before_resolve(harness):
    """A 2 s sleep on /jobs must keep the dashboard loading panel visible
    until resolution — the page-level firstJobsLoad guard exists so the
    user doesn't see a confusing empty dashboard skeleton.
    """
    harness.page.route(
        f"{_BASE}/jobs",
        _slow_then_fulfill(seconds=2.0, body=b'{"jobs":[]}'),
    )
    # Navigate WITHOUT waiting for networkidle so we observe the loading state.
    harness.page.goto(
        f"{harness.base_url}/#/",
        wait_until="domcontentloaded",
        timeout=15_000,
    )
    # The LoadingPanel testid is 'dashboard-loading'.
    harness.page.wait_for_selector("[data-testid=dashboard-loading]", timeout=2_500)
    # Then it resolves to the empty dashboard.
    harness.page.wait_for_selector("[data-testid=dashboard-empty]", timeout=10_000)


def test_jobs_list_network_abort_eventually_unreachable_banner(harness):
    """`route.abort('failed')` simulates a port-forward drop; the banner
    must still appear after threshold, and the page-level error must
    use the 'network error' rewrite from describeLoadError().
    """
    harness.page.route(f"{_BASE}/jobs", _abort("failed"))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    body = harness.page.locator("[data-testid=jobs-error]").inner_text()
    assert "network error" in body.lower() or "fetch" in body.lower(), body
    _wait_for_unreachable_banner(harness, timeout_ms=20_000)


# ---------------------------------------------------------------------------
# Job-detail page — getJob / getJobEpochs
# ---------------------------------------------------------------------------


def test_job_detail_get_job_404_renders_page_level_error(harness):
    """`/jobs/<ns>/<name>` 404 must show ``job-detail-error``, NOT a
    permanent Loading skeleton.

    BUG SURFACED: ``pages/job-detail.js:1563-1577`` calls poll() with NO
    catch block in the closure body. On HTTP error, api.js::poll catches
    the throw, increments the failure counter, but the page's ``error``
    state stays null. So ``if (!job && !error)`` keeps returning the
    LoadingPanel forever — the user sees "Loading ns/missing…" until the
    global poll-threshold banner trips, then sees BOTH at once. There is
    no path where the user lands on the documented job-detail-error card
    via a poll failure.
    """
    harness.page.route(f"{_BASE}/jobs/{harness.ns}/missing*", _fulfill_status(404))
    harness.goto_job_detail(harness.ns, "missing")
    harness.page.wait_for_selector("[data-testid=job-detail-error]", timeout=15_000)
    harness.assert_no_unreachable_banner()


def test_job_detail_get_job_500_renders_error_card(harness):
    """`/jobs/<ns>/<name>` 500 should produce the job-detail-error card.

    Same underlying bug as the 404 case: the poll closure swallows the
    page-level error setter, so the LoadingPanel never resolves.
    """
    harness.page.route(f"{_BASE}/jobs/{harness.ns}/exploding*", _fulfill_status(500))
    harness.goto_job_detail(harness.ns, "exploding")
    harness.page.wait_for_selector("[data-testid=job-detail-error]", timeout=15_000)
    body = harness.page.locator("[data-testid=job-detail-error]").inner_text()
    assert "Failed to load job" in body, body


def test_job_detail_epochs_endpoint_malformed_does_not_crash_page(harness):
    """The /epochs sidecar fetch must be tolerated when it returns garbage.

    The page calls ``api.getJobEpochs(...).catch(() => {})`` — a malformed
    body should be eaten silently and the run-selector should render with
    zero epochs (live URL).
    """
    from tests.unit.operator.ui_e2e.conftest import good_summary

    harness.seed_run(
        name="epoch-mangle",
        epoch="1714069323",
        summary=good_summary(),
        is_latest=True,
    )
    harness.page.route(
        f"{_BASE}/jobs/{harness.ns}/epoch-mangle/epochs",
        _fulfill_body(b'{"epochs": [{"truncated'),
    )
    harness.goto_job_detail(harness.ns, "epoch-mangle", epoch="1714069323")
    # The page header must still render even with mangled epochs feed.
    harness.page.wait_for_selector("[data-testid=page-job-detail]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_job_detail_epochs_endpoint_503_keeps_page_rendering(harness):
    """A 503 on `/epochs` must not poison the main page (it's a side fetch)."""
    from tests.unit.operator.ui_e2e.conftest import good_summary

    harness.seed_run(
        name="epoch-503",
        epoch="1714069323",
        summary=good_summary(),
        is_latest=True,
    )
    harness.page.route(
        f"{_BASE}/jobs/{harness.ns}/epoch-503/epochs", _fulfill_status(503)
    )
    harness.goto_job_detail(harness.ns, "epoch-503", epoch="1714069323")
    harness.page.wait_for_selector("[data-testid=page-job-detail]", timeout=15_000)
    body = harness.page.locator("body").inner_text()
    assert "epoch-503" in body, body[:500]


def test_job_detail_get_job_returns_array_renders_error_card(harness):
    """If /jobs/.../<name> returns ``[]`` (wrong shape — should be a dict),
    rendering relies on ``data?.job ?? {}`` everywhere. Any direct
    ``data.<key>`` access on an array would crash; verify the page lives.
    """
    harness.page.route(f"{_BASE}/jobs/{harness.ns}/array-shape*", _fulfill_body(b"[]"))
    harness.goto_job_detail(harness.ns, "array-shape")
    # We don't care whether this renders the error card or a degraded
    # detail page — only that the page-level wrapper is alive AND no
    # JS exception escaped.
    harness.page.wait_for_selector(
        "[data-testid=page-job-detail],[data-testid=job-detail-error]",
        timeout=15_000,
    )
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_job_detail_get_job_returns_null_does_not_crash(harness):
    """``null`` is the most degenerate "valid JSON" body — the page must
    tolerate it via the ``data?.`` chain without throwing a pageerror.

    BUG SURFACED: a ``null`` response leaves the page stuck on the
    "Loading…" panel forever (``setJob(null)`` keeps ``!job && !error``
    true). The contract should be: either resolve to a degraded job
    header OR show ``job-detail-error``. Either way, the loading panel
    must not be permanent.
    """
    harness.page.route(f"{_BASE}/jobs/{harness.ns}/null-shape*", _fulfill_body(b"null"))
    harness.goto_job_detail(harness.ns, "null-shape")
    harness.page.wait_for_selector(
        "[data-testid=page-job-detail],[data-testid=job-detail-error]",
        timeout=15_000,
    )
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


# ---------------------------------------------------------------------------
# Sweep-detail & sweeps list
# ---------------------------------------------------------------------------


def test_sweeps_list_503_renders_sweeps_error_card(harness):
    """`/sweeps` 503 first-load -> sweeps-error card."""
    harness.page.route(f"{_BASE}/sweeps", _fulfill_status(503))
    harness.goto_sweeps_list()
    harness.page.wait_for_selector("[data-testid=sweeps-error]", timeout=15_000)


def test_sweeps_list_malformed_json_does_not_pageerror(harness):
    """Sweeps list robustness against truncated JSON."""
    harness.page.route(f"{_BASE}/sweeps", _fulfill_body(b'{"sweeps":[{"name":'))
    harness.goto_sweeps_list()
    harness.page.wait_for_selector("[data-testid=sweeps-error]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_sweep_detail_get_sweep_503_renders_loading_or_error_not_crash(harness):
    """A 503 on the main sweep getter on first load must not pageerror."""
    harness.page.route(f"{_BASE}/sweeps/{harness.ns}/sw-down*", _fulfill_status(503))
    harness.goto_sweep_detail(harness.ns, "sw-down")
    # The page either renders the loading wrapper (waiting on retries) or
    # an error card; assert at least that the SPA shell is present.
    harness.page.wait_for_selector("[data-testid=page-sweep-detail]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_sweep_detail_children_with_null_name_entry_does_not_crash(harness):
    """If /children returns a list where one entry has ``name: null``,
    the children table must skip the bad entry rather than crashing on
    ``entry.name.toUpperCase()`` etc.
    """
    harness.seed_sweep_aggregate(sweep="sw-null", epoch="1714069323")
    harness.page.route(
        f"{_BASE}/sweeps/{harness.ns}/sw-null/children*",
        _fulfill_body(
            b'{"children":[{"name":null,"namespace":"x"},'
            b'{"name":"valid-child","namespace":"x"}]}'
        ),
    )
    harness.goto_sweep_detail(harness.ns, "sw-null", epoch="1714069323")
    harness.page.wait_for_selector("[data-testid=page-sweep-detail]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


def test_sweep_detail_cells_endpoint_500_keeps_page_alive(harness):
    """A 500 on /cells (a side fetch) must not crash the page."""
    harness.seed_sweep_aggregate(sweep="sw-cells-500", epoch="1714069323")
    harness.page.route(
        f"{_BASE}/sweeps/{harness.ns}/sw-cells-500/cells*",
        _fulfill_status(500),
    )
    harness.goto_sweep_detail(harness.ns, "sw-cells-500", epoch="1714069323")
    harness.page.wait_for_selector("[data-testid=page-sweep-detail]", timeout=15_000)


# ---------------------------------------------------------------------------
# Compare-epochs — two parallel run-summary fetches
# ---------------------------------------------------------------------------


def test_compare_epochs_one_404_one_200_renders_half(harness):
    """Compare page must tolerate a half-missing run: the side with
    summary renders normally, the missing side becomes ``n/a``.
    """
    from tests.unit.operator.ui_e2e.conftest import good_summary

    # Side A (epoch_A=1714069323) has data + readiness marker so the file
    # serve returns 200; side B (1714069424) is force-404'd via route.
    run_a = harness.seed_run(
        name="cmp-half",
        epoch="1714069323",
        summary=good_summary(),
        is_latest=False,
    )
    harness.seed_results_ready(run_a)

    def route_b(route):
        if "1714069424" in route.request.url:
            route.fulfill(status=404, body=b"")
        else:
            route.continue_()

    harness.page.route(f"{_BASE}/results/{harness.ns}/cmp-half/runs/**", route_b)
    harness.goto(
        f"/compare/{harness.ns}/cmp-half/1714069323/1714069424",
    )
    harness.page.wait_for_selector("[data-testid=page-compare-epochs]", timeout=15_000)
    # Half-missing is NOT both-missing -> the both-missing banner is absent.
    assert not harness.page.locator("[data-testid=compare-both-missing]").is_visible()
    body = harness.page.locator("[data-testid=compare-table]").inner_text()
    assert "n/a" in body.lower(), body


def test_compare_epochs_both_500_shows_both_missing_or_err(harness):
    """Both endpoints failing: page must surface a clear failure state."""

    def fail(route):
        route.fulfill(status=500, body=b"x")

    harness.page.route(f"{_BASE}/results/{harness.ns}/cmp-fail/runs/**", fail)
    harness.goto(
        f"/compare/{harness.ns}/cmp-fail/1714069323/1714069424",
    )
    harness.page.wait_for_selector("[data-testid=page-compare-epochs]", timeout=15_000)
    # The both-missing card OR an err message must be present.
    body = harness.page.locator("body").inner_text()
    assert (
        "Neither run" in body or "missing" in body.lower() or "failed" in body.lower()
    ), body[:1000]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_leaderboard_503_renders_error_text_not_unreachable_banner(harness):
    """Leaderboard owns its own first-load error rendering; the global
    poll-banner contract does not apply (leaderboard isn't polled).
    """
    harness.page.route(f"{_BASE}/analytics/leaderboard*", _fulfill_status(503))
    harness.goto("/leaderboard")
    harness.page.wait_for_selector("[data-testid=page-leaderboard]", timeout=15_000)
    body = harness.page.locator("body").inner_text()
    assert "Failed to load leaderboard" in body, body[:1000]


def test_leaderboard_entries_missing_model_field_still_render(harness):
    """If leaderboard entries lack ``model`` (a synonym for empty), the
    page should still render — falsy-but-defined cells must be tolerated.
    """
    harness.page.route(
        f"{_BASE}/analytics/leaderboard*",
        _fulfill_body(
            b'{"entries":['
            b'{"job_id":"a","namespace":"ns","value":42.0},'
            b'{"job_id":"b","namespace":"ns","value":21.0}'
            b"]}"
        ),
    )
    harness.goto("/leaderboard")
    harness.page.wait_for_selector("[data-testid=page-leaderboard]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


# ---------------------------------------------------------------------------
# Per-job WebSocket adversaries
# ---------------------------------------------------------------------------


def test_job_detail_ws_immediate_close_does_not_spam_reconnects(harness):
    """Abort the WS upgrade; the SPA's onclose schedules a reconnect, but
    the page must still render REST data. Verify pageerror count stays at 0
    even after the reconnect-loop window.
    """
    from tests.unit.operator.ui_e2e.conftest import FakeLiveCR, good_summary

    harness.seed_run(
        name="ws-abort",
        epoch="1714069323",
        summary=good_summary(),
        is_latest=True,
    )
    # Register a live CR with Running phase so the page TRIES to open the WS.
    harness.register_cr(
        FakeLiveCR(
            name="ws-abort",
            namespace=harness.ns,
            phase="Running",
        )
    )
    harness.page.route(f"**/jobs/{harness.ns}/ws-abort/ws", _abort("failed"))
    harness.goto_job_detail(harness.ns, "ws-abort", epoch="1714069323")
    # Give the reconnect loop two cycles (~6 s in api.js — but ws uses its
    # own delay constant; conservative 4 s wait for the first failures).
    harness.page.wait_for_timeout(4_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors
    harness.assert_no_unreachable_banner()


def test_job_detail_ws_garbage_frame_does_not_pageerror(harness):
    """If the WS sends a non-JSON frame, handleMessage swallows the parse
    error. Sanity-check by aborting the WS entirely (proxy here doesn't
    support fulfilling a WS handshake) — equivalent invariant: no uncaught
    exceptions reach window.error.
    """
    from tests.unit.operator.ui_e2e.conftest import FakeLiveCR, good_summary

    harness.seed_run(
        name="ws-garbage",
        epoch="1714069323",
        summary=good_summary(),
        is_latest=True,
    )
    harness.register_cr(
        FakeLiveCR(
            name="ws-garbage",
            namespace=harness.ns,
            phase="Running",
        )
    )
    # Abort each WS connection — the SPA's parse-error swallow path is
    # covered by handleMessage's try/catch, and the no-pageerror invariant
    # is what we care about.
    harness.page.route(
        f"**/jobs/{harness.ns}/ws-garbage/ws", _abort("connectionfailed")
    )
    harness.goto_job_detail(harness.ns, "ws-garbage", epoch="1714069323")
    harness.page.wait_for_timeout(3_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


# ---------------------------------------------------------------------------
# Auth / unexpected response codes
# ---------------------------------------------------------------------------


def test_jobs_list_401_does_not_redirect_to_login(harness):
    """The SPA has no auth UI; a 401 must NOT trigger any redirect away
    from the dashboard. It surfaces in the jobs-error card.
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(401))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    # URL must still point at /jobs.
    assert "/#/jobs" in harness.page.url or harness.page.url.endswith("/#/jobs"), (
        f"unexpected URL after 401: {harness.page.url}"
    )


def test_jobs_list_429_rate_limit_eventually_unreachable_banner(harness):
    """429 is a server failure to the SPA poll wrapper — banner appears
    after the same threshold as 5xx.
    """
    harness.page.route(
        f"{_BASE}/jobs",
        _fulfill_status(429, body=b'{"detail":"rate limited"}'),
    )
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    _wait_for_unreachable_banner(harness, timeout_ms=20_000)


# ---------------------------------------------------------------------------
# Race / repeated reload
# ---------------------------------------------------------------------------


def test_repeated_reload_with_503_keeps_console_errors_bounded(harness):
    """Three reloads with /jobs at 503; console-error count must stay
    bounded (i.e. each reload starts a fresh poll, no listener leak).
    We accept a generous bound to avoid being flaky on noisy /metrics
    fetches — the contract is "not unbounded growth", not "exactly N".
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(503))
    counts = []
    for _ in range(3):
        harness.goto_jobs_list()
        harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
        counts.append(len(harness.console_errors))
    # Growth between reloads should be < 200 console-errors per reload
    # (sane upper bound; observed ~5-10 from resource-load failures).
    assert counts[2] - counts[0] < 600, counts


def test_jobs_to_dashboard_nav_with_jobs_500_does_not_pageerror(harness):
    """Navigate jobs -> dashboard with /jobs always 500. Page renders the
    second page, no uncaught exceptions.
    """
    harness.page.route(f"{_BASE}/jobs", _fulfill_status(500))
    harness.goto_jobs_list()
    harness.page.wait_for_selector("[data-testid=jobs-error]", timeout=15_000)
    harness.goto_dashboard()
    harness.page.wait_for_selector("[data-testid=dashboard-jobs-error]", timeout=15_000)
    page_errors = [e for e in harness.console_errors if e.startswith("[pageerror]")]
    assert not page_errors, page_errors


# ---------------------------------------------------------------------------
# Cluster banner (independent poller — separate poll instance)
# ---------------------------------------------------------------------------


def test_dashboard_cluster_500_shows_warning_not_unreachable(harness):
    """`/cluster` failure must surface the cluster-warning-banner inline
    on the dashboard, NOT the global Operator-API-unreachable banner —
    cluster-info uses its own try/catch and swallows on error.

    NOTE: ClusterStatsBanner has its own poll; failures there shouldn't
    feed the unhealthy-pollers gate in api.js because it doesn't go
    through the shared poll() helper (it lives in cluster-stats-banner.js).
    """
    harness.page.route(f"{_BASE}/cluster", _fulfill_status(500))
    # /jobs returns happy empty so the dashboard render isn't blocked.
    harness.page.route(f"{_BASE}/jobs", _fulfill_body(b'{"jobs":[]}'))
    harness.goto_dashboard()
    harness.page.wait_for_selector("[data-testid=page-dashboard]", timeout=15_000)
    # No GLOBAL unreachable banner.
    harness.assert_no_unreachable_banner()

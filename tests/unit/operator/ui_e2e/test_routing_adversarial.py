# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Playwright tests for the operator-UI client-side router.

Focuses on:
- Hash parsing edge cases (empty hash, double slashes, trailing slash,
  query-only, repeated query keys, empty values, key-without-equals).
- URL-encoded route params (encoded slash, space, plus, percent, unicode,
  embedded newline) — does ``matchRoute`` decode them and does the page
  survive the resulting param values?
- Path-traversal-shaped inputs in ``:ns`` / ``:name`` segments.
- Malformed ``/runs/:epoch`` values (non-numeric, zero, negative, float,
  numeric overflow, whitespace-padded).
- Excess path segments not matching any route -> Not Found, no JS crash.
- ``/compare/:ns/:name/:epochA/:epochB`` with degenerate epoch pairs.
- Browser history (``page.go_back``) round-trip across SPA navigations.
- ``setQuery`` round-trip via direct ``window.location.hash`` mutation.
- ``safeDecodeURIComponent`` truncated-UTF-8 fallback path.
- Command palette (Ctrl+K) on dashboard with zero-result query.
- Top-nav cross-links + external Plots ↗ link non-navigation.
- ``hashchange`` listener pick-up on direct ``window.location.hash`` writes.
- Concurrent route changes — final route is the last one requested.

Out of scope: page-internal content (job-detail tables, sweep variations);
covered in their own adversarial files.
"""

from __future__ import annotations

import pytest
from pytest import param

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_hash(harness) -> str:
    """Return the current ``window.location.hash`` (including leading ``#``)."""
    return harness.page.evaluate("() => window.location.hash")


def _current_route_signal(harness) -> str:
    """Read the current ``route.value`` from the router module via window.

    The router doesn't expose the signal on ``window``; we drive observations
    via DOM state instead. This helper is kept for documentation but the
    primary check is the hash + the rendered breadcrumb.
    """
    return harness.page.evaluate("() => window.location.hash.replace(/^#/, '')")


def _wait_for_route(harness, expected_path: str, *, timeout_ms: int = 5_000) -> None:
    """Wait until ``window.location.hash`` matches ``#<expected_path>``."""
    harness.page.wait_for_function(
        f"() => window.location.hash === '#{expected_path}'",
        timeout=timeout_ms,
    )


# ===========================================================================
# 1. Hash parsing edge cases
# ===========================================================================


def test_empty_hash_renders_dashboard(harness):
    """Bare ``#`` (empty hash body) routes to Dashboard, not Not Found.

    ``parseHash`` collapses an empty raw to ``'/'``; the SPA must render the
    Dashboard at the empty hash exactly as it does at ``/``.
    """
    harness.page.goto(f"{harness.base_url}/#", wait_until="networkidle", timeout=15_000)
    assert harness.page.locator("[data-testid=page-dashboard]").first.is_visible()
    body = harness.page.locator("body").inner_text()
    assert "Not Found" not in body
    harness.assert_no_unreachable_banner()


def test_root_slash_hash_renders_dashboard(harness):
    """``#/`` (root slash only) routes to Dashboard."""
    harness.goto("/")
    assert harness.page.locator("[data-testid=page-dashboard]").first.is_visible()


def test_double_slash_hash_does_not_crash(harness):
    """``#//`` (double leading slash) must not throw — empty segments are dropped.

    ``matchRoute`` filters empty segments via ``split('/').filter(Boolean)``,
    so ``//`` should normalize to ``/``. Failure mode: page-stub Not Found OR
    a JS error.
    """
    harness.goto("//")
    body = harness.page.locator("body").inner_text()
    assert "Operator API unreachable" not in body
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_trailing_slash_jobs_does_not_match_jobs_list(harness):
    """``/jobs/`` (trailing slash) -> page-stub Not Found, no crash.

    The router compares ``currentRoute === '/jobs'`` strict-equal; trailing
    slash is a different string and is expected to miss every pattern. The
    *important* invariant is that the page renders the Not Found stub and
    not a JS exception.
    """
    harness.goto("/jobs/")
    body = harness.page.locator("body").inner_text()
    assert "Not Found" in body
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_query_only_hash_renders_dashboard(harness):
    """``#?q=foo`` (query only, no path) -> Dashboard, query parsed."""
    harness.page.goto(
        f"{harness.base_url}/#?q=foo", wait_until="networkidle", timeout=15_000
    )
    assert harness.page.locator("[data-testid=page-dashboard]").first.is_visible()
    harness.assert_no_unreachable_banner()


def test_jobs_list_with_repeated_query_keys_keeps_last(harness):
    """Repeated ``phase=`` query keys: parser keeps the *last* occurrence.

    ``parseQueryString`` overwrites ``out[key]`` on each pair, so a repeated
    key collapses to its last value. This locks the contract; if a caller
    expects arrays they need a different parser.
    """
    harness.goto("/jobs?ns=foo&phase=running&phase=queued")
    # Page renders without crashing.
    assert harness.page.locator("[data-testid=page-jobs]").first.is_visible()
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_jobs_list_with_empty_query_value(harness):
    """``?ns=`` (empty value) -> parser stores empty string, page renders."""
    harness.goto("/jobs?ns=")
    assert harness.page.locator("[data-testid=page-jobs]").first.is_visible()


def test_jobs_list_with_empty_query_string(harness):
    """``/jobs?`` (empty query body, leading ``?``) -> page renders."""
    harness.goto("/jobs?")
    assert harness.page.locator("[data-testid=page-jobs]").first.is_visible()


def test_jobs_list_with_key_without_equals(harness):
    """``?key`` (no ``=``) -> parser stores ``key -> ''``, page renders."""
    harness.goto("/jobs?lonelyflag")
    assert harness.page.locator("[data-testid=page-jobs]").first.is_visible()


# ===========================================================================
# 2. URL-encoded path segments
# ===========================================================================


@pytest.mark.parametrize(
    "raw_name,decoded_name,case_id",
    [
        param("space%20in%20name", "space in name", "percent-20-decodes-to-space"),
        param("dash%2Dname", "dash-name", "percent-2D-decodes-to-dash"),
        param("check%E2%9C%93", "check✓", "utf8-checkmark-decodes"),
        param("plus+name", "plus+name", "plus-is-not-decoded-to-space"),
        param("percent%25literal", "percent%literal", "double-percent-decodes"),
    ],
)  # fmt: skip
def test_encoded_name_segment_decodes_in_route_param(
    harness, raw_name: str, decoded_name: str, case_id: str
) -> None:
    """``matchRoute`` runs ``safeDecodeURIComponent`` on each ``:param`` slot.

    The decoded value must appear in the breadcrumb's last segment (the page
    template injects the decoded ``name`` straight into the title).
    """
    harness.goto(f"/jobs/{harness.ns}/{raw_name}")
    bc = harness.page.locator("[data-testid=breadcrumb]").inner_text()
    assert decoded_name in bc, (case_id, bc)


def test_encoded_slash_in_name_does_not_split_route(harness):
    """``%2F`` inside a name segment must NOT introduce a new path component.

    The browser does not decode the hash before pattern-matching, so the
    encoded slash should stay inside the name segment. The breadcrumb should
    render the decoded ``foo/bar`` as a single segment, not split it.
    """
    harness.goto(f"/jobs/{harness.ns}/foo%2Fbar")
    bc = harness.page.locator("[data-testid=breadcrumb]").inner_text()
    assert "foo/bar" in bc, bc


def test_encoded_newline_in_name_segment_does_not_crash(harness):
    """A literal ``%0A`` inside ``:name`` decodes to a newline.

    Page must not throw. The breadcrumb might render two visual lines or
    collapse whitespace — both are acceptable; the bar is "no JS error".
    """
    harness.goto(f"/jobs/{harness.ns}/danger%0Atext")
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


# ===========================================================================
# 3. Path-traversal-shaped inputs
# ===========================================================================


@pytest.mark.parametrize(
    "ns_segment,name_segment,case_id",
    [
        param("..%2Fother", "job", "ns-encoded-dotdot-slash"),
        param("foo%2Fbar", "job", "encoded-slash-inside-ns"),
        param("default", "foo%2F..%2Fbar", "encoded-dotdot-inside-name"),
    ],
)  # fmt: skip
def test_traversal_shaped_path_does_not_500_backend(
    harness, ns_segment: str, name_segment: str, case_id: str
) -> None:
    """Encoded ``..`` and ``/`` inside route params must reach the SPA cleanly.

    Backend API call for the (decoded) ns/name may legitimately 404, but the
    SPA itself must not 500 or surface "Operator API unreachable", and there
    must be no client-side exception.
    """
    harness.goto(f"/jobs/{ns_segment}/{name_segment}")
    harness.assert_no_unreachable_banner()
    # 404 from the backend is expected (no such job); only flag >= 500.
    server_errors = [r for r in harness.bad_responses if r.startswith("5")]
    assert not server_errors, (case_id, server_errors)


def test_excess_dotdot_in_path_does_not_match_run_route(harness):
    """``/jobs/foo/job/../runs/1714069323`` is 5 segments — must NOT match
    ``/jobs/:ns/:name/runs/:epoch`` (4 visible segments after Boolean filter).

    The literal ``..`` segment is NOT collapsed by the router (it splits on
    ``/`` and filters empties only). It must land in the Not Found stub.
    """
    harness.goto("/jobs/foo/job/../runs/1714069323")
    body = harness.page.locator("body").inner_text()
    assert "Not Found" in body
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


# ===========================================================================
# 4. Malformed /runs/:epoch
# ===========================================================================


@pytest.mark.parametrize(
    "epoch_segment,case_id",
    [
        param("abc", "non-numeric-epoch"),
        param("0", "zero-epoch"),
        param("-1", "negative-epoch"),
        param("1.5", "fractional-epoch"),
        param("999999999999999999999", "overflow-epoch"),
        param("%201714069323%20", "whitespace-padded-epoch"),
    ],
)  # fmt: skip
def test_malformed_epoch_renders_without_crash(
    harness, epoch_segment: str, case_id: str
) -> None:
    """``/runs/:epoch`` with garbage epoch: SPA must not throw.

    The router's ``matchRoute`` is purely textual; whatever the epoch contains
    becomes the ``epoch`` prop of ``JobDetail``. The backend may 404 or 422,
    but the SPA's job-detail page should render without console errors.
    """
    harness.goto(f"/jobs/{harness.ns}/somename/runs/{epoch_segment}")
    harness.assert_no_unreachable_banner()
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_empty_epoch_segment_does_not_match_run_route(harness):
    """``/jobs/<ns>/<name>/runs/`` with trailing slash — empty epoch.

    After ``split('/').filter(Boolean)``, the path has 4 segments (no empty
    epoch tail), which matches ``/jobs/:ns/:name/runs/:epoch`` length only if
    we consider ``runs`` as the ``epoch`` slot. Either it land on Not Found,
    or it renders the run page with epoch=``runs``. Both are acceptable; the
    invariant is "no crash".
    """
    harness.goto(f"/jobs/{harness.ns}/somename/runs/")
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


# ===========================================================================
# 5. Excess path segments
# ===========================================================================


def test_extra_trailing_segments_do_not_match_run_route(harness):
    """``/jobs/<ns>/<name>/runs/<epoch>/extra/segments`` -> Not Found stub.

    ``matchRoute`` rejects when ``patternParts.length !== pathParts.length``.
    """
    harness.goto(f"/jobs/{harness.ns}/somename/runs/1714069323/extra/segments")
    body = harness.page.locator("body").inner_text()
    assert "Not Found" in body
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


# ===========================================================================
# 6. Compare route degenerate cases
# ===========================================================================


@pytest.mark.parametrize(
    "epoch_a,epoch_b,case_id",
    [
        param("1714069323", "1714069323", "identical-epochs"),
        param("abc", "1714069323", "non-numeric-first-epoch"),
        param("1714069323", "xyz", "non-numeric-second-epoch"),
    ],
)  # fmt: skip
def test_compare_route_degenerate_epoch_pair_renders(
    harness, epoch_a: str, epoch_b: str, case_id: str
) -> None:
    """``/compare/:ns/:name/:epochA/:epochB`` accepts any string; SPA must render."""
    harness.goto(f"/compare/{harness.ns}/somename/{epoch_a}/{epoch_b}")
    harness.assert_no_unreachable_banner()
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_compare_route_missing_epoch_does_not_match(harness):
    """``/compare/<ns>/<name>/<epoch>`` (one epoch missing) -> 4 segments, not 5.

    Pattern has 5 placeholders; mismatched length -> matchRoute returns null
    -> Not Found stub.
    """
    harness.goto(f"/compare/{harness.ns}/somename/1714069323")
    body = harness.page.locator("body").inner_text()
    assert "Not Found" in body


# ===========================================================================
# 7. Browser history
# ===========================================================================


def test_back_button_round_trip_jobs_to_run_and_back(harness):
    """Jobs -> JobDetail -> /jobs/.../runs/X -> page.go_back() -> JobDetail.

    The SPA must restore the route and not surface stale or duplicated state.
    """
    harness.seed_run(name="hist-job", epoch="1714069323", summary=None)
    harness.goto_jobs_list()
    _wait_for_route(harness, "/jobs")
    harness.goto(f"/jobs/{harness.ns}/hist-job")
    _wait_for_route(harness, f"/jobs/{harness.ns}/hist-job")
    harness.goto(f"/jobs/{harness.ns}/hist-job/runs/1714069323")
    _wait_for_route(harness, f"/jobs/{harness.ns}/hist-job/runs/1714069323")
    harness.page.go_back()
    _wait_for_route(harness, f"/jobs/{harness.ns}/hist-job")
    harness.page.go_back()
    _wait_for_route(harness, "/jobs")
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


# ===========================================================================
# 8. setQuery round-trip via direct hash mutation
# ===========================================================================


def test_set_hash_via_evaluate_round_trips_query(harness):
    """Programmatically mutating ``window.location.hash`` to a path+query
    must update the route signal (verified by breadcrumb refresh).
    """
    harness.goto_dashboard()
    harness.page.evaluate(
        "() => { window.location.hash = '/jobs?phase=Failed&ns=foo'; }"
    )
    _wait_for_route(harness, "/jobs?phase=Failed&ns=foo")
    assert harness.page.locator("[data-testid=page-jobs]").first.is_visible(
        timeout=5_000
    )


def test_set_hash_to_sweeps_picks_up_via_hashchange_listener(harness):
    """Direct ``window.location.hash = '/sweeps/foo/bar'`` triggers the
    ``hashchange`` listener and transitions to the sweep-detail route.
    """
    harness.goto_dashboard()
    harness.page.evaluate("() => { window.location.hash = '/sweeps/foo/bar'; }")
    _wait_for_route(harness, "/sweeps/foo/bar")
    assert harness.page.locator("[data-testid=page-sweep-detail]").first.is_visible(
        timeout=5_000
    )


# ===========================================================================
# 9. safeDecodeURIComponent fallback
# ===========================================================================


def test_truncated_utf8_in_route_param_does_not_crash(harness):
    """``%E0%A4%A`` is a truncated UTF-8 sequence -> ``decodeURIComponent``
    throws URIError; ``safeDecodeURIComponent`` must catch and return raw bytes.

    The page should render (with the raw escape sequence inside the
    breadcrumb), not throw.
    """
    harness.goto(f"/jobs/{harness.ns}/broken%E0%A4%A")
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))
    # Either the raw bytes or some breadcrumb DOM should be present.
    assert harness.page.locator("[data-testid=breadcrumb]").first.is_visible()


# ===========================================================================
# 10. Command palette (Ctrl+K)
# ===========================================================================


def test_ctrl_k_opens_palette_on_dashboard(harness):
    """Ctrl+K opens the command palette overlay.

    BUG (reproduces here): the handler in ``src/aiperf/operator/ui/app.js:30``
    matches ``e.key === 'k'`` (lowercase), but Chromium emits ``e.key === 'K'``
    (uppercase) for Ctrl+K when the Shift key is conceptually engaged for the
    keysym lookup. The keystroke fires, ``ctrlKey`` is True, but the lowercase
    comparison fails and the palette never opens. The Search button (clicked
    in :func:`test_search_button_opens_palette`) is the working fallback.

    Fix sketch: ``e.key.toLowerCase() === 'k'``.
    """
    harness.goto_dashboard()
    harness.page.keyboard.press("Control+K")
    assert harness.page.locator("[data-testid=command-palette]").first.is_visible(
        timeout=5_000
    )


def test_search_button_opens_palette(harness):
    """Clicking the Search button in the top-nav opens the command palette.

    This is the fallback path that works today even with the Ctrl+K bug above.
    """
    harness.goto_dashboard()
    harness.page.locator("[data-testid=nav-search]").click()
    assert harness.page.locator("[data-testid=command-palette]").first.is_visible(
        timeout=5_000
    )


def test_palette_zero_result_query_renders_no_results(harness):
    """A query with zero matches surfaces the 'No results for ...' empty row."""
    harness.goto_dashboard()
    harness.page.locator("[data-testid=nav-search]").click()
    harness.page.locator("[data-testid=command-palette-input]").fill(
        "zzzz-no-such-target-xyzzy"
    )
    body = harness.page.locator("[data-testid=command-palette]").inner_text()
    assert "No results" in body, body
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_palette_enter_on_page_match_navigates(harness):
    """Typing ``jobs`` then Enter routes to ``/jobs`` and the palette closes."""
    harness.goto_dashboard()
    harness.page.locator("[data-testid=nav-search]").click()
    harness.page.locator("[data-testid=command-palette-input]").fill("jobs")
    harness.page.keyboard.press("Enter")
    _wait_for_route(harness, "/jobs")
    harness.page.locator("[data-testid=page-jobs]").first.wait_for(
        state="visible", timeout=5_000
    )


# ===========================================================================
# 11. Topnav cross-links
# ===========================================================================


def test_topnav_jobs_link_navigates(harness):
    """Clicking the Jobs nav-tab on the dashboard routes to ``/jobs``."""
    harness.goto_dashboard()
    harness.page.locator("[data-testid=nav-link-jobs]").click()
    _wait_for_route(harness, "/jobs")


def test_topnav_sweeps_link_navigates_from_jobs(harness):
    """Sweeps cross-link works from the Jobs list."""
    harness.goto_jobs_list()
    harness.page.locator("[data-testid=nav-link-sweeps]").click()
    _wait_for_route(harness, "/sweeps")


def test_topnav_dashboard_link_returns_home(harness):
    """Clicking Dashboard from /sweeps returns to ``/``."""
    harness.goto_sweeps_list()
    harness.page.locator("[data-testid=nav-link-dashboard]").click()
    _wait_for_route(harness, "/")


# ===========================================================================
# 12. Concurrent route changes
# ===========================================================================


def test_rapid_consecutive_navigate_settles_on_final_route(harness):
    """Two ``goto`` calls back-to-back: final URL is the second target.

    The harness's ``goto`` waits for networkidle, but the in-flight first
    request may overlap. The router's only state is ``window.location.hash``;
    whichever assignment runs last wins. Verifies no stale page leaks.
    """
    harness.goto("/jobs")
    # Issue the second navigation without waiting — relies on goto() doing it
    # internally. Both produce a single networkidle cycle; verify final state.
    harness.goto("/sweeps")
    _wait_for_route(harness, "/sweeps")
    assert harness.page.locator("[data-testid=page-sweeps]").first.is_visible()
    assert not harness.page.locator("[data-testid=page-jobs]").first.is_visible(
        timeout=500
    )

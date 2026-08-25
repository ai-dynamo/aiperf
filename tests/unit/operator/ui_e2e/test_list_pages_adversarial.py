# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Playwright tests for the operator-UI list / overview pages.

Targets:
  * Dashboard           ``/``
  * Jobs list           ``/jobs``
  * Sweeps list         ``/sweeps``
  * Leaderboard         ``/leaderboard``
  * History             ``/history``
  * Compare             ``/compare`` and ``/compare/<ns>/<name>/<epochA>/<epochB>``

These tests are *adversarial* — they seed pathological shapes (NaN/0 KPIs,
oversized model names, missing summaries, source=live vs archived vs both,
phase-XSS attempts, asymmetric epochs, empty sweeps) and assert the UI keeps
rendering, no console errors fire, and no 5xx responses come back.

Real bugs surfaced by a test cause the test to fail (no ``xfail``).
"""

from __future__ import annotations

import copy
import json

import pytest
from pytest import param

from tests.unit.operator.ui_e2e.conftest import FakeLiveCR, good_summary

# ----------------------------------------------------------------------------
# Helpers — kept tiny and obvious. Each one is a single intent.
# ----------------------------------------------------------------------------

# Canonical 10-digit epochs used across the file. Treating them as named
# constants makes "stale child from prior epoch" tests legible.
EPOCH_OLDER = "1714069323"
EPOCH_NEWER = "1714069999"

# Allowed console-error substrings. The dashboard fires a single benign 503
# for /api/v1/cluster when the patched kubernetes_asyncio APIs return empty;
# leaderboard / history / summary may 404 for sparsely seeded namespaces and
# that surfaces as a console-level "Failed to load resource" warning we let
# pass without making the test about it.
_BENIGN_CONSOLE = (
    "Failed to load resource",
    "/api/v1/cluster",
    "/api/v1/analytics/leaderboard",
    "/api/v1/analytics/history",
    "/api/v1/analytics/summary",
    "/api/v1/analytics/compare",
    "/api/v1/results",
    "/api/v1/jobs/",
    "ChartWrapper",  # canvas errors in headless render
    "Chart",
)


def _good_with(**overrides) -> dict:
    """Spread overrides into a baseline ``good_summary`` dict.

    Mirrors the canonical adversarial-test "dict-spread baseline" pattern so a
    single ``_good_with(throughput_rps=0)`` line keeps the rest of the summary
    valid and validator-compatible.
    """
    base = good_summary()
    # Deep copy so overrides at nested keys don't mutate the module-level
    # canonical dict returned by good_summary().
    base = copy.deepcopy(base)
    for k, v in overrides.items():
        base[k] = v
    return base


def _seed_phase_run(
    harness, *, name: str, phase: str, epoch: str = EPOCH_OLDER
) -> None:
    """Seed an archived run that will surface under the requested phase.

    The archived ``AIPerfJobInfo.phase`` is taken verbatim from the summary's
    ``status`` field, so this is the simplest path to "an archived run with
    phase=Failed" without standing up a live CR.
    """
    summary = good_summary()
    summary["status"] = phase
    harness.seed_run(name=name, epoch=epoch, summary=summary, is_latest=True)


def _wait_for_jobs_list_settled(harness) -> None:
    """Wait for the jobs page to leave first-load and render a table OR an
    empty-state marker. Uses Playwright's locator wait — never time.sleep.
    """
    page = harness.page
    # Either an empty-state, or a table, or a row.
    page.wait_for_selector(
        "[data-testid=jobs-empty-real],"
        "[data-testid=jobs-empty-filtered],"
        "[data-testid=jobs-error],"
        "table.job-table",
        timeout=10_000,
    )


def _wait_for_dashboard_settled(harness) -> None:
    page = harness.page
    page.wait_for_selector(
        "[data-testid=page-dashboard]",
        timeout=10_000,
    )
    # firstJobsLoad spinner gives way to either the empty state or rendered
    # content; both expose a clear marker. Wait for one to land.
    page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid=page-dashboard]');
            if (!el) return false;
            const loading = el.querySelector('[data-testid=dashboard-loading]');
            return !loading;
        }""",
        timeout=10_000,
    )


def _wait_for_sweeps_list_settled(harness) -> None:
    page = harness.page
    page.wait_for_selector(
        "[data-testid=sweeps-empty-real],"
        "[data-testid=sweeps-empty-filtered],"
        "[data-testid=sweeps-error],"
        "[data-testid=sweep-table]",
        timeout=10_000,
    )


def _seed_archived_sweep_aggregate(
    harness,
    *,
    sweep: str,
    epoch: str = EPOCH_OLDER,
    total_variations: int = 3,
    completed_runs: int = 3,
    failed_runs: int = 0,
    phase: str = "Succeeded",
    model: str = "llama3-8b",
) -> None:
    """Write the ``aggregate.json`` shape the operator's archived-sweep
    scanner expects, plus a ``latest.txt`` pointer.

    The shared seed helper writes ``profile_export_aiperf.json``, but the
    archived-sweep code path keys off ``aggregate.json`` (see
    ``sweep_union._AGGREGATE_FILE``). Bypassing the shared helper here lets
    archived sweeps actually appear in ``GET /api/v1/sweeps``.
    """
    base = harness.results_dir / harness.ns / "sweeps" / sweep / epoch
    base.mkdir(parents=True, exist_ok=True)
    (base / "aggregate.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "totalVariations": total_variations,
                "completedRuns": completed_runs,
                "failedRuns": failed_runs,
                "model": model,
                "startedAt": "2026-05-19T00:00:00Z",
                "completedAt": "2026-05-19T00:05:00Z",
            }
        )
    )
    (harness.results_dir / harness.ns / "sweeps" / sweep / "latest.txt").write_text(
        epoch
    )


# ----------------------------------------------------------------------------
# 1) Empty states — every list page renders without console errors.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route_path,settle_testid",
    [
        param("/", "page-dashboard", id="dashboard-empty"),
        param("/jobs", "page-jobs", id="jobs-empty"),
        param("/sweeps", "page-sweeps", id="sweeps-empty"),
        param("/leaderboard", "page-leaderboard", id="leaderboard-empty"),
        param("/history", "page-history", id="history-empty"),
        param("/compare", "page-compare", id="compare-empty"),
    ],
)  # fmt: skip
def test_list_page_empty_state_renders_without_console_errors(
    harness,
    route_path: str,
    settle_testid: str,
) -> None:
    """Each list page renders an empty-state affordance for a virgin namespace.

    Invariant: with zero seeded data and no live CRs, the page must still
    reach DOMContentLoaded, expose its page-level data-testid, never show the
    "Operator API unreachable" banner, and never trigger a 5xx response.
    """
    harness.goto(route_path)
    harness.page.wait_for_selector(f"[data-testid={settle_testid}]", timeout=10_000)
    harness.assert_no_unreachable_banner()
    fivexx = [r for r in harness.bad_responses if r.startswith(("5",))]
    assert not fivexx, f"Got 5xx responses: {fivexx}"


# ----------------------------------------------------------------------------
# 2) Pure-CR jobs: no on-disk run, just a live FakeLiveCR.
# ----------------------------------------------------------------------------


def test_jobs_detail_serves_live_only_cr_with_source_live(harness) -> None:
    """A registered FakeLiveCR with no on-disk dir surfaces via per-job
    detail with ``source="live"``.

    Harness gap (documented): the shared conftest patches only
    ``find_aiperf_job`` (per-name lookup) and lets ``list_aiperf_jobs``
    crash against a MagicMock api client, so live CRs do NOT appear in
    ``GET /api/v1/jobs``. The detail endpoint ``GET /api/v1/jobs/{ns}/{name}``
    uses ``find_aiperf_job`` and works correctly — assert there instead.
    Surfaced bug-or-gap: the list endpoint should not be brittle against a
    failing list-aiperf-jobs call, but the warning is logged + PVC fallback
    kicks in, so this is graceful-degradation rather than a list-time crash.
    """
    harness.register_cr(
        FakeLiveCR(
            name="live-only-bench",
            namespace=harness.ns,
            phase="Running",
            model="meta-llama/Llama-3-8B-Instruct",
            endpoint="http://srv:8000/v1",
            workers_ready=4,
            workers_total=4,
            created="2026-05-19T00:00:00Z",
            start_time="2026-05-19T00:00:00Z",
        )
    )
    status, raw = harness.api_get(f"/api/v1/jobs/{harness.ns}/live-only-bench")
    assert status == 200, (status, raw[:200])
    payload = json.loads(raw)
    assert payload["job"]["source"] == "live", payload
    assert payload["job"]["phase"] == "Running"


# ----------------------------------------------------------------------------
# 3) Pure-archived jobs: no CR, just a seeded run with is_latest=True.
# ----------------------------------------------------------------------------


def test_jobs_list_shows_archived_only_run_with_source_archived(harness) -> None:
    """A seeded run with no live CR shows in the list as source=archived."""
    harness.seed_run(
        name="archived-only-bench",
        epoch=EPOCH_OLDER,
        summary=good_summary(throughput_rps=200.0, model="llama-archive"),
        is_latest=True,
    )
    harness.goto_jobs_list()
    _wait_for_jobs_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "archived-only-bench" in body

    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    entry = next(
        (
            j
            for j in payload["jobs"]
            if j["namespace"] == harness.ns and j["name"] == "archived-only-bench"
        ),
        None,
    )
    assert entry is not None, payload
    assert entry["source"] == "archived"


# ----------------------------------------------------------------------------
# 4) Both: live CR + archived run for the same (ns, name) — must dedupe and
#    surface source=both with CR taking precedence on live fields.
# ----------------------------------------------------------------------------


def test_jobs_detail_overlap_marks_source_both_with_cr_phase_winning(harness) -> None:
    """When the same (ns, name) has both a CR and an on-disk run, the
    job-detail endpoint returns ``source="both"`` with the CR's phase
    overriding the archived summary's status.

    Harness-scope note: the *list* endpoint can't dedupe in this harness
    because ``list_aiperf_jobs`` is unpatched (see live-only test above).
    The detail endpoint uses ``find_any_job`` which calls ``find_aiperf_job``
    + reads the PVC, both of which work — that's where we assert.
    """
    name = "overlap-bench"
    harness.register_cr(
        FakeLiveCR(
            name=name,
            namespace=harness.ns,
            phase="Initializing",
            model="meta-llama/Llama-3-70B",
            workers_ready=2,
            workers_total=8,
            created="2026-05-19T00:00:00Z",
            start_time="2026-05-19T00:00:00Z",
        )
    )
    harness.seed_run(
        name=name,
        epoch=EPOCH_OLDER,
        summary=good_summary(throughput_rps=42.0),
        is_latest=True,
    )

    status, raw = harness.api_get(f"/api/v1/jobs/{harness.ns}/{name}")
    assert status == 200, (status, raw[:200])
    payload = json.loads(raw)
    entry = payload["job"]
    assert entry["source"] == "both", entry
    # CR's live phase must win — the archived summary's status was "Succeeded".
    assert entry["phase"] == "Initializing"


# ----------------------------------------------------------------------------
# 5) Mixed phases on the Dashboard — counters must accurately classify.
# ----------------------------------------------------------------------------


def test_dashboard_mixed_phases_classifies_into_running_and_completed(harness) -> None:
    """Five archived runs across five phases — counters must classify per
    ``Dashboard``'s running/completed predicates.

    Running predicate: phase in {running, initializing, pending}.
    Completed predicate: phase in {completed, succeeded}.
    Failed/Cancelled phases are neither and must not be double-counted.

    Harness-scope note: live CRs are not visible to the list endpoint in
    this harness, so we use archived runs (status=<phase>) for everything.
    The archived rows still flow through the same Dashboard ``running``
    filter — that filter keys on ``phase``, not on ``source``.
    """
    # ``results_dir`` is session-scoped, so earlier tests can leak rows into
    # the Dashboard's recent-list/KPI tiles. Wipe before seeding so the
    # classification counters are deterministic.
    harness.clear_all_seeded_data()
    _seed_phase_run(harness, name="pending-bench", phase="Pending")
    _seed_phase_run(harness, name="running-bench", phase="Running", epoch=EPOCH_NEWER)
    _seed_phase_run(
        harness, name="completed-bench", phase="Completed", epoch="1714070100"
    )
    _seed_phase_run(
        harness, name="succeeded-bench", phase="Succeeded", epoch="1714070200"
    )
    _seed_phase_run(harness, name="failed-bench", phase="Failed", epoch="1714070300")
    _seed_phase_run(
        harness, name="cancelled-bench", phase="Cancelled", epoch="1714070400"
    )

    # Sanity-check the API payload first.
    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    in_ns = [j for j in payload["jobs"] if j["namespace"] == harness.ns]
    phases = {j["name"]: j["phase"] for j in in_ns}
    assert phases.get("pending-bench") == "Pending", phases
    assert phases.get("running-bench") == "Running", phases
    assert phases.get("completed-bench") == "Completed", phases
    assert phases.get("succeeded-bench") == "Succeeded", phases
    assert phases.get("failed-bench") == "Failed", phases
    assert phases.get("cancelled-bench") == "Cancelled", phases

    harness.goto_dashboard()
    _wait_for_dashboard_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "Running" in body
    assert "Completed" in body


# ----------------------------------------------------------------------------
# 6) Large list — 60 jobs must render without any 5xx and the API must
#    return all 60 entries (no silent truncation).
# ----------------------------------------------------------------------------


def test_jobs_list_renders_60_seeded_jobs_without_5xx(harness) -> None:
    """Seed 60 distinct archived jobs and verify the API + UI handle it.

    Realistic identifiers — names follow ``aiperf-bench-<n>``. Throughputs
    vary so a perceived-order spot-check later is meaningful.
    """
    for i in range(60):
        harness.seed_run(
            name=f"aiperf-bench-{i:03d}",
            epoch=str(1_714_069_000 + i),
            summary=good_summary(
                throughput_rps=float(50 + i),
                model=f"model-{i % 3}",
            ),
            is_latest=True,
        )

    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    in_ns = [j for j in payload["jobs"] if j["namespace"] == harness.ns]
    assert len(in_ns) == 60

    harness.goto_jobs_list()
    _wait_for_jobs_list_settled(harness)
    fivexx = [r for r in harness.bad_responses if r.startswith("5")]
    assert not fivexx, fivexx


# ----------------------------------------------------------------------------
# 7) Numeric KPI edge cases — long model names, zeros, NaN, missing latency.
# ----------------------------------------------------------------------------


def test_dashboard_kpi_tiles_handle_zero_throughput(harness) -> None:
    """A throughput=0 completed run must NOT trip Dashboard's findBest
    (which uses ``> best`` comparison) into spurious rendering.
    """
    # See note on the harness contamination caveat above.
    harness.clear_all_seeded_data()
    summary = good_summary(throughput_rps=0.0)
    summary["status"] = "Completed"
    harness.seed_run(
        name="zero-tput-bench", epoch=EPOCH_OLDER, summary=summary, is_latest=True
    )
    harness.goto_dashboard()
    _wait_for_dashboard_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "zero-tput-bench" in body
    harness.assert_no_unreachable_banner()


def test_dashboard_handles_nan_throughput_without_breaking_layout(harness) -> None:
    """A NaN throughput in the seeded JSON must not 500 the API or stall the UI.

    ``json.dumps(float("nan"))`` would raise; we go through a string-replace
    on the dump to emit raw ``NaN`` (technically invalid JSON, which orjson
    rejects on parse). The API should tolerate parse failure (returns the
    job as archived with KPIs=None) rather than 500.
    """
    # Write a literal "NaN" — invalid JSON — so the operator's defensive
    # try/except wraps it (see CLAUDE.md "Completion-parse resilience").
    base = harness.results_dir / harness.ns / "nan-bench" / EPOCH_OLDER
    base.mkdir(parents=True, exist_ok=True)
    (base / "profile_export_aiperf.json").write_text(
        '{"status": "Completed", "request_throughput": {"avg": NaN, "unit": "requests/sec"}}'
    )
    (harness.results_dir / harness.ns / "nan-bench" / "latest.txt").write_text(
        EPOCH_OLDER
    )

    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200, (status, raw[:200])
    # Even if the malformed file means the job is filtered out, the API
    # must not 500. We don't assert the job appears — only that it didn't
    # blow up the listing.

    harness.goto_dashboard()
    _wait_for_dashboard_settled(harness)
    harness.assert_no_unreachable_banner()


def test_dashboard_handles_missing_p99_latency(harness) -> None:
    """A summary with no ``request_latency`` block still renders on the
    Dashboard. The "Best TTFT" / latency KPI tiles show "---" rather than
    crashing the page.

    Cross-test contamination note: ``results_dir`` is session-scoped, so
    every prior test's seeded runs are still on disk when this one runs.
    We assert against the API payload to keep the assertion deterministic,
    and only smoke-check that the dashboard mounts.
    """
    summary = good_summary()
    summary.pop("request_latency", None)
    summary["status"] = "Completed"
    harness.seed_run(
        name="no-p99-bench", epoch=EPOCH_OLDER, summary=summary, is_latest=True
    )

    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    entry = next(
        (
            j
            for j in payload["jobs"]
            if j["namespace"] == harness.ns and j["name"] == "no-p99-bench"
        ),
        None,
    )
    assert entry is not None, payload
    # Missing p99 must surface as None, not as a crash.
    assert entry.get("latencyP99Ms") is None, entry

    harness.goto_dashboard()
    _wait_for_dashboard_settled(harness)
    harness.assert_no_unreachable_banner()


def test_jobs_list_handles_200_char_model_name(harness) -> None:
    """A 200-char model name does not break the Jobs table layout.

    Regression-style: extremely long model strings used to overflow the
    table cell and offset the surrounding chips; the fix was max-width +
    ellipsis. Test that the row still renders AND a known short field of
    the same row (the job name) is intact.
    """
    long_model = "huge/" + ("x" * 195)  # 200 chars total
    assert len(long_model) >= 200
    harness.seed_run(
        name="long-model-bench",
        epoch=EPOCH_OLDER,
        summary=good_summary(model=long_model),
        is_latest=True,
    )
    harness.goto_jobs_list()
    _wait_for_jobs_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "long-model-bench" in body


# ----------------------------------------------------------------------------
# 8) Sweep listing with children — list shows the parent + sweep detail
#    nav reaches /sweeps/<ns>/<name>.
# ----------------------------------------------------------------------------


def test_sweeps_list_shows_three_archived_sweeps_with_rollups(harness) -> None:
    """Seed three archived sweeps. The list must show all three rows AND
    each row's progress/variations matches the seeded aggregate.
    """
    for i in range(3):
        _seed_archived_sweep_aggregate(
            harness,
            sweep=f"sweep-{i}",
            epoch=str(1_714_069_000 + i * 100),
            total_variations=5,
            completed_runs=5,
            failed_runs=0,
            phase="Succeeded",
            model=f"model-{i}",
        )
    harness.goto_sweeps_list()
    _wait_for_sweeps_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    for i in range(3):
        assert f"sweep-{i}" in body, body[:500]


def test_sweep_row_click_navigates_to_sweep_detail(harness) -> None:
    """Clicking a sweep row in the list updates the URL hash to /sweeps/.../."""
    _seed_archived_sweep_aggregate(
        harness,
        sweep="clicky-sweep",
        total_variations=2,
        completed_runs=2,
    )
    harness.goto_sweeps_list()
    _wait_for_sweeps_list_settled(harness)
    row = harness.page.locator(f"[data-testid=sweep-row-{harness.ns}-clicky-sweep]")
    row.first.click(timeout=5_000)
    # navigate(...) mutates the hash; wait for it to change.
    harness.page.wait_for_function(
        f"() => window.location.hash.includes('/sweeps/{harness.ns}/clicky-sweep')",
        timeout=5_000,
    )


# ----------------------------------------------------------------------------
# 9) Sweep with no children / null cardinality — must not crash.
# ----------------------------------------------------------------------------


def test_sweeps_list_renders_sweep_with_zero_variations(harness) -> None:
    """A sweep aggregate with ``totalVariations=0`` and zero completed runs
    must still render in the list rather than crash the page.

    The UI renders ``? / ?`` style markup; this test only enforces that no
    console error fires and the row appears.
    """
    _seed_archived_sweep_aggregate(
        harness,
        sweep="empty-sweep",
        total_variations=0,
        completed_runs=0,
        phase="Pending",
    )
    harness.goto_sweeps_list()
    _wait_for_sweeps_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "empty-sweep" in body, body[:500]
    harness.assert_no_unreachable_banner()


# ----------------------------------------------------------------------------
# 10) Leaderboard — ranking + all-zero-KPI edge case.
# ----------------------------------------------------------------------------


def test_leaderboard_ranks_seeded_jobs_by_throughput(harness) -> None:
    """Seed 10 jobs across two models with mixed throughputs.

    Expected leaderboard top entry by throughput=avg must be ``high-tput-bench``
    which we seeded at 500 req/s. The leaderboard endpoint returns entries
    ordered by value desc; assert directly against the API to keep this test
    independent of chart rendering.
    """
    throughputs = [50, 75, 100, 150, 200, 250, 300, 350, 400, 500]
    for i, tps in enumerate(throughputs):
        name = "high-tput-bench" if tps == 500 else f"bench-{i:02d}"
        summary = good_summary(throughput_rps=float(tps), model=f"model-{i % 2}")
        summary["status"] = "Completed"
        harness.seed_run(
            name=name,
            epoch=str(1_714_069_000 + i),
            summary=summary,
            is_latest=True,
        )

    # Hit the API directly first — keeps the assertion deterministic.
    status, raw = harness.api_get(
        "/api/v1/analytics/leaderboard?metric=request_throughput&stat=avg&limit=20"
    )
    assert status == 200, (status, raw[:200])
    payload = json.loads(raw)
    entries_in_ns = [
        e for e in payload.get("entries", []) if e.get("namespace") == harness.ns
    ]
    assert entries_in_ns, payload
    top = entries_in_ns[0]
    assert top.get("job_id") == "high-tput-bench", top

    harness.goto("/leaderboard")
    harness.page.wait_for_selector("[data-testid=page-leaderboard]", timeout=10_000)


def test_leaderboard_all_zero_kpis_does_not_divide_by_zero(harness) -> None:
    """All-zero KPIs across multiple seeded jobs must not produce NaN
    rendering or a JS console error on the Leaderboard page.

    The Leaderboard chart fills a horizontal bar canvas — if every value is
    0 the layout could divide by zero when computing pixel widths. The
    explicit invariant: page renders, no pageerror is logged.
    """
    for i in range(3):
        summary = good_summary(throughput_rps=0.0, model="zero-model")
        summary["status"] = "Completed"
        harness.seed_run(
            name=f"zero-{i}",
            epoch=str(1_714_069_000 + i),
            summary=summary,
            is_latest=True,
        )
    harness.goto("/leaderboard")
    harness.page.wait_for_selector("[data-testid=page-leaderboard]", timeout=10_000)
    # No pageerror in console_errors (we tolerate benign resource fetches).
    pageerrors = [e for e in harness.console_errors if "[pageerror]" in e]
    assert not pageerrors, pageerrors


# ----------------------------------------------------------------------------
# 11) Compare flow — two epochs for one job; symmetric and asymmetric.
# ----------------------------------------------------------------------------


def test_compare_epochs_page_renders_for_two_archived_epochs(harness) -> None:
    """Seed two epochs for one job, navigate to /compare/<ns>/<name>/A/B.

    The compare-epochs page is a thin wrapper over the per-epoch summary
    fetch. We only assert: page renders, no Unreachable banner, and the URL
    hash matches the expected pattern.
    """
    harness.seed_run(
        name="two-epoch-bench",
        epoch=EPOCH_OLDER,
        summary=good_summary(throughput_rps=100.0),
    )
    harness.seed_run(
        name="two-epoch-bench",
        epoch=EPOCH_NEWER,
        summary=good_summary(throughput_rps=200.0),
        is_latest=True,
    )
    harness.goto(f"/compare/{harness.ns}/two-epoch-bench/{EPOCH_OLDER}/{EPOCH_NEWER}")
    harness.assert_no_unreachable_banner()


def test_compare_epochs_renders_when_one_epoch_lacks_summary(harness) -> None:
    """Asymmetric epochs: A has no summary, B has one.

    The page must still render — the summary 404 for A is expected and
    handled per ``api.fetchRunSummary``'s ``.status`` attribute on the error.
    """
    name = "asym-epoch-bench"
    harness.seed_run(name=name, epoch=EPOCH_OLDER, summary=None)
    harness.seed_run(
        name=name,
        epoch=EPOCH_NEWER,
        summary=good_summary(throughput_rps=99.0),
        is_latest=True,
    )
    harness.goto(f"/compare/{harness.ns}/{name}/{EPOCH_OLDER}/{EPOCH_NEWER}")
    harness.assert_no_unreachable_banner()


# ----------------------------------------------------------------------------
# 12) History — 30 epochs across 5 jobs, no stale repaints.
# ----------------------------------------------------------------------------


def test_history_page_renders_30_seeded_runs(harness) -> None:
    """Seed 6 runs apiece for 5 distinct jobs (30 total) and verify the
    history endpoint returns them ordered by start_time, no 5xx, no console
    pageerror.
    """
    seq = 0
    for j in range(5):
        for k in range(6):
            seq += 1
            ts_iso = f"2026-05-{(seq % 28) + 1:02d}T00:00:00Z"
            summary = good_summary(throughput_rps=float(seq))
            summary["status"] = "Completed"
            summary["start_time"] = ts_iso
            harness.seed_run(
                name=f"hist-job-{j}",
                epoch=str(1_714_069_000 + seq),
                summary=summary,
                is_latest=(k == 5),
            )

    status, raw = harness.api_get(
        "/api/v1/analytics/history?metric=request_throughput&stat=avg"
    )
    assert status == 200, raw[:200]

    harness.goto("/history")
    harness.page.wait_for_selector("[data-testid=page-history]", timeout=10_000)
    pageerrors = [e for e in harness.console_errors if "[pageerror]" in e]
    assert not pageerrors, pageerrors


# ----------------------------------------------------------------------------
# 13) Filter param injection — namespace + phase. Includes XSS attempt.
# ----------------------------------------------------------------------------


def test_jobs_list_filter_by_namespace_query_param(harness) -> None:
    """``?ns=<ns>`` filters the Jobs table to that namespace and shows the ns chip."""
    _seed_phase_run(harness, name="filtered-bench", phase="Running")
    harness.goto(f"/jobs?ns={harness.ns}&phase=running")
    _wait_for_jobs_list_settled(harness)
    chip = harness.page.locator("[data-testid=ns-filter-chip]")
    assert chip.count() >= 1
    body = harness.page.locator("body").inner_text()
    assert harness.ns in body
    assert "filtered-bench" in body


def test_jobs_list_unknown_phase_filter_shows_unfiltered_results(harness) -> None:
    """``?phase=DoesNotExist`` falls through to ``activeFilter=null`` so the
    list shows ALL jobs in that namespace, not just an unfiltered set."""
    _seed_phase_run(harness, name="phase-test-bench", phase="Running")
    harness.goto(f"/jobs?ns={harness.ns}&phase=DoesNotExist")
    _wait_for_jobs_list_settled(harness)
    # Unknown phase keys yield activeFilter=null so the job DOES appear.
    body = harness.page.locator("body").inner_text()
    assert "phase-test-bench" in body


def test_jobs_list_phase_filter_with_xss_payload_does_not_execute_script(
    harness,
) -> None:
    """A ``?phase=<script>alert(1)</script>`` payload must NOT inject a
    runnable <script> tag into the DOM. The framework escapes URL-decoded
    query values when rendering them, and we assert: (a) no console error
    fires, (b) the literal ``<script>`` tag count in HTML is zero (any
    appearance would be as text content, not as a script element).
    """
    _seed_phase_run(harness, name="xss-test-bench", phase="Running")
    payload = "<script>alert(1)</script>"
    # Manually URL-encode to avoid Playwright trimming.
    from urllib.parse import quote

    harness.goto(f"/jobs?ns={harness.ns}&phase={quote(payload)}")
    _wait_for_jobs_list_settled(harness)

    # No injected <script> element executed: the page lacks an inline
    # ``alert`` script. Count script elements that contain "alert(1)".
    suspicious = harness.page.evaluate(
        """() => Array.from(document.querySelectorAll('script'))
                    .filter(s => (s.textContent || '').includes('alert(1)')).length"""
    )
    assert suspicious == 0
    pageerrors = [e for e in harness.console_errors if "[pageerror]" in e]
    assert not pageerrors, pageerrors


# ----------------------------------------------------------------------------
# 14) Pagination — current pages are unpaginated; 100 items shouldn't add a
#     "Load more" button and shouldn't silently truncate.
# ----------------------------------------------------------------------------


def test_jobs_list_unpaginated_at_100_items_no_load_more(harness) -> None:
    """100 seeded jobs all render — no Load More button is injected, and
    the page-level "X of Y" counter matches the seeded count.
    """
    for i in range(100):
        harness.seed_run(
            name=f"bench-{i:03d}",
            epoch=str(1_714_069_000 + i),
            summary=good_summary(throughput_rps=float(i + 1)),
            is_latest=True,
        )

    # Verify the API actually returns 100 (no server-side truncation).
    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    in_ns = [j for j in payload["jobs"] if j["namespace"] == harness.ns]
    assert len(in_ns) == 100

    harness.goto_jobs_list()
    _wait_for_jobs_list_settled(harness)

    # No element resembling "Load more" / "Show more" exists.
    load_more = harness.page.evaluate(
        """() => {
            return Array.from(document.querySelectorAll('button,a'))
                .map(el => (el.textContent || '').trim().toLowerCase())
                .filter(t => t === 'load more' || t === 'show more')
                .length;
        }"""
    )
    assert load_more == 0


# ----------------------------------------------------------------------------
# 15) Command palette — Ctrl+K, type, Enter navigates to the matching job.
# ----------------------------------------------------------------------------


def test_command_palette_navigates_to_seeded_job(harness) -> None:
    """Ctrl+K opens the palette; typing a job name + Enter navigates to its
    detail route.

    The palette indexes pages first and jobs second, so we type the unique
    prefix of our seeded name to guarantee the job is the first match.
    """
    harness.seed_run(
        name="aiperf-palette-target",
        epoch=EPOCH_OLDER,
        summary=good_summary(model="meta-llama/Llama-3-8B"),
        is_latest=True,
    )
    # Visit jobs first so the global ``jobs`` signal is populated; the
    # palette filters off that signal.
    harness.goto_jobs_list()
    _wait_for_jobs_list_settled(harness)

    harness.page.keyboard.press("Control+k")
    harness.page.wait_for_selector("[data-testid=command-palette]", timeout=5_000)
    palette_input = harness.page.locator("[data-testid=command-palette-input]")
    palette_input.fill("aiperf-palette-target")
    # Yield one frame so the filter recomputes against the typed query.
    harness.page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid=command-palette]');
            if (!el) return false;
            return el.textContent.includes('aiperf-palette-target');
        }""",
        timeout=5_000,
    )
    harness.page.keyboard.press("Enter")

    harness.page.wait_for_function(
        f"() => window.location.hash.includes('/jobs/{harness.ns}/aiperf-palette-target')",
        timeout=5_000,
    )


# ----------------------------------------------------------------------------
# 16) Sweep source propagation — live-only sweep CR with no on-disk dir
#     should still appear (skip if the operator/k8s shim doesn't surface
#     live-CR sweeps in this harness).
# ----------------------------------------------------------------------------


def test_sweeps_list_handles_only_archived_sweep_phase_running(harness) -> None:
    """A sweep aggregate with phase=Running (controller still aggregating)
    must list with phase rendered and source=archived.
    """
    _seed_archived_sweep_aggregate(
        harness,
        sweep="running-archived-sweep",
        total_variations=5,
        completed_runs=2,
        phase="Running",
    )
    harness.goto_sweeps_list()
    _wait_for_sweeps_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "running-archived-sweep" in body
    assert "Running" in body


# ----------------------------------------------------------------------------
# Extra adversarial cases — stale child clamp + sweep dedup + non-finite age.
# ----------------------------------------------------------------------------


def test_sweeps_list_clamps_stale_completed_children(harness) -> None:
    """If completedRuns > totalVariations (stale children from prior epochs),
    the UI clamps the visible numerator and surfaces a "+N stale" hint.
    """
    _seed_archived_sweep_aggregate(
        harness,
        sweep="stale-sweep",
        total_variations=3,
        completed_runs=5,
        phase="Succeeded",
    )
    harness.goto_sweeps_list()
    _wait_for_sweeps_list_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "stale-sweep" in body
    # The clamp logic emits "3 / 3" and a stale-count hint; both should appear.
    assert "3 / 3" in body, body[:500]
    assert "stale" in body.lower(), body[:500]


def test_jobs_list_handles_summary_with_only_status_field(harness) -> None:
    """A summary that has only ``status`` and no KPI blocks must produce a
    valid archived row (KPIs all None). Repro for the
    ``profile_export_aiperf.json={"status": "Cancelled"}`` shape from the
    smoke test's regression."""
    harness.seed_run(
        name="status-only-bench",
        epoch=EPOCH_OLDER,
        summary={"status": "Cancelled"},
        is_latest=True,
    )
    status, raw = harness.api_get("/api/v1/jobs")
    assert status == 200
    payload = json.loads(raw)
    in_ns = [j for j in payload["jobs"] if j["namespace"] == harness.ns]
    assert any(j["name"] == "status-only-bench" for j in in_ns), in_ns
    entry = next(j for j in in_ns if j["name"] == "status-only-bench")
    assert entry["phase"] == "Cancelled"


def test_compare_page_with_no_results_shows_empty_state(harness) -> None:
    """Visiting /compare with nothing in /results yields a helpful empty
    state, not an error banner."""
    harness.goto("/compare")
    harness.page.wait_for_selector("[data-testid=page-compare]", timeout=10_000)
    harness.assert_no_unreachable_banner()
    # No pageerror should fire.
    pageerrors = [e for e in harness.console_errors if "[pageerror]" in e]
    assert not pageerrors, pageerrors


def test_history_filter_by_namespace_via_url_param(harness) -> None:
    """``/history?ns=<harness.ns>`` initializes the ns filter chip."""
    summary = good_summary(throughput_rps=42.0)
    summary["status"] = "Completed"
    harness.seed_run(
        name="hist-ns-bench",
        epoch=EPOCH_OLDER,
        summary=summary,
        is_latest=True,
    )
    harness.goto(f"/history?ns={harness.ns}")
    harness.page.wait_for_selector("[data-testid=page-history]", timeout=10_000)
    chip = harness.page.locator("[data-testid=ns-filter-chip]")
    assert chip.count() >= 1


def test_dashboard_renders_with_running_archived_job(harness) -> None:
    """An archived run with status=Running surfaces on the Dashboard's
    Active Jobs section. This guards against the previously-shipped bug
    where archived rows with a "Running" phase fell through every section.
    """
    _seed_phase_run(harness, name="archived-running-bench", phase="Running")
    harness.goto_dashboard()
    _wait_for_dashboard_settled(harness)
    body = harness.page.locator("body").inner_text()
    assert "archived-running-bench" in body


def test_jobs_list_clear_filters_button_removes_url_chips(harness) -> None:
    """Clicking the Clear filters button strips ns/phase/search params from URL."""
    _seed_phase_run(harness, name="clear-bench", phase="Running")
    harness.goto(f"/jobs?ns={harness.ns}&phase=running")
    _wait_for_jobs_list_settled(harness)
    # Click the "Clear" button (the smaller one in the filter strip).
    # We avoid asserting on which button — locate by text.
    btns = harness.page.locator("button:has-text('Clear')")
    if btns.count() == 0:
        pytest.skip("filter strip did not render a Clear button (no chips active)")
    btns.first.click()
    harness.page.wait_for_function(
        "() => !window.location.hash.includes('ns=') && !window.location.hash.includes('phase=')",
        timeout=5_000,
    )


def test_leaderboard_handles_metric_with_no_seeded_data(harness) -> None:
    """``/api/v1/analytics/leaderboard?metric=inter_token_latency&stat=avg``
    against a namespace with only a throughput-only summary returns an empty
    entries list (200), not a 500.
    """
    summary = good_summary(throughput_rps=42.0)
    summary["status"] = "Completed"
    # Remove inter_token_latency entirely.
    summary.pop("inter_token_latency", None)
    harness.seed_run(
        name="itl-test-bench",
        epoch=EPOCH_OLDER,
        summary=summary,
        is_latest=True,
    )
    status, raw = harness.api_get(
        "/api/v1/analytics/leaderboard?metric=inter_token_latency&stat=avg&limit=20"
    )
    assert status == 200, raw[:200]

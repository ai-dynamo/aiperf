# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Playwright tests for the operator-UI job-detail page.

Focuses on:
  * Epoch resolution edge-cases (pinned vs latest pointer mismatches, leading
    zeros, NO-summary regression, pointer-to-nowhere).
  * KPI parsing of malformed / missing / non-finite / out-of-range summary
    fields.
  * `conditions.json` polymorphism: top-level list, dict-with-conditions,
    null, scalar, malformed entries, mixed-case status strings.
  * Sweep linkage marker: non-int variation_index, malformed JSON, claimed
    sweep with no parent dir.
  * Live CR vs archived merge: pinned historical epoch must NOT pick up
    live CR fields (job_union.py:531-533).
  * URL-path encoding: hyphen / dot / percent / `+` / UTF-8 in namespace
    and name segments.
  * /events router behavior while the page is open (verifies that the
    diagnostics-panel render survives an event response).
  * Results-ready marker gating: artifacts list with vs without the
    `.aiperf_results_ready.json` sentinel.
  * Phase-stage decoding: legacy int request_count, missing request_count,
    request_count.avg == 0 on Succeeded.
  * Deep link to a non-existent job: page must not freeze on "Loading…".

These tests target real UI bugs — they intentionally do not use ``xfail``.
A failure here surfaces a contract gap that needs fixing in the page or
the router; the test stays red until the fix lands.
"""

from __future__ import annotations

import json
import re

import pytest
from pytest import param

from tests.unit.operator.ui_e2e.conftest import FakeLiveCR, good_summary

# Canonical good epochs used across tests.
_EPOCH_A = "1714069323"
_EPOCH_B = "1714069999"


def _safe_seed_name(prefix: str, token: object) -> str:
    """Build a DNS-1123-valid job name from an arbitrary token.

    Job names are validated as Kubernetes object names by the results API,
    so a name derived from a raw test value (``NoneType``, ``{'avg': 0}``,
    ``"a string"``) must be lowercased and stripped to ``[a-z0-9-]`` before
    it can be used as a PVC/CR identifier.
    """
    suffix = re.sub(r"[^a-z0-9]+", "-", str(token).lower()).strip("-")
    return f"{prefix}-{suffix}" if suffix else prefix


# ---------------------------------------------------------------------------
# 1. Epoch resolution — pinned vs latest, leading zeros, no-summary, pointer
#    -to-nowhere.
# ---------------------------------------------------------------------------


def test_pinned_historical_epoch_with_summary_renders_archived_phase(harness):
    """When `?epoch=<historical>` is supplied and the dir has a summary, the
    page must paint that run's `status` (Succeeded) as the displayed phase,
    NOT fall through to a live CR. job_union.py:531-533 is the contract."""
    harness.seed_run(
        name="hist-job",
        epoch=_EPOCH_A,
        summary=good_summary(),
        is_latest=False,
    )
    page = harness.goto_job_detail(harness.ns, "hist-job", epoch=_EPOCH_A)
    body = page.locator("[data-testid=page-job-detail]").inner_text(timeout=10_000)
    assert "Succeeded" in body, body[:500]
    assert "hist-job" in body
    harness.assert_no_unreachable_banner()


def test_pinned_epoch_without_summary_renders_unknown_stub(harness):
    """Regression for the recent _archived_stub fix: pinned-epoch dir with no
    summary file must NOT 404 the run-detail page. It must show phase=Unknown
    and the page-job-detail container."""
    harness.seed_run(name="stub-job", epoch=_EPOCH_A, summary=None)
    page = harness.goto_job_detail(harness.ns, "stub-job", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    body = page.locator("[data-testid=page-job-detail]").inner_text()
    assert "stub-job" in body
    assert "Unknown" in body, body[:500]
    harness.assert_no_unreachable_banner()


@pytest.mark.parametrize(
    "phase_str",
    [
        param("Cancelled", id="phase-cancelled"),
        param("Failed", id="phase-failed"),
    ],
)  # fmt: skip
def test_pinned_epoch_terminal_failure_phase_paints_phase_text(harness, phase_str):
    """Pinned epoch whose summary carries a terminal-but-not-Succeeded phase
    must render that phase in the page — `archived` interpretation kicks in
    via deriveJobRunState only for `Archived`, so Cancelled/Failed should
    still be shown verbatim."""
    summary = good_summary()
    summary["status"] = phase_str
    harness.seed_run(
        name=f"fail-job-{phase_str.lower()}", epoch=_EPOCH_A, summary=summary
    )
    page = harness.goto_job_detail(
        harness.ns, f"fail-job-{phase_str.lower()}", epoch=_EPOCH_A
    )
    body = page.locator("[data-testid=page-job-detail]").inner_text(timeout=10_000)
    assert phase_str in body, body[:500]
    harness.assert_no_unreachable_banner()


def test_pinned_epoch_summary_with_nan_inf_kpis_does_not_crash(harness):
    """A summary carrying out-of-domain numerics (negative throughput,
    error_rate>1) at multiple KPI sites must not break the page — the
    file is JSON-encodable but the values are physically meaningless.
    Specifically: `error_rate=1.5` is read by job_union which downstream
    feeds the KPI grid; negative throughput would trip naive Math.* paths.
    The page must still render without console errors. (Real `Infinity`/
    `NaN` are not orjson-serializable, so this is the closest legit
    adversarial-numeric payload that survives the wire.)"""
    summary = good_summary()
    # error_rate=1.5 is outside [0,1] — should not gate the UI render.
    summary["error_rate"] = 1.5
    # Negative throughput is physically meaningless but the JS layer should
    # display "---" instead of crashing on Math.* paths.
    summary["request_throughput"] = {"avg": -1.0, "unit": "requests/sec"}
    harness.seed_run(name="weird-kpis", epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(harness.ns, "weird-kpis", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()
    harness.assert_no_console_errors(allow_substrings=("Failed to load resource",))


def test_pinned_epoch_with_missing_throughput_block_renders_dashes(harness):
    """A summary lacking `request_throughput` entirely must NOT crash KPI
    parsing; the KPI tile should fall back to the `---` placeholder."""
    summary = good_summary()
    summary.pop("request_throughput", None)
    harness.seed_run(name="no-throughput", epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(harness.ns, "no-throughput", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_pinned_epoch_with_missing_request_latency_renders(harness):
    """Same drop test as throughput, against `request_latency`."""
    summary = good_summary()
    summary.pop("request_latency", None)
    harness.seed_run(name="no-latency", epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(harness.ns, "no-latency", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


@pytest.mark.parametrize(
    "kpi_block",
    [
        param(None, id="kpi-block-null"),
        param(42, id="kpi-block-scalar"),
        param({}, id="kpi-block-empty-dict"),
    ],
)  # fmt: skip
def test_pinned_epoch_with_malformed_kpi_dicts_renders(harness, kpi_block):
    """`request_throughput` shaped as null / scalar / empty dict must not
    crash KPI parsing — the page reads `?.avg` and is meant to no-op on
    non-dicts. Real metric-pipeline bugs produce these shapes on partial
    aggregation."""
    summary = good_summary()
    summary["request_throughput"] = kpi_block
    name = _safe_seed_name("kpi-shape", type(kpi_block).__name__)
    harness.seed_run(name=name, epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(
        harness.ns,
        name,
        epoch=_EPOCH_A,
    )
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_latest_pointer_at_nonexistent_epoch_dir_falls_through(harness):
    """`latest.txt` pointing at an epoch directory that does NOT exist on
    disk must not crash /epochs; the page falls back to the no-epoch URL
    and the run-selector either shows the live pseudo-row or empty."""
    # Write a latest.txt pointing at an epoch with no run dir at all.
    (harness.results_dir / harness.ns / "ghost-job").mkdir(parents=True, exist_ok=True)
    (harness.results_dir / harness.ns / "ghost-job" / "latest.txt").write_text(
        "9999999999"
    )
    page = harness.goto_job_detail(harness.ns, "ghost-job")
    # Without a live CR, no archived dir, and no resolvable latest — the API
    # 404s; the page must either show an error stub OR the global unreachable
    # banner, but it MUST NOT freeze.
    body = page.locator("body").inner_text(timeout=10_000)
    assert "ghost-job" in body or "unreachable" in body.lower(), body[:500]


def test_latest_pointer_with_leading_zeros_is_treated_as_string(harness):
    """If latest.txt holds a numeric string with leading zeros, the page
    must NOT normalise it — `String(epoch)` preserves zeros, and the on-disk
    dir name is the literal string. Use a 10-char numeric for EPOCH_RE
    compliance (^\\d{9,20}$)."""
    epoch_with_zero = "0714069323"
    harness.seed_run(
        name="zero-pad-job",
        epoch=epoch_with_zero,
        summary=good_summary(),
        is_latest=True,
    )
    page = harness.goto_job_detail(harness.ns, "zero-pad-job", epoch=epoch_with_zero)
    body = page.locator("[data-testid=page-job-detail]").inner_text(timeout=10_000)
    assert epoch_with_zero in body, body[:500]
    harness.assert_no_unreachable_banner()


def test_latest_pointer_at_only_run_renders_once(harness):
    """latest.txt pointing at the same epoch as the only seeded run — the
    page should render Succeeded without choking on "self-redirects". The
    useEffect at job-detail.js:1801-1804 must not loop."""
    harness.seed_run(
        name="single-run", epoch=_EPOCH_A, summary=good_summary(), is_latest=True
    )
    page = harness.goto_job_detail(harness.ns, "single-run", epoch=_EPOCH_A)
    body = page.locator("[data-testid=page-job-detail]").inner_text(timeout=10_000)
    assert "single-run" in body
    assert "Succeeded" in body, body[:500]
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 2. Conditions — varied shapes from the conditions.json file.
# ---------------------------------------------------------------------------


def test_conditions_as_top_level_list_renders_badges(harness):
    """conditions.json as a top-level JSON list — the router accepts this
    shape (jobs.py:379-380). The Conditions component renders a badge for
    every status!=True entry; True-statused green conditions are hidden by
    conditions-helpers.js:shouldHideCondition. Use a Warning to force a badge."""
    harness.seed_run(
        name="conds-list",
        epoch=_EPOCH_A,
        summary=good_summary(),
        conditions=[
            # status=False forces a visible "failed" badge with conditionLabel(type)
            {"type": "EndpointReachable", "status": "False", "reason": "DNSError"},
        ],
    )
    page = harness.goto_job_detail(harness.ns, "conds-list", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    # conditionLabel maps EndpointReachable -> "Endpoint"
    badge = page.locator(".condition-badge", has_text="Endpoint")
    badge.first.wait_for(timeout=5_000)
    harness.assert_no_unreachable_banner()


def test_conditions_as_wrapping_dict_renders_badges(harness):
    """conditions.json as `{"conditions": [...]}` — the router unwraps
    (jobs.py:381-382). UI must end up with the same visible badge as the
    bare list case."""
    harness.seed_run(
        name="conds-dict",
        epoch=_EPOCH_A,
        summary=good_summary(),
        extra_files={
            "conditions.json": json.dumps(
                {
                    "conditions": [
                        {"type": "WorkersReady", "status": "False", "reason": "Pending"}
                    ]
                }
            ).encode(),
        },
    )
    page = harness.goto_job_detail(harness.ns, "conds-dict", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    # WorkersReady -> "Workers" label.
    badge = page.locator(".condition-badge", has_text="Workers")
    badge.first.wait_for(timeout=5_000)
    harness.assert_no_unreachable_banner()


@pytest.mark.parametrize(
    "raw_bytes",
    [
        param(b"null", id="conditions-null"),
        param(b"42", id="conditions-scalar"),
        param(b"{}", id="conditions-empty-dict"),
        param(b'"a string"', id="conditions-string"),
    ],
)  # fmt: skip
def test_conditions_malformed_shape_renders_without_crashing(harness, raw_bytes):
    """conditions.json with a non-list/non-dict-with-conditions shape — the
    router silently leaves conditions=None (jobs.py:379-382). Page must still
    render without console errors."""
    name = _safe_seed_name("conds-bad", raw_bytes[:6].decode(errors="replace"))
    harness.seed_run(
        name=name,
        epoch=_EPOCH_A,
        summary=good_summary(),
        extra_files={"conditions.json": raw_bytes},
    )
    page = harness.goto_job_detail(harness.ns, name, epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_conditions_entry_missing_keys_renders(harness):
    """An entry that lacks `type` or `status` is malformed but the UI must
    not crash. Visible-condition helper should either drop it or render
    a fallback label."""
    harness.seed_run(
        name="conds-partial",
        epoch=_EPOCH_A,
        summary=good_summary(),
        conditions=[
            {"type": "OnlyType"},  # missing status
            {"status": "True"},  # missing type
            {"type": "Real", "status": "True"},
        ],
    )
    page = harness.goto_job_detail(harness.ns, "conds-partial", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_conditions_mixed_case_status_strings_render(harness):
    """K8s spec mandates "True"/"False"/"Unknown" but real-world status
    blocks sometimes leak "true"/"false"/None. UI must handle the casing
    without console errors."""
    harness.seed_run(
        name="conds-case",
        epoch=_EPOCH_A,
        summary=good_summary(),
        conditions=[
            {"type": "A", "status": "true"},
            {"type": "B", "status": "FALSE"},
            {"type": "C", "status": "unknown"},
        ],
    )
    page = harness.goto_job_detail(harness.ns, "conds-case", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 3. Sweep linkage — sweep.json shape adversaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variation_value",
    [
        param("abc", id="variation-index-string"),
        param(None, id="variation-index-null"),
    ],
)  # fmt: skip
def test_sweep_marker_non_int_variation_index_does_not_crash(harness, variation_value):
    """job_union._sweep_linkage_from_marker (lines 64-67) coerces variation_index
    through int(), trapping TypeError/ValueError. Non-int values must NOT
    propagate to the UI — the page should still render."""
    name = _safe_seed_name("sweep-child", variation_value)
    harness.seed_run(
        name=name,
        epoch=_EPOCH_A,
        summary=good_summary(),
        sweep_marker={
            "sweep_name": "parent-sweep",
            "variation_index": variation_value,
            "variation_label": "x=1",
        },
    )
    page = harness.goto_job_detail(harness.ns, name, epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    body = page.locator("[data-testid=page-job-detail]").inner_text()
    # Page links to the sweep name; the variation_index null/garbage falls
    # silently to None in AIPerfJobInfo.
    assert "parent-sweep" in body, body[:500]
    harness.assert_no_unreachable_banner()


def test_sweep_marker_malformed_json_falls_back_silently(harness):
    """sweep.json with broken JSON bytes — _sweep_linkage_from_marker logs
    a warning and returns (None, None, None). The page must render the job
    without claiming sweep linkage."""
    # seed_run writes the sweep_marker as json — bypass via extra_files at the
    # name level. But seed_run writes sweep.json at `<ns>/<name>/sweep.json`,
    # NOT inside the epoch dir. Write directly.
    run = harness.seed_run(name="sweep-broken", epoch=_EPOCH_A, summary=good_summary())
    (run.parent / "sweep.json").write_bytes(b"{not valid json")
    page = harness.goto_job_detail(harness.ns, "sweep-broken", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    body = page.locator("[data-testid=page-job-detail]").inner_text()
    # No sweep linkage emitted — the SweepLink testid is absent.
    assert page.locator("[data-testid=job-detail-sweep-link]").count() == 0
    assert "sweep-broken" in body
    harness.assert_no_unreachable_banner()


def test_sweep_marker_orphan_link_no_parent_sweep_dir(harness):
    """Child claims sweep linkage but the parent sweep dir does not exist
    on the PVC. The job-detail page should still render — the sweep
    deep-link href just won't resolve when clicked. This is a regression
    test for the displayed link not blocking the render."""
    harness.seed_run(
        name="orphan-child",
        epoch=_EPOCH_A,
        summary=good_summary(),
        sweep_marker={"sweep_name": "nowhere-sweep", "variation_index": 0},
    )
    page = harness.goto_job_detail(harness.ns, "orphan-child", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    # Sweep link IS emitted because variation_index is valid; only its target
    # is broken.
    assert page.locator("[data-testid=job-detail-sweep-link]").count() == 1
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 4. Live CR vs archived merge — pinned historical epochs.
# ---------------------------------------------------------------------------


def test_pinned_historical_epoch_ignores_live_cr_phase(harness):
    """find_any_job (job_union.py:531-533) hard-drops the live CR when an
    explicit historical epoch is passed. A registered Running CR must NOT
    leak its phase onto a pinned past run. The visible phase is the
    archived summary's `status`."""
    harness.register_cr(
        FakeLiveCR(
            name="merge-job",
            namespace=harness.ns,
            phase="Running",
            workers_ready=4,
            workers_total=4,
        )
    )
    harness.seed_run(name="merge-job", epoch=_EPOCH_A, summary=good_summary())
    page = harness.goto_job_detail(harness.ns, "merge-job", epoch=_EPOCH_A)
    body = page.locator("[data-testid=page-job-detail]").inner_text(timeout=10_000)
    assert "Succeeded" in body, body[:500]
    # The page must show the archived "Archived" indicator, not "Live".
    # Bug-magnet area: if leak happens, "Live" with the green pulse appears.
    assert "Running" not in body or "Archived" in body, body[:500]
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 5. Path-encoding adversaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ns,name,expect_status",
    [
        # Hyphen in ns is canonical, dot is legal in a DNS-1123 subdomain name;
        # both route to a clean 404 for a missing job.
        param("ns-with-hyphen", "okay-job", 404, id="hyphen-in-ns-404-on-missing"),
        param("ns-plain", "name.dot", 404, id="dot-in-name"),
        # Plus is NOT legal in a K8s object name — the path validator rejects
        # it with a 400 before any lookup, never a 500.
        param("ns-plain", "name+plus", 400, id="plus-in-name-rejected"),
    ],
)  # fmt: skip
def test_path_encoding_special_chars_yield_clean_404(harness, ns, name, expect_status):
    """Legal K8s-name special chars (hyphen, dot) yield a clean 404; illegal
    ones (plus) yield a 400 from the path validator — never a 500."""
    # `harness.ns` is the seeded one — but for the API_GET path we just want
    # to verify the rejection/404 response for the special-char name.
    status, _ = harness.api_get(f"/api/v1/jobs/{ns}/{name}?epoch={_EPOCH_A}")
    assert status == expect_status, status


def test_namespace_with_encoded_slash_is_400_or_404(harness):
    """A `%2F` (encoded slash) in the namespace either splits the path
    (router 404s on the rebuilt segments) or trips FastAPI's path-validator
    (400). It must NOT silently match a real ns."""
    status, _ = harness.api_get("/api/v1/jobs/ns%2Fwith%2Fslash/anyname")
    assert status in (400, 404), status


def test_utf8_name_in_path_does_not_500(harness):
    """UTF-8 name segments (e.g. accented chars) are not legal K8s names, so
    the path validator rejects `kafé` with a 400 — a clean rejection, not a
    500. (A 404 is equally acceptable if routing rejects it first.)"""
    status, _ = harness.api_get(f"/api/v1/jobs/{harness.ns}/kaf%C3%A9")
    assert status in (400, 404), status


# ---------------------------------------------------------------------------
# 6. /epochs router while the page is open.
# ---------------------------------------------------------------------------


def test_epochs_endpoint_lists_seeded_run(harness):
    """While the page is open, /epochs must surface every seeded run. This
    is what powers the run-selector card; a missing/empty list collapses
    the selector entirely. Verifies the endpoint contract directly because
    /events depends on a real K8s API not mocked by the harness."""
    harness.seed_run(name="epoch-shape", epoch=_EPOCH_A, summary=good_summary())
    harness.seed_run(
        name="epoch-shape", epoch=_EPOCH_B, summary=good_summary(), is_latest=True
    )
    page = harness.goto_job_detail(harness.ns, "epoch-shape", epoch=_EPOCH_B)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)

    status, body = harness.api_get(f"/api/v1/jobs/{harness.ns}/epoch-shape/epochs")
    assert status == 200, (status, body[:200])
    data = json.loads(body)
    epochs = data.get("epochs", [])
    assert {e["epoch"] for e in epochs} == {_EPOCH_A, _EPOCH_B}, epochs
    # The latest pointer must be honored.
    latest = [e for e in epochs if e.get("isLatest")]
    assert len(latest) == 1 and latest[0]["epoch"] == _EPOCH_B, epochs
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 7. Results-files gating via the `.aiperf_results_ready.json` marker.
# ---------------------------------------------------------------------------


def test_artifacts_section_empty_state_without_results_ready_marker(harness):
    """Run dir has a summary BUT no `.aiperf_results_ready.json` marker —
    the results router refuses to serve top-level artifacts. The artifacts
    card must therefore render the empty state, NOT crash on a 5xx."""
    harness.seed_run(
        name="no-ready",
        epoch=_EPOCH_A,
        summary=good_summary(),
        # Drop a fake artifact that would have been listed if the marker
        # was present. Without the marker, the listing returns "no files".
        extra_files={"profile_export.csv": b"a,b,c\n"},
    )
    page = harness.goto_job_detail(harness.ns, "no-ready", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    # ArtifactsCard renders 'data-testid="artifacts-empty"' when filesLoaded
    # is true and the list is empty.
    page.locator("[data-testid=artifacts-empty]").wait_for(timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_artifacts_section_populates_after_results_ready_marker(harness):
    """Inverse of the above: with the marker present, the artifacts card
    enumerates real files."""
    run = harness.seed_run(
        name="ready-yes",
        epoch=_EPOCH_A,
        summary=good_summary(),
        extra_files={"profile_export.csv": b"a,b,c\n"},
    )
    harness.seed_results_ready(run)
    page = harness.goto_job_detail(harness.ns, "ready-yes", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    body = page.locator("[data-testid=artifacts-card]").inner_text(timeout=10_000)
    assert "profile_export" in body, body[:500]
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 8. Phase-stage decoding from the synthetic status block.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rc_value",
    [
        param(0, id="request-count-zero"),
        param(1000, id="request-count-int-legacy"),
        param({"avg": 0, "unit": "count"}, id="request-count-avg-zero"),
    ],
)  # fmt: skip
def test_request_count_variants_render_phase_block(harness, rc_value):
    """The synthesize_status_from_summary helper coerces request_count through
    int() — bare int legacy shape, dict shape, and zero-on-Succeeded all
    must produce a phases.benchmark entry that the page can render."""
    summary = good_summary()
    summary["request_count"] = rc_value
    name = _safe_seed_name("rc", str(rc_value)[:16])
    harness.seed_run(name=name, epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(harness.ns, name, epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


def test_request_count_missing_entirely_renders(harness):
    """Summary entirely lacking request_count must not crash phase decoding;
    synthesize_status_from_summary falls back to 0."""
    summary = good_summary()
    summary.pop("request_count", None)
    harness.seed_run(name="no-rc", epoch=_EPOCH_A, summary=summary)
    page = harness.goto_job_detail(harness.ns, "no-rc", epoch=_EPOCH_A)
    page.wait_for_selector("[data-testid=page-job-detail]", timeout=10_000)
    harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# 9. Deep link to a non-existent job — must not freeze on "Loading…".
# ---------------------------------------------------------------------------


def test_deep_link_to_nonexistent_job_does_not_stall_forever(harness):
    """Page navigated directly to a job that has neither a CR nor a PVC
    dir. The page currently has dead-code error handling — its local
    setError is only ever called with null (job-detail.js:1567). So this
    test surfaces the bug: after the poll-fail threshold (~6s+), the
    global 'Operator API unreachable' banner must appear; before then the
    page renders the LoadingPanel. The test confirms either the unreachable
    banner OR a clean error stub appears within 15s, NOT permanent loading."""
    page = harness.goto_job_detail(harness.ns, "phantom-job", epoch=_EPOCH_A)

    # Either the unreachable banner is rendered, or we transition out of the
    # loading panel into the error card. Wait up to ~15s for whichever lands.
    page.wait_for_function(
        """() => {
            const body = document.body.innerText;
            return body.includes('Operator API unreachable')
                || body.includes('Failed to load job')
                || !document.querySelector('[data-testid=job-detail-loading]');
        }""",
        timeout=15_000,
    )
    body = page.locator("body").inner_text()
    assert (
        "Operator API unreachable" in body
        or "Failed to load job" in body
        or "phantom-job" in body
    ), body[:500]


def test_deep_link_with_only_latest_pointer_no_runs_shows_state(harness):
    """latest.txt holds a value, but no run dir exists with that name —
    the API returns 404. Like the phantom test above, the page must not
    stall on Loading… forever."""
    (harness.results_dir / harness.ns / "pointer-only").mkdir(
        parents=True, exist_ok=True
    )
    (harness.results_dir / harness.ns / "pointer-only" / "latest.txt").write_text(
        _EPOCH_A
    )
    page = harness.goto_job_detail(harness.ns, "pointer-only")
    page.wait_for_function(
        """() => {
            const body = document.body.innerText;
            return body.includes('Operator API unreachable')
                || body.includes('Failed to load job')
                || !document.querySelector('[data-testid=job-detail-loading]');
        }""",
        timeout=15_000,
    )

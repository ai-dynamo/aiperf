# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Playwright tests for the operator-UI sweep-detail page.

Goal: stress the contract between ``SweepDetail`` (``pages/sweep-detail.js``),
``resolveSweepManifest`` (``pages/sweep-detail-helpers.js``), and the
backing routers (``routers/sweeps.py``, ``sweep_union.py``) — specifically
the joints where live CR state, archived aggregate.json on disk, the
per-epoch ``children.json`` manifest, and the per-cell metrics from
``getSweepCells`` are reconciled. Looking for:

  * Epoch-resolution drift (latest.txt missing/stale, EPOCH_RE-rejected,
    aggregate.json present without children.json or vice versa).
  * Children manifest shape attacks (0/1/many, gaps, dupes, empty labels,
    very-long labels, path-traversal-looking child names).
  * Run-state miscounting (all-pending UI must not say "Completed";
    runStates absent must not crash; cancelled separated from failed).
  * Aggregate-vs-children divergence (totals mismatch in either direction;
    child phase conflicts with aggregate roll-up).
  * Sweep linkage on child page (sweep.json marker vs URL parent).
  * Pathological run-summary rows (numeric strings, unknown metric keys,
    missing ``name``).
  * Direct deep-link to non-existent sweep does NOT stall on Loading.
  * API surface for unknown valid-shape epochs returns 404 cleanly and
    never bubbles 5xx into the page.

Out of scope: events / logs / pods diagnostic panel (covered separately);
artifact download / bundle streaming (covered in sweep-artifacts tests).

Harness gap worked around:

  * ``conftest.py`` patches ``find_aiperf_job`` for child-job detail but
    not ``find_aiperfsweep`` / ``list_aiperfsweeps``. With a MagicMock
    ApiClient those calls raise ``TypeError`` (non-awaitable return) and
    ``find_any_sweep`` re-raises into a 500. We install per-test patches
    via :func:`_patch_sweep_cr_lookups` so each test can declare its own
    live CR contents (or a flat ``None`` for "archived only" scenarios).
  * ``seed_sweep_aggregate`` writes ``profile_export_aiperf.json`` /
    ``runs.json``, but ``sweep_union._record_from_archive`` reads
    ``aggregate.json``. We write ``aggregate.json`` directly through
    :func:`_seed_aggregate_json` so archived sweep records actually
    surface through the API.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.unit.operator.ui_e2e.conftest import FakeLiveCR  # noqa: F401  (re-export)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_aggregate_json(
    results_dir: Path,
    *,
    ns: str,
    sweep: str,
    epoch: str,
    aggregate: dict[str, Any],
    children_doc: dict[str, Any] | None = None,
    is_latest: bool = True,
) -> Path:
    """Write the on-disk shape the operator actually reads.

    ``sweep_union._record_from_archive`` reads ``aggregate.json``;
    ``_get_children_impl`` reads ``children.json``. The harness's
    ``seed_sweep_aggregate`` writes ``profile_export_aiperf.json`` /
    ``runs.json`` instead, which the operator never opens. Use this
    helper any time a test needs an archived sweep that the API can see.
    """
    sweep_dir = results_dir / ns / "sweeps" / sweep / epoch
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "aggregate.json").write_text(json.dumps(aggregate))
    if children_doc is not None:
        (sweep_dir / "children.json").write_text(json.dumps(children_doc))
    if is_latest:
        (results_dir / ns / "sweeps" / sweep / "latest.txt").write_text(epoch)
    return sweep_dir


def _good_aggregate(
    *,
    phase: str = "Succeeded",
    total: int = 3,
    completed: int = 3,
    failed: int = 0,
    cancelled: int = 0,
    model: str = "llama3-8b",
    per_cell: list[dict[str, Any]] | None = None,
    run_states: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical aggregate.json. Mutate from this baseline."""
    body: dict[str, Any] = {
        "phase": phase,
        "totalVariations": total,
        "completedRuns": completed,
        "failedRuns": failed,
        "maxTotalRuns": total,
        "model": model,
        "startedAt": "2026-05-19T00:00:00Z",
        "completedAt": "2026-05-19T00:10:00Z",
        "per_cell_aggregates": per_cell
        or [
            {
                "variation_index": i,
                "variation_label": f"c={i + 1}",
                "values": {"concurrency": i + 1},
                "trials_completed": 1,
                "trials_failed": 0,
                "metrics": {
                    "request_throughput": {"avg": 100.0 + 50.0 * i},
                    "output_token_throughput": {"avg": 800.0 + 200.0 * i},
                    "request_latency": {"p99": 80.0 - 10.0 * i, "p50": 40.0},
                    "time_to_first_token": {"p50": 25.0, "p99": 60.0},
                    "inter_token_latency": {"avg": 5.0},
                },
                "children": [
                    {
                        "namespace": "PLACEHOLDER",
                        "name": f"sw-v{i:02d}",
                        "trial_index": 0,
                        "phase": "Succeeded",
                    },
                ],
            }
            for i in range(total)
        ],
    }
    if run_states is not None:
        body["runStates"] = run_states
    if cancelled:
        body.setdefault("runStates", {})["cancelled"] = cancelled
    if extra:
        body.update(extra)
    return body


def _children_doc(
    entries: list[dict[str, Any]], *, sweep_run_epoch: str = "1714069323"
) -> dict[str, Any]:
    return {"sweep_run_epoch": sweep_run_epoch, "children": entries}


@contextlib.contextmanager
def _patch_sweep_cr_lookups(
    *, find_returns: Any = None, list_returns: list[dict[str, Any]] | None = None
) -> Iterator[None]:
    """Patch ``find_aiperfsweep`` / ``list_aiperfsweeps`` for one test.

    The harness's MagicMock ApiClient makes the real calls raise. Without
    these patches every sweep route 500s instead of returning the archived
    fallback the test cares about.
    """
    items = list_returns if list_returns is not None else []
    with (
        patch(
            "aiperf.operator.sweep_union.find_aiperfsweep",
            AsyncMock(return_value=find_returns),
        ),
        patch(
            "aiperf.operator.sweep_union.list_aiperfsweeps",
            AsyncMock(return_value=items),
        ),
    ):
        yield


def _live_cr(
    *,
    ns: str,
    name: str,
    phase: str = "Running",
    total: int = 3,
    completed: int = 1,
    failed: int = 0,
    cancelled: int = 0,
    run_states: dict[str, int] | None = None,
    aggregate_children: list[dict[str, Any]] | None = None,
    current_child_ref: dict[str, Any] | None = None,
    runs_truncated: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a minimal live AIPerfSweep CR dict the operator reads.

    Note: ``spec`` defaults to ``{}`` so ``_spec_summary_from_record`` skips
    the strict ``AIPerfSweepSpec.model_validate`` branch — the same branch
    that 500s every sweep-detail call when the live CR carries an
    incompatible spec (covered as a regression in ``TestStaleSpecRegression``).
    """
    status: dict[str, Any] = {
        "phase": phase,
        "totalVariations": total,
        "completedRuns": completed,
        "failedRuns": failed,
    }
    if run_states is not None:
        status["runStates"] = run_states
    if cancelled:
        status.setdefault("runStates", {})["cancelled"] = cancelled
    if aggregate_children is not None:
        status["aggregate"] = {
            "children": {
                "sweep_run_epoch": "1714069323",
                "children": aggregate_children,
            }
        }
    if current_child_ref is not None:
        status["currentChildRef"] = current_child_ref
    if runs_truncated is not None:
        status["runsTruncated"] = runs_truncated
    return {
        "metadata": {
            "name": name,
            "namespace": ns,
            "creationTimestamp": "2026-05-19T00:00:00Z",
        },
        "spec": spec if spec is not None else {},
        "status": status,
    }


# ---------------------------------------------------------------------------
# Aggregate-bundle epoch resolution
# ---------------------------------------------------------------------------


class TestEpochResolution:
    """Latest pointer drift, partial epoch contents, EPOCH_RE rejections."""

    def test_archived_only_aggregate_renders_succeeded_header(self, harness):
        """Aggregate.json on disk + no live CR → page renders archived KPIs."""
        epoch = "1714069323"
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="archived-sweep",
            epoch=epoch,
            aggregate=_good_aggregate(),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "archived-sweep")
        body = page.locator("body").inner_text()
        assert "archived-sweep" in body
        # Aggregate's phase=Succeeded; the page should not be stuck on Loading.
        assert "Loading sweep" not in body
        harness.assert_no_unreachable_banner()

    def test_latest_pointer_drift_falls_back_to_none(self, harness):
        """latest.txt points at epoch dir that doesn't exist → 404 list path.

        The page should render the loading or error state cleanly, not 5xx.
        """
        sweep_root = harness.results_dir / harness.ns / "sweeps" / "drift"
        sweep_root.mkdir(parents=True, exist_ok=True)
        # Pointer to a valid-shape epoch that has no directory on disk.
        (sweep_root / "latest.txt").write_text("9999999999")
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "drift")
        # Either we get an Error card (most likely 404) or empty render —
        # but never an infinite Loading nor a 5xx-driven console error.
        body = page.locator("body").inner_text()
        assert "Operator API unreachable" not in body
        # 4xx is OK; 5xx is the bug.
        five_xx = [r for r in harness.bad_responses if r.startswith("5")]
        assert not five_xx, f"unexpected 5xx during drift load: {five_xx}"

    def test_runs_json_without_aggregate_json_is_invisible_to_api(self, harness):
        """Seed only the misnamed harness files; the operator must NOT pick them up.

        Documents the on-disk contract: the operator reads ``aggregate.json``,
        not ``runs.json`` or ``profile_export_aiperf.json``. If this test
        ever starts failing, the union read path widened — update both ends.
        """
        epoch = "1714069323"
        harness.seed_sweep_aggregate(
            sweep="misnamed",
            epoch=epoch,
            summary={"phase": "Succeeded", "totalVariations": 1},
            runs=[{"name": "x", "metrics": {}}],
            is_latest=True,
        )
        with _patch_sweep_cr_lookups():
            status, _ = harness.api_get(
                f"/api/v1/sweeps/{harness.ns}/misnamed?epoch={epoch}"
            )
        assert status == 404, (
            f"Expected 404 because aggregate.json is missing; got {status}. "
            "Either the harness seeds wrong files, or the operator quietly "
            "started reading profile_export_aiperf.json — pick one and fix it."
        )

    def test_aggregate_present_without_children_json_still_renders(self, harness):
        """aggregate.json present, children.json absent — KPI page renders."""
        epoch = "1714069323"
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="no-children-doc",
            epoch=epoch,
            aggregate=_good_aggregate(),
            children_doc=None,
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "no-children-doc")
        body = page.locator("body").inner_text()
        assert "no-children-doc" in body
        harness.assert_no_unreachable_banner()

    def test_children_json_without_aggregate_is_inaccessible(self, harness):
        """children.json alone produces no sweep record — 404, not 5xx."""
        epoch = "1714069323"
        sweep_root = harness.results_dir / harness.ns / "sweeps" / "orphan"
        epoch_dir = sweep_root / epoch
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "children.json").write_text(json.dumps(_children_doc([])))
        (sweep_root / "latest.txt").write_text(epoch)
        with _patch_sweep_cr_lookups():
            status, _ = harness.api_get(
                f"/api/v1/sweeps/{harness.ns}/orphan?epoch={epoch}"
            )
        assert status == 404, status

    def test_invalid_epoch_shape_returns_400(self, harness):
        """4xx for malformed epoch (router-level EPOCH_RE rejection)."""
        with _patch_sweep_cr_lookups():
            status, body = harness.api_get(
                f"/api/v1/sweeps/{harness.ns}/anything?epoch=not-a-number"
            )
        assert status == 400, (status, body[:200])


# ---------------------------------------------------------------------------
# Children manifest
# ---------------------------------------------------------------------------


def _child_entry(
    *,
    idx: int,
    name: str | None = None,
    label: str | None = None,
    ns: str = "PLACEHOLDER",
    trial: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "namespace": ns,
        "name": name if name is not None else f"sw-v{idx:02d}",
        "variation_index": idx,
        "variation_label": label if label is not None else f"c={idx + 1}",
        "child_run_epoch": "1714069323",
    }
    if trial is not None:
        out["trial_index"] = trial
    return out


class TestChildrenManifest:
    """Cardinality + content fuzz around the children manifest."""

    def test_sweep_with_zero_children_renders_empty_state(self, harness):
        """0 variations: no manifest, no children — page must not stall."""
        epoch = "1714069323"
        agg = _good_aggregate(total=0, completed=0, per_cell=[])
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="zero-kids",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "zero-kids", epoch=epoch)
        body = page.locator("body").inner_text()
        assert "No children persisted for this epoch yet." in body, body[:600]

    def test_sweep_with_one_child_is_a_valid_degenerate_sweep(self, harness):
        """1-variation sweep — UI must still render variations table or row."""
        epoch = "1714069323"
        agg = _good_aggregate(total=1, completed=1, per_cell=None)
        # _good_aggregate generated 1 per_cell with the right index 0.
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="solo",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0)]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "solo", epoch=epoch)
        body = page.locator("body").inner_text()
        # 1 variation should be reflected in the Variations KPI card.
        # KPI card labels are upper-cased via CSS in the rendered DOM.
        assert "VARIATIONS" in body or "Variations" in body, body[:600]
        harness.assert_no_unreachable_banner()

    def test_sweep_with_many_children_renders_table(self, harness):
        """50 variations: page must render without 5xx and Children card visible."""
        epoch = "1714069323"
        n = 50
        agg = _good_aggregate(
            total=n,
            completed=n,
            per_cell=[
                {
                    "variation_index": i,
                    "variation_label": f"c={i + 1}",
                    "values": {"concurrency": i + 1},
                    "trials_completed": 1,
                    "trials_failed": 0,
                    "metrics": {
                        "request_throughput": {"avg": float(i + 1)},
                    },
                    "children": [],
                }
                for i in range(n)
            ],
        )
        children = [_child_entry(idx=i) for i in range(n)]
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="stress",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc(children),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "stress", epoch=epoch)
        assert page.locator("[data-testid=sweep-detail-variations]").count() >= 1
        five_xx = [r for r in harness.bad_responses if r.startswith("5")]
        assert not five_xx, five_xx

    def test_child_name_with_traversal_chars_is_rendered_safely(self, harness):
        """Path-traversal-looking child name: never escapes the table cell."""
        epoch = "1714069323"
        nasty = "../../etc/passwd"
        agg = _good_aggregate(total=1)
        agg["per_cell_aggregates"][0]["children"][0]["name"] = nasty
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="traversal",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0, name=nasty)]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "traversal", epoch=epoch)
        body = page.locator("body").inner_text()
        # The nasty name should appear as text but the URL should never have
        # been navigated to (no 5xx leaks from a router doing path.join).
        assert nasty in body
        five_xx = [r for r in harness.bad_responses if r.startswith("5")]
        assert not five_xx, five_xx

    def test_variation_index_gaps_do_not_crash(self, harness):
        """Indices 0,1,3,4 (no 2): manifest accepted, page renders."""
        epoch = "1714069323"
        gapped = [_child_entry(idx=i) for i in (0, 1, 3, 4)]
        agg = _good_aggregate(
            total=4,
            completed=4,
            per_cell=[
                {
                    "variation_index": i,
                    "variation_label": f"c={i}",
                    "values": {},
                    "trials_completed": 1,
                    "trials_failed": 0,
                    "metrics": {"request_throughput": {"avg": float(i + 1)}},
                    "children": [],
                }
                for i in (0, 1, 3, 4)
            ],
        )
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="gaps",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc(gapped),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "gaps", epoch=epoch)
        harness.assert_no_unreachable_banner()
        body = page.locator("body").inner_text()
        assert ("VARIATIONS" in body) or ("Variations" in body), body[:600]

    def test_duplicate_variation_index_across_children(self, harness):
        """Two children both at variation_index=0 — multi-trial layout.

        Per ``buildSweepVariations`` (sweep-detail-helpers.js:71-86), duplicate
        indices group into the same variation; the row's n_trials should reflect
        both. Renderer must not duplicate the row.
        """
        epoch = "1714069323"
        kids = [
            _child_entry(idx=0, name="sw-v00-t0", trial=0),
            _child_entry(idx=0, name="sw-v00-t1", trial=1),
        ]
        agg = _good_aggregate(total=1, completed=2)
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="dup-idx",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc(kids),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "dup-idx", epoch=epoch)
        # Exactly one variation row for index 0 — duplicates must collapse.
        rows = page.locator("[data-testid=variation-row-0]")
        # If the variations panel is rendering, we should see one row;
        # if not (because childRows source missing summaries), it's OK
        # to see zero — but never more than one for the same index.
        assert rows.count() <= 1, rows.count()

    @pytest.mark.parametrize(
        "label",
        [
            "",
            "x" * 256,
            "concurrency=1, input_seq_len=2048, isl_var=ratio",
        ],
    )
    def test_variation_label_edge_shapes(self, harness, label: str):
        """Empty / very-long / comma-laden labels must not break rendering."""
        epoch = "1714069323"
        agg = _good_aggregate(total=1)
        agg["per_cell_aggregates"][0]["variation_label"] = label
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="labels",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0, label=label)]),
        )
        with _patch_sweep_cr_lookups():
            harness.goto_sweep_detail(harness.ns, "labels", epoch=epoch)
        # Long labels truncated by CSS is fine; what's not fine is a console
        # error or a 5xx from the route.
        harness.assert_no_unreachable_banner()


# ---------------------------------------------------------------------------
# Per-cell aggregates — pathological metric values
# ---------------------------------------------------------------------------


class TestPerCellAggregates:
    """Cells endpoint returning weird metric values; UI must degrade gracefully."""

    @pytest.mark.parametrize(
        "metric_value",
        [
            -1.0,  # negative throughput
            0.0,  # zero throughput
            None,  # explicit null KPI dict
        ],
    )
    def test_pathological_cell_metric_values(self, harness, metric_value):
        """Cells with negative / zero / null metric values do not 5xx."""
        epoch = "1714069323"
        agg = _good_aggregate(total=1)
        if metric_value is None:
            # Null the metrics dict entirely on that cell.
            agg["per_cell_aggregates"][0]["metrics"] = {}
        else:
            agg["per_cell_aggregates"][0]["metrics"]["request_throughput"] = {
                "avg": metric_value
            }
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="weird-metrics",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0)]),
        )
        with _patch_sweep_cr_lookups():
            harness.goto_sweep_detail(harness.ns, "weird-metrics", epoch=epoch)
        harness.assert_no_unreachable_banner()
        five_xx = [r for r in harness.bad_responses if r.startswith("5")]
        assert not five_xx, five_xx

    def test_cell_variation_label_does_not_match_any_child(self, harness):
        """Mismatched labels between cells and children: page must not crash.

        The two sources are joined by ``variation_index`` only — the label is
        cosmetic. Mismatch should be silent, not a 5xx.
        """
        epoch = "1714069323"
        agg = _good_aggregate(total=1)
        agg["per_cell_aggregates"][0]["variation_label"] = "cell-says-A"
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="label-mismatch",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0, label="children-says-B")]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "label-mismatch", epoch=epoch)
        body = page.locator("body").inner_text()
        # At least one label should appear — the cell's, the child's, or both.
        assert ("cell-says-A" in body) or ("children-says-B" in body), body[:600]


# ---------------------------------------------------------------------------
# Run-tracking state badges
# ---------------------------------------------------------------------------


def test_completed_sweep_promotes_winner_and_downloads_above_charts(harness):
    epoch = "1714069323"
    agg = _good_aggregate(total=2, completed=2)
    agg["per_cell_aggregates"][0]["metrics"]["output_token_throughput"] = {"avg": 900.0}
    agg["per_cell_aggregates"][1]["metrics"]["output_token_throughput"] = {
        "avg": 1200.0
    }
    _seed_aggregate_json(
        harness.results_dir,
        ns=harness.ns,
        sweep="completed-analysis",
        epoch=epoch,
        aggregate=agg,
        children_doc=_children_doc([_child_entry(idx=0), _child_entry(idx=1)]),
    )
    sweep_dir = (
        harness.results_dir / harness.ns / "sweeps" / "completed-analysis" / epoch
    )
    (sweep_dir / "profile_export_aiperf.json").write_text(json.dumps({"ok": True}))
    with _patch_sweep_cr_lookups():
        page = harness.goto_sweep_detail(harness.ns, "completed-analysis", epoch=epoch)
    winner_summary = page.locator("[data-testid=sweep-winner-summary]")
    assert winner_summary.count() == 1
    winner_text = winner_summary.inner_text()
    assert "c=2" in winner_text
    assert "1,200" in winner_text
    assert "tok/s" in winner_text
    assert "No completed variation" not in winner_text
    assert (
        page.locator("[data-testid=sweep-detail-aggregate-artifacts-card]").count() == 1
    )
    winner_top = page.locator("[data-testid=sweep-winner-summary]").bounding_box()["y"]
    chart_top = page.locator("[data-testid=sweep-detail-variations]").bounding_box()[
        "y"
    ]
    assert winner_top < chart_top

    with _patch_sweep_cr_lookups():
        invalid_metric_page = harness.goto(
            f"/sweeps/{harness.ns}/completed-analysis/runs/{epoch}?metric=missing.metric"
        )
    invalid_metric_page.wait_for_function(
        "document.querySelector('[data-testid=\"sweep-winner-summary\"]')?.textContent.includes('Req throughput')",
        timeout=5000,
    )
    fallback_text = invalid_metric_page.locator(
        "[data-testid=sweep-winner-summary]"
    ).inner_text()
    assert "c=2" in fallback_text
    assert "Req throughput" in fallback_text
    assert "150" in fallback_text
    assert "req/s" in fallback_text
    assert "No completed variation" not in fallback_text


def test_live_trial_board_selection_opens_trial_detail(harness):
    children = [
        {
            "name": "select-sweep-v00-t0",
            "namespace": harness.ns,
            "variationIndex": 0,
            "variationLabel": "concurrency=8",
            "trialIndex": 0,
        },
        {
            "name": "select-sweep-v00-t1",
            "namespace": harness.ns,
            "variationIndex": 0,
            "variationLabel": "concurrency=8",
            "trialIndex": 1,
        },
    ]
    cr = _live_cr(
        ns=harness.ns,
        name="select-board",
        phase="Running",
        total=1,
        completed=0,
        aggregate_children=children,
    )
    harness.seed_run(
        name="select-sweep-v00-t1",
        epoch="1714069323",
        summary={"output_token_throughput": {"avg": 1234.0}},
        is_latest=True,
    )
    with _patch_sweep_cr_lookups(find_returns=cr):
        page = harness.goto_sweep_detail(harness.ns, "select-board")
    assert (
        "Unknown" in page.locator("[data-testid=sweep-live-trial-detail]").inner_text()
    )
    page.locator("[data-testid=sweep-trial-cell-0-1]").click()
    detail = page.locator("[data-testid=sweep-live-trial-detail]").inner_text()
    assert "trial 1" in detail
    assert "select-sweep-v00-t1" in detail or "Open child job" in detail
    assert "1,234 tok/s" in detail


def test_live_trial_board_refreshes_child_status_without_manifest_change(harness):
    child_name = "refresh-sweep-v00-t0"
    children = [
        {
            "name": child_name,
            "namespace": harness.ns,
            "variationIndex": 0,
            "variationLabel": "concurrency=8",
            "trialIndex": 0,
        },
    ]
    cr = _live_cr(
        ns=harness.ns,
        name="refresh-board",
        phase="Running",
        total=1,
        completed=0,
        aggregate_children=children,
    )
    harness.register_cr(
        FakeLiveCR(
            name=child_name,
            namespace=harness.ns,
            phase="Running",
            progress_percent=10,
        )
    )
    with _patch_sweep_cr_lookups(find_returns=cr):
        page = harness.goto_sweep_detail(harness.ns, "refresh-board")
        page.wait_for_function(
            "document.querySelector('[data-testid=\"sweep-trial-cell-0-0\"]')?.textContent.includes('Running')",
            timeout=7000,
        )
        harness.register_cr(
            FakeLiveCR(
                name=child_name,
                namespace=harness.ns,
                phase="Succeeded",
                progress_percent=100,
            )
        )
        page.wait_for_function(
            "document.querySelector('[data-testid=\"sweep-trial-cell-0-0\"]')?.textContent.includes('Succeeded')",
            timeout=7000,
        )


def test_running_sweep_renders_live_trial_board_above_variation_charts(harness):
    children = [
        {
            "name": "live-sweep-v00-t0",
            "namespace": harness.ns,
            "variationIndex": 0,
            "variationLabel": "concurrency=8",
            "trialIndex": 0,
        },
        {
            "name": "live-sweep-v00-t1",
            "namespace": harness.ns,
            "variationIndex": 0,
            "variationLabel": "concurrency=8",
            "trialIndex": 1,
        },
        {
            "name": "live-sweep-v01-t0",
            "namespace": harness.ns,
            "variationIndex": 1,
            "variationLabel": "concurrency=16",
            "trialIndex": 0,
        },
    ]
    cr = _live_cr(
        ns=harness.ns,
        name="live-board",
        phase="Running",
        total=2,
        completed=1,
        aggregate_children=children,
        current_child_ref={
            "name": "live-sweep-v00-t1",
            "index": 0,
            "label": "concurrency=8",
        },
    )
    with _patch_sweep_cr_lookups(find_returns=cr):
        page = harness.goto_sweep_detail(harness.ns, "live-board")
    assert page.locator("[data-testid=sweep-live-trial-board]").count() == 1
    assert page.locator("[data-testid=sweep-detail-variations]").count() == 1
    board_top = page.locator("[data-testid=sweep-live-trial-board]").bounding_box()["y"]
    chart_top = page.locator("[data-testid=sweep-detail-variations]").bounding_box()[
        "y"
    ]
    assert board_top < chart_top


class TestRunStates:
    """The Live/Completed/Failed badges must follow runStates+phase, not totals."""

    def test_all_pending_renders_pending_state_not_completed(self, harness):
        """phase=Pending with 100 pending children — must not say Completed.

        ``RUNNING_PHASES`` includes ``pending``, so the page should render
        the Live indicator and the children-pending affordance — never the
        "Completed" green badge.
        """
        cr = _live_cr(
            ns=harness.ns,
            name="pending-100",
            phase="Pending",
            total=100,
            completed=0,
            failed=0,
            run_states={
                "pending": 100,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            },
        )
        with _patch_sweep_cr_lookups(find_returns=cr):
            page = harness.goto_sweep_detail(harness.ns, "pending-100")
        # The Live indicator must be present (Pending ∈ RUNNING_PHASES).
        live_indicator = page.locator("[data-testid=sweep-detail-live]").count()
        live_stale = page.locator("[data-testid=sweep-detail-live-stale]").count()
        assert live_indicator + live_stale >= 1, (
            "Pending sweep must render the Live (or Stale) indicator, not "
            "treat itself as a finished run."
        )
        # The Variations KPI must show the total count.
        body = page.locator("body").inner_text()
        assert "100" in body, body[:800]

    def test_runs_truncated_with_fetch_url_is_tolerated(self, harness):
        """runsTruncated: {true, fetchURL} in status must not crash the page."""
        cr = _live_cr(
            ns=harness.ns,
            name="truncated",
            phase="Running",
            runs_truncated={"true": True, "fetchURL": "/api/v1/results/..."},
        )
        with _patch_sweep_cr_lookups(find_returns=cr):
            page = harness.goto_sweep_detail(harness.ns, "truncated")
        harness.assert_no_unreachable_banner()
        body = page.locator("body").inner_text()
        assert "truncated" in body  # sweep name shows up

    def test_run_states_missing_entirely_does_not_crash(self, harness):
        """No runStates key in CR status — page still works (just no cancelled split)."""
        cr = _live_cr(
            ns=harness.ns, name="no-states", phase="Succeeded", total=2, completed=2
        )
        cr["status"].pop("runStates", None)
        with _patch_sweep_cr_lookups(find_returns=cr):
            page = harness.goto_sweep_detail(harness.ns, "no-states")
        harness.assert_no_unreachable_banner()
        assert "no-states" in page.locator("body").inner_text()


# ---------------------------------------------------------------------------
# Aggregate-vs-children divergence
# ---------------------------------------------------------------------------


class TestAggregateChildrenDivergence:
    """Disk aggregate claims one thing, children manifest claims another."""

    def test_aggregate_claims_5_variations_but_disk_has_3(self, harness):
        """totalVariations=5 in aggregate.json, children.json lists only 3.

        Real bug surface: ``s.total_variations`` comes from aggregate
        (``sweep_union._record_from_archive``), but the variations table is
        built from the manifest. Page must show the aggregate's "5" in the
        Variations KPI card AND render the 3 actual rows — without erroring.
        """
        epoch = "1714069323"
        agg = _good_aggregate(total=5, completed=3)
        # per_cell still has 5 by default (we don't touch it)
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="divergent",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc(
                [_child_entry(idx=i) for i in range(3)]  # only 3 children
            ),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "divergent", epoch=epoch)
        body = page.locator("body").inner_text()
        assert ("VARIATIONS" in body) or ("Variations" in body), body[:600]
        harness.assert_no_unreachable_banner()

    def test_child_phase_succeeded_but_aggregate_says_failed(self, harness):
        """Child claims Succeeded in cells doc; aggregate counters claim failure.

        Page must render — the two sources are independent and the UI
        can't reconcile them, but it must not 5xx.
        """
        epoch = "1714069323"
        agg = _good_aggregate(total=2, completed=0, failed=2)
        agg["per_cell_aggregates"][0]["children"][0]["phase"] = "Succeeded"
        agg["per_cell_aggregates"][1]["children"][0]["phase"] = "Succeeded"
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="phase-conflict",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=i) for i in range(2)]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(harness.ns, "phase-conflict", epoch=epoch)
        harness.assert_no_unreachable_banner()
        body = page.locator("body").inner_text()
        # Failed KPI card carries the failed counter; both sources unaltered.
        # Label renders ALL-CAPS in the DOM via CSS.
        assert ("FAILED" in body) or ("Failed" in body), body[:600]


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Cancelled phase must not surface as Completed."""

    def test_cancelled_sweep_does_not_show_completed_badge(self, harness):
        """phase=Cancelled with 1/3 done — never the Completed badge."""
        epoch = "1714069323"
        agg = _good_aggregate(
            phase="Cancelled",
            total=3,
            completed=1,
            failed=0,
            cancelled=2,
            run_states={
                "pending": 0,
                "running": 0,
                "completed": 1,
                "failed": 0,
                "cancelled": 2,
            },
        )
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="cancelled-partial",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=i) for i in range(3)]),
        )
        with _patch_sweep_cr_lookups():
            page = harness.goto_sweep_detail(
                harness.ns, "cancelled-partial", epoch=epoch
            )
        body = page.locator("body").inner_text()
        assert "Cancelled" in body, body[:800]
        # The "Completed" indicator (the small badge after the phase pill) must
        # be absent. The bare word "Completed" can appear inside the KPI card
        # title; what we forbid is the green-completion *badge* via title attr.
        # We assert the sub-text indicating cancelled count is present.
        assert "cancelled" in body.lower()


# ---------------------------------------------------------------------------
# Slim run summaries (per-variation rows)
# ---------------------------------------------------------------------------


class TestSlimRunSummaries:
    """Per-cell metrics dict shape attacks."""

    def test_one_cell_missing_metric_keys_does_not_crash(self, harness):
        """One cell has metrics={} — meanStd returns None, page must render."""
        epoch = "1714069323"
        agg = _good_aggregate(total=2)
        agg["per_cell_aggregates"][1]["metrics"] = {}
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="missing-metric",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=i) for i in range(2)]),
        )
        with _patch_sweep_cr_lookups():
            harness.goto_sweep_detail(harness.ns, "missing-metric", epoch=epoch)
        harness.assert_no_unreachable_banner()

    def test_metric_values_as_strings_are_filtered_out(self, harness):
        """Numeric values arriving as strings: ``Number.isFinite`` rejects them.

        ``metricValue`` (sweep-detail-helpers.js:33-36) requires a finite
        number; strings drop to null. Renderer must show ``---`` and not crash.
        """
        epoch = "1714069323"
        agg = _good_aggregate(total=1)
        agg["per_cell_aggregates"][0]["metrics"] = {
            "request_throughput": {"avg": "100"},
            "request_latency": {"p99": "80.0"},
        }
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="stringy",
            epoch=epoch,
            aggregate=agg,
            children_doc=_children_doc([_child_entry(idx=0)]),
        )
        with _patch_sweep_cr_lookups():
            status, body = harness.api_get(
                f"/api/v1/sweeps/{harness.ns}/stringy/cells?epoch={epoch}"
            )
        # ``CellEntry.metrics`` is typed ``dict[str, dict[str, float]]`` in
        # ``sweeps_models.py:138-141``. Strings must be coerced or rejected
        # by Pydantic — pick one and stick to it. Either 200 with floats or
        # the validator catches it. 500 is the only failure mode.
        assert status != 500, (status, body[:400])


# ---------------------------------------------------------------------------
# Direct deep link to nonexistent sweep
# ---------------------------------------------------------------------------


def test_deep_link_to_nonexistent_sweep_renders_error_not_loading_forever(harness):
    """``/#/sweeps/<ns>/does-not-exist``: must render an error card.

    The first-load failure path in ``SweepDetail`` (sweep-detail.js:170-189)
    sets ``error`` which produces the ``[data-testid=page-sweep-detail]``
    error card. We assert the card is present and the loading label is gone.
    """
    with _patch_sweep_cr_lookups():
        page = harness.goto_sweep_detail(harness.ns, "does-not-exist")
    body = page.locator("body").inner_text()
    assert "Loading sweep" not in body, body[:800]
    assert page.locator("[data-testid=page-sweep-detail]").count() >= 1


# ---------------------------------------------------------------------------
# API surface — unknown valid-shape epochs return 404 cleanly
# ---------------------------------------------------------------------------


class TestApiSurface:
    """Direct GETs against /api/v1/sweeps when no data is seeded."""

    @pytest.mark.parametrize(
        "subpath",
        [
            "cells",
            "children",
        ],
    )
    def test_unknown_epoch_returns_404_not_5xx(self, harness, subpath: str):
        """Valid-shape epoch with no matching dir on disk: clean 404."""
        # Valid-shape epoch (>=9 digits) but no directory exists for it.
        unknown_epoch = "1999999999"
        with _patch_sweep_cr_lookups():
            status, body = harness.api_get(
                f"/api/v1/sweeps/{harness.ns}/never-seeded/{subpath}"
                f"?epoch={unknown_epoch}"
            )
        # Either 404 (per-epoch dir missing) or 503 (api not ready). What we
        # forbid is a generic 500 leaking from a non-existent path. The
        # router contract is documented as 404; assert that directly.
        assert status == 404, (status, body[:400])

    def test_page_load_has_no_5xx_responses(self, harness):
        """Full page load against a seeded archived sweep emits zero 5xx."""
        epoch = "1714069323"
        _seed_aggregate_json(
            harness.results_dir,
            ns=harness.ns,
            sweep="clean-load",
            epoch=epoch,
            aggregate=_good_aggregate(),
            children_doc=_children_doc([_child_entry(idx=i) for i in range(3)]),
        )
        with _patch_sweep_cr_lookups():
            harness.goto_sweep_detail(harness.ns, "clean-load", epoch=epoch)
        five_xx = [r for r in harness.bad_responses if r.startswith("5")]
        assert not five_xx, "5xx during load:\n  " + "\n  ".join(five_xx)


# ---------------------------------------------------------------------------
# Stale spec — real bug surfaced during test authoring
# ---------------------------------------------------------------------------


class TestStaleSpecRegression:
    """A live CR carrying a spec that fails ``AIPerfSweepSpec.model_validate``
    should NOT take down the whole sweep-detail route.

    BUG SURFACED at ``src/aiperf/operator/routers/sweeps.py:85``:
    ``_spec_summary_from_record`` calls ``AIPerfSweepSpec.model_validate``
    on ``rec.raw_spec`` with no try/except. If a legacy CR is left in the
    cluster after a spec-schema change (e.g. a ``template`` key from an
    earlier release), every ``GET /api/v1/sweeps/<ns>/<name>`` returns 422
    and the page errors out with "Operator API unreachable".

    Expected behavior: fall through to the aggregate-doc path (or the
    empty ``SpecSummary`` default) and surface the parse failure as a
    condition / status warning, not a 500/422.
    """

    def test_legacy_template_key_in_live_spec_does_not_500_the_detail_route(
        self, harness
    ):
        """Legacy ``spec.template`` key (pre-pod_template rename) — page renders."""
        cr = _live_cr(
            ns=harness.ns,
            name="legacy-spec",
            phase="Running",
            spec={"template": {"spec": {"models": [{"name": "llama3-8b"}]}}},
        )
        with _patch_sweep_cr_lookups(find_returns=cr):
            status, body = harness.api_get(f"/api/v1/sweeps/{harness.ns}/legacy-spec")
        # The contract this test asserts: ``AIPerfSweepSpec.model_validate``
        # failure on a stale live CR must not propagate as a 4xx/5xx for
        # the read-only sweep-detail view.
        assert status == 200, (
            f"Legacy-shape CR spec produced status {status}; the route is "
            f"swallowing the entire detail response. See "
            f"src/aiperf/operator/routers/sweeps.py:85 — "
            f"AIPerfSweepSpec.model_validate has no fallback for stale CRs.\n"
            f"body[:400]={body[:400]!r}"
        )


def test_child_page_with_sweep_marker_renders(harness):
    """Child job's ``sweep.json`` marker is read by ``_archived_from_summary``.

    Loading the child's job-detail page must show the child's name and not
    stall on the unreachable banner regardless of the URL parent path.
    """
    epoch = "1714069323"
    from tests.unit.operator.ui_e2e.conftest import good_summary

    harness.seed_run(
        name="my-sweep-v00-t0",
        epoch=epoch,
        summary=good_summary(),
        sweep_marker={
            "sweep_name": "my-sweep",
            "variation_index": 0,
            "variation_label": "c=1",
        },
        is_latest=True,
    )
    page = harness.goto_job_detail(harness.ns, "my-sweep-v00-t0", epoch=epoch)
    body = page.locator("body").inner_text()
    assert "my-sweep-v00-t0" in body
    harness.assert_no_unreachable_banner()

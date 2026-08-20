# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SweepDetailResponse.search_summary`` — the planner's own verdict.

The adaptive planner records its feasibility verdict, its stopping reason, and
the empirical SLA boundary only in the ``search_history.json`` artifact
(``exporters/search_history.py``). These tests lock the projection of that file
onto the detail route, and — more importantly — lock the fail-soft contract: a
missing, torn, or nonsense artifact must degrade the field to ``null`` and never
fail the request, because the file is written with a bare ``write_bytes`` and is
absent outright for every grid-family sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import zstandard
from aiperf.operator.routers.sweeps import create_sweeps_router
from aiperf.operator.routers.sweeps_models import (
    MAX_BEST_TRIALS,
    SearchBestTrial,
    SearchBoundaryEdge,
    SearchSLABreach,
    SweepSearchSummary,
)
from aiperf.operator.sweep_union import SweepRecord
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pytest import param

NAMESPACE = "bench"
SWEEP_NAME = "gemma-bo4"
EPOCH = "1714150923"


@pytest.mark.parametrize(
    "model,payload",
    [
        param(
            SearchBestTrial,
            {"iteration_idx": 0, "objective_values": [float("nan")]},
            id="best-trial-objective",
        ),
        param(
            SearchBoundaryEdge,
            {"value": float("inf")},
            id="boundary-value",
        ),
        param(
            SearchBoundaryEdge,
            {"objective_value": float("-inf")},
            id="boundary-objective",
        ),
        param(
            SearchSLABreach,
            {"threshold": float("nan")},
            id="sla-threshold",
        ),
        param(
            SearchSLABreach,
            {"observed": float("inf")},
            id="sla-observed",
        ),
    ],
)  # fmt: skip
def test_search_summary_models_reject_non_finite_measurements(
    model: type[Any],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        model.model_validate(payload)


@pytest.mark.parametrize(
    "model,payload",
    [
        param(SearchBestTrial, {"iteration_idx": -1}, id="best-trial-index"),
        param(
            SearchBestTrial,
            {"iteration_idx": 0, "feasible_count": -1},
            id="feasible-count",
        ),
        param(
            SearchBestTrial,
            {"iteration_idx": 0, "pareto_rank": -1},
            id="pareto-rank",
        ),
        param(SearchBoundaryEdge, {"iteration_idx": -1}, id="boundary-index"),
        param(SweepSearchSummary, {"iteration_count": -1}, id="iteration-count"),
        param(
            SweepSearchSummary,
            {"feasible_iteration_count": -1},
            id="feasible-iteration-count",
        ),
        param(SweepSearchSummary, {"sla_filter_count": -1}, id="sla-filter-count"),
    ],
)  # fmt: skip
def test_search_summary_models_reject_negative_counts_and_indices(
    model: type[Any],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        model.model_validate(payload)


def _search_history_doc(**overrides: Any) -> dict[str, Any]:
    """A single-objective, SLA-constrained trajectory shaped like gemma-bo4."""
    doc: dict[str, Any] = {
        "config": {
            "planner": "optuna",
            "objectives": [
                {
                    "metric": "request_throughput",
                    "stat": "avg",
                    "direction": "MAXIMIZE",
                    "threshold": None,
                }
            ],
            "sla_filters": [
                {
                    "metric_tag": "time_to_first_token",
                    "stat": "p99",
                    "op": "lt",
                    "threshold": 500.0,
                }
            ],
            "search_space": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 512,
                    "kind": "int",
                }
            ],
        },
        "iterations": [
            {
                "iteration_idx": 0,
                "variation_values": {"phases.profiling.concurrency": 17},
                "objective_values": [42.5],
                "feasible": True,
                "non_monotonic_warning": False,
            },
            {
                "iteration_idx": 1,
                "variation_values": {"phases.profiling.concurrency": 309},
                "objective_values": [180.0],
                "feasible": False,
                "non_monotonic_warning": False,
            },
        ],
        "best_trials": [
            {
                "iteration_idx": 0,
                "objective_values": [42.5],
                "variation_values": {"phases.profiling.concurrency": 17},
                "feasible": True,
                "feasible_count": 1,
                "pareto_rank": 0,
            }
        ],
        "boundary_summary": {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": {
                "value": 17,
                "iteration_idx": 0,
                "objective_value": 42.5,
            },
            "infeasible_min": {
                "value": 309,
                "iteration_idx": 1,
                "first_breach": {
                    "metric_tag": "time_to_first_token",
                    "stat": "p99",
                    "op": "lt",
                    "threshold": 500.0,
                    "observed": 18500.0,
                },
            },
        },
        "recipe": "max-concurrency-under-sla",
        "convergence_reason": "improvement_patience",
    }
    doc.update(overrides)
    return doc


def _seed_epoch(base_dir: Path) -> Path:
    """Create the sweep epoch dir with an aggregate.json; return the dir."""
    epoch_dir = base_dir / NAMESPACE / "sweeps" / SWEEP_NAME / EPOCH
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "aggregate.json").write_bytes(orjson.dumps({"phase": "Completed"}))
    return epoch_dir


def _archived_record(base_dir: Path) -> SweepRecord:
    """An archived record pointing at the seeded epoch's aggregate.json.

    ``aggregate_path`` is what the reader derives the epoch dir from, mirroring
    ``_read_conditions``.
    """
    return SweepRecord(
        namespace=NAMESPACE,
        name=SWEEP_NAME,
        source="archived",
        phase="Completed",
        total_variations=22,
        completed_runs=14,
        failed_runs=0,
        age_seconds=99,
        model="gemma",
        aggregate_path=str(
            base_dir / NAMESPACE / "sweeps" / SWEEP_NAME / EPOCH / "aggregate.json"
        ),
        aggregate_doc={"phase": "Completed"},
    )


def _get_detail(base_dir: Path, rec: SweepRecord) -> dict[str, Any]:
    app = FastAPI()
    app.include_router(create_sweeps_router([MagicMock()], base_dir))
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        response = TestClient(app).get(f"/api/v1/sweeps/{NAMESPACE}/{SWEEP_NAME}")
    assert response.status_code == 200, response.text
    return response.json()


def _detail_with_history(tmp_path: Path, doc: Any) -> dict[str, Any]:
    epoch_dir = _seed_epoch(tmp_path)
    (epoch_dir / "search_history.json").write_bytes(orjson.dumps(doc))
    return _get_detail(tmp_path, _archived_record(tmp_path))


def test_search_summary_absent_when_no_artifact(tmp_path: Path) -> None:
    """Grid sweeps never write the artifact; the field must simply be null."""
    _seed_epoch(tmp_path)
    body = _get_detail(tmp_path, _archived_record(tmp_path))
    assert body["search_summary"] is None


def test_search_summary_absent_for_live_only_record(tmp_path: Path) -> None:
    """A record with no archived epoch must not borrow a previous run's file.

    The trajectory only reaches the operator's PVC via the sweep-aggregate
    harvest, so a live-only record has no epoch of its own; captioning it with
    whatever ``latest.txt`` points at would show the prior run's verdict beside
    this run's counters.
    """
    epoch_dir = _seed_epoch(tmp_path)
    (epoch_dir / "search_history.json").write_bytes(orjson.dumps(_search_history_doc()))
    rec = _archived_record(tmp_path)
    rec.aggregate_path = None
    body = _get_detail(tmp_path, rec)
    assert body["search_summary"] is None


def test_search_summary_projects_planner_verdict(tmp_path: Path) -> None:
    body = _detail_with_history(tmp_path, _search_history_doc())
    summary = body["search_summary"]

    assert summary["convergence_reason"] == "improvement_patience"
    assert summary["stop_kind"] == "converged"
    assert summary["recipe"] == "max-concurrency-under-sla"
    assert summary["iteration_count"] == 2
    assert summary["feasible_iteration_count"] == 1

    best = summary["best_trials"]
    assert len(best) == 1
    assert best[0]["iteration_idx"] == 0
    assert best[0]["objective_values"] == [42.5]
    assert best[0]["variation_values"] == {"phases.profiling.concurrency": 17}
    assert best[0]["feasible"] is True
    assert best[0]["feasible_count"] == 1
    assert summary["best_trials_truncated"] is False


def test_sla_filter_count_is_reported(tmp_path: Path) -> None:
    body = _detail_with_history(tmp_path, _search_history_doc())
    assert body["search_summary"]["sla_filter_count"] == 1


def test_unconstrained_search_reports_zero_sla_filters(tmp_path: Path) -> None:
    """Without filters every ``feasible`` flag is vacuously true.

    ``write_search_history`` defaults the verdict to true when nothing
    constrains it, so a client that renders "meets SLA" off ``feasible`` alone
    would claim an unconstrained search passed an SLA it never had.
    """
    doc = _search_history_doc()
    doc["config"]["sla_filters"] = []
    body = _detail_with_history(tmp_path, doc)
    assert body["search_summary"]["sla_filter_count"] == 0
    assert body["search_summary"]["best_trials"][0]["feasible"] is True


def test_objective_direction_is_normalized_to_lowercase(tmp_path: Path) -> None:
    """The artifact writes ``MAXIMIZE``; ``spec_summary`` writes ``maximize``.

    One response carrying both spellings of one enum forces every client into a
    case-insensitive comparison, so the router normalizes at the boundary.
    """
    body = _detail_with_history(tmp_path, _search_history_doc())
    assert body["search_summary"]["objectives"] == [
        {"metric": "request_throughput", "stat": "avg", "direction": "maximize"}
    ]


def test_boundary_summary_carries_both_edges_and_first_breach(tmp_path: Path) -> None:
    body = _detail_with_history(tmp_path, _search_history_doc())
    boundary = body["search_summary"]["boundary_summary"]

    assert boundary["swept_dim_path"] == "phases.profiling.concurrency"
    assert boundary["feasible_max"]["value"] == 17.0
    assert boundary["feasible_max"]["iteration_idx"] == 0
    assert boundary["infeasible_min"]["value"] == 309.0
    assert boundary["infeasible_min"]["first_breach"] == {
        "metric_tag": "time_to_first_token",
        "stat": "p99",
        "op": "lt",
        "threshold": 500.0,
        "observed": 18500.0,
    }


def test_smooth_isotonic_boundary_extras_are_carried(tmp_path: Path) -> None:
    """``boundary_type``/``binding_constraint`` are absent, not null, for BO."""
    doc = _search_history_doc()
    doc["boundary_summary"]["boundary_type"] = "cliff"
    doc["boundary_summary"]["binding_constraint"] = "time_to_first_token:p99"
    body = _detail_with_history(tmp_path, doc)
    boundary = body["search_summary"]["boundary_summary"]
    assert boundary["boundary_type"] == "cliff"
    assert boundary["binding_constraint"] == "time_to_first_token:p99"


@pytest.mark.parametrize(
    "reason,expected",
    [
        param("improvement_patience", "converged", id="convergence-rule-fired"),
        param("plateau_cv", "converged", id="plateau"),
        param("monotonic_precision_reached", "converged", id="sla-planner-precision"),
        param("unknown", "converged", id="clean-exit-without-recorded-reason"),
        param("max_iterations", "budget_exhausted", id="budget"),
        param(None, "incomplete", id="null-means-mid-loop-or-abnormal-exit"),
    ],
)  # fmt: skip
def test_stop_kind_classification(
    tmp_path: Path, reason: str | None, expected: str
) -> None:
    """``stop_kind`` is the server-side classification of a growing string enum.

    ``"unknown"`` is a clean terminal exit whose planner recorded no structured
    reason — the planner still decided to stop, so it is a convergence. Only a
    null reason (mid-loop write, cancellation, crash) is ``incomplete``; a
    cancelled sweep must not be captioned "converged early".
    """
    body = _detail_with_history(
        tmp_path, _search_history_doc(convergence_reason=reason)
    )
    assert body["search_summary"]["stop_kind"] == expected


def test_zero_feasible_count_survives_projection(tmp_path: Path) -> None:
    """``feasible_count == 0`` is the "no servable point was found" signal.

    The planner falls back to ranking the whole scored pool in that case, so the
    winner is a least-bad point. Losing the zero would let a client present an
    SLA-breaching configuration as optimal.
    """
    doc = _search_history_doc()
    doc["best_trials"][0]["feasible"] = False
    doc["best_trials"][0]["feasible_count"] = 0
    body = _detail_with_history(tmp_path, doc)
    best = body["search_summary"]["best_trials"][0]
    assert best["feasible"] is False
    assert best["feasible_count"] == 0


def test_pareto_front_is_truncated_with_a_marker(tmp_path: Path) -> None:
    doc = _search_history_doc()
    doc["best_trials"] = [
        {
            "iteration_idx": idx,
            "objective_values": [float(idx)],
            "variation_values": {"phases.profiling.concurrency": idx},
            "feasible": True,
            "feasible_count": MAX_BEST_TRIALS + 5,
            "pareto_rank": 0,
        }
        for idx in range(MAX_BEST_TRIALS + 5)
    ]
    body = _detail_with_history(tmp_path, doc)
    summary = body["search_summary"]
    assert len(summary["best_trials"]) == MAX_BEST_TRIALS
    assert summary["best_trials_truncated"] is True


def test_null_inside_objective_values_keeps_its_slot(tmp_path: Path) -> None:
    """An unscored objective must NOT be compacted out of the vector.

    ``objective_values`` is positional against ``config.objectives`` and the
    exporter writes explicit nulls on purpose: ``scrub_non_finite`` maps a NaN
    score to null so the artifact keeps "the scorer returned NaN for this
    objective" distinct from "this iteration was never scored"
    (exporters/search_history.py:135-138). Dropping the null shortens the vector
    and shifts every later objective onto the wrong label -- on this two-
    objective run, objective #2's slot would render objective #1's 42.5.
    """
    doc = _search_history_doc()
    doc["config"]["objectives"].append(
        {
            "metric": "time_to_first_token",
            "stat": "p99",
            "direction": "MINIMIZE",
            "threshold": None,
        }
    )
    doc["best_trials"][0]["objective_values"] = [42.5, None]
    body = _detail_with_history(tmp_path, doc)
    summary = body["search_summary"]

    assert [o["metric"] for o in summary["objectives"]] == [
        "request_throughput",
        "time_to_first_token",
    ]
    values = summary["best_trials"][0]["objective_values"]
    assert values == [42.5, None]
    assert len(values) == len(summary["objectives"])


def test_leading_null_in_objective_values_keeps_its_slot(tmp_path: Path) -> None:
    """The compaction was worst at index 0: objective #1 read #2's number."""
    doc = _search_history_doc()
    doc["config"]["objectives"].append(
        {
            "metric": "time_to_first_token",
            "stat": "p99",
            "direction": "MINIMIZE",
            "threshold": None,
        }
    )
    doc["best_trials"][0]["objective_values"] = [None, 180.0]
    body = _detail_with_history(tmp_path, doc)

    assert body["search_summary"]["best_trials"][0]["objective_values"] == [None, 180.0]


def test_non_numeric_objective_value_becomes_null_not_a_gap(tmp_path: Path) -> None:
    """A junk entry degrades in place; it never shortens the vector."""
    doc = _search_history_doc()
    doc["best_trials"][0]["objective_values"] = ["not-a-number"]
    body = _detail_with_history(tmp_path, doc)

    assert body["search_summary"]["best_trials"][0]["objective_values"] == [None]


def test_null_best_trials_degrades_to_empty_list(tmp_path: Path) -> None:
    """``best_trials`` is null until an iteration produces a usable objective."""
    body = _detail_with_history(tmp_path, _search_history_doc(best_trials=None))
    assert body["search_summary"]["best_trials"] == []
    assert body["search_summary"]["convergence_reason"] == "improvement_patience"


def test_null_boundary_summary_degrades_to_null(tmp_path: Path) -> None:
    """Multi-dimensional searches have no orderable axis, so no boundary."""
    body = _detail_with_history(tmp_path, _search_history_doc(boundary_summary=None))
    assert body["search_summary"]["boundary_summary"] is None


def test_boundary_without_swept_dim_path_is_dropped(tmp_path: Path) -> None:
    """A boundary with no axis name is not renderable; drop the block, keep the rest."""
    doc = _search_history_doc()
    doc["boundary_summary"] = {"feasible_max": {"value": 17}}
    body = _detail_with_history(tmp_path, doc)
    assert body["search_summary"]["boundary_summary"] is None
    assert body["search_summary"]["convergence_reason"] == "improvement_patience"


def test_zstd_compressed_artifact_is_read(tmp_path: Path) -> None:
    """COMPRESS_ON_DISK lands the harvested artifact as ``.zst``."""
    epoch_dir = _seed_epoch(tmp_path)
    raw = orjson.dumps(_search_history_doc())
    (epoch_dir / "search_history.json.zst").write_bytes(
        zstandard.ZstdCompressor().compress(raw)
    )
    body = _get_detail(tmp_path, _archived_record(tmp_path))
    assert body["search_summary"]["convergence_reason"] == "improvement_patience"


def test_torn_write_degrades_instead_of_500(tmp_path: Path) -> None:
    """The exporter does a bare ``write_bytes``; readers can see partial JSON."""
    epoch_dir = _seed_epoch(tmp_path)
    (epoch_dir / "search_history.json").write_bytes(b'{"config": {"objecti')
    body = _get_detail(tmp_path, _archived_record(tmp_path))
    assert body["search_summary"] is None
    assert body["sweep"]["name"] == SWEEP_NAME


@pytest.mark.parametrize(
    "payload",
    [
        param(b"", id="zero-bytes"),
        param(b"[]", id="json-array-not-object"),
        param(b"null", id="json-null"),
        param(b'"a string"', id="json-string"),
    ],
)  # fmt: skip
def test_nonsense_artifact_degrades_to_null(tmp_path: Path, payload: bytes) -> None:
    epoch_dir = _seed_epoch(tmp_path)
    (epoch_dir / "search_history.json").write_bytes(payload)
    body = _get_detail(tmp_path, _archived_record(tmp_path))
    assert body["search_summary"] is None


def test_corrupt_zstd_falls_through_to_plain_file(tmp_path: Path) -> None:
    """An unreadable ``.zst`` must not shadow a readable plain sibling."""
    epoch_dir = _seed_epoch(tmp_path)
    (epoch_dir / "search_history.json.zst").write_bytes(b"not-a-zstd-frame")
    (epoch_dir / "search_history.json").write_bytes(orjson.dumps(_search_history_doc()))
    body = _get_detail(tmp_path, _archived_record(tmp_path))
    assert body["search_summary"]["convergence_reason"] == "improvement_patience"


def test_wrongly_typed_fields_do_not_500(tmp_path: Path) -> None:
    """Every scalar is defensively coerced; a hostile artifact still renders."""
    doc = _search_history_doc()
    doc["iterations"] = "not-a-list"
    doc["config"] = ["not", "a", "dict"]
    doc["recipe"] = 17
    doc["best_trials"] = [{"iteration_idx": "not-an-int"}, "not-a-dict"]
    doc["boundary_summary"]["feasible_max"] = "not-a-dict"
    doc["boundary_summary"]["infeasible_min"] = {"value": "high", "first_breach": 3}
    body = _detail_with_history(tmp_path, doc)
    summary = body["search_summary"]
    assert summary["iteration_count"] == 0
    assert summary["objectives"] == []
    assert summary["sla_filter_count"] == 0
    assert summary["best_trials"] == []
    assert summary["recipe"] is None
    assert summary["boundary_summary"]["feasible_max"] is None
    assert summary["boundary_summary"]["infeasible_min"]["value"] is None
    assert summary["boundary_summary"]["infeasible_min"]["first_breach"] is None

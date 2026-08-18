# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Winner-card presentation of the planner's own verdict.

Exercises ``components/sweep-winner-summary-helpers.js``, which turns
``detail.search_summary`` into the card's model. The rules under test are all
honesty rules: never claim SLA feasibility for an unconstrained search, never
present the best of a set of failing points as a recommendation, never say
"converged" when no stopping reason was recorded, and never assert a boundary
bracket the search only half-observed.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "sweep-winner-summary-helpers.js"
)


def _search_summary(**overrides: object) -> dict:
    doc: dict = {
        "convergence_reason": "improvement_patience",
        "stop_kind": "converged",
        "iteration_count": 14,
        "feasible_iteration_count": 9,
        "sla_filter_count": 1,
        "objectives": [
            {"metric": "request_throughput", "stat": "avg", "direction": "maximize"}
        ],
        "best_trials": [
            {
                "iteration_idx": 8,
                "objective_values": [42.5],
                "variation_values": {"phases.profiling.concurrency": 17},
                "feasible": True,
                "feasible_count": 9,
                "pareto_rank": 0,
            }
        ],
        "best_trials_truncated": False,
        "boundary_summary": {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": {"value": 17, "iteration_idx": 8, "objective_value": 42.5},
            "infeasible_min": {
                "value": 309,
                "iteration_idx": 11,
                "first_breach": {
                    "metric_tag": "time_to_first_token",
                    "stat": "p99",
                    "op": "lt",
                    "threshold": 500.0,
                    "observed": 18500.0,
                },
            },
            "boundary_type": None,
            "binding_constraint": None,
        },
        "recipe": "max-concurrency-under-sla",
    }
    doc.update(overrides)
    return doc


def _call(fn: str, search: object) -> object:
    script = f"""
        import {{ {fn} }} from {HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify({fn}({json.dumps(search)})));
    """
    return json.loads(run_node(script))


def test_planner_verdict_leads_with_swept_values_not_the_cell_id() -> None:
    """`concurrency=17` tells a reader what was tried; `search_iter_0008` does not."""
    verdict = _call("plannerVerdict", _search_summary())
    assert verdict["headline"] == "concurrency=17"
    assert verdict["iterationIdx"] == 8


def test_planner_verdict_labels_the_optimized_objective() -> None:
    verdict = _call("plannerVerdict", _search_summary())
    assert verdict["objectives"] == [
        {
            "metric": "request_throughput",
            "stat": "avg",
            "label": "request throughput avg",
            # Resolved from the objective's own metric tag, never borrowed from
            # whichever series the chart selector happens to be on.
            "unit": "req/s",
            "higherIsBetter": True,
            "value": 42.5,
        }
    ]


def _two_objective_summary(values: list[object]) -> dict:
    """A two-objective verdict whose winning trial scored ``values``."""
    summary = _search_summary()
    summary["objectives"].append(
        {"metric": "time_to_first_token", "stat": "p99", "direction": "minimize"}
    )
    summary["best_trials"][0]["objective_values"] = values
    return summary


def test_objective_values_are_read_positionally_not_by_compaction() -> None:
    """An unscored objective keeps its slot, so labels never shift onto it.

    The API preserves explicit nulls inside ``objective_values`` precisely so
    this map stays aligned with ``objectives``. If the null were compacted out,
    the second objective would index the FIRST one's 42.5 and the card would
    report a throughput number as a TTFT.
    """
    verdict = _call("plannerVerdict", _two_objective_summary([42.5, None]))
    assert [o["metric"] for o in verdict["objectives"]] == [
        "request_throughput",
        "time_to_first_token",
    ]
    assert [o["value"] for o in verdict["objectives"]] == [42.5, None]


def test_a_null_first_objective_does_not_inherit_the_second_value() -> None:
    """The compaction was worst at index 0: the headline read the wrong metric."""
    verdict = _call("plannerVerdict", _two_objective_summary([None, 180.0]))
    assert [o["value"] for o in verdict["objectives"]] == [None, 180.0]
    assert verdict["objectives"][0]["unit"] == "req/s"
    assert verdict["objectives"][1]["unit"] == "ms"


def test_objective_unit_comes_from_the_objective_not_the_chart() -> None:
    """The unit is a property of the metric, so no selection can erase it."""
    verdict = _call("plannerVerdict", _search_summary())
    assert verdict["objectives"][0]["unit"] == "req/s"

    latency = _search_summary(
        objectives=[
            {"metric": "time_to_first_token", "stat": "p99", "direction": "minimize"}
        ]
    )
    assert _call("plannerVerdict", latency)["objectives"][0]["unit"] == "ms"


def test_unknown_metric_tag_claims_no_unit() -> None:
    """Omitting a unit is honest; inventing one is not."""
    summary = _search_summary(
        objectives=[{"metric": "some_future_metric", "stat": "avg"}]
    )
    assert _call("plannerVerdict", summary)["objectives"][0]["unit"] == ""


def test_planner_verdict_absent_without_best_trials() -> None:
    """No verdict means the card falls back to its own metric-ranked pick."""
    assert _call("plannerVerdict", _search_summary(best_trials=[])) is None
    assert _call("plannerVerdict", None) is None


def test_feasibility_is_not_claimed_for_an_unconstrained_search() -> None:
    """`feasible` defaults to true when nothing constrains it.

    Rendering that flag unguarded would tell an operator the winner "meets SLA"
    for a search that never had one. `null` means "make no claim".
    """
    verdict = _call("plannerVerdict", _search_summary(sla_filter_count=0))
    assert verdict["feasible"] is None
    assert verdict["constrained"] is False
    assert verdict["noFeasiblePoint"] is False


def test_zero_feasible_count_is_flagged_as_no_servable_point() -> None:
    """The planner ranked the full pool; the winner is the least-bad breach."""
    summary = _search_summary()
    summary["best_trials"][0]["feasible"] = False
    summary["best_trials"][0]["feasible_count"] = 0
    verdict = _call("plannerVerdict", summary)
    assert verdict["noFeasiblePoint"] is True
    assert verdict["feasible"] is False


def test_pareto_front_is_reported_as_one_of_several() -> None:
    summary = _search_summary()
    summary["best_trials"] = summary["best_trials"] * 3
    summary["best_trials_truncated"] = True
    verdict = _call("plannerVerdict", summary)
    assert verdict["isFront"] is True
    assert verdict["frontSize"] == 3
    assert verdict["truncated"] is True


def test_convergence_note_names_the_rule_that_fired() -> None:
    note = _call("convergenceNote", _search_summary())
    assert note["kind"] == "converged"
    assert note["text"] == (
        "Converged after 14 iterations because no iteration improved on the "
        "best result for the configured patience window."
    )


def test_budget_exhaustion_is_not_described_as_convergence() -> None:
    """Hitting the iteration cap means the search ran out of budget, not that it
    found the answer -- a better point may sit outside the explored region."""
    note = _call(
        "convergenceNote",
        _search_summary(
            convergence_reason="max_iterations", stop_kind="budget_exhausted"
        ),
    )
    assert note["kind"] == "budget_exhausted"
    assert "full iteration budget" in note["text"]
    assert "not because it converged" in note["text"]
    assert not note["text"].startswith("Converged")


def test_missing_reason_is_never_called_converged() -> None:
    """A null reason covers cancellation and crash as well as still-running.

    This is the exact mislabelling the variations KPI tile had to hedge around
    by saying "stopped early" instead of "converged early".
    """
    note = _call(
        "convergenceNote",
        _search_summary(convergence_reason=None, stop_kind="incomplete"),
    )
    assert note["kind"] == "incomplete"
    assert "onverged" not in note["text"]
    assert "cancelled, interrupted, or is still going" in note["text"]


def test_unmapped_reason_is_shown_verbatim_rather_than_dropped() -> None:
    """New planners add new reason strings; an unphrased reason is still data."""
    note = _call(
        "convergenceNote",
        _search_summary(convergence_reason="some_future_planner_rule"),
    )
    assert "some_future_planner_rule" in note["text"]


def test_clean_exit_without_a_recorded_reason_says_so() -> None:
    note = _call("convergenceNote", _search_summary(convergence_reason="unknown"))
    assert note["kind"] == "converged"
    assert "recorded no specific reason" in note["text"]


def test_sla_boundary_reports_both_edges_and_the_first_breach() -> None:
    boundary = _call("slaBoundaryNote", _search_summary())
    assert boundary["dimension"] == "concurrency"
    assert boundary["passText"] == "concurrency 17"
    assert boundary["failText"] == "concurrency 309"
    assert boundary["bracketed"] is True
    assert boundary["breachText"] == "time to first token p99 < 500"
    assert boundary["observed"] == 18500.0


def test_sla_boundary_suppressed_for_an_unconstrained_search() -> None:
    """Without filters `feasible_max` is just "highest value tried", which the
    variations table already shows and which is not an SLA statement."""
    assert _call("slaBoundaryNote", _search_summary(sla_filter_count=0)) is None


def test_one_sided_boundary_is_not_presented_as_a_bracket() -> None:
    """With only a passing edge the search proved a lower bound, not a limit."""
    summary = _search_summary()
    summary["boundary_summary"]["infeasible_min"] = None
    boundary = _call("slaBoundaryNote", summary)
    assert boundary["bracketed"] is False
    assert boundary["failText"] is None
    assert boundary["passText"] == "concurrency 17"


def test_cliff_boundary_is_flagged() -> None:
    """A discontinuous boundary must not be interpolated across."""
    summary = _search_summary()
    summary["boundary_summary"]["boundary_type"] = "cliff"
    boundary = _call("slaBoundaryNote", summary)
    assert boundary["isCliff"] is True


def test_boundary_absent_for_multi_dimensional_search() -> None:
    assert _call("slaBoundaryNote", _search_summary(boundary_summary=None)) is None


def test_helpers_tolerate_a_null_search_summary() -> None:
    """Grid sweeps carry `search_summary: null`; nothing may throw."""
    assert _call("plannerVerdict", None) is None
    assert _call("convergenceNote", None) is None
    assert _call("slaBoundaryNote", None) is None

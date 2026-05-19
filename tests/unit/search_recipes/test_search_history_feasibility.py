# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for write_search_history's lexicographic feasibility-first best.

write_search_history must:
- prefer the best feasible iteration when any feasible iterations exist;
- fall back to the best of the full pool when none are feasible (with
  feasible_count == 0 so readers can tell the two cases apart);
- surface ``recipe`` (the recipe name) and ``sla_filters`` for post-run audit.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import (
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.base import SearchIteration
from aiperf.orchestrator.search_planner.monotonic import MonotonicSLASearchPlanner


def _cfg(**overrides) -> AdaptiveSearchSweep:
    kwargs: dict = dict(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            ),
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ],
        max_iterations=5,
        n_initial_points=2,
    )
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def _read(base_dir: Path) -> dict:
    return orjson.loads((base_dir / "search_history.json").read_bytes())


def test_write_search_history_picks_feasible_best_over_higher_objective(
    tmp_path: Path,
):
    """MAXIMIZE: a feasible point with low objective beats an infeasible high one."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            objective_values=[50.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"phases.profiling.concurrency": 100},
            objective_value=1000.0,
            objective_values=[1000.0],
            feasible=False,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["best_trials"][0]["iteration_idx"] == 0
    assert payload["best_trials"][0]["objective_values"][0] == 50.0
    assert payload["best_trials"][0]["feasible"] is True
    assert payload["best_trials"][0]["feasible_count"] == 1


def test_write_search_history_falls_back_to_best_infeasible_when_none_feasible(
    tmp_path: Path,
):
    """All-infeasible: pick the best of the full pool, feasible_count == 0."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            objective_values=[50.0],
            feasible=False,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"phases.profiling.concurrency": 100},
            objective_value=1000.0,
            objective_values=[1000.0],
            feasible=False,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["best_trials"][0]["iteration_idx"] == 1
    assert payload["best_trials"][0]["objective_values"][0] == 1000.0
    assert payload["best_trials"][0]["feasible"] is False
    assert payload["best_trials"][0]["feasible_count"] == 0


def test_write_search_history_records_recipe_name_and_sla_filters(tmp_path: Path):
    """Recipe metadata lands in the top-level payload + config block."""
    cfg = _cfg(
        recipe_name="max-throughput-ttft-sla",
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            ),
        ],
    )
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["recipe"] == "max-throughput-ttft-sla"
    assert payload["config"]["sla_filters"] == [
        {
            "metric_tag": "time_to_first_token",
            "stat": "p95",
            "op": "lt",
            "threshold": 200.0,
        }
    ]


def test_write_search_history_recipe_is_none_for_explicit_search_flags(
    tmp_path: Path,
):
    """Without a recipe, payload['recipe'] is null and sla_filters is empty."""
    cfg = _cfg()
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 10},
            objective_value=50.0,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["recipe"] is None
    assert payload["config"]["sla_filters"] == []


# ----------------------------------------------------------------------------
# boundary_summary block
# ----------------------------------------------------------------------------


SWEPT = "phases.profiling.concurrency"
TTFT_SLA = SLAFilter(
    metric_tag="time_to_first_token",
    stat="p95",
    op="lt",
    threshold=200.0,
)


def _ttft_run(value: int, *, ttft_p95: float, success: bool = True) -> RunResult:
    """Build a successful (or failed) RunResult with a TTFT p95 measurement."""
    return RunResult(
        label=f"c{value}",
        success=success,
        summary_metrics={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft_p95),
        },
        variation_label=f"search_iter_{value:04d}",
        variation_values={SWEPT: value},
    )


def _iter(
    idx: int,
    value: int,
    *,
    feasible: bool,
    objective: float | None = 100.0,
    results: list[RunResult] | None = None,
) -> SearchIteration:
    return SearchIteration(
        iteration_idx=idx,
        variation_values={SWEPT: value},
        objective_value=objective,
        objective_values=[objective] if objective is not None else None,
        feasible=feasible,
        results=results or [],
    )


def test_boundary_summary_1d_mixed_feasibility(tmp_path: Path):
    """1D BO with mixed feasibility populates feasible_max and infeasible_min."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    history = [
        _iter(0, 10, feasible=True, objective=50.0),
        _iter(
            1,
            100,
            feasible=False,
            objective=900.0,
            results=[_ttft_run(100, ttft_p95=210.0)],
        ),
        _iter(2, 50, feasible=True, objective=400.0),
        _iter(3, 200, feasible=True, objective=800.0),
        _iter(
            4,
            500,
            feasible=False,
            objective=1500.0,
            results=[_ttft_run(500, ttft_p95=350.0)],
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    bs = payload["boundary_summary"]
    assert bs is not None
    assert bs["swept_dim_path"] == SWEPT
    assert bs["feasible_max"] == {
        "value": 200,
        "iteration_idx": 3,
        "objective_value": 800.0,
    }
    assert bs["infeasible_min"]["value"] == 100
    assert bs["infeasible_min"]["iteration_idx"] == 1
    assert bs["infeasible_min"]["first_breach"] == {
        "metric_tag": "time_to_first_token",
        "stat": "p95",
        "op": "lt",
        "threshold": 200.0,
        "observed": 210.0,
    }


def test_boundary_summary_multi_dim_returns_null(tmp_path: Path):
    """2D search space - boundary_summary is null (only 1D has a scalar boundary)."""
    cfg = _cfg(
        search_space=[
            SearchSpaceDimension(path=SWEPT, lo=1, hi=100, kind="int"),
            SearchSpaceDimension(
                path="phases.profiling.requests", lo=10, hi=1000, kind="int"
            ),
        ],
        sla_filters=[TTFT_SLA],
    )
    history = [
        SearchIteration(
            iteration_idx=0,
            variation_values={SWEPT: 10, "phases.profiling.requests": 100},
            objective_value=50.0,
            feasible=True,
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert payload["boundary_summary"] is None


def test_boundary_summary_all_feasible_1d(tmp_path: Path):
    """All feasible: feasible_max set, infeasible_min null."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    history = [
        _iter(0, 10, feasible=True, objective=50.0),
        _iter(1, 50, feasible=True, objective=200.0),
        _iter(2, 100, feasible=True, objective=300.0),
    ]
    write_search_history(tmp_path, history, cfg)
    bs = _read(tmp_path)["boundary_summary"]

    assert bs["feasible_max"]["value"] == 100
    assert bs["feasible_max"]["iteration_idx"] == 2
    assert bs["infeasible_min"] is None


def test_boundary_summary_all_infeasible_1d(tmp_path: Path):
    """All infeasible: feasible_max null, infeasible_min populated with first_breach."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    history = [
        _iter(
            0,
            50,
            feasible=False,
            objective=300.0,
            results=[_ttft_run(50, ttft_p95=250.0)],
        ),
        _iter(
            1,
            10,
            feasible=False,
            objective=200.0,
            results=[_ttft_run(10, ttft_p95=220.0)],
        ),
    ]
    write_search_history(tmp_path, history, cfg)
    bs = _read(tmp_path)["boundary_summary"]

    assert bs["feasible_max"] is None
    assert bs["infeasible_min"]["value"] == 10
    assert bs["infeasible_min"]["iteration_idx"] == 1
    assert bs["infeasible_min"]["first_breach"]["observed"] == 220.0


def test_boundary_summary_empty_history(tmp_path: Path):
    """Zero iterations - boundary_summary is null regardless of search-space dim."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    write_search_history(tmp_path, [], cfg)
    assert _read(tmp_path)["boundary_summary"] is None


def test_boundary_summary_first_breach_picks_first_failing_filter(tmp_path: Path):
    """Two filters, only the second breaches - first_breach reports the second."""
    throughput_sla = SLAFilter(
        metric_tag="output_token_throughput",
        stat="avg",
        op="gt",
        threshold=10.0,
    )
    cfg = _cfg(sla_filters=[throughput_sla, TTFT_SLA])
    # The run satisfies the throughput filter (avg > 10) but breaches TTFT.
    run = RunResult(
        label="c100",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=500.0),
            "time_to_first_token": JsonMetricResult(unit="ms", p95=240.0),
        },
        variation_label="search_iter_0001",
        variation_values={SWEPT: 100},
    )
    history = [
        _iter(0, 50, feasible=True, objective=400.0),
        _iter(1, 100, feasible=False, objective=900.0, results=[run]),
    ]
    write_search_history(tmp_path, history, cfg)
    bs = _read(tmp_path)["boundary_summary"]

    breach = bs["infeasible_min"]["first_breach"]
    assert breach["metric_tag"] == "time_to_first_token"
    assert breach["observed"] == 240.0


def test_boundary_summary_unmeasurable_breach_observed_null(tmp_path: Path):
    """All-failed runs: first_breach populated, observed is null."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    failed = _ttft_run(100, ttft_p95=999.0, success=False)
    history = [
        _iter(0, 50, feasible=True, objective=400.0),
        _iter(1, 100, feasible=False, objective=None, results=[failed]),
    ]
    write_search_history(tmp_path, history, cfg)
    bs = _read(tmp_path)["boundary_summary"]

    assert bs["infeasible_min"]["first_breach"]["observed"] is None


def test_boundary_summary_monotonic_planner_matches_internal_state(tmp_path: Path):
    """Monotonic planner end-to-end: written boundary_summary == planner.boundary_summary()."""
    base = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path=SWEPT, lo=1, hi=1000, kind="int"),
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ],
        max_iterations=30,
        n_initial_points=2,
        random_seed=42,
        sla_filters=[TTFT_SLA],
        monotonic_stability_trials=1,
    )
    planner = MonotonicSLASearchPlanner(base, cfg)
    # Boundary at concurrency = 128: feasible iff c < 128.
    for _ in range(20):
        if planner.is_converged():
            break
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        c = variation.values[SWEPT]
        ttft = 100.0 if c < 128 else 300.0
        run = RunResult(
            label=variation.label,
            success=True,
            summary_metrics={
                "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft),
                "output_token_throughput": JsonMetricResult(unit="tok/s", avg=200.0),
            },
            variation_label=variation.label,
            variation_values=variation.values,
        )
        planner.tell(variation, [run])

    write_search_history(tmp_path, planner.history(), cfg, planner=planner)
    written = _read(tmp_path)["boundary_summary"]
    expected = planner.boundary_summary()

    assert expected is not None
    assert written == orjson.loads(orjson.dumps(expected))
    assert written["feasible_max"] is not None
    assert written["infeasible_min"] is not None
    # Boundary brackets the synthetic threshold.
    assert written["feasible_max"]["value"] < 128
    assert written["infeasible_min"]["value"] >= 128


@pytest.mark.parametrize(
    "feasibility_pattern",
    [
        pytest.param([True, True, True], id="all_feasible"),
        pytest.param([False, False, False], id="all_infeasible"),
        pytest.param([True, False, True], id="mixed_non_monotonic"),
    ],
)  # fmt: skip
def test_boundary_summary_iterations_payload_keeps_non_monotonic_warning(
    tmp_path: Path, feasibility_pattern: list[bool]
):
    """non_monotonic_warning is not dropped from iterations payload."""
    cfg = _cfg(sla_filters=[TTFT_SLA])
    history = [
        SearchIteration(
            iteration_idx=i,
            variation_values={SWEPT: 10 * (i + 1)},
            objective_value=float(100 * (i + 1)),
            feasible=feas,
            non_monotonic_warning=(
                i == 2 and feasibility_pattern == [True, False, True]
            ),
        )
        for i, feas in enumerate(feasibility_pattern)
    ]
    write_search_history(tmp_path, history, cfg)
    payload = _read(tmp_path)

    assert "non_monotonic_warning" in payload["iterations"][0]
    if feasibility_pattern == [True, False, True]:
        assert payload["iterations"][2]["non_monotonic_warning"] is True

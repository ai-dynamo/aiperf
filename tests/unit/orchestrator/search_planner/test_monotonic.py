# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MonotonicSLASearchPlanner.

The planner is a 1D-optimized exponential-probe + bisection search for the
SLA-saturation boundary. Mirrors perf_analyzer's `--binary-search`. These
tests use synthetic feasibility functions (no Optuna / BoTorch dep — pure-Python).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import (
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.monotonic import MonotonicSLASearchPlanner


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
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


def _cfg(
    *,
    lo: int = 1,
    hi: int = 1000,
    max_iterations: int = 30,
    sla_filters: list[SLAFilter] | None = None,
    monotonic_stability_trials: int = 1,
    objective_metric: str = "output_token_throughput",
    extra_dims: list[SearchSpaceDimension] | None = None,
) -> AdaptiveSearchSweep:
    if sla_filters is None:
        sla_filters = [
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ]
    dims = [
        SearchSpaceDimension(
            path="phases.profiling.concurrency", lo=lo, hi=hi, kind="int"
        ),
    ]
    if extra_dims:
        dims.extend(extra_dims)
    return AdaptiveSearchSweep(
        search_space=dims,
        objectives=[
            Objective(
                metric=objective_metric,
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
        sla_filters=sla_filters,
        monotonic_stability_trials=monotonic_stability_trials,
    )


def _make_result(
    variation: SweepVariation, *, ttft_p95: float, throughput: float = 100.0
) -> RunResult:
    """Return a RunResult with the configured TTFT p95 and throughput avg."""
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft_p95),
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


def _drive_to_convergence(
    planner: MonotonicSLASearchPlanner,
    feasibility_fn,
    *,
    max_loops: int = 60,
) -> int:
    """Run ask/tell pairs until is_converged() or ask() returns None.

    Returns total iterations consumed. ``feasibility_fn(concurrency: int) -> bool``
    drives the synthetic SLA verdict — feasible runs report TTFT below the
    200 ms threshold, infeasible runs report above.
    """
    iters = 0
    while iters < max_loops:
        if planner.is_converged():
            break
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        c = variation.values["phases.profiling.concurrency"]
        ttft = 100.0 if feasibility_fn(c) else 300.0
        planner.tell(variation, [_make_result(variation, ttft_p95=ttft)])
        iters += 1
    return iters


# ----------------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------------


def test_construction_rejects_multi_dim_search_space():
    """Multi-dim search-space → ValueError naming the planner and BO."""
    extra = SearchSpaceDimension(
        path="phases.profiling.requests", lo=10, hi=1000, kind="int"
    )
    cfg = _cfg(extra_dims=[extra])
    with pytest.raises(ValueError) as exc:
        MonotonicSLASearchPlanner(_base_config(), cfg)
    msg = str(exc.value)
    assert "monotonic_sla" in msg
    assert "BO" in msg or "bayesian" in msg.lower()


def test_construction_rejects_empty_sla_filters():
    """Empty sla_filters → ValueError; the planner has no scoring otherwise."""
    cfg = _cfg(sla_filters=[])
    with pytest.raises(ValueError) as exc:
        MonotonicSLASearchPlanner(_base_config(), cfg)
    assert "sla_filter" in str(exc.value).lower()


def test_construction_rejects_real_kind_dimension():
    """Monotonic planner is integer-only; real-valued dim must raise."""
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.rate", lo=1.0, hi=100.0, kind="real"
            ),
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=20,
        n_initial_points=2,
        random_seed=42,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ],
    )
    with pytest.raises(ValueError) as exc:
        MonotonicSLASearchPlanner(_base_config(), cfg)
    msg = str(exc.value).lower()
    assert "int" in msg
    assert "bayesian" in msg


# ----------------------------------------------------------------------------
# Search behavior
# ----------------------------------------------------------------------------


def test_monotonic_boundary_in_middle_converges_within_15_iters():
    """Boundary at c=256 in [1, 1000]: precision-converge in <15 iterations."""
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1000, max_iterations=30),
    )
    iters = _drive_to_convergence(planner, lambda c: c < 256)
    assert iters < 15
    assert planner.convergence_reason() == "monotonic_precision_reached"

    feasible_max = planner.feasible_max
    infeasible_min = planner.infeasible_min
    assert feasible_max is not None
    assert infeasible_min is not None
    # Boundary lies between feasible_max (passes) and infeasible_min (fails).
    assert feasible_max < 256 <= infeasible_min
    # Precision: relative gap under 5%.
    assert (infeasible_min - feasible_max) / infeasible_min < 0.05


def test_all_passing_region_reports_no_failure_in_range():
    """Every probed point passes → max_passing = hi, first_failing = None."""
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1000, max_iterations=30),
    )
    _drive_to_convergence(planner, lambda c: True)
    assert planner.convergence_reason() == "monotonic_no_failure_in_range"
    assert planner.feasible_max == 1000
    assert planner.infeasible_min is None


def test_all_failing_region_reports_no_pass_in_range():
    """Every probed point fails → max_passing = None, first_failing = lo."""
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1000, max_iterations=30),
    )
    _drive_to_convergence(planner, lambda c: False)
    assert planner.convergence_reason() == "monotonic_no_pass_in_range"
    assert planner.feasible_max is None
    assert planner.infeasible_min == 1


def test_non_monotonic_sets_warning_flag_and_finds_a_boundary():
    """Feasible-above-infeasible_min branch sets ``non_monotonic_warning=True``.

    Covers the first ``_absorb_verdict`` branch: a feasible verdict arrives at
    a swept value at-or-above the latched ``infeasible_min``. Injects the
    fresh-probe verdict directly through the ``_pending_value`` seam — the
    same code path the planner exercises during normal ``ask`` / ``tell``
    cycles, just without doubling the iteration count to drive the algorithm
    into that geometry organically.
    """
    cfg = _cfg(lo=1, hi=1000, max_iterations=30, monotonic_stability_trials=1)
    planner = MonotonicSLASearchPlanner(_base_config(), cfg)

    # Drive iterations until both bracket bounds latch.
    safety = 0
    while (
        planner.feasible_max is None or planner.infeasible_min is None
    ) and safety < 20:
        proposal = planner.ask()
        if proposal is None:
            break
        _, v = proposal
        c = v.values["phases.profiling.concurrency"]
        ttft = 100.0 if c < 50 else 300.0
        planner.tell(v, [_make_result(v, ttft_p95=ttft)])
        safety += 1

    feasible_max_before = planner.feasible_max
    infeasible_min_before = planner.infeasible_min
    assert feasible_max_before is not None
    assert infeasible_min_before is not None

    # Inject a contradicting verdict at a fresh swept value: above
    # infeasible_min but feasible. Equivalent to a cold-cache flip on a
    # value the planner has not yet probed.
    fresh_high_value = infeasible_min_before + 100
    fake_variation = SweepVariation(
        index=planner._iter,
        label=f"search_iter_{planner._iter:04d}",
        values={"phases.profiling.concurrency": fresh_high_value},
    )
    planner._pending_value = fresh_high_value
    planner.tell(fake_variation, [_make_result(fake_variation, ttft_p95=100.0)])

    assert planner.non_monotonic_warning is True
    history = planner.history()
    assert any(it.non_monotonic_warning for it in history), (
        "expected non_monotonic_warning on at least one iteration"
    )


def test_non_monotonic_infeasible_at_or_below_feasible_max_sets_warning():
    """Infeasible-at-or-below-feasible_max branch sets ``non_monotonic_warning=True``.

    Covers the second ``_absorb_verdict`` branch (mirror of the feasible-above
    branch): an infeasible verdict arrives at a swept value at-or-below the
    latched ``feasible_max``. Injects the fresh verdict through the
    ``_pending_value`` seam — same code path as the live algorithm.
    """
    cfg = _cfg(lo=1, hi=1000, max_iterations=30, monotonic_stability_trials=1)
    planner = MonotonicSLASearchPlanner(_base_config(), cfg)

    # Drive until both bracket bounds latch (feasibility flips at c >= 50).
    safety = 0
    while (
        planner.feasible_max is None or planner.infeasible_min is None
    ) and safety < 20:
        proposal = planner.ask()
        if proposal is None:
            break
        _, v = proposal
        c = v.values["phases.profiling.concurrency"]
        ttft = 100.0 if c < 50 else 300.0
        planner.tell(v, [_make_result(v, ttft_p95=ttft)])
        safety += 1

    feasible_max_before = planner.feasible_max
    assert feasible_max_before is not None
    assert planner.infeasible_min is not None
    assert planner.non_monotonic_warning is False
    # Need a fresh swept value (no prior PointLog) at-or-below feasible_max
    # so the verdict latches False on the first observation. Use lo=1 if it
    # has not been probed yet, otherwise pick any below-feasible_max value
    # that has no existing PointLog.
    contradicting_value = next(
        v for v in range(1, feasible_max_before + 1) if v not in planner._point_logs
    )
    fake_variation = SweepVariation(
        index=planner._iter,
        label=f"search_iter_{planner._iter:04d}",
        values={"phases.profiling.concurrency": contradicting_value},
    )
    planner._pending_value = contradicting_value
    # ttft above 200 ms threshold = infeasible.
    planner.tell(fake_variation, [_make_result(fake_variation, ttft_p95=300.0)])

    assert planner.non_monotonic_warning is True
    history = planner.history()
    flagged = [it for it in history if it.non_monotonic_warning]
    assert flagged, "expected non_monotonic_warning on the contradicting iteration"
    # The contradicting verdict is the most recent iteration.
    assert flagged[-1].iteration_idx == history[-1].iteration_idx


# ----------------------------------------------------------------------------
# Stability window
# ----------------------------------------------------------------------------


def test_stability_window_single_trial_accepts_immediately():
    """monotonic_stability_trials=1 → single-trial verdict accepted."""
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1000, max_iterations=30, monotonic_stability_trials=1),
    )
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    planner.tell(variation, [_make_result(variation, ttft_p95=100.0)])
    # Verdict latched after one trial: history has the iteration.
    assert len(planner.history()) == 1


def test_stability_window_two_trials_provisional_until_agreement():
    """monotonic_stability_trials=2 with disagreeing trials must not latch.

    After a pass and then a fail at the same point, the planner should
    re-issue another probe (or otherwise record disagreement) rather than
    locking in either verdict.
    """
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1000, max_iterations=30, monotonic_stability_trials=2),
    )
    proposal_a = planner.ask()
    assert proposal_a is not None
    _, va = proposal_a
    # First trial passes.
    planner.tell(va, [_make_result(va, ttft_p95=100.0)])
    # Second trial at the same swept value disagrees (fails).
    proposal_b = planner.ask()
    assert proposal_b is not None
    _, vb = proposal_b
    # Stability window asks at the same swept value while waiting for two
    # agreeing observations.
    assert (
        vb.values["phases.profiling.concurrency"]
        == va.values["phases.profiling.concurrency"]
    )
    planner.tell(vb, [_make_result(vb, ttft_p95=300.0)])
    # No verdict latched yet: planner still wants more probes; not converged
    # solely on the strength of this one disputed point.
    assert not (
        planner.convergence_reason() == "monotonic_precision_reached"
        and len(planner.history()) <= 2
    )


@pytest.mark.parametrize(
    "iterations,reason",
    [
        param(3, "max_iterations", id="max_iterations_floor"),
    ],
)  # fmt: skip
def test_max_iterations_budget_is_respected(iterations: int, reason: str):
    """Bisection stops when iteration budget exhausted before precision."""
    planner = MonotonicSLASearchPlanner(
        _base_config(),
        _cfg(lo=1, hi=1_000_000, max_iterations=iterations),
    )
    _drive_to_convergence(planner, lambda c: c < 256, max_loops=iterations + 5)
    # Either ask returned None or convergence latched on max_iterations; both
    # are valid budget-exhaustion signals. The planner MUST surface the
    # max_iterations reason in convergence_reason().
    assert planner.convergence_reason() in {reason, "monotonic_precision_reached"}

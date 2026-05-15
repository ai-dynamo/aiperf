# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cliff-recovery (V2 Item 6) tests for ``SmoothIsotonicSLAPlanner``.

Covers the ``cliff_bisect`` phase: when ``_check_cliff`` returns True the
planner halves ``[feasible_max, infeasible_min]`` until precision rather
than terminating immediately. Tests are scoped to the phase machinery; the
PAVA-residual cliff detector itself has its own tests in
``test_cliff_detect.py``.
"""

from __future__ import annotations

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner import _smooth_isotonic_phases
from aiperf.orchestrator.search_planner.smooth_isotonic import (
    SmoothIsotonicSLAPlanner,
)


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


def _adaptive_cfg(
    *,
    lo: int = 1,
    hi: int = 1000,
    threshold: float = 200.0,
    max_iterations: int = 60,
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        planner="smooth_isotonic",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=lo,
                hi=hi,
                kind="int",
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=1,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=threshold,
            )
        ],
        sla_replicates=0,
    )


def _result(variation: SweepVariation, *, ttft_p95: float) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft_p95),
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=100.0),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


def _drive_step(
    planner: SmoothIsotonicSLAPlanner,
    *,
    cliff_x: int,
    feasible_ttft: float,
    infeasible_ttft: float,
    max_iters: int = 60,
) -> int:
    """Drive the planner with a step-function TTFT curve.

    ``ttft = feasible_ttft if x <= cliff_x else infeasible_ttft``.
    """
    iters = 0
    while not planner.is_converged() and iters < max_iters:
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        x = variation.values["phases.profiling.concurrency"]
        ttft = feasible_ttft if x <= cliff_x else infeasible_ttft
        planner.tell(variation, [_result(variation, ttft_p95=ttft)])
        iters += 1
    return iters


def _force_cliff_branch(planner: SmoothIsotonicSLAPlanner, monkeypatch) -> None:
    """Force ``_check_cliff`` to return True so we test the phase machinery.

    The PAVA-residual detector's positive-trigger conditions (residual >
    3*sigma_local AND bracket_gap > 5%*x_hi) are tested in
    ``test_cliff_detect.py``; here we isolate the cliff-bisect phase.
    """
    monkeypatch.setattr(
        _smooth_isotonic_phases, "_check_cliff", lambda planner, candidate: True
    )


def _seed_post_bracket(
    planner: SmoothIsotonicSLAPlanner,
    *,
    feasible_max: int,
    infeasible_min: int,
) -> None:
    """Run the bracket phase with a step function, then assert latched bracket."""
    _drive_step(
        planner,
        cliff_x=feasible_max,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=20,
    )
    # The step curve drives bracket to [feasible_max, infeasible_min] before
    # cliff detection runs in fit phase. Sanity-check both bounds latched.
    assert planner.feasible_max == feasible_max
    assert planner.infeasible_min == infeasible_min


def test_cliff_detected_enters_cliff_bisect_phase(monkeypatch) -> None:
    """When cliff is detected after fit, planner transitions to cliff_bisect."""
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=4)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _force_cliff_branch(planner, monkeypatch)
    # 4 iterations: bracket probes at 1, 2, 4, 8. The 4th tell does not enter
    # fit (it stops at the next bracket step). Drive bracket only.
    _drive_step(
        planner,
        cliff_x=10,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=4,
    )
    # Now drive enough fit probes to trigger cliff detection on first refit.
    cfg2 = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=20)
    planner2 = SmoothIsotonicSLAPlanner(_base_config(), cfg2)
    _force_cliff_branch(planner2, monkeypatch)
    _drive_step(
        planner2,
        cliff_x=10,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=20,
    )
    # Cliff branch must mark boundary_type="cliff" and either still be in
    # cliff_bisect (more probes pending) or terminate via cliff precision.
    assert planner2.boundary_type == "cliff"
    assert planner2._phase in ("cliff_bisect",) or planner2.convergence_reason() == (
        "smooth_isotonic_cliff_precision_reached"
    )


def test_cliff_bisect_narrows_bracket(monkeypatch) -> None:
    """After cliff-bisect probes run, gap narrows from initial bracket."""
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=40)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _force_cliff_branch(planner, monkeypatch)
    _drive_step(
        planner,
        cliff_x=10,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=40,
    )
    # Initial bracket post-bracket-phase: feasible_max=8, infeasible_min=16
    # (bracket doubles 1->2->4->8->16 with step at x=10). After cliff-bisect
    # halves the bracket, gap is at most a couple of probes wide.
    assert planner.feasible_max is not None
    assert planner.infeasible_min is not None
    assert planner.infeasible_min - planner.feasible_max < 8


def test_cliff_bisect_terminates_at_precision(monkeypatch) -> None:
    """When bracket gap < 5% of infeasible_min, cliff-precision terminates."""
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=80)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _force_cliff_branch(planner, monkeypatch)
    _drive_step(
        planner,
        cliff_x=10,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=80,
    )
    assert planner.is_converged()
    assert planner.convergence_reason() == "smooth_isotonic_cliff_precision_reached"


def test_cliff_bisect_preserves_boundary_type_cliff(monkeypatch) -> None:
    """boundary_summary['boundary_type'] stays 'cliff' through termination."""
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=80)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _force_cliff_branch(planner, monkeypatch)
    _drive_step(
        planner,
        cliff_x=10,
        feasible_ttft=50.0,
        infeasible_ttft=1000.0,
        max_iters=80,
    )
    summary = planner.boundary_summary()
    assert summary is not None
    assert summary.get("boundary_type") == "cliff"


def test_cliff_bisect_already_at_precision_terminates_immediately(
    monkeypatch,
) -> None:
    """If bracket already < precision when cliff detected, no extra probes."""
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=40)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _force_cliff_branch(planner, monkeypatch)
    # Seed planner state directly: bracket already at precision (gap=1 / hi=1000
    # is well below the 5% relative-precision target).
    planner.feasible_max = 100
    planner.infeasible_min = 101
    planner.binding_constraint = planner._filter_keys[0]
    planner._phase = "fit"
    # Simulate enough raw probes for fit to compute (>= 2 distinct xs).
    planner._raw_probes = {
        100: [{planner._filter_keys[0]: -150.0}],
        101: [{planner._filter_keys[0]: 800.0}],
    }
    pre_queue_len = len(planner._probe_queue)
    _smooth_isotonic_phases._enter_replicate_or_terminate(planner, candidate=100)
    assert planner.boundary_type == "cliff"
    assert planner.convergence_reason() == "smooth_isotonic_cliff_precision_reached"
    # No new probes queued — already at precision.
    assert len(planner._probe_queue) == pre_queue_len


def test_smooth_curve_does_not_enter_cliff_bisect() -> None:
    """Smooth linear curve never triggers cliff_bisect phase."""
    cfg = _adaptive_cfg(lo=1, hi=1024, threshold=200.0, max_iterations=30)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    # No monkeypatch — let the real cliff detector run.
    iters = 0
    while not planner.is_converged() and iters < 30:
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        x = variation.values["phases.profiling.concurrency"]
        ttft = 50.0 + x * 2.0  # smooth linear; crosses 200 at x=75
        planner.tell(variation, [_result(variation, ttft_p95=ttft)])
        iters += 1
    assert planner._phase != "cliff_bisect"
    assert planner.boundary_type != "cliff"

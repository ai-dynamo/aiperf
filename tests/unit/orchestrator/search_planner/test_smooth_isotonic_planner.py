# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``SmoothIsotonicSLAPlanner``.

Covers the ABC contract (ask / tell / is_converged / history /
boundary_summary), the bracket -> fit -> terminate state machine, edge
cases (no-pass, no-failure), and the boundary_summary export shape.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.phases import PhaseType
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
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


def _base_config_rate() -> BenchmarkConfig:
    """Base config with a rate-controlled profiling phase for real-axis dims."""
    cfg_dict = _base_config().model_dump(exclude_none=True)
    cfg_dict["phases"][0] = {
        "name": "profiling",
        "type": "poisson",
        "rate": 1.0,
        "requests": 10,
    }
    return BenchmarkConfig.model_validate(cfg_dict)


def _adaptive_cfg(
    *,
    lo: int = 1,
    hi: int = 1000,
    threshold: float = 200.0,
    max_iterations: int = 30,
    sla_replicates: int = 0,
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
        sla_replicates=sla_replicates,
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


def _drive(
    planner: SmoothIsotonicSLAPlanner,
    ttft_curve: Callable[[float], float],
    max_iters: int = 50,
    path: str = "phases.profiling.concurrency",
) -> int:
    """Run the planner; return iter count when converged."""
    iters = 0
    while not planner.is_converged() and iters < max_iters:
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        x = variation.values[path]
        ttft = ttft_curve(float(x))
        planner.tell(variation, [_result(variation, ttft_p95=ttft)])
        iters += 1
    return iters


def test_construction_rejects_multi_dim() -> None:
    cfg = _adaptive_cfg()
    cfg.search_space.append(
        SearchSpaceDimension(
            path="phases.profiling.request_rate", lo=1, hi=10, kind="int"
        )
    )
    with pytest.raises(ValueError, match="exactly one search-space"):
        SmoothIsotonicSLAPlanner(_base_config(), cfg)


def test_construction_accepts_real_dim() -> None:
    cfg = _adaptive_cfg()
    cfg.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=1.0, hi=1000.0, kind="real"
    )
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    assert isinstance(planner._lo, float)
    assert isinstance(planner._hi, float)


def test_construction_rejects_real_dim_with_nonpositive_lo() -> None:
    cfg = _adaptive_cfg()
    cfg.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=0.0, hi=1000.0, kind="real"
    )
    with pytest.raises(ValueError, match="lo > 0"):
        SmoothIsotonicSLAPlanner(_base_config(), cfg)


def test_construction_rejects_int_dim_with_lo_below_one() -> None:
    cfg = _adaptive_cfg(lo=0)
    with pytest.raises(ValueError, match="lo >= 1"):
        SmoothIsotonicSLAPlanner(_base_config(), cfg)


def test_construction_rejects_empty_sla_filters() -> None:
    cfg = _adaptive_cfg()
    cfg.sla_filters = []
    with pytest.raises(ValueError, match="at least one SLA filter"):
        SmoothIsotonicSLAPlanner(_base_config(), cfg)


def test_no_pass_in_range_when_first_probe_fails() -> None:
    """First (lo) probe already infeasible -> no_pass_in_range."""
    cfg = _adaptive_cfg(lo=100, hi=1000, threshold=50.0)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _drive(planner, lambda x: 200.0)
    assert planner.is_converged()
    assert planner.convergence_reason() == "smooth_isotonic_no_pass_in_range"


def test_no_failure_in_range_when_all_probes_pass() -> None:
    """Every probe up to hi passes -> no_failure_in_range."""
    cfg = _adaptive_cfg(lo=1, hi=64, threshold=200.0)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _drive(planner, lambda x: 50.0)
    assert planner.is_converged()
    assert planner.convergence_reason() == "smooth_isotonic_no_failure_in_range"


def test_finds_boundary_on_smooth_curve() -> None:
    """Smoothly-increasing TTFT crosses 200ms around concurrency=75."""
    cfg = _adaptive_cfg(lo=1, hi=1024, threshold=200.0)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    # TTFT = 50 + concurrency * 2 (linear). Crosses 200 at concurrency=75.
    _drive(planner, lambda x: 50.0 + 2.0 * x)
    assert planner.is_converged()
    summary = planner.boundary_summary()
    assert summary is not None
    fmax = summary["feasible_max"]
    imin = summary["infeasible_min"]
    assert fmax is not None
    assert imin is not None
    assert 64 <= fmax["value"] <= 80, f"feasible_max={fmax['value']!r}"
    assert imin["value"] - fmax["value"] <= max(4, fmax["value"] // 20)


def test_finds_boundary_on_real_axis() -> None:
    """Real-valued dim: TTFT = 50 + rate*2 crosses 200 at rate=75.

    Exercises the float bracket/fit/bisect path end-to-end; the boundary
    band must straddle 75 and respect the relative precision target.
    """
    cfg = _adaptive_cfg()
    cfg.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=1.0, hi=1024.0, kind="real"
    )
    planner = SmoothIsotonicSLAPlanner(_base_config_rate(), cfg)
    _drive(planner, lambda x: 50.0 + 2.0 * x, path="phases.profiling.rate")
    assert planner.is_converged()
    assert planner.convergence_reason() in (
        "smooth_isotonic_precision_reached",
        "smooth_isotonic_cliff_precision_reached",
    )
    summary = planner.boundary_summary()
    assert summary is not None
    fmax = summary["feasible_max"]
    imin = summary["infeasible_min"]
    assert fmax is not None
    assert imin is not None
    assert 60 <= fmax["value"] <= 80, f"feasible_max={fmax['value']!r}"
    assert 60 <= imin["value"] <= 90, f"infeasible_min={imin['value']!r}"
    assert (imin["value"] - fmax["value"]) / imin["value"] < 0.10


def test_boundary_summary_includes_smooth_isotonic_fields() -> None:
    cfg = _adaptive_cfg(lo=1, hi=512, threshold=200.0)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _drive(planner, lambda x: 50.0 + 2.0 * x)
    summary = planner.boundary_summary()
    assert summary is not None
    assert summary["swept_dim_path"] == "phases.profiling.concurrency"
    # boundary_type latches to "smooth" or "cliff" once the fit step ran.
    assert summary.get("boundary_type") in ("smooth", "cliff", None)
    # Smooth linear curve should not trip the cliff guard.
    assert summary.get("boundary_type") != "cliff"


def test_history_grows_per_iteration() -> None:
    cfg = _adaptive_cfg(lo=1, hi=512, threshold=200.0, max_iterations=8)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    iters = _drive(planner, lambda x: 50.0 + 2.0 * x)
    assert iters == len(planner.history())
    assert iters >= 3
    for i, entry in enumerate(planner.history()):
        assert entry.iteration_idx == i
        assert "phases.profiling.concurrency" in entry.variation_values


def test_max_iterations_reason() -> None:
    cfg = _adaptive_cfg(lo=1, hi=4096, threshold=200.0, max_iterations=3)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    _drive(planner, lambda x: 50.0 + 2.0 * x)
    assert planner.is_converged()
    assert planner.convergence_reason() == "max_iterations"


def test_tell_without_ask_raises() -> None:
    cfg = _adaptive_cfg()
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
    fake_variation = SweepVariation(
        index=0, label="x", values={"phases.profiling.concurrency": 1}
    )
    with pytest.raises(RuntimeError, match="without matching ask"):
        planner.tell(fake_variation, [_result(fake_variation, ttft_p95=50.0)])


def test_sla_warmup_mirrors_rate_phase_for_real_dim() -> None:
    """Auto warmup must run at the swept rate for a rate-controlled
    profiling phase, not a hardcoded (non-integer) concurrency."""
    cfg = _adaptive_cfg()
    cfg.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=1.0, hi=1000.0, kind="real"
    )
    planner = SmoothIsotonicSLAPlanner(_base_config_rate(), cfg)
    mutated = planner._mutate_base(50.5)
    warmup = mutated.phases[0]
    assert warmup.name == "warmup"
    assert warmup.type == PhaseType.POISSON
    assert warmup.rate == 50.5
    assert warmup.exclude_from_results is True
    assert mutated.phases[1].rate == 50.5


def test_non_monotonic_warning_threaded_per_iteration() -> None:
    """Issue #12: a non-monotonic verdict must be flagged on the iteration
    that triggered it.

    Mirrors ``MonotonicSLASearchPlanner``'s per-iteration semantic: only the
    iteration whose verdict revealed non-monotonicity carries
    ``non_monotonic_warning=True``; earlier (consistent) iterations stay
    False even after the planner-level cumulative flag latches True.
    """
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=30)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)

    # Drive bracket until both bounds latch: feasibility flips at c >= 50.
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
        planner.tell(v, [_result(v, ttft_p95=ttft)])
        safety += 1

    pre_history = planner.history()
    assert pre_history, "expected the planner to have recorded bracket iterations"
    assert all(not it.non_monotonic_warning for it in pre_history), (
        "no bracket iteration should be flagged before the contradicting verdict"
    )
    assert planner.infeasible_min is not None

    # Inject a fresh, contradicting verdict: feasible at a swept value
    # above ``infeasible_min``. Drives the first ``_absorb_verdict`` branch
    # without re-running the bracket.
    fresh_high_value = planner.infeasible_min + 100
    fake_variation = SweepVariation(
        index=planner._iter,
        label=f"search_iter_{planner._iter:04d}",
        values={"phases.profiling.concurrency": fresh_high_value},
    )
    planner._pending_value = fresh_high_value
    planner.tell(fake_variation, [_result(fake_variation, ttft_p95=50.0)])

    history = planner.history()
    flagged = [it for it in history if it.non_monotonic_warning]
    assert flagged, (
        "expected non_monotonic_warning on the iteration that revealed the "
        "non-monotonic boundary"
    )
    # Only the most recent iteration carries the flag; previous-iteration
    # records remain False.
    assert flagged[-1].iteration_idx == history[-1].iteration_idx
    assert len(flagged) == 1


def test_non_monotonic_warning_per_iteration_does_not_back_propagate() -> None:
    """Once a non-monotonic verdict latches the planner-level cumulative
    flag, a subsequent monotonic iteration must NOT inherit
    ``non_monotonic_warning=True`` on its own ``SearchIteration`` record.
    """
    cfg = _adaptive_cfg(lo=1, hi=1000, threshold=200.0, max_iterations=30)
    planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)

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
        planner.tell(v, [_result(v, ttft_p95=ttft)])
        safety += 1

    assert planner.infeasible_min is not None
    fresh_high_value = planner.infeasible_min + 100
    bad = SweepVariation(
        index=planner._iter,
        label=f"search_iter_{planner._iter:04d}",
        values={"phases.profiling.concurrency": fresh_high_value},
    )
    planner._pending_value = fresh_high_value
    planner.tell(bad, [_result(bad, ttft_p95=50.0)])
    assert planner.non_monotonic_warning is True

    # A subsequent monotonic-consistent verdict (infeasible above the new
    # feasible_max) should not be tagged.
    assert planner.feasible_max is not None
    consistent_value = max(planner.feasible_max + 1, fresh_high_value + 1)
    good = SweepVariation(
        index=planner._iter,
        label=f"search_iter_{planner._iter:04d}",
        values={"phases.profiling.concurrency": consistent_value},
    )
    planner._pending_value = consistent_value
    planner.tell(good, [_result(good, ttft_p95=300.0)])

    last = planner.history()[-1]
    assert last.iteration_idx == planner._iter - 1
    assert last.non_monotonic_warning is False, (
        "a monotonic-consistent iteration must not back-inherit the cumulative flag"
    )
    # Cumulative planner-level flag stays True.
    assert planner.non_monotonic_warning is True

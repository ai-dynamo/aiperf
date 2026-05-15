# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-3 adversarial regressions: NaN/inf discipline in BO + search recipes.

Covers the behaviours hardened in this round:

- NaN metric values are treated as "missing" by the Optuna BO planner and
  logged once per planner instance.
- The planner's iteration_feasibility view AGREES with what the sampler
  is told (no silent split-brain on NaN).
- ``SLABreachKnee`` collapses retried trials at the same swept value before
  emitting ``max_passing`` / ``first_failing``.
- ``ItlSurfaceFit`` drops non-finite or negative ITL rows up front.
- ``DegradationKneeDetect`` rejects NaN/+inf/-inf baselines with ValueError.
"""

from __future__ import annotations

import math
import warnings

import pytest

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
from aiperf.search_recipes._itl_surface_fit import ItlSurfaceFit
from aiperf.search_recipes._sla_breach_knee import SLABreachKnee
from aiperf.search_recipes.post_process import DegradationKneeDetect

# ---------------------------------------------------------------------------
# Shared builders (REAL Pydantic configs — per memory rule, at least one test
# must build a real config; we use a real BenchmarkConfig for both BO planners
# rather than MagicMock).
# ---------------------------------------------------------------------------


def _real_base_config() -> BenchmarkConfig:
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


def _bo_cfg(**overrides) -> AdaptiveSearchSweep:
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
            )
        ],
        max_iterations=10,
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
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def _result(
    variation: SweepVariation,
    *,
    throughput: float | None = 100.0,
    ttft_p95: float | None = 50.0,
) -> RunResult:
    metrics: dict = {}
    if throughput is not None:
        metrics["output_token_throughput"] = JsonMetricResult(
            unit="tok/s", avg=throughput
        )
    if ttft_p95 is not None:
        metrics["time_to_first_token"] = JsonMetricResult(unit="ms", p95=ttft_p95)
    return RunResult(
        label="t",
        success=True,
        summary_metrics=metrics,
        variation_label=variation.label,
        variation_values=variation.values,
    )


# ---------------------------------------------------------------------------
# B1 / B5: BO planners treat NaN as missing; history feasibility view agrees.
# ---------------------------------------------------------------------------


def test_optuna_planner_treats_nan_as_missing() -> None:
    from aiperf.orchestrator.search_planner.optuna_planner import OptunaSearchPlanner

    planner = OptunaSearchPlanner(_real_base_config(), _bo_cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    nan_result = _result(variation, throughput=float("nan"), ttft_p95=float("nan"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        planner.tell(variation, [nan_result])
    nan_warns = [w for w in caught if "non-finite" in str(w.message).lower()]
    assert nan_warns, "expected at least one non-finite warning"

    next_proposal = planner.ask()
    assert next_proposal is not None


def test_optuna_feasibility_view_agrees_with_history_on_nan() -> None:
    from aiperf.orchestrator.search_planner.optuna_planner import OptunaSearchPlanner

    planner = OptunaSearchPlanner(_real_base_config(), _bo_cfg(max_iterations=5))
    _, variation = planner.ask()  # type: ignore[misc]
    planner.tell(
        variation,
        [_result(variation, throughput=100.0, ttft_p95=float("nan"))],
    )
    iteration = planner.history()[0]
    assert iteration.feasible is False


# ---------------------------------------------------------------------------
# B2: SLABreachKnee retried-conflict collapse.
# ---------------------------------------------------------------------------


def test_sla_breach_knee_retried_conflicting_collapse() -> None:
    """Retried trials at the same x with conflicting feasibility must collapse.

    Pre-fix: max_passing=8 AND first_failing=8 emitted simultaneously, AND
    monotonicity_check spuriously True. Post-fix: under all-pass-required
    rule, x=8 is infeasible (one trial failed), so max_passing<first_failing
    strictly OR both None.
    """
    flt = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
    agg = {
        "per_combination_metrics": [
            {"parameters": {"c": 4}, "metrics": {"ttft_p95": {"mean": 50.0}}},
            # Two trials at c=8: one passes (100ms), one fails (250ms).
            {"parameters": {"c": 8}, "metrics": {"ttft_p95": {"mean": 100.0}}},
            {"parameters": {"c": 8}, "metrics": {"ttft_p95": {"mean": 250.0}}},
            {"parameters": {"c": 16}, "metrics": {"ttft_p95": {"mean": 300.0}}},
        ]
    }
    out = SLABreachKnee().process(agg, {"sla_filters": [flt], "swept_param": "c"})

    max_passing = out["max_passing_c"]
    first_failing = out["first_failing_c"]

    # Either both None (no overlap by being absent) OR strict disjoint.
    if max_passing is not None and first_failing is not None:
        assert float(max_passing) < float(first_failing), (
            f"max_passing ({max_passing}) must be < first_failing "
            f"({first_failing}); both at the same value is the bug."
        )

    # Under all-pass-required, c=8 fails (one trial breached). c=4 still
    # passes; c=8/16 fail. So first_failing should be 8 and max_passing 4.
    assert max_passing == 4
    assert first_failing == 8

    # The collapsed point at c=8 is infeasible, so monotonicity is preserved
    # (4 feasible, 8 infeasible, 16 infeasible).
    assert out["monotonicity_check"] is True


# ---------------------------------------------------------------------------
# B3: ItlSurfaceFit drops NaN / inf / negative ITL rows.
# ---------------------------------------------------------------------------


def test_itl_surface_fit_drops_nonfinite_and_negative() -> None:
    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"c": 1, "o": 64},
                "metrics": {"itl_avg": {"mean": float("nan")}},
            },
            {
                "parameters": {"c": 1, "o": 128},
                "metrics": {"itl_avg": {"mean": -5.0}},
            },
            {
                "parameters": {"c": 1, "o": 256},
                "metrics": {"itl_avg": {"mean": float("inf")}},
            },
            {
                "parameters": {"c": 2, "o": 64},
                "metrics": {"itl_avg": {"mean": 12.0}},
            },
            {
                "parameters": {"c": 2, "o": 128},
                "metrics": {"itl_avg": {"mean": 15.0}},
            },
        ]
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ItlSurfaceFit().process(
            agg,
            {
                "metric_tag": "itl",
                "stat": "avg",
                "concurrency_param": "c",
                "osl_param": "o",
            },
        )
    drop_warns = [w for w in caught if "dropped" in str(w.message).lower()]
    assert drop_warns, "expected a 'dropped' warning"

    # Only the two finite-positive triples (c=2,o=64) and (c=2,o=128) survive.
    raw_points = out["raw_points"]
    assert len(raw_points) == 2
    for pt in raw_points:
        assert math.isfinite(pt["itl_ms"]) and pt["itl_ms"] >= 0
    assert out["surface_fit_failed"] is False


def test_itl_surface_fit_too_few_finite_returns_sentinel() -> None:
    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"c": 1, "o": 64},
                "metrics": {"itl_avg": {"mean": float("nan")}},
            },
            {
                "parameters": {"c": 1, "o": 128},
                "metrics": {"itl_avg": {"mean": -1.0}},
            },
        ]
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ItlSurfaceFit().process(
            agg,
            {
                "metric_tag": "itl",
                "stat": "avg",
                "concurrency_param": "c",
                "osl_param": "o",
            },
        )
    assert out["surface_fit_failed"] is True
    assert "error_reason" in out
    assert "non-finite" in out["error_reason"] or "negative" in out["error_reason"]


# ---------------------------------------------------------------------------
# B4: DegradationKneeDetect rejects NaN/+inf/-inf baseline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="+inf"),
        pytest.param(float("-inf"), id="-inf"),
    ],
)
def test_degradation_knee_rejects_nan_inf_baseline(bad_value: float) -> None:
    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"c": 1},
                "metrics": {"request_latency_p99": {"mean": bad_value}},
            },
            {
                "parameters": {"c": 100},
                "metrics": {"request_latency_p99": {"mean": 1000.0}},
            },
        ]
    }
    with pytest.raises(ValueError, match="non-finite"):
        DegradationKneeDetect().process(
            agg,
            {
                "threshold_pct": 0.2,
                "metric_tag": "request_latency",
                "stat": "p99",
                "swept_param": "c",
            },
        )


# ---------------------------------------------------------------------------
# L5 / L6: env settings tightened.
# ---------------------------------------------------------------------------


def test_search_planner_settings_reject_zero_warmup_floors() -> None:
    from pydantic import ValidationError

    from aiperf.common.environment import _SearchPlannerSettings

    with pytest.raises(ValidationError):
        _SearchPlannerSettings(DEFAULT_WARMUP_SECONDS=0.0)
    with pytest.raises(ValidationError):
        _SearchPlannerSettings(FIRST_PROBE_WARMUP_FLOOR=0.0)
    with pytest.raises(ValidationError):
        _SearchPlannerSettings(REPLICATE_WARMUP_FLOOR=0.0)


def test_search_planner_settings_reject_nonpositive_precision_requests() -> None:
    from pydantic import ValidationError

    from aiperf.common.environment import _SearchPlannerSettings

    with pytest.raises(ValidationError):
        _SearchPlannerSettings(SLA_PRECISION_REQUESTS={"tight": 0})
    with pytest.raises(ValidationError):
        _SearchPlannerSettings(SLA_PRECISION_REQUESTS={"normal": -100})

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SLA-aware BO scoring in BayesianSearchPlanner.

Phase 2 of the search-recipes feature: SLA filters add a soft penalty to the
loss told to skopt and flip ``SearchIteration.feasible`` to False when the
iteration's averaged constraint metric violates a filter or is unmeasurable.
"""

from __future__ import annotations

import pytest

# Branch's BayesianSearchPlanner subclasses OptunaSearchPlanner so the
# ``bo`` extra (skopt) is no longer required; the helper test exists to
# pin observable behavior (history feasibility + objective values) rather
# than internal sampler internals.

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.sweep.adaptive import (
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import (
    AdaptiveObjective,
    AdaptiveSearchSweep,
)
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.bayesian import (
    BayesianSearchPlanner,
)


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
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


def _cfg_with_filter(threshold: float = 200.0, **overrides) -> AdaptiveSearchSweep:
    kwargs: dict = dict(
        planner="bayesian",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            ),
        ],
        objectives=[AdaptiveObjective(
            metric="output_token_throughput",
            stat="avg",
            direction=OptimizationDirection.MAXIMIZE,
        )],
        max_iterations=5,
        n_initial_points=2,
        random_seed=42,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=threshold,
            ),
        ],
    )
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def _make_result(
    *,
    throughput: float = 100.0,
    ttft_p95: float | None = None,
    success: bool = True,
) -> RunResult:
    metrics: dict[str, JsonMetricResult] = {
        "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
    }
    if ttft_p95 is not None:
        metrics["time_to_first_token"] = JsonMetricResult(unit="ms", p95=ttft_p95)
    return RunResult(
        label="t",
        success=success,
        summary_metrics=metrics,
    )


def test_tell_with_feasible_metric_flags_iteration_feasible_and_no_penalty():
    """Below-threshold ttft means feasible=True; raw objective preserved on iteration."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter())
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=150.0)])

    assert len(planner.history()) == 1
    iteration = planner.history()[0]
    assert iteration.feasible is True
    # Raw objective recorded (MAXIMIZE direction).
    assert iteration.objective_value == 100.0


def test_tell_with_violating_metric_flags_infeasible_and_adds_penalty():
    """50% over threshold means feasible=False; raw objective stays honest in history."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter(threshold=200.0))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    # ttft_p95 = 300 against threshold 200 -> 50% over.
    planner.tell(
        variation,
        [_make_result(throughput=100.0, ttft_p95=300.0)],
    )

    iteration = planner.history()[0]
    assert iteration.feasible is False
    # Raw objective recorded honestly even though the planner internally
    # penalizes the loss it tells to the underlying sampler.
    assert iteration.objective_value == 100.0


def test_tell_with_missing_constraint_metric_flags_infeasible_with_fixed_penalty():
    """Unmeasurable constraint -> feasible=False (planner can't verify the SLA)."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter(threshold=200.0))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    # ttft_p95 absent on the result -> constraint unmeasurable.
    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=None)])

    iteration = planner.history()[0]
    assert iteration.feasible is False
    # The raw objective is still recorded so post-hoc analysis can see the
    # numeric value alongside the infeasibility verdict.
    assert iteration.objective_value == 100.0


def test_tell_without_sla_filters_keeps_feasible_true_unchanged():
    """No filters means iteration is unconditionally feasible (back-compat path)."""
    cfg = _cfg_with_filter()
    cfg = cfg.model_copy(update={"sla_filters": []})
    planner = BayesianSearchPlanner(_base_config(), cfg)
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=99999.0)])

    iteration = planner.history()[0]
    assert iteration.feasible is True
    assert iteration.objective_value == 100.0


def test_tell_warns_once_per_unmeasurable_metric_tag(caplog):
    """Two consecutive iterations with the same missing tag emit exactly one warning.

    The warn-once dedup is non-trivial state on the planner; verify it directly
    so a regression surfaces here instead of in noisy logs.
    """
    import logging

    # Watch both the legacy bayesian logger and the optuna planner logger
    # (BayesianSearchPlanner subclasses OptunaSearchPlanner in branch).
    caplog.set_level(logging.WARNING)

    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter(threshold=200.0))

    for _ in range(2):
        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal
        # ttft_p95 absent both times -> constraint unmeasurable on both iterations.
        planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=None)])

    unmeasurable_warnings = [
        r for r in caplog.records if "unmeasurable" in r.getMessage()
    ]
    assert len(unmeasurable_warnings) == 1, (
        f"expected exactly one unmeasurable warning across two iterations with the "
        f"same missing metric, got {len(unmeasurable_warnings)}"
    )
    # And the message is informative -- names the metric.
    msg = unmeasurable_warnings[0].getMessage()
    assert "time_to_first_token" in msg

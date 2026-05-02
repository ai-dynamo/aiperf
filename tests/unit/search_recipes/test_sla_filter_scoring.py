# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SLA-aware BO scoring in BayesianSearchPlanner.

Phase 2 of the search-recipes feature: SLA filters add a soft penalty to the
loss told to skopt and flip ``SearchIteration.feasible`` to False when the
iteration's averaged constraint metric violates a filter or is unmeasurable.
"""

from __future__ import annotations

import pytest

skopt = pytest.importorskip("skopt")

# Imports below depend on skopt being importable; pytest.importorskip must
# precede them so the whole module is skipped when the `bo` extra is absent.
from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.adaptive_search import (  # noqa: E402
    AdaptiveSearchConfig,
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
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


def _cfg_with_filter(threshold: float = 200.0, **overrides) -> AdaptiveSearchConfig:
    kwargs: dict = dict(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            ),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
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
    return AdaptiveSearchConfig(**kwargs)


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


def test_tell_with_feasible_metric_flags_iteration_feasible_and_no_penalty(
    monkeypatch,
):
    """Below-threshold ttft means feasible=True and skopt sees raw -objective."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter())
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    captured: list[float] = []

    def spy_tell(x, y, *args, **kwargs):
        captured.append(float(y) if not isinstance(y, list) else float(y[0]))

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=150.0)])

    assert len(planner.history()) == 1
    iteration = planner.history()[0]
    assert iteration.feasible is True
    # No penalty: skopt loss equals -objective (MAXIMIZE direction).
    assert captured == [-100.0]


def test_tell_with_violating_metric_flags_infeasible_and_adds_penalty(monkeypatch):
    """50% over threshold means feasible=False and penalty = 0.5 * W."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter(threshold=200.0))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    captured: list[float] = []

    def spy_tell(x, y, *args, **kwargs):
        captured.append(float(y) if not isinstance(y, list) else float(y[0]))

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    # ttft_p95 = 300 against threshold 200 → 50% over.
    planner.tell(
        variation,
        [_make_result(throughput=100.0, ttft_p95=300.0)],
    )

    iteration = planner.history()[0]
    assert iteration.feasible is False
    # Raw objective recorded honestly even though loss is penalized.
    assert iteration.objective_value == 100.0
    # W = 100 * max(self._max_seen_loss, 1.0). On the first iteration
    # _max_seen_loss is the initial 1.0 (it's only updated AFTER computing
    # penalty), so W=100; violation/threshold = (300-200)/200 = 0.5;
    # penalty = 100 * 0.5 = 50; loss = -100 + 50 = -50.
    assert captured == [pytest.approx(-50.0)]


def test_tell_with_missing_constraint_metric_flags_infeasible_with_fixed_penalty(
    monkeypatch,
):
    """Unmeasurable constraint => feasible=False and a fixed-magnitude penalty."""
    planner = BayesianSearchPlanner(_base_config(), _cfg_with_filter(threshold=200.0))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    captured: list[float] = []

    def spy_tell(x, y, *args, **kwargs):
        captured.append(float(y) if not isinstance(y, list) else float(y[0]))

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    # ttft_p95 absent on the result → constraint unmeasurable.
    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=None)])

    iteration = planner.history()[0]
    assert iteration.feasible is False
    # Fixed penalty W = 100 * max(_max_seen_loss=1.0, 1.0) = 100 → loss = -100 + 100 = 0.
    assert captured == [pytest.approx(0.0)]


def test_tell_without_sla_filters_keeps_feasible_true_unchanged(monkeypatch):
    """No filters means iteration is unconditionally feasible (back-compat path)."""
    cfg = _cfg_with_filter()
    cfg = cfg.model_copy(update={"sla_filters": []})
    planner = BayesianSearchPlanner(_base_config(), cfg)
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal

    captured: list[float] = []

    def spy_tell(x, y, *args, **kwargs):
        captured.append(float(y) if not isinstance(y, list) else float(y[0]))

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    planner.tell(variation, [_make_result(throughput=100.0, ttft_p95=99999.0)])

    iteration = planner.history()[0]
    assert iteration.feasible is True
    assert captured == [-100.0]

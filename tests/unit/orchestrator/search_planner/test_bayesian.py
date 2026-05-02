# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BayesianSearchPlanner.

Skopt is a soft dep; tests skip when not installed. Local CI must install
the `bo` extra.
"""

from __future__ import annotations

import pytest

skopt = pytest.importorskip("skopt")

# Imports below depend on skopt being importable. pytest.importorskip must
# precede them so the whole module is skipped when the `bo` extra is absent.
from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.adaptive_search import (  # noqa: E402
    AdaptiveSearchConfig,
    SearchSpaceDimension,
)
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.sweep import SweepVariation  # noqa: E402
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


def _cfg(max_iterations: int = 5, **overrides) -> AdaptiveSearchConfig:
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
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
    )
    kwargs.update(overrides)
    return AdaptiveSearchConfig(**kwargs)


def test_ask_returns_cfg_and_variation():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    cfg, variation = proposal
    assert variation.index == 0
    assert variation.label.startswith("search_iter_")
    assert "phases.profiling.concurrency" in variation.values
    proposed = variation.values["phases.profiling.concurrency"]
    assert 1 <= proposed <= 100
    assert isinstance(proposed, int)  # int dim → integer
    # The mutated cfg must reflect the proposed value.
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.concurrency == proposed


def test_ask_returns_none_after_max_iterations():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=3))
    for _ in range(3):
        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal
        planner.tell(variation, [_make_result(variation, throughput=100.0)])
    assert planner.ask() is None


def test_record_extracts_avg_from_summary_metrics_and_signs_for_maximize():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    planner.tell(variation, [_make_result(variation, throughput=42.5)])
    history = planner.history()
    assert len(history) == 1
    assert history[0].objective_value == pytest.approx(42.5)


def test_record_skips_failed_runs():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    failed = RunResult(label="x", success=False, error="boom")
    planner.tell(variation, [failed, _make_result(variation, throughput=10.0)])
    assert planner.history()[0].objective_value == pytest.approx(10.0)


def test_record_with_no_successful_runs_records_none():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    planner.tell(variation, [RunResult(label="x", success=False)])
    assert planner.history()[0].objective_value is None


def test_minimize_direction_signs_correctly():
    cfg = _cfg(max_iterations=3, objective_direction=OptimizationDirection.MINIMIZE)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    _, v1 = planner.ask()
    planner.tell(v1, [_make_result(v1, throughput=10.0)])
    _, v2 = planner.ask()
    # If skopt sees signed values correctly, asking again does not crash.
    assert v2 is not None


def test_is_converged_on_max_iterations_exhausted():
    planner = BayesianSearchPlanner(
        _base_config(), _cfg(max_iterations=2, n_initial_points=1, plateau_window=2)
    )
    assert not planner.is_converged()
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=1.0)])
    assert planner.is_converged()


def test_is_converged_on_plateau():
    cfg = _cfg(max_iterations=20, plateau_window=3, plateau_threshold=0.05)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0)])
    assert planner.is_converged()


def _make_result(variation: SweepVariation, *, throughput: float) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )

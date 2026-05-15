# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-objective Optuna planner tests.

Verifies the Optuna planner accepts ``objectives=[obj_a, obj_b, ...]`` and
plumbs an objective vector through ``study.tell`` end-to-end:
  - the underlying study is built with ``directions=[..., ...]`` (one per objective)
  - ``SearchIteration.objective_values`` records the full vector
  - the legacy ``SearchIteration.objective_value`` mirrors ``objective_values[0]``
    for backward-compat consumers
"""

from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

# Imports below depend on optuna being importable.
from aiperf.common.enums import OptimizationDirection  # noqa: E402
from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.sweep import (  # noqa: E402
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.search_planner.optuna_planner import (  # noqa: E402
    OptunaSearchPlanner,
)
from aiperf.plugin.enums import SearchPlannerType  # noqa: E402


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


def _two_obj_cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        planner=SearchPlannerType.OPTUNA,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
            ),
            Objective(
                metric="time_to_first_token",
                direction=OptimizationDirection.MINIMIZE,
            ),
        ],
        max_iterations=5,
        n_initial_points=2,
        # Multi-objective TPE; doesn't require the optional BoTorch stack.
        optuna_sampler="tpe",
        optuna_acquisition=None,
        random_seed=42,
        sla_filters=[],
    )


def _make_result(
    variation: SweepVariation,
    *,
    throughput: float,
    ttft_avg: float,
) -> RunResult:
    summary: dict[str, JsonMetricResult] = {
        "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        "time_to_first_token": JsonMetricResult(unit="ms", avg=ttft_avg),
    }
    return RunResult(
        label="t",
        success=True,
        summary_metrics=summary,
        variation_label=variation.label,
        variation_values=variation.values,
    )


def test_optuna_planner_uses_directions_list_for_multi_objective():
    cfg = _two_obj_cfg()
    planner = OptunaSearchPlanner(_base_config(), cfg)
    directions = [d.name for d in planner._study.directions]
    assert directions == ["MAXIMIZE", "MINIMIZE"]


def test_optuna_planner_tell_accepts_vector_for_multi_objective():
    cfg = _two_obj_cfg()
    planner = OptunaSearchPlanner(_base_config(), cfg)
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    fake_results = [_make_result(variation, throughput=100.0, ttft_avg=50.0)]
    planner.tell(variation, fake_results)
    history = planner.history()
    assert history[-1].objective_values == [100.0, 50.0]
    # Backward-compat scalar mirrors the first element.
    assert history[-1].objective_value == 100.0


def test_optuna_planner_single_objective_still_records_vector():
    """Single-objective configs still populate the vector field."""
    cfg = AdaptiveSearchSweep(
        planner=SearchPlannerType.OPTUNA,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
            ),
        ],
        max_iterations=5,
        n_initial_points=2,
        optuna_sampler="tpe",
        random_seed=42,
        sla_filters=[],
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)
    assert [d.name for d in planner._study.directions] == ["MAXIMIZE"]
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    planner.tell(variation, [_make_result(variation, throughput=42.0, ttft_avg=10.0)])
    last = planner.history()[-1]
    assert last.objective_values == [42.0]
    assert last.objective_value == 42.0


def test_build_qnehvi_candidates_func_returns_callable():
    pytest.importorskip("botorch")
    from aiperf.orchestrator.search_planner._optuna_helpers import (
        build_qnehvi_candidates_func,
    )

    func = build_qnehvi_candidates_func(reference_point=[0.0, 0.0])
    assert callable(func)


def test_qnehvi_candidates_func_runs_on_synthetic_2obj():
    pytest.importorskip("botorch")
    import torch

    from aiperf.orchestrator.search_planner._optuna_helpers import (
        build_qnehvi_candidates_func,
    )

    func = build_qnehvi_candidates_func(reference_point=[-100.0, -100.0])
    train_x = torch.tensor([[0.1], [0.5], [0.9]])
    train_obj = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    bounds = torch.tensor([[0.0], [1.0]])
    candidates = func(train_x, train_obj, None, bounds, None)
    assert candidates.shape[-1] == 1  # 1D search space


def test_multi_objective_improvement_patience_uses_hypervolume_delta():
    """Identical observations -> hypervolume flatlines -> patience triggers.

    plateau_window is set high so the scalar plateau_cv signal is
    structurally unable to fire — only hypervolume tracking can drive the
    counter to ``improvement_patience``.
    """
    pytest.importorskip("botorch")
    cfg = AdaptiveSearchSweep(
        planner=SearchPlannerType.OPTUNA,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
                threshold=0.0,
            ),
            Objective(
                metric="time_to_first_token",
                direction=OptimizationDirection.MINIMIZE,
                threshold=1000.0,
            ),
        ],
        max_iterations=30,
        n_initial_points=2,
        improvement_patience=3,
        plateau_window=20,
        optuna_sampler="tpe",
        random_seed=42,
        sla_filters=[],
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)
    for _ in range(15):
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        planner.tell(
            variation,
            [_make_result(variation, throughput=100.0, ttft_avg=50.0)],
        )
    assert planner.is_converged() is True
    assert planner.convergence_reason() == "improvement_patience"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator.execute_adaptive_search."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("botorch")

# Imports below depend on Optuna+BoTorch being importable. pytest.importorskip
# must precede them so the whole module is skipped when the optional BoTorch
# stack is absent.
import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun  # noqa: E402
from aiperf.config.sweep import (  # noqa: E402
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension  # noqa: E402
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection  # noqa: E402
from aiperf.orchestrator.executor import RunExecutor  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
    BayesianSearchPlanner,
)


class _RecordingExecutor(RunExecutor):
    """Returns synthetic RunResult with a configurable objective metric value."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, dict]] = []

    def derive_id(self, plan, var_idx: int, trial: int) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        self.calls.append((run.variation.index, run.trial, dict(run.variation.values)))
        # Linear objective so the BO learns: throughput = concurrency * 10.
        concurrency = run.variation.values.get("phases.profiling.concurrency", 1)
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tok/s",
                    avg=float(concurrency) * 10.0,
                ),
            },
            artifacts_path=run.artifact_dir,
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
                    "requests": 1,
                    "concurrency": 1,
                },
            ],
        }
    )


def _plan_with_bo(max_iterations: int = 4, trials: int = 1) -> BenchmarkPlan:
    sweep = AdaptiveSearchSweep(
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
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
    )
    return BenchmarkPlan(
        configs=[_base_config()],
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=trials,
        sweep=sweep,
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_adaptive_search_runs_max_iterations_iterations(tmp_path: Path):
    plan = _plan_with_bo(max_iterations=4, trials=1)
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    executor = _RecordingExecutor()

    results = await orch.execute_adaptive_search(plan, executor, planner)
    assert len(results) == 4  # max_iterations × trials
    assert all(r.success for r in results)
    # Variations distinct per iteration:
    seen_idx = sorted({r.variation_label for r in results})
    assert seen_idx == [
        "search_iter_0000",
        "search_iter_0001",
        "search_iter_0002",
        "search_iter_0003",
    ]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_adaptive_search_writes_search_history_incrementally(
    tmp_path: Path,
):
    plan = _plan_with_bo(max_iterations=3)
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    await orch.execute_adaptive_search(plan, _RecordingExecutor(), planner)
    search_history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert len(search_history["iterations"]) == 3
    assert search_history["best_trials"] is not None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_execute_dispatches_to_adaptive_when_adaptive_search_set(tmp_path: Path):
    """The top-level execute() must route plans-with-AdaptiveSearchSweep to the BO path."""
    # max_iterations must be > n_initial_points (2) per AdaptiveSearchSweep validator.
    plan = _plan_with_bo(max_iterations=3)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    # Call execute() (not execute_adaptive_search): the dispatch should kick in.
    # Pass planner via a kwarg the orchestrator forwards.
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    results = await orch.execute(plan, _RecordingExecutor(), search_planner=planner)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_execute_adaptive_search_respects_cancel_check(tmp_path: Path):
    plan = _plan_with_bo(max_iterations=10)
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    state = {"calls": 0}

    def cancel_check() -> bool:
        state["calls"] += 1
        return state["calls"] > 4  # cancel after a few iterations

    results = await orch.execute_adaptive_search(
        plan,
        _RecordingExecutor(),
        planner,
        cancel_check=cancel_check,
    )
    assert len(results) < 10  # cancelled early

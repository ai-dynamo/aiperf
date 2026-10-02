# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""execute_adaptive_search must not block the event loop when flushing checkpoints.

Uses a minimal fake planner (no Optuna/BoTorch dependency) that converges on
the first `ask()`, so `_flush_history` runs exactly once without needing a
real `_run_independent_cell` pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import AdaptiveSearchSweep, Objective, SweepVariation
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


class _NeverCalledExecutor(RunExecutor):
    """Fails the test if a trial is actually dispatched."""

    def derive_id(self, plan, var_idx: int, trial: int) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run):  # pragma: no cover - must not be reached
        raise AssertionError(
            "executor should not run when planner converges immediately"
        )


class _ImmediatelyConvergedPlanner:
    """Fake adaptive planner that converges on the very first `ask()`."""

    iter_count = 0

    def history(self):
        return []

    def ask(self):
        return None

    def convergence_reason(self) -> str:
        return "test_converged"


def _plan_with_bo() -> BenchmarkPlan:
    config = BenchmarkConfig.model_validate(
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
        max_iterations=4,
        n_initial_points=2,
        random_seed=42,
    )
    return BenchmarkPlan(
        configs=[config],
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=1,
        sweep=sweep,
    )


@pytest.mark.asyncio
async def test_execute_adaptive_search_flushes_history_via_to_thread(
    tmp_path: Path,
):
    """_flush_history must offload write_search_history/write_search_checkpoint
    to a worker thread via asyncio.to_thread, mirroring the read-side
    read_search_checkpoint call in sweep_controller/main.py, instead of
    calling them synchronously from the async loop."""
    plan = _plan_with_bo()
    planner = _ImmediatelyConvergedPlanner()
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    executor = _NeverCalledExecutor()

    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        results = await orch.execute_adaptive_search(plan, executor, planner)

    assert results == []
    called_targets = [call.args[0] for call in mock_to_thread.call_args_list]
    called_names = {getattr(target, "__name__", None) for target in called_targets}
    assert "write_search_history" in called_names
    assert "write_search_checkpoint" in called_names

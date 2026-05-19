# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that _run_multi_benchmark instantiates a BO planner when plan.sweep is adaptive."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("optuna")
pytest.importorskip("botorch")

from aiperf.cli_runner import _run_multi_benchmark  # noqa: E402  (after importorskip)


def test_run_multi_benchmark_with_bo_invokes_orchestrator_with_planner(
    tmp_path, monkeypatch
):
    """A plan with an AdaptiveSearchSweep on .sweep should reach the orchestrator
    with a BayesianSearchPlanner."""
    monkeypatch.delenv("AIPERF_OPERATOR_MANAGED", raising=False)
    from aiperf.common.enums import OptimizationDirection
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkPlan
    from aiperf.config.sweep import (
        AdaptiveSearchSweep,
        Objective,
        SweepVariation,
    )
    from aiperf.config.sweep.adaptive import SearchSpaceDimension

    plan = BenchmarkPlan(
        configs=[
            BenchmarkConfig.model_validate(
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
                        }
                    ],
                    "runtime": {"ui": "none"},
                }
            )
        ],
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=1,
        sweep=AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path="phases.profiling.concurrency", lo=1, hi=10, kind="int"
                )
            ],
            objectives=[
                Objective(
                    metric="m", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=2,
            n_initial_points=1,
        ),
    )

    with patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator") as orch_cls:
        instance = orch_cls.return_value
        # Async method must use AsyncMock — asyncio.run() awaits the return value.
        instance.execute = AsyncMock(return_value=[])
        # Stub the LocalSubprocessExecutor to avoid spawning subprocesses.
        # Mock os._exit so the multi-run hang-protection terminator is a
        # no-op under the test harness.
        with (
            patch("aiperf.orchestrator.local_executor.LocalSubprocessExecutor"),
            patch("aiperf.cli_runner._multi_run._summarize_and_export", return_value=0),
            patch("os._exit"),
        ):
            _run_multi_benchmark(plan)

    # Inspect the kwargs passed to execute(): search_planner must be a BayesianSearchPlanner.
    from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner

    call_kwargs = instance.execute.call_args.kwargs
    assert "search_planner" in call_kwargs
    assert isinstance(call_kwargs["search_planner"], BayesianSearchPlanner)

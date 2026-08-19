# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restart determinism for Kubernetes adaptive sweeps."""

from __future__ import annotations

import pytest

pytest.importorskip("optuna")

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.search_planner.optuna_planner import (  # noqa: E402
    OptunaSearchPlanner,
)
from aiperf.sweep_controller.plan_builder import build_plan_from_sweep  # noqa: E402


def _adaptive_sweep_cr() -> dict:
    return {
        "metadata": {
            "name": "restartable-search",
            "namespace": "default",
            "uid": "f19bb6d7-2d91-4d04-a3c8-7b93724c7e42",
        },
        "spec": {
            "sweep": {
                "type": "adaptive_search",
                "planner": "optuna",
                "optunaSampler": "tpe",
                "searchSpace": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 64,
                        "kind": "int",
                    }
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    }
                ],
                "maxIterations": 4,
                "nInitialPoints": 2,
            },
            "benchmark": {
                "models": "mock",
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
        },
    }


def _result_for(values: dict[str, object]) -> list[RunResult]:
    concurrency = float(values["phases.profiling.concurrency"])
    return [
        RunResult(
            label="run_0000",
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tok/s", avg=concurrency * 10.0
                )
            },
        )
    ]


def test_restarted_planner_replays_terminal_history_before_same_next_proposal():
    """Terminal child results reconstruct planner state before new child creation."""
    original_plan = build_plan_from_sweep(_adaptive_sweep_cr())
    assert original_plan.sweep is not None
    original = OptunaSearchPlanner(original_plan.configs[0], original_plan.sweep)
    terminal_history: list[tuple[dict[str, object], list[RunResult]]] = []

    for _ in range(2):
        proposal = original.ask()
        assert proposal is not None
        _, variation = proposal
        results = _result_for(variation.values)
        terminal_history.append((dict(variation.values), results))
        original.tell(variation, results)

    expected_next = original.ask()
    assert expected_next is not None

    restarted_plan = build_plan_from_sweep(_adaptive_sweep_cr())
    assert restarted_plan.sweep is not None
    restarted = OptunaSearchPlanner(restarted_plan.configs[0], restarted_plan.sweep)
    for expected_values, results in terminal_history:
        replayed = restarted.ask()
        assert replayed is not None
        _, variation = replayed
        assert variation.values == expected_values
        restarted.tell(variation, results)

    actual_next = restarted.ask()
    assert actual_next is not None
    assert actual_next[1].values == expected_next[1].values

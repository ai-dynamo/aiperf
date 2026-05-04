# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for build_benchmark_plan with BO adaptive_search."""

from __future__ import annotations

import pytest

from aiperf.config.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def _make_config_with_bo() -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "default", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
            "multi_run": {
                "num_runs": 2,
                "adaptive_search": {
                    "algorithm": "bayes",
                    "search_space": [
                        {
                            "path": "benchmark.phases.profiling.concurrency",
                            "lo": 1,
                            "hi": 1000,
                            "kind": "int",
                        },
                    ],
                    "objective_metric": "output_token_throughput",
                    "objective_stat": "avg",
                    "objective_direction": "maximize",
                    "max_iterations": 15,
                },
            },
        }
    )


def test_build_plan_with_bo_skips_grid_expansion():
    plan = build_benchmark_plan(_make_config_with_bo())
    assert len(plan.configs) == 1
    assert plan.is_adaptive_search is True
    assert plan.is_sweep is False
    assert plan.adaptive_search is not None
    assert plan.adaptive_search.max_iterations == 15
    assert plan.adaptive_search.objective_direction == OptimizationDirection.MAXIMIZE
    assert plan.trials == 2  # multi_run.num_runs preserved


def test_build_plan_rejects_bo_with_sweep_block():
    cfg = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "default", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [1, 2]},
            },
            "multi_run": {
                "adaptive_search": {
                    "algorithm": "bayes",
                    "search_space": [
                        {
                            "path": "benchmark.phases.profiling.concurrency",
                            "lo": 1,
                            "hi": 1000,
                            "kind": "int",
                        },
                    ],
                    "objective_metric": "x",
                    "objective_stat": "avg",
                    "objective_direction": "maximize",
                    "max_iterations": 10,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_benchmark_plan(cfg)

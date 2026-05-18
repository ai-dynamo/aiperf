# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for build_benchmark_plan with an adaptive_search sweep."""

from __future__ import annotations

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.config.sweep import AdaptiveSearchSweep


def _make_config_with_bo() -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "profiling", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
            "multi_run": {"num_runs": 2},
            "sweep": {
                "type": "adaptive_search",
                "search_space": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 1000,
                        "kind": "int",
                    },
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    }
                ],
                "max_iterations": 15,
            },
        }
    )


def test_build_plan_with_bo_skips_grid_expansion():
    plan = build_benchmark_plan(_make_config_with_bo())
    assert len(plan.configs) == 1
    assert plan.is_adaptive_search is True
    assert plan.is_sweep is False
    assert isinstance(plan.sweep, AdaptiveSearchSweep)
    assert plan.sweep.max_iterations == 15
    assert plan.sweep.objectives[0].direction == OptimizationDirection.MAXIMIZE
    assert plan.trials == 2  # multi_run.num_runs preserved


def test_build_plan_rejects_grid_dict_with_adaptive_type():
    """A grid sweep is mutually exclusive with adaptive search; the
    discriminator picks one variant — mixing fields belonging to the other
    raises a structural validation error.
    """
    with pytest.raises(ValueError):
        AIPerfConfig.model_validate(
            {
                "benchmark": {
                    "models": ["m"],
                    "endpoint": {"urls": ["http://x"], "type": "chat"},
                    "datasets": [{"name": "profiling", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "concurrency": 1,
                            "requests": 1,
                        }
                    ],
                },
                # Grid sweep variant rejects adaptive-only fields like
                # `objective` / `max_iterations`.
                "sweep": {
                    "type": "grid",
                    "parameters": {"phases.profiling.concurrency": [1, 2]},
                    "objectives": [
                        {
                            "metric": "x",
                            "stat": "avg",
                            "direction": "maximize",
                        }
                    ],
                    "max_iterations": 10,
                },
            }
        )

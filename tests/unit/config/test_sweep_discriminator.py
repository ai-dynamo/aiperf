# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    GridSweep,
    Objective,  # noqa: F401  (sanity-import check)
    ScenarioSweep,
    SweepConfig,
)


def _adapter():
    return TypeAdapter(SweepConfig)


def test_grid_sweep_with_base_fields():
    sweep = _adapter().validate_python(
        {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": [1, 4, 16]},
            "cooldown_seconds": 30.0,
            "same_seed": True,
            "iteration_order": "independent",
        }
    )
    assert isinstance(sweep, GridSweep)
    assert sweep.cooldown_seconds == 30.0
    assert sweep.same_seed is True


def test_scenarios_sweep_with_base_fields():
    sweep = _adapter().validate_python(
        {
            "type": "scenarios",
            "runs": [{"name": "a", "benchmark": {}}],
            "cooldown_seconds": 5.0,
        }
    )
    assert isinstance(sweep, ScenarioSweep)
    assert sweep.cooldown_seconds == 5.0


def test_adaptive_search_sweep_basic():
    sweep = _adapter().validate_python(
        {
            "type": "adaptive_search",
            "search_space": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 1000,
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
            "max_iterations": 30,
        }
    )
    assert isinstance(sweep, AdaptiveSearchSweep)
    assert sweep.objectives[0].metric == "output_token_throughput"
    assert sweep.objectives[0].direction == OptimizationDirection.MAXIMIZE
    assert sweep.cooldown_seconds == 0.0
    assert sweep.sla_filters == []


def test_adaptive_search_rejects_iteration_order():
    with pytest.raises(ValidationError, match=r"iteration_order"):
        _adapter().validate_python(
            {
                "type": "adaptive_search",
                "search_space": [{"path": "x", "lo": 1, "hi": 2, "kind": "int"}],
                "objectives": [{"metric": "m", "direction": "maximize"}],
                "max_iterations": 5,
                "iteration_order": "repeated",
            }
        )


def test_unknown_type_rejected():
    with pytest.raises(ValidationError, match=r"(?:type|tag|discrim)"):
        _adapter().validate_python({"type": "bogus"})


def test_adaptive_search_n_initial_points_must_be_less_than_max_iterations():
    with pytest.raises(ValidationError, match="must be <"):
        _adapter().validate_python(
            {
                "type": "adaptive_search",
                "search_space": [{"path": "x", "lo": 1, "hi": 2, "kind": "int"}],
                "objectives": [{"metric": "m", "direction": "maximize"}],
                "max_iterations": 5,
                "n_initial_points": 5,
            }
        )

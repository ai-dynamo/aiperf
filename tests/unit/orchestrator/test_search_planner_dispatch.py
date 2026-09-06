# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that the shared planner factory dispatches via the plugin registry."""

import pytest

from aiperf.config import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.search_planner import build_search_planner
from aiperf.orchestrator.search_planner.base import SearchPlanner
from aiperf.plugin.enums import SearchPlannerType


@pytest.fixture
def adaptive_plan() -> BenchmarkPlan:
    """A plan with the fields ``build_search_planner`` reads."""
    return BenchmarkPlan(
        configs=[
            BenchmarkConfig.model_validate(
                {
                    "models": ["m"],
                    "endpoint": {"urls": ["http://x"], "type": "chat"},
                    "datasets": [{"name": "profiling", "type": "synthetic"}],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "poisson",
                            "rate": 1.0,
                            "requests": 10,
                        }
                    ],
                }
            )
        ],
        sweep=AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path="phases.profiling.concurrency", lo=1, hi=10, kind="int"
                )
            ],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    direction="maximize",
                )
            ],
            planner=SearchPlannerType.MONOTONIC_SLA,
            max_iterations=3,
            n_initial_points=2,
            sla_filters=[
                SLAFilter(
                    metric_tag="time_to_first_token",
                    stat="p95",
                    op="lt",
                    threshold=200.0,
                )
            ],
        ),
    )


def test_build_search_planner_returns_none_when_not_adaptive() -> None:
    """The factory returns None for non-adaptive plans."""
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
                            "type": "poisson",
                            "rate": 1.0,
                            "requests": 10,
                        }
                    ],
                }
            )
        ],
        sweep=None,
    )
    assert build_search_planner(plan) is None


def test_build_search_planner_dispatches_via_plugin_registry(
    adaptive_plan: BenchmarkPlan,
) -> None:
    """The factory returns a SearchPlanner via plugin lookup."""
    planner = build_search_planner(adaptive_plan)
    assert isinstance(planner, SearchPlanner)


def test_build_search_planner_rejects_real_dim_on_int_typed_field(
    adaptive_plan: BenchmarkPlan,
) -> None:
    """A kind='real' dimension targeting an int-typed phase field fails fast
    instead of crashing (or silently coercing) mid-search."""
    assert adaptive_plan.sweep is not None
    adaptive_plan.sweep.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.requests", lo=10, hi=1000, kind="real"
    )
    with pytest.raises(ValueError, match="int-typed field 'requests'"):
        build_search_planner(adaptive_plan)


def test_build_search_planner_accepts_real_dim_on_float_typed_field(
    adaptive_plan: BenchmarkPlan,
) -> None:
    """A kind='real' dimension on a float-typed phase field builds normally."""
    assert adaptive_plan.sweep is not None
    adaptive_plan.sweep.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=1.0, hi=10.0, kind="real"
    )
    assert isinstance(build_search_planner(adaptive_plan), SearchPlanner)


def test_build_search_planner_rejects_real_dim_on_non_numeric_field(
    adaptive_plan: BenchmarkPlan,
) -> None:
    """A kind='real' dimension on a non-numeric field fails fast."""
    assert adaptive_plan.sweep is not None
    adaptive_plan.sweep.search_space[0] = SearchSpaceDimension(
        path="endpoint.type", lo=1, hi=10, kind="real"
    )
    with pytest.raises(ValueError, match="int-typed field 'type'"):
        build_search_planner(adaptive_plan)

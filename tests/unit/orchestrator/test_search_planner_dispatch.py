# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that the shared planner factory dispatches via the plugin registry."""

from unittest.mock import MagicMock

import pytest

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.plugin.enums import SearchPlannerType


@pytest.fixture
def adaptive_plan():
    """A MagicMock plan with the fields ``build_search_planner`` reads."""
    plan = MagicMock()
    plan.sweep = AdaptiveSearchSweep(
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
    )
    plan.configs = [MagicMock()]
    return plan


def test_build_search_planner_returns_none_when_not_adaptive():
    """The factory returns None for non-adaptive plans."""
    from aiperf.orchestrator.search_planner import build_search_planner

    plan = MagicMock()
    plan.sweep = None
    assert build_search_planner(plan) is None


def test_build_search_planner_dispatches_via_plugin_registry(adaptive_plan):
    """The factory returns a SearchPlanner via plugin lookup."""
    from aiperf.orchestrator.search_planner import build_search_planner
    from aiperf.orchestrator.search_planner.base import SearchPlanner

    planner = build_search_planner(adaptive_plan)
    assert isinstance(planner, SearchPlanner)


def test_build_search_planner_rejects_real_dim_on_int_typed_field(adaptive_plan):
    """A kind='real' dimension targeting an int-typed phase field fails fast
    instead of crashing (or silently coercing) mid-search."""
    from aiperf.orchestrator.search_planner import build_search_planner

    adaptive_plan.sweep.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.requests", lo=10, hi=1000, kind="real"
    )
    with pytest.raises(ValueError, match="int-typed phase field 'requests'"):
        build_search_planner(adaptive_plan)


def test_build_search_planner_accepts_real_dim_on_float_typed_field(adaptive_plan):
    """A kind='real' dimension on a float-typed phase field builds normally."""
    from aiperf.orchestrator.search_planner import build_search_planner
    from aiperf.orchestrator.search_planner.base import SearchPlanner

    adaptive_plan.sweep.search_space[0] = SearchSpaceDimension(
        path="phases.profiling.rate", lo=1.0, hi=10.0, kind="real"
    )
    assert isinstance(build_search_planner(adaptive_plan), SearchPlanner)

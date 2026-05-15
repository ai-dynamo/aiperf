# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that `_build_search_planner` dispatches via the plugin registry."""

from unittest.mock import MagicMock

import pytest

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.plugin.enums import SearchPlannerType


@pytest.fixture
def adaptive_plan():
    """A MagicMock plan with the fields `_build_search_planner` reads."""
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
    """`_build_search_planner` returns None for non-adaptive plans."""
    from aiperf.cli_runner._strategy import _build_search_planner

    plan = MagicMock()
    plan.sweep = None
    assert _build_search_planner(plan) is None


def test_build_search_planner_dispatches_via_plugin_registry(adaptive_plan):
    """`_build_search_planner(plan)` returns a SearchPlanner via plugin lookup."""
    from aiperf.cli_runner._strategy import _build_search_planner
    from aiperf.orchestrator.search_planner.base import SearchPlanner

    planner = _build_search_planner(adaptive_plan)
    assert isinstance(planner, SearchPlanner)

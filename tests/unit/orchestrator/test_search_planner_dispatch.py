# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that `_build_search_planner` dispatches via the plugin registry."""

from unittest.mock import MagicMock

import pytest

from aiperf.config.adaptive_search import (
    AdaptiveSearchConfig,
    SearchSpaceDimension,
)


@pytest.fixture
def adaptive_plan():
    """A MagicMock plan with the fields `_build_search_planner` reads."""
    plan = MagicMock()
    plan.is_adaptive_search = True
    plan.adaptive_search = AdaptiveSearchConfig(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=10, kind="int"
            )
        ],
        objective_metric="output_token_throughput",
        objective_direction="maximize",
        max_iterations=3,
        n_initial_points=2,
    )
    plan.configs = [MagicMock()]
    return plan


def test_build_search_planner_returns_none_when_not_adaptive():
    """`_build_search_planner` returns None for non-adaptive plans."""
    from aiperf._cli_runner_helpers import _build_search_planner

    plan = MagicMock()
    plan.is_adaptive_search = False
    assert _build_search_planner(plan) is None


def test_build_search_planner_dispatches_via_plugin_registry(adaptive_plan):
    """`_build_search_planner(plan)` returns a SearchPlanner via plugin lookup.

    Skips if skopt isn't installed (the [bo] extra is optional).
    """
    pytest.importorskip("skopt")
    from aiperf._cli_runner_helpers import _build_search_planner
    from aiperf.orchestrator.search_planner.base import SearchPlanner

    planner = _build_search_planner(adaptive_plan)
    assert isinstance(planner, SearchPlanner)

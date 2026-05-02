# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SearchRecipe base types."""

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.config.v1 import UserConfig
from aiperf.search_recipes import (
    PostProcessSpec,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
)


def _adaptive() -> AdaptiveSearchConfig:
    return AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
            ),
        ],
        objective_metric="output_token_throughput",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=20,
    )


def test_sla_filter_accepts_float_threshold_and_default_stat():
    f = SLAFilter(metric_tag="time_to_first_token", op="lt", threshold=200.0)
    assert f.threshold == 200.0
    assert f.stat == "p95"
    assert f.op == "lt"


def test_post_process_spec_defaults_params_to_empty_dict():
    p = PostProcessSpec(handler="ttft_sla_curve", output_filename="out.json")
    assert p.params == {}
    assert p.handler == "ttft_sla_curve"


def test_search_recipe_output_rejects_when_neither_branch_set():
    with pytest.raises(
        ValueError, match="exactly one of 'adaptive_search' or 'sweep_variables'"
    ):
        SearchRecipeOutput()


def test_search_recipe_output_rejects_when_both_branches_set():
    with pytest.raises(
        ValueError, match="exactly one of 'adaptive_search' or 'sweep_variables'"
    ):
        SearchRecipeOutput(
            adaptive_search=_adaptive(),
            sweep_variables={"phases.profiling.concurrency": [1, 10, 100]},
        )


def test_search_recipe_output_accepts_adaptive_search_only():
    out = SearchRecipeOutput(adaptive_search=_adaptive())
    assert out.adaptive_search is not None
    assert out.sweep_variables is None
    assert out.sla_filters == []
    assert out.post_process is None


def test_search_recipe_output_accepts_sweep_variables_only():
    out = SearchRecipeOutput(
        sweep_variables={"phases.profiling.concurrency": [1, 10, 100]}
    )
    assert out.sweep_variables == {"phases.profiling.concurrency": [1, 10, 100]}
    assert out.adaptive_search is None


def test_search_recipe_context_round_trips_user_config():
    user = UserConfig()
    ctx = SearchRecipeContext(
        user_config=user,
        sla_targets={"ttft_sla_ms": 200.0},
        sweep_overrides={"concurrency_max": 500},
    )
    assert ctx.user_config is user
    assert ctx.sla_targets["ttft_sla_ms"] == 200.0
    assert ctx.sweep_overrides["concurrency_max"] == 500

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxThroughputUnderTTFTSLA built-in recipe."""

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes import SearchRecipeContext
from aiperf.search_recipes.builtins import MaxThroughputUnderTTFTSLA


def test_recipe_expand_builds_expected_adaptive_search_and_sla_filter():
    recipe = MaxThroughputUnderTTFTSLA()
    ctx = SearchRecipeContext(
        user_config=UserConfig(),
        sla_targets={"ttft_sla_ms": 200.0},
    )
    out = recipe.expand(ctx)

    assert out.adaptive_search is not None
    assert out.sweep_variables is None
    assert out.adaptive_search.objective_metric == "output_token_throughput"
    assert out.adaptive_search.objective_direction == OptimizationDirection.MAXIMIZE
    assert out.adaptive_search.objective_stat == "avg"
    assert out.adaptive_search.max_iterations == 30
    assert out.adaptive_search.n_initial_points == 5
    assert len(out.adaptive_search.search_space) == 1
    dim = out.adaptive_search.search_space[0]
    assert dim.path == "phases.profiling.concurrency"
    assert dim.lo == 1
    assert dim.hi == 1000
    assert dim.kind == "int"

    assert len(out.sla_filters) == 1
    sla = out.sla_filters[0]
    assert sla.metric_tag == "time_to_first_token"
    assert sla.op == "lt"
    assert sla.stat == "p95"
    assert sla.threshold == 200.0


def test_recipe_expand_rejects_when_ttft_sla_ms_missing():
    recipe = MaxThroughputUnderTTFTSLA()
    ctx = SearchRecipeContext(user_config=UserConfig(), sla_targets={})
    with pytest.raises(ValueError, match="--ttft-sla-ms"):
        recipe.expand(ctx)


def test_recipe_expand_rejects_non_streaming_endpoint():
    recipe = MaxThroughputUnderTTFTSLA()
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=False))
    ctx = SearchRecipeContext(user_config=user, sla_targets={"ttft_sla_ms": 200.0})
    with pytest.raises(ValueError, match="--streaming"):
        recipe.expand(ctx)


def test_recipe_expand_accepts_streaming_endpoint():
    recipe = MaxThroughputUnderTTFTSLA()
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=True))
    ctx = SearchRecipeContext(user_config=user, sla_targets={"ttft_sla_ms": 200.0})
    out = recipe.expand(ctx)
    assert out.adaptive_search is not None


def test_recipe_name_and_description_are_classvars():
    assert MaxThroughputUnderTTFTSLA.name == "max-throughput-ttft-sla"
    assert "TTFT" in MaxThroughputUnderTTFTSLA.description


def test_recipe_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "max-throughput-ttft-sla")
    assert resolved is MaxThroughputUnderTTFTSLA

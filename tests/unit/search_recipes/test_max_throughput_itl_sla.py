# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxThroughputUnderITLSLA built-in recipe."""

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes import SearchRecipeContext
from aiperf.search_recipes.builtins import MaxThroughputUnderITLSLA


def test_recipe_expand_builds_expected_adaptive_search_and_sla_filter():
    recipe = MaxThroughputUnderITLSLA()
    ctx = SearchRecipeContext(
        user_config=UserConfig(),
        sla_targets={"itl_sla_ms": 50.0},
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
    assert sla.metric_tag == "inter_token_latency"
    assert sla.op == "lt"
    assert sla.stat == "p95"
    assert sla.threshold == 50.0


def test_recipe_expand_rejects_when_itl_sla_ms_missing():
    recipe = MaxThroughputUnderITLSLA()
    ctx = SearchRecipeContext(user_config=UserConfig(), sla_targets={})
    with pytest.raises(ValueError, match="--itl-sla-ms"):
        recipe.expand(ctx)


def test_recipe_expand_rejects_non_streaming_endpoint():
    recipe = MaxThroughputUnderITLSLA()
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=False))
    ctx = SearchRecipeContext(user_config=user, sla_targets={"itl_sla_ms": 50.0})
    with pytest.raises(ValueError, match="--streaming"):
        recipe.expand(ctx)


def test_recipe_expand_accepts_streaming_endpoint():
    recipe = MaxThroughputUnderITLSLA()
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=True))
    ctx = SearchRecipeContext(user_config=user, sla_targets={"itl_sla_ms": 50.0})
    out = recipe.expand(ctx)
    assert out.adaptive_search is not None


def test_recipe_name_and_description_are_classvars():
    assert MaxThroughputUnderITLSLA.name == "max-throughput-itl-sla"
    assert "ITL" in MaxThroughputUnderITLSLA.description


def test_recipe_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "max-throughput-itl-sla")
    assert resolved is MaxThroughputUnderITLSLA


def test_recipe_through_converter_populates_sla_filters_and_recipe_name():
    """Converter integration: AdaptiveSearchConfig carries the recipe contract."""
    from aiperf.config.adaptive_search import AdaptiveSearchConfig
    from aiperf.config.v1 import UserConfig
    from aiperf.config.v1._converter_optionals import build_multi_run
    from aiperf.config.v1._loadgen import LoadGeneratorConfig

    user = UserConfig(
        loadgen=LoadGeneratorConfig(
            search_recipe="max-throughput-itl-sla", itl_sla_ms=50.0
        )
    )
    out = build_multi_run(user)
    assert out is not None
    cfg = AdaptiveSearchConfig.model_validate(out["adaptive_search"])
    assert cfg.recipe_name == "max-throughput-itl-sla"
    assert len(cfg.sla_filters) == 1
    sla = cfg.sla_filters[0]
    assert sla.metric_tag == "inter_token_latency"
    assert sla.op == "lt"
    assert sla.threshold == 50.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxThroughputUnderTTFTSLA built-in recipe.

Targets the post-merge shape: recipes emit ``AdaptiveSearchSweep`` with a
nested ``Objective`` (replacing the deleted v1 ``AdaptiveSearchConfig``
flat-objective fields).

This branch's recipes consume a ``SearchRecipeContext`` carrying a validated
``BenchmarkConfig`` (not the ajc/k8s structural ``RecipeCLIConfigView``); the
``make_ctx`` helper in ``conftest.py`` builds one.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import MaxThroughputUnderTTFTSLA
from tests.unit.search_recipes.conftest import make_ctx


def test_max_throughput_ttft_sla_expand_with_sla_target_returns_adaptive_search() -> (
    None
):
    out = MaxThroughputUnderTTFTSLA().expand(
        make_ctx(sla_targets={"ttft_sla_ms": 200.0})
    )
    assert out.adaptive_search is not None
    assert out.sweep_parameters is None
    assert out.adaptive_search.objectives[0].metric == "output_token_throughput"
    assert out.adaptive_search.objectives[0].direction == OptimizationDirection.MAXIMIZE
    assert out.adaptive_search.objectives[0].stat == "avg"
    assert out.adaptive_search.max_iterations == 30
    assert out.adaptive_search.n_initial_points == 5
    assert len(out.adaptive_search.search_space) == 1
    dim = out.adaptive_search.search_space[0]
    assert dim.path == "phases.profiling.concurrency"
    assert dim.lo == 1
    assert dim.hi == 1000
    assert dim.kind == "int"


def test_max_throughput_ttft_sla_expand_emits_ttft_sla_filter() -> None:
    out = MaxThroughputUnderTTFTSLA().expand(
        make_ctx(sla_targets={"ttft_sla_ms": 200.0})
    )
    assert len(out.sla_filters) == 1
    sla = out.sla_filters[0]
    assert sla.metric_tag == "time_to_first_token"
    assert sla.op == "lt"
    assert sla.stat == "p95"
    assert sla.threshold == 200.0


def test_max_throughput_ttft_sla_expand_missing_sla_target_raises() -> None:
    with pytest.raises(ValueError, match="--ttft-sla-ms"):
        MaxThroughputUnderTTFTSLA().expand(make_ctx(sla_targets={}))


def test_max_throughput_ttft_sla_expand_non_streaming_endpoint_raises() -> None:
    with pytest.raises(ValueError, match="streaming"):
        MaxThroughputUnderTTFTSLA().expand(
            make_ctx(streaming=False, sla_targets={"ttft_sla_ms": 200.0})
        )


def test_max_throughput_ttft_sla_name_and_description_are_classvars() -> None:
    assert MaxThroughputUnderTTFTSLA.name == "max-throughput-ttft-sla"
    assert MaxThroughputUnderTTFTSLA.description
    assert "TTFT" in MaxThroughputUnderTTFTSLA.description


def test_max_throughput_ttft_sla_resolves_through_plugin_registry() -> None:
    cls = plugins.get_class(PluginType.SEARCH_RECIPE, "max-throughput-ttft-sla")
    assert cls is MaxThroughputUnderTTFTSLA

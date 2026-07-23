# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxThroughputUnderITLSLA built-in recipe.

Mirrors the TTFT-SLA twin against the post-merge ``AdaptiveSearchSweep`` shape
with nested ``Objective``. Adapted to this branch's
``SearchRecipeContext.benchmark_config`` shape via the ``make_ctx`` helper in
``conftest.py``.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import OptimizationDirection
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import MaxThroughputUnderITLSLA
from tests.unit.search_recipes.conftest import make_ctx


def test_max_throughput_itl_sla_expand_with_sla_target_returns_adaptive_search() -> (
    None
):
    out = MaxThroughputUnderITLSLA().expand(make_ctx(sla_targets={"itl_sla_ms": 50.0}))
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


def test_max_throughput_itl_sla_expand_emits_inter_token_latency_sla_filter() -> None:
    out = MaxThroughputUnderITLSLA().expand(make_ctx(sla_targets={"itl_sla_ms": 50.0}))
    assert len(out.sla_filters) == 1
    sla = out.sla_filters[0]
    assert sla.metric_tag == "inter_token_latency"
    assert sla.op == "lt"
    assert sla.stat == "p95"
    assert sla.threshold == 50.0


def test_max_throughput_itl_sla_expand_with_tpot_sla_alias_returns_same_filter() -> (
    None
):
    """``--tpot-sla-ms`` is an alias for ``--itl-sla-ms``; either populates the
    same ``inter_token_latency`` SLA filter via ``get_inter_token_sla_ms``."""
    out = MaxThroughputUnderITLSLA().expand(make_ctx(sla_targets={"tpot_sla_ms": 50.0}))
    assert len(out.sla_filters) == 1
    sla = out.sla_filters[0]
    assert sla.metric_tag == "inter_token_latency"
    assert sla.threshold == 50.0


def test_max_throughput_itl_sla_expand_with_conflicting_aliases_raises() -> None:
    """Passing both --itl-sla-ms and --tpot-sla-ms with different values raises."""
    with pytest.raises(ValueError, match="aliases"):
        MaxThroughputUnderITLSLA().expand(
            make_ctx(sla_targets={"itl_sla_ms": 50.0, "tpot_sla_ms": 100.0})
        )


def test_max_throughput_itl_sla_expand_missing_sla_target_raises() -> None:
    with pytest.raises(ValueError, match="--itl-sla-ms"):
        MaxThroughputUnderITLSLA().expand(make_ctx(sla_targets={}))


def test_max_throughput_itl_sla_expand_non_streaming_endpoint_raises() -> None:
    with pytest.raises(ValueError, match="streaming"):
        MaxThroughputUnderITLSLA().expand(
            make_ctx(streaming=False, sla_targets={"itl_sla_ms": 50.0})
        )


def test_max_throughput_itl_sla_name_and_description_are_classvars() -> None:
    assert MaxThroughputUnderITLSLA.name == "max-throughput-itl-sla"
    assert MaxThroughputUnderITLSLA.description
    assert "ITL" in MaxThroughputUnderITLSLA.description


def test_max_throughput_itl_sla_resolves_through_plugin_registry() -> None:
    cls = plugins.get_class(PluginType.SEARCH_RECIPE, "max-throughput-itl-sla")
    assert cls is MaxThroughputUnderITLSLA

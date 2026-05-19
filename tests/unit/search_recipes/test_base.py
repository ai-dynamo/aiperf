# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SearchRecipe base types.

Targets the post-merge shape: ``SearchRecipeOutput.adaptive_search`` is
typed ``AdaptiveSearchSweep | None`` (not ``Any``), and ``SearchRecipeContext``
takes a validated ``BenchmarkConfig`` (not the deleted v1 ``CLIConfig`` or
the ajc/k8s ``RecipeCLIConfigView`` SimpleNamespace shape).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.search_recipes import (
    PostProcessSpec,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
)
from tests.unit.search_recipes.conftest import make_ctx


def _adaptive() -> AdaptiveSearchSweep:
    """Build a valid AdaptiveSearchSweep for branch-population tests."""
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
            ),
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=20,
    )


def test_sla_filter_accepts_float_threshold_and_default_stat() -> None:
    f = SLAFilter(metric_tag="time_to_first_token", op="lt", threshold=200.0)
    assert f.threshold == 200.0
    assert f.stat == "p95"
    assert f.op == "lt"


def test_post_process_spec_defaults_params_to_empty_dict() -> None:
    p = PostProcessSpec(handler="ttft_sla_curve", output_filename="out.json")
    assert p.params == {}
    assert p.handler == "ttft_sla_curve"


def test_search_recipe_output_rejects_when_neither_branch_set() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one of 'adaptive_search', 'sweep_parameters', or 'scenarios'",
    ):
        SearchRecipeOutput()


def test_search_recipe_output_rejects_when_both_branches_set() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one of 'adaptive_search', 'sweep_parameters', or 'scenarios'",
    ):
        SearchRecipeOutput(
            adaptive_search=_adaptive(),
            sweep_parameters={"phases.profiling.concurrency": [1, 10, 100]},
        )


def test_search_recipe_output_accepts_adaptive_search_only() -> None:
    out = SearchRecipeOutput(adaptive_search=_adaptive())
    assert out.adaptive_search is not None
    assert isinstance(out.adaptive_search, AdaptiveSearchSweep)
    assert out.sweep_parameters is None
    assert out.sla_filters == []
    assert out.post_process is None


def test_search_recipe_output_accepts_sweep_parameters_only() -> None:
    out = SearchRecipeOutput(
        sweep_parameters={"phases.profiling.concurrency": [1, 10, 100]}
    )
    assert out.sweep_parameters == {"phases.profiling.concurrency": [1, 10, 100]}
    assert out.adaptive_search is None


def test_search_recipe_output_adaptive_search_field_is_typed_adaptive_search_sweep() -> (
    None
):
    """The post-merge contract: ``adaptive_search`` is no longer ``Any | None``."""
    field = SearchRecipeOutput.model_fields["adaptive_search"]
    annotation = field.annotation
    # Pydantic stores as Optional[AdaptiveSearchSweep] -> AdaptiveSearchSweep | None.
    assert AdaptiveSearchSweep in getattr(annotation, "__args__", (annotation,))


def test_search_recipe_output_propagates_sla_filters() -> None:
    sla = SLAFilter(metric_tag="time_to_first_token", op="lt", threshold=200.0)
    out = SearchRecipeOutput(adaptive_search=_adaptive(), sla_filters=[sla])
    assert len(out.sla_filters) == 1
    assert out.sla_filters[0].metric_tag == "time_to_first_token"
    assert out.sla_filters[0].threshold == 200.0


def test_search_recipe_context_round_trips_benchmark_config() -> None:
    """Ctx round-trips its inputs: a validated BenchmarkConfig is accessible
    via ``ctx.benchmark_config``, and the recipe-specific dicts pass through
    unmodified."""
    ctx = make_ctx(
        streaming=True,
        sla_targets={"ttft_sla_ms": 200.0},
        concurrency_max=500,
    )
    assert ctx.benchmark_config is not None
    assert ctx.benchmark_config.endpoint.streaming is True
    assert ctx.sla_targets["ttft_sla_ms"] == 200.0
    assert ctx.sweep_overrides["concurrency_max"] == 500


def test_search_recipe_context_minimal_construction() -> None:
    """A bare context with just a benchmark_config validates cleanly; the two
    dict fields default to empty."""
    ctx = make_ctx()
    assert ctx.sla_targets == {}
    assert ctx.sweep_overrides == {}
    assert ctx.benchmark_config is not None


def test_search_recipe_context_rejects_missing_benchmark_config() -> None:
    """``benchmark_config`` is required (no default); omitting it raises."""
    with pytest.raises(ValidationError):
        SearchRecipeContext()  # type: ignore[call-arg]

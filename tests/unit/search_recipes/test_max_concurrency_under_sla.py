# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxConcurrencyUnderSLA built-in recipe (closes GH issue #883).

Five search styles dispatch off ``ctx.sweep_overrides["search_style"]``:

- ``smooth_isotonic`` (default): planner=SMOOTH_ISOTONIC, BO branch, max_iterations=30.
- ``monotonic``: planner=MONOTONIC_SLA, BO branch, max_iterations=20.
- ``bo``: planner=BAYESIAN, BO branch, max_iterations=30.
- ``optuna``: planner=OPTUNA, BO branch, max_iterations=30.
- ``grid``: sweep_parameters branch + sla_breach_knee post-process.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, SearchPlannerType
from aiperf.search_recipes.builtins import MaxConcurrencyUnderSLA
from tests.unit.search_recipes.conftest import make_ctx


@pytest.mark.parametrize(
    "style,expected_planner,expected_max_iters",
    [
        param("smooth_isotonic", SearchPlannerType.SMOOTH_ISOTONIC, 30, id="smooth_isotonic"),
        param("monotonic", SearchPlannerType.MONOTONIC_SLA, 20, id="monotonic"),
        param("bo", SearchPlannerType.BAYESIAN, 30, id="bo"),
    ],
)  # fmt: skip
def test_max_concurrency_under_sla_expand_adaptive_styles_have_expected_planner(
    style: str, expected_planner, expected_max_iters: int
) -> None:
    out = MaxConcurrencyUnderSLA().expand(
        make_ctx(
            sla_targets={"ttft_sla_ms": 200.0},
            search_style=style,
        )
    )
    assert out.adaptive_search is not None
    assert out.sweep_parameters is None
    assert out.adaptive_search.planner == expected_planner
    assert out.adaptive_search.max_iterations == expected_max_iters
    dims = out.adaptive_search.search_space
    assert len(dims) == 1
    assert dims[0].path == "phases.profiling.concurrency"
    assert dims[0].lo == 1
    assert dims[0].hi == 1000
    assert dims[0].kind == "int"
    # SLA filter lands on AdaptiveSearchSweep.sla_filters and ALSO the output's
    # top-level sla_filters (the recipe sets both for consumer flexibility).
    assert len(out.adaptive_search.sla_filters) == 1
    assert out.adaptive_search.sla_filters[0].metric_tag == "time_to_first_token"
    assert len(out.sla_filters) == 1
    assert out.sla_filters[0].metric_tag == "time_to_first_token"


def test_max_concurrency_under_sla_default_style_is_smooth_isotonic() -> None:
    out = MaxConcurrencyUnderSLA().expand(make_ctx(sla_targets={"ttft_sla_ms": 200.0}))
    assert out.adaptive_search is not None
    assert out.adaptive_search.planner == SearchPlannerType.SMOOTH_ISOTONIC


def test_max_concurrency_under_sla_bo_uses_throughput_objective_maximize() -> None:
    out = MaxConcurrencyUnderSLA().expand(
        make_ctx(
            sla_targets={"ttft_sla_ms": 200.0},
            search_style="bo",
        )
    )
    assert out.adaptive_search is not None
    assert out.adaptive_search.objectives[0].metric == "output_token_throughput"
    assert out.adaptive_search.objectives[0].stat == "avg"
    assert out.adaptive_search.objectives[0].direction == OptimizationDirection.MAXIMIZE
    assert out.adaptive_search.n_initial_points == 5


def test_max_concurrency_under_sla_grid_emits_sweep_parameters_and_post_process() -> (
    None
):
    out = MaxConcurrencyUnderSLA().expand(
        make_ctx(
            sla_targets={"ttft_sla_ms": 200.0},
            search_style="grid",
        )
    )
    assert out.adaptive_search is None
    assert out.sweep_parameters is not None
    # Grid sweep_parameters are body-rooted (no "benchmark." prefix); the
    # converter writes them straight into the top-level sweep.parameters
    # block at the v1->v2 boundary.
    assert "phases.profiling.concurrency" in out.sweep_parameters
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert len(values) == 8
    assert values[0] == 1
    assert values[-1] == 1000
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    assert out.post_process is not None
    assert out.post_process.handler == "sla_breach_knee"
    assert out.post_process.output_filename == "sla_breach.json"
    assert out.post_process.params["swept_param"] == "phases.profiling.concurrency"
    sla_filters_param = out.post_process.params["sla_filters"]
    assert isinstance(sla_filters_param, list)
    assert sla_filters_param[0]["metric_tag"] == "time_to_first_token"
    assert sla_filters_param[0]["threshold"] == 200.0


def test_max_concurrency_under_sla_no_sla_target_raises_listing_all_flags() -> None:
    with pytest.raises(ValueError) as exc:
        MaxConcurrencyUnderSLA().expand(make_ctx())
    msg = str(exc.value)
    for flag in ("--ttft-sla-ms", "--tpot-sla-ms", "--e2e-sla-ms", "--error-rate-sla"):
        assert flag in msg


def test_max_concurrency_under_sla_multi_filter_composition() -> None:
    out = MaxConcurrencyUnderSLA().expand(
        make_ctx(
            sla_targets={
                "ttft_sla_ms": 200.0,
                "e2e_sla_ms": 1000.0,
                "error_rate_sla": 0.05,
            },
        )
    )
    assert out.adaptive_search is not None
    filters = out.adaptive_search.sla_filters
    assert len(filters) == 3
    by_tag = {f.metric_tag: f for f in filters}
    assert by_tag["time_to_first_token"].threshold == 200.0
    assert by_tag["request_latency"].threshold == 1000.0
    assert by_tag["request_error_rate"].threshold == 0.05


def test_max_concurrency_under_sla_streaming_required_for_ttft_filter() -> None:
    with pytest.raises(ValueError, match="streaming"):
        MaxConcurrencyUnderSLA().expand(
            make_ctx(streaming=False, sla_targets={"ttft_sla_ms": 200.0})
        )


def test_max_concurrency_under_sla_streaming_not_required_for_e2e_only() -> None:
    out = MaxConcurrencyUnderSLA().expand(
        make_ctx(
            streaming=False,
            sla_targets={"e2e_sla_ms": 1000.0, "error_rate_sla": 0.01},
        )
    )
    assert out.adaptive_search is not None
    assert len(out.adaptive_search.sla_filters) == 2


def test_max_concurrency_under_sla_unknown_style_raises() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        MaxConcurrencyUnderSLA().expand(
            make_ctx(
                sla_targets={"ttft_sla_ms": 200.0},
                search_style="not_a_real_style",
            )
        )


def test_max_concurrency_under_sla_resolves_through_plugin_registry() -> None:
    cls = plugins.get_class(PluginType.SEARCH_RECIPE, "max-concurrency-under-sla")
    assert cls is MaxConcurrencyUnderSLA


def test_max_concurrency_under_sla_name_and_description_are_classvars() -> None:
    assert MaxConcurrencyUnderSLA.name == "max-concurrency-under-sla"
    assert "SLA" in MaxConcurrencyUnderSLA.description

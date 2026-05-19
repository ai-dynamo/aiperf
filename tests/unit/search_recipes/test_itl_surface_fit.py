# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ItlSurfaceFit post-process handler.

Recipes on this branch emit envelope-prefixed swept_param keys
(``phases.profiling.concurrency`` etc.); fixtures here mirror that.
"""

from __future__ import annotations

import pytest

from aiperf.search_recipes.post_process import ItlSurfaceFit, PostProcessHandler

CONCURRENCY = "phases.profiling.concurrency"
OSL = "datasets.main.prompts.osl"
ITL_AVG = "inter_token_latency_avg"


def _row(c: float, o: float, itl: float) -> dict:
    return {
        "parameters": {CONCURRENCY: c, OSL: o},
        "metrics": {ITL_AVG: {"mean": itl}},
    }


def _params() -> dict:
    return {
        "metric_tag": "inter_token_latency",
        "stat": "avg",
        "concurrency_param": CONCURRENCY,
        "osl_param": OSL,
    }


def test_itl_surface_fit_complete_grid_returns_dense_surface():
    handler = ItlSurfaceFit()
    rows = [
        _row(1, 64, 10.0),
        _row(1, 256, 12.0),
        _row(10, 64, 11.0),
        _row(10, 256, 14.0),
    ]
    out = handler.process({"per_combination_metrics": rows}, _params())

    assert out["swept_metric"] == "inter_token_latency"
    assert out["stat"] == "avg"
    assert out["swept_params"] == [CONCURRENCY, OSL]
    assert out["surface"]["concurrency_axis"] == [1.0, 10.0]
    assert out["surface"]["osl_axis"] == [64.0, 256.0]
    assert out["surface"]["itl_grid"] == [[10.0, 12.0], [11.0, 14.0]]
    assert len(out["raw_points"]) == 4


def test_itl_surface_fit_partial_grid_emits_null_for_missing_cells():
    handler = ItlSurfaceFit()
    # Missing cell: (concurrency=10, osl=256).
    rows = [_row(1, 64, 10.0), _row(1, 256, 12.0), _row(10, 64, 11.0)]
    out = handler.process({"per_combination_metrics": rows}, _params())
    assert out["surface"]["concurrency_axis"] == [1.0, 10.0]
    assert out["surface"]["osl_axis"] == [64.0, 256.0]
    grid = out["surface"]["itl_grid"]
    assert grid[0] == [10.0, 12.0]
    assert grid[1][0] == 11.0
    assert grid[1][1] is None


def test_itl_surface_fit_tolerates_tag_only_layout():
    """Single-trial sweeps store metrics under the bare metric tag, not '<tag>_<stat>'."""
    handler = ItlSurfaceFit()
    rows = [
        {
            "parameters": {CONCURRENCY: 1, OSL: 64},
            "metrics": {"inter_token_latency": {"mean": 10.0}},
        },
        {
            "parameters": {CONCURRENCY: 1, OSL: 256},
            "metrics": {"inter_token_latency": {"mean": 12.0}},
        },
    ]
    out = handler.process({"per_combination_metrics": rows}, _params())
    assert out["surface"]["itl_grid"] == [[10.0, 12.0]]


def test_itl_surface_fit_multi_trial_layout_uses_flat_key():
    """Multi-trial sweeps emit '<metric_tag>_<stat>' keys; the handler reads 'mean'."""
    handler = ItlSurfaceFit()
    rows = [_row(1, 64, 10.0), _row(2, 64, 11.0)]
    out = handler.process({"per_combination_metrics": rows}, _params())
    assert out["surface"]["concurrency_axis"] == [1.0, 2.0]
    assert out["surface"]["osl_axis"] == [64.0]
    assert out["surface"]["itl_grid"] == [[10.0], [11.0]]


def test_itl_surface_fit_raises_when_no_matching_rows():
    handler = ItlSurfaceFit()
    rows = [
        {
            "parameters": {"other": 1, OSL: 64},
            "metrics": {ITL_AVG: {"mean": 10.0}},
        },
    ]
    with pytest.raises(ValueError, match="no rows with parameters"):
        handler.process({"per_combination_metrics": rows}, _params())


def test_itl_surface_fit_implements_post_process_handler_protocol():
    assert isinstance(ItlSurfaceFit(), PostProcessHandler)


def test_itl_surface_fit_raw_points_are_sorted_by_concurrency_then_osl():
    handler = ItlSurfaceFit()
    rows = [
        _row(10, 256, 14.0),
        _row(1, 64, 10.0),
        _row(10, 64, 11.0),
        _row(1, 256, 12.0),
    ]
    out = handler.process({"per_combination_metrics": rows}, _params())
    triples = [(p["concurrency"], p["osl"]) for p in out["raw_points"]]
    assert triples == sorted(triples)

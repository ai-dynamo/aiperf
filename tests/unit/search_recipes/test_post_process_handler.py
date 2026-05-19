# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for built-in post-process handlers.

Tests the handlers' ``process()`` methods directly against synthetic sweep
aggregates. Recipes on this branch emit envelope-prefixed swept_param keys
(``phases.profiling.concurrency`` etc.); fixtures here mirror that
so they reflect what real recipes feed into the hook.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.search_recipes.post_process import (
    DegradationKneeDetect,
    PostProcessHandler,
    TTFTCurveFit,
)


def _make_aggregate(
    swept_param: str, flat_metric: str, points: list[tuple[float, float]]
) -> dict:
    return {
        "per_combination_metrics": [
            {
                "parameters": {swept_param: x},
                "metrics": {flat_metric: {"mean": y}},
            }
            for x, y in points
        ]
    }


def test_degradation_knee_detect_finds_first_breach():
    handler = DegradationKneeDetect()
    agg = _make_aggregate(
        "phases.profiling.concurrency",
        "request_latency_p99",
        [(1, 10.0), (10, 11.0), (50, 11.5), (100, 12.5)],
    )
    out = handler.process(
        agg,
        {
            "threshold_pct": 0.20,
            "metric_tag": "request_latency",
            "stat": "p99",
            "swept_param": "phases.profiling.concurrency",
        },
    )
    assert out["baseline_concurrency"] == 1
    assert out["baseline_p99"] == 10.0
    assert out["knee_concurrency"] == 100
    assert out["knee_p99"] == 12.5
    assert out["threshold_pct"] == 0.20
    assert out["swept_metric"] == "request_latency"
    assert out["stat"] == "p99"
    assert len(out["all_points"]) == 4


def test_degradation_knee_detect_returns_null_when_no_breach():
    handler = DegradationKneeDetect()
    agg = _make_aggregate(
        "phases.profiling.concurrency",
        "request_latency_p99",
        [(1, 10.0), (10, 10.5), (50, 11.0)],
    )
    out = handler.process(
        agg,
        {
            "threshold_pct": 0.20,
            "metric_tag": "request_latency",
            "stat": "p99",
            "swept_param": "phases.profiling.concurrency",
        },
    )
    assert out["knee_concurrency"] is None
    assert out["knee_p99"] is None


def test_degradation_knee_detect_raises_when_no_matching_rows():
    handler = DegradationKneeDetect()
    agg = {
        "per_combination_metrics": [
            {"parameters": {"other_param": 1}, "metrics": {"x_p99": {"mean": 10.0}}},
        ]
    }
    with pytest.raises(ValueError, match="no rows with parameter"):
        handler.process(
            agg,
            {
                "threshold_pct": 0.20,
                "metric_tag": "request_latency",
                "stat": "p99",
                "swept_param": "phases.profiling.concurrency",
            },
        )


@pytest.mark.parametrize(
    "points,expected_form",
    [
        param(
            [(256.0, 12.0), (512.0, 24.0), (1024.0, 48.0), (2048.0, 96.0)],
            "linear",
            id="perfectly_linear",
        ),
        param(
            # Symmetric parabola y = (x - 1000)^2 / 10 -- linear fit gets ~0
            # r^2, quadratic fits perfectly.
            [(0.0, 100000.0), (500.0, 25000.0), (1000.0, 0.0),
             (1500.0, 25000.0), (2000.0, 100000.0)],
            "quadratic",
            id="symmetric_parabola",
        ),
    ],
)  # fmt: skip
def test_ttft_curve_fit_chooses_form_by_r_squared(points, expected_form):
    handler = TTFTCurveFit()
    agg = _make_aggregate(
        "datasets.main.prompts.isl",
        "time_to_first_token_avg",
        points,
    )
    out = handler.process(
        agg,
        {
            "metric_tag": "time_to_first_token",
            "stat": "avg",
            "swept_param": "datasets.main.prompts.isl",
        },
    )
    assert out["fit_form"] == expected_form
    assert 0.0 <= out["r_squared"] <= 1.0
    assert len(out["raw_points"]) == len(points)


def test_ttft_curve_fit_returns_r_squared_above_floor_for_linear_data():
    handler = TTFTCurveFit()
    points = [(256.0, 12.0), (512.0, 24.0), (1024.0, 48.0), (2048.0, 96.0)]
    agg = _make_aggregate(
        "datasets.main.prompts.isl",
        "time_to_first_token_avg",
        points,
    )
    out = handler.process(
        agg,
        {
            "metric_tag": "time_to_first_token",
            "stat": "avg",
            "swept_param": "datasets.main.prompts.isl",
        },
    )
    assert out["r_squared"] > 0.99


def test_ttft_curve_fit_raises_with_single_point():
    handler = TTFTCurveFit()
    agg = _make_aggregate(
        "datasets.main.prompts.isl",
        "time_to_first_token_avg",
        [(256.0, 12.0)],
    )
    with pytest.raises(ValueError, match=">= 2 sweep points"):
        handler.process(
            agg,
            {
                "metric_tag": "time_to_first_token",
                "stat": "avg",
                "swept_param": "datasets.main.prompts.isl",
            },
        )


def test_handlers_implement_post_process_handler_protocol():
    assert isinstance(DegradationKneeDetect(), PostProcessHandler)
    assert isinstance(TTFTCurveFit(), PostProcessHandler)

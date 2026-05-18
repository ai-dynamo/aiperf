# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SLABreachKnee post-process handler."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.search_recipes.post_process import (
    PostProcessHandler,
    SLABreachKnee,
)

SWEPT = "phases.profiling.concurrency"


def _row(concurrency: float | int, metrics: dict) -> dict:
    return {
        "parameters": {SWEPT: concurrency},
        "metrics": metrics,
    }


def _ttft_p95(value: float) -> dict:
    return {"time_to_first_token_p95": {"mean": value}}


def _ttft_filter(threshold: float = 200.0) -> SLAFilter:
    return SLAFilter(
        metric_tag="time_to_first_token",
        stat="p95",
        op="lt",
        threshold=threshold,
    )


def _latency_filter(threshold: float = 1000.0) -> SLAFilter:
    return SLAFilter(
        metric_tag="request_latency",
        stat="p99",
        op="lt",
        threshold=threshold,
    )


def test_sla_breach_knee_implements_protocol():
    assert isinstance(SLABreachKnee(), PostProcessHandler)


def test_boundary_in_middle():
    # Concurrencies 1, 64, 256 pass; 384, 512 fail.
    agg = {
        "per_combination_metrics": [
            _row(1, _ttft_p95(50.0)),
            _row(64, _ttft_p95(120.0)),
            _row(256, _ttft_p95(195.0)),
            _row(384, _ttft_p95(213.4)),
            _row(512, _ttft_p95(280.0)),
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["swept_param"] == SWEPT
    assert out["max_passing_concurrency"] == 256
    assert out["first_failing_concurrency"] == 384
    assert out["monotonicity_check"] is True
    assert len(out["all_points"]) == 5
    assert [p["feasible"] for p in out["all_points"]] == [
        True,
        True,
        True,
        False,
        False,
    ]
    breach = out["first_failing_breach"]
    assert breach["metric_tag"] == "time_to_first_token"
    assert breach["stat"] == "p95"
    assert breach["op"] == "lt"
    assert breach["threshold"] == 200.0
    assert breach["observed"] == pytest.approx(213.4)


def test_all_passing():
    agg = {
        "per_combination_metrics": [
            _row(c, _ttft_p95(v))
            for c, v in [(1, 50.0), (8, 80.0), (32, 110.0), (128, 150.0), (256, 190.0)]
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] == 256
    assert out["first_failing_concurrency"] is None
    assert out["first_failing_breach"] is None
    assert out["monotonicity_check"] is True
    assert all(p["feasible"] for p in out["all_points"])


def test_all_failing():
    agg = {
        "per_combination_metrics": [
            _row(c, _ttft_p95(v))
            for c, v in [
                (1, 250.0),
                (8, 260.0),
                (32, 270.0),
                (128, 290.0),
                (256, 320.0),
            ]
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] is None
    assert out["first_failing_concurrency"] == 1
    assert out["first_failing_breach"] is not None
    assert out["first_failing_breach"]["observed"] == pytest.approx(250.0)
    assert out["monotonicity_check"] is True
    assert not any(p["feasible"] for p in out["all_points"])


def test_non_monotonic_alternating():
    # Pattern: pass, fail, pass, fail, pass -> non-monotonic.
    agg = {
        "per_combination_metrics": [
            _row(1, _ttft_p95(50.0)),
            _row(8, _ttft_p95(250.0)),
            _row(32, _ttft_p95(150.0)),
            _row(128, _ttft_p95(260.0)),
            _row(256, _ttft_p95(180.0)),
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["monotonicity_check"] is False
    # Smallest swept value where any filter fails: 8.
    assert out["first_failing_concurrency"] == 8
    # Largest feasible swept value: 256 (even though boundary alternates).
    assert out["max_passing_concurrency"] == 256
    assert out["first_failing_breach"]["observed"] == pytest.approx(250.0)


def test_multi_filter_partial_breach():
    # Two filters: TTFT p95 lt 200, request_latency p99 lt 1000.
    # Point 1 (c=128): TTFT 150 (pass), latency 800 (pass) -> feasible.
    # Point 2 (c=256): TTFT 250 (FAIL), latency 900 (pass) -> infeasible (TTFT first).
    # Point 3 (c=384): TTFT 180 (pass), latency 1200 (FAIL) -> infeasible (latency only).
    agg = {
        "per_combination_metrics": [
            _row(
                128,
                {
                    "time_to_first_token_p95": {"mean": 150.0},
                    "request_latency_p99": {"mean": 800.0},
                },
            ),
            _row(
                256,
                {
                    "time_to_first_token_p95": {"mean": 250.0},
                    "request_latency_p99": {"mean": 900.0},
                },
            ),
            _row(
                384,
                {
                    "time_to_first_token_p95": {"mean": 180.0},
                    "request_latency_p99": {"mean": 1200.0},
                },
            ),
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {
            "sla_filters": [_ttft_filter(), _latency_filter()],
            "swept_param": SWEPT,
        },
    )
    # max_passing = 128, first_failing = 256.
    assert out["max_passing_concurrency"] == 128
    assert out["first_failing_concurrency"] == 256
    # First filter (TTFT) breached at c=256.
    assert out["first_failing_breach"]["metric_tag"] == "time_to_first_token"
    # Point 2 has only one breach (TTFT); latency passed.
    point_256 = next(p for p in out["all_points"] if p["concurrency"] == 256)
    assert len(point_256["breaches"]) == 1
    assert point_256["breaches"][0]["metric_tag"] == "time_to_first_token"
    # Point 3 has only one breach (latency); TTFT passed.
    point_384 = next(p for p in out["all_points"] if p["concurrency"] == 384)
    assert len(point_384["breaches"]) == 1
    assert point_384["breaches"][0]["metric_tag"] == "request_latency"


@pytest.mark.parametrize(
    "as_dict",
    [
        param(False, id="typed_SLAFilter"),
        param(True, id="dumped_dict"),
    ],
)  # fmt: skip
def test_filter_input_shape_tolerance(as_dict: bool):
    agg = {
        "per_combination_metrics": [
            _row(64, _ttft_p95(150.0)),
            _row(256, _ttft_p95(250.0)),
        ]
    }
    raw_filter = _ttft_filter()
    filter_arg = raw_filter.model_dump(mode="json") if as_dict else raw_filter
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [filter_arg], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] == 64
    assert out["first_failing_concurrency"] == 256
    assert out["first_failing_breach"]["metric_tag"] == "time_to_first_token"


def test_missing_metric_treated_as_infeasible():
    # Row 2 has no time_to_first_token metric at all.
    agg = {
        "per_combination_metrics": [
            _row(64, _ttft_p95(150.0)),
            _row(256, {"some_other_metric_avg": {"mean": 1.0}}),
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] == 64
    assert out["first_failing_concurrency"] == 256
    point_256 = next(p for p in out["all_points"] if p["concurrency"] == 256)
    assert point_256["feasible"] is False
    assert len(point_256["breaches"]) == 1
    assert point_256["breaches"][0]["metric_tag"] == "time_to_first_token"
    assert point_256["breaches"][0]["observed"] is None
    assert out["first_failing_breach"]["observed"] is None


def test_filters_field_serialized_as_dicts():
    # The output ``filters`` field should always be a list of dicts (round-tripped),
    # regardless of whether the input was typed or dict.
    agg = {"per_combination_metrics": [_row(64, _ttft_p95(100.0))]}
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert isinstance(out["filters"], list)
    assert isinstance(out["filters"][0], dict)
    assert out["filters"][0]["metric_tag"] == "time_to_first_token"


@pytest.mark.parametrize(
    "concurrency_values,expected_type",
    [
        param([1, 64, 256, 384, 512], int, id="int_inputs"),
        param([1.0, 64.0, 256.0, 384.0, 512.0], float, id="float_inputs"),
    ],
)  # fmt: skip
def test_swept_value_type_preserved(concurrency_values, expected_type):
    # Input value types (int vs float) must be preserved into the output
    # JSON; coercing int->float would yield e.g. `256.0` in user output.
    agg = {
        "per_combination_metrics": [
            _row(c, _ttft_p95(v))
            for c, v in zip(
                concurrency_values, [50.0, 120.0, 195.0, 213.4, 280.0], strict=True
            )
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] == 256
    assert isinstance(out["max_passing_concurrency"], expected_type)
    assert out["first_failing_concurrency"] == 384
    assert isinstance(out["first_failing_concurrency"], expected_type)
    for point in out["all_points"]:
        assert isinstance(point["concurrency"], expected_type)


def test_empty_per_combination_metrics():
    # No rows -> no feasible/infeasible bookkeeping; monotonicity is vacuously true.
    agg: dict = {"per_combination_metrics": []}
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] is None
    assert out["first_failing_concurrency"] is None
    assert out["first_failing_breach"] is None
    assert out["monotonicity_check"] is True
    assert out["all_points"] == []


def test_single_point_passing():
    agg = {"per_combination_metrics": [_row(128, _ttft_p95(100.0))]}
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["max_passing_concurrency"] == 128
    assert out["first_failing_concurrency"] is None
    assert out["first_failing_breach"] is None
    assert out["monotonicity_check"] is True
    assert len(out["all_points"]) == 1
    assert out["all_points"][0]["feasible"] is True


def test_late_non_monotonic_break():
    # Pattern: pass, pass, fail, fail, pass -> single late non-monotonic transition.
    agg = {
        "per_combination_metrics": [
            _row(1, _ttft_p95(50.0)),
            _row(8, _ttft_p95(80.0)),
            _row(32, _ttft_p95(250.0)),
            _row(128, _ttft_p95(260.0)),
            _row(256, _ttft_p95(180.0)),
        ]
    }
    out = SLABreachKnee().process(
        agg,
        {"sla_filters": [_ttft_filter()], "swept_param": SWEPT},
    )
    assert out["monotonicity_check"] is False
    # First swept value where any filter fails: 32.
    assert out["first_failing_concurrency"] == 32
    # Largest feasible swept value: 256 (the post-failure pass).
    assert out["max_passing_concurrency"] == 256
    assert out["first_failing_breach"]["observed"] == pytest.approx(250.0)

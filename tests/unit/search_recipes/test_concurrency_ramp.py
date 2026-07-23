# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ConcurrencyRamp grid recipe."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import ConcurrencyRamp
from tests.unit.search_recipes.conftest import make_ctx


def test_concurrency_ramp_default_grid_uses_endpoints_1_and_1000():
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.adaptive_search is None
    assert out.sweep_parameters is not None
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[0] == 1
    assert values[-1] == 1000
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_concurrency_ramp_emits_post_process_with_threshold():
    out = ConcurrencyRamp().expand(make_ctx(degradation_threshold=0.30))
    assert out.post_process is not None
    assert out.post_process.handler == "degradation_knee_detect"
    assert out.post_process.params["threshold_pct"] == 0.30
    assert out.post_process.params["metric_tag"] == "request_latency"
    assert out.post_process.params["stat"] == "p99"
    assert out.post_process.output_filename == "degradation_knee.json"


def test_concurrency_ramp_default_threshold_is_20_percent():
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.post_process.params["threshold_pct"] == 0.20


def test_concurrency_ramp_overrides_extend_grid_range():
    out = ConcurrencyRamp().expand(
        make_ctx(concurrency_min=4, concurrency_max=64, concurrency_steps=4)
    )
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[0] == 4
    assert values[-1] == 64


def test_concurrency_ramp_does_not_require_streaming():
    out = ConcurrencyRamp().expand(make_ctx(streaming=False))
    assert out.sweep_parameters is not None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        param({"concurrency_steps": 1}, "steps must be >= 2", id="steps_one"),
        param({"concurrency_steps": 0}, "steps must be >= 2", id="steps_zero"),
        param({"concurrency_steps": -3}, "steps must be >= 2", id="steps_negative"),
        param(
            {"concurrency_min": 10, "concurrency_max": 10},
            r"concurrency-min.*must be <",
            id="lo_eq_hi",
        ),
        param(
            {"concurrency_min": 100, "concurrency_max": 4},
            r"concurrency-min.*must be <",
            id="lo_gt_hi",
        ),
    ],
)  # fmt: skip
def test_concurrency_ramp_invalid_grid_raises(overrides, match):
    with pytest.raises(ValueError, match=match):
        ConcurrencyRamp().expand(make_ctx(**overrides))


def test_concurrency_ramp_unknown_override_keys_silently_ignored():
    out = ConcurrencyRamp().expand(
        make_ctx(unrecognized_knob=42, another_extra="ignored")
    )
    assert out.sweep_parameters is not None


def test_concurrency_ramp_ignores_sla_targets():
    out = ConcurrencyRamp().expand(make_ctx(sla_targets={"ttft_sla_ms": 250.0}))
    assert out.sla_filters == []


def test_concurrency_ramp_string_overrides_coerce_via_int():
    out = ConcurrencyRamp().expand(
        make_ctx(concurrency_min="2", concurrency_max="50", concurrency_steps="3")
    )
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[0] == 2
    assert values[-1] == 50
    assert len(values) == 3


def test_concurrency_ramp_unparseable_string_override_raises():
    with pytest.raises(ValueError, match="invalid literal"):
        ConcurrencyRamp().expand(make_ctx(concurrency_min="not-an-int"))


def test_concurrency_ramp_output_is_deterministic():
    a = ConcurrencyRamp().expand(make_ctx())
    b = ConcurrencyRamp().expand(make_ctx())
    assert a.sweep_parameters == b.sweep_parameters
    assert a.post_process == b.post_process


def test_concurrency_ramp_grid_values_strictly_ascending_and_unique():
    out = ConcurrencyRamp().expand(make_ctx())
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values == sorted(set(values))


def test_concurrency_ramp_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "concurrency-ramp")
    assert resolved is ConcurrencyRamp


def test_concurrency_ramp_sweep_parameters_only_no_adaptive_search():
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.adaptive_search is None
    assert out.sweep_parameters is not None
    assert set(out.sweep_parameters.keys()) == {"phases.profiling.concurrency"}


def test_concurrency_ramp_high_range_does_not_overflow():
    out = ConcurrencyRamp().expand(
        make_ctx(concurrency_min=1, concurrency_max=10_000_000, concurrency_steps=4)
    )
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[0] == 1
    assert values[-1] == 10_000_000


def test_concurrency_ramp_two_steps_is_minimum_valid_input():
    out = ConcurrencyRamp().expand(make_ctx(concurrency_steps=2))
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values == [1, 1000]


def test_concurrency_ramp_default_post_process_metric_and_stat():
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.post_process.params["metric_tag"] == "request_latency"
    assert out.post_process.params["stat"] == "p99"


def test_concurrency_ramp_metric_tag_override_flows_to_post_process():
    out = ConcurrencyRamp().expand(
        make_ctx(degradation_metric_tag="time_to_first_token")
    )
    assert out.post_process.params["metric_tag"] == "time_to_first_token"
    assert out.post_process.params["stat"] == "p99"


def test_concurrency_ramp_stat_override_flows_to_post_process():
    out = ConcurrencyRamp().expand(make_ctx(degradation_stat="p95"))
    assert out.post_process.params["stat"] == "p95"
    assert out.post_process.params["metric_tag"] == "request_latency"


def test_concurrency_ramp_metric_and_stat_overrides_compose():
    out = ConcurrencyRamp().expand(
        make_ctx(
            degradation_metric_tag="time_to_first_token",
            degradation_stat="p95",
        )
    )
    assert out.post_process.params["metric_tag"] == "time_to_first_token"
    assert out.post_process.params["stat"] == "p95"

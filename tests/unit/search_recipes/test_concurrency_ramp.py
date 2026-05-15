# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ConcurrencyRamp grid recipe.

Pins the recipe's default-grid endpoints, post-process spec, and override
behavior so the user-facing contract (default ramp 1->1000, 20% threshold,
post-process emits ``degradation_knee.json``) cannot drift silently.

Ported from ``ajc/k8s`` to the BenchmarkConfig-based ``SearchRecipeContext``.
Recipe sweep keys are envelope-prefixed (``phases.profiling.<...>``).
"""

from __future__ import annotations

import pytest

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


def test_concurrency_ramp_invalid_step_count_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        ConcurrencyRamp().expand(make_ctx(concurrency_steps=1))


def test_concurrency_ramp_does_not_require_streaming():
    out = ConcurrencyRamp().expand(make_ctx(streaming=False))
    assert out.sweep_parameters is not None


# ---- Adversarial cases ----


def test_concurrency_ramp_lo_equals_hi_raises():
    with pytest.raises(ValueError, match=r"concurrency-min.*must be <"):
        ConcurrencyRamp().expand(make_ctx(concurrency_min=10, concurrency_max=10))


def test_concurrency_ramp_lo_greater_than_hi_raises():
    with pytest.raises(ValueError, match=r"concurrency-min.*must be <"):
        ConcurrencyRamp().expand(make_ctx(concurrency_min=100, concurrency_max=4))


def test_concurrency_ramp_zero_steps_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        ConcurrencyRamp().expand(make_ctx(concurrency_steps=0))


def test_concurrency_ramp_negative_steps_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        ConcurrencyRamp().expand(make_ctx(concurrency_steps=-3))


def test_concurrency_ramp_unknown_override_keys_silently_ignored():
    # Recipe reads only the keys it knows; extras don't break expansion. Lets the
    # click+assemble layer evolve sweep_overrides without per-recipe coupling.
    out = ConcurrencyRamp().expand(
        make_ctx(unrecognized_knob=42, another_extra="ignored")
    )
    assert out.sweep_parameters is not None


def test_concurrency_ramp_ignores_sla_targets():
    # ConcurrencyRamp is grid-only; sla_targets is a recipe input the BO recipes
    # consume. Passing one must not affect the recipe's output.
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
    # Same inputs -> identical sweep_parameters; no hidden RNG / global state.
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
    # Mutual-exclusivity invariant: SearchRecipeOutput requires exactly one
    # branch set. Pin the grid branch so a future BO refactor of this recipe
    # surfaces here loudly.
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.adaptive_search is None
    assert out.sweep_parameters is not None
    assert set(out.sweep_parameters.keys()) == {"phases.profiling.concurrency"}


def test_concurrency_ramp_high_range_does_not_overflow():
    # Pin behavior at the upper end of the int domain; logspace must cope.
    out = ConcurrencyRamp().expand(
        make_ctx(concurrency_min=1, concurrency_max=10_000_000, concurrency_steps=4)
    )
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[0] == 1
    assert values[-1] == 10_000_000


def test_concurrency_ramp_two_steps_is_minimum_valid_input():
    # `_logspace_int_steps` requires steps >= 2; pin the boundary.
    out = ConcurrencyRamp().expand(make_ctx(concurrency_steps=2))
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values == [1, 1000]


# ---- Post-process metric/stat overrides (#16) ----


def test_concurrency_ramp_default_post_process_metric_and_stat():
    # Defaults: request_latency / p99. Pinned so a future ergonomic refactor of
    # the recipe params surfaces here loudly.
    out = ConcurrencyRamp().expand(make_ctx())
    assert out.post_process.params["metric_tag"] == "request_latency"
    assert out.post_process.params["stat"] == "p99"


def test_concurrency_ramp_metric_tag_override_flows_to_post_process():
    out = ConcurrencyRamp().expand(
        make_ctx(degradation_metric_tag="time_to_first_token")
    )
    assert out.post_process.params["metric_tag"] == "time_to_first_token"
    # Stat untouched by metric-tag-only override.
    assert out.post_process.params["stat"] == "p99"


def test_concurrency_ramp_stat_override_flows_to_post_process():
    out = ConcurrencyRamp().expand(make_ctx(degradation_stat="p95"))
    assert out.post_process.params["stat"] == "p95"
    # Metric tag untouched by stat-only override.
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

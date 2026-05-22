# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PrefillTTFTCurve grid recipe.

Pins the ISL-sweep grid defaults ([256, 32768]), the concurrency=1 isolation
rule, the ``ttft_curve_fit`` post-process spec, and the streaming-required
guard so the user-facing contract cannot drift.

Ported from ``ajc/k8s`` to the BenchmarkConfig-based ``SearchRecipeContext``.
Sweep keys are body-rooted (``datasets.main.prompts.isl``,
``phases.profiling.concurrency``).
"""

from __future__ import annotations

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import PrefillTTFTCurve
from tests.unit.search_recipes.conftest import make_ctx


def test_prefill_ttft_curve_default_grid_uses_isl_min_max_endpoints():
    out = PrefillTTFTCurve().expand(make_ctx())
    assert out.sweep_parameters is not None
    isl_values = out.sweep_parameters["datasets.main.prompts.isl"]
    assert isl_values[0] == 256
    assert isl_values[-1] == 32768


def test_prefill_ttft_curve_pins_concurrency_to_one():
    out = PrefillTTFTCurve().expand(make_ctx())
    concurrency_values = out.sweep_parameters["phases.profiling.concurrency"]
    assert concurrency_values == [1]


def test_prefill_ttft_curve_emits_ttft_curve_fit_post_process():
    out = PrefillTTFTCurve().expand(make_ctx())
    assert out.post_process is not None
    assert out.post_process.handler == "ttft_curve_fit"
    assert out.post_process.params["metric_tag"] == "time_to_first_token"
    assert out.post_process.params["stat"] == "avg"
    assert out.post_process.output_filename == "prefill_curve.json"


def test_prefill_ttft_curve_overrides_isl_range():
    out = PrefillTTFTCurve().expand(make_ctx(isl_min=128, isl_max=4096, isl_steps=4))
    isl_values = out.sweep_parameters["datasets.main.prompts.isl"]
    assert isl_values[0] == 128
    assert isl_values[-1] == 4096


def test_prefill_ttft_curve_rejects_no_streaming():
    with pytest.raises(ValueError, match="streaming-only"):
        PrefillTTFTCurve().expand(make_ctx(streaming=False))


def test_prefill_ttft_curve_allows_default_streaming():
    out = PrefillTTFTCurve().expand(make_ctx(streaming=True))
    assert out.sweep_parameters is not None


# ---- Adversarial cases ----


def test_prefill_ttft_curve_isl_lo_eq_hi_raises():
    with pytest.raises(ValueError, match=r"hi .* must be > lo"):
        PrefillTTFTCurve().expand(make_ctx(isl_min=1024, isl_max=1024))


def test_prefill_ttft_curve_isl_lo_gt_hi_raises():
    with pytest.raises(ValueError, match=r"hi .* must be > lo"):
        PrefillTTFTCurve().expand(make_ctx(isl_min=4096, isl_max=256))


def test_prefill_ttft_curve_zero_isl_steps_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        PrefillTTFTCurve().expand(make_ctx(isl_steps=0))


def test_prefill_ttft_curve_concurrency_overrides_silently_ignored():
    # The recipe pins concurrency to [1] and intentionally does NOT consult
    # sweep_overrides for concurrency_min/max/steps. Pin this so a future
    # "make it configurable" change has to update this test on purpose.
    out = PrefillTTFTCurve().expand(
        make_ctx(concurrency_min=4, concurrency_max=64, concurrency_steps=3)
    )
    assert out.sweep_parameters["phases.profiling.concurrency"] == [1]


def test_prefill_ttft_curve_unknown_override_keys_silently_ignored():
    out = PrefillTTFTCurve().expand(make_ctx(unrecognized=42, isl_max=1024))
    assert out.sweep_parameters["datasets.main.prompts.isl"][-1] == 1024


def test_prefill_ttft_curve_ignores_sla_targets():
    out = PrefillTTFTCurve().expand(make_ctx(sla_targets={"ttft_sla_ms": 250.0}))
    assert out.sla_filters == []


def test_prefill_ttft_curve_string_overrides_coerce_via_int():
    out = PrefillTTFTCurve().expand(
        make_ctx(isl_min="64", isl_max="256", isl_steps="3")
    )
    isl_values = out.sweep_parameters["datasets.main.prompts.isl"]
    assert isl_values[0] == 64
    assert isl_values[-1] == 256
    assert len(isl_values) == 3


def test_prefill_ttft_curve_unparseable_string_override_raises():
    with pytest.raises(ValueError, match="invalid literal"):
        PrefillTTFTCurve().expand(make_ctx(isl_min="not-a-number"))


def test_prefill_ttft_curve_output_is_deterministic():
    a = PrefillTTFTCurve().expand(make_ctx())
    b = PrefillTTFTCurve().expand(make_ctx())
    assert a.sweep_parameters == b.sweep_parameters
    assert a.post_process == b.post_process


def test_prefill_ttft_curve_sweep_parameters_only_no_adaptive_search():
    out = PrefillTTFTCurve().expand(make_ctx())
    assert out.adaptive_search is None
    assert set(out.sweep_parameters.keys()) == {
        "datasets.main.prompts.isl",
        "phases.profiling.concurrency",
    }


def test_prefill_ttft_curve_streaming_error_message_names_recipe_and_metric():
    with pytest.raises(ValueError) as exc:
        PrefillTTFTCurve().expand(make_ctx(streaming=False))
    msg = str(exc.value)
    assert "prefill-ttft-curve" in msg
    assert "TTFT is a streaming-only metric" in msg


def test_prefill_ttft_curve_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "prefill-ttft-curve")
    assert resolved is PrefillTTFTCurve


def test_prefill_ttft_curve_isl_grid_strictly_ascending_and_unique():
    out = PrefillTTFTCurve().expand(make_ctx())
    isl_values = out.sweep_parameters["datasets.main.prompts.isl"]
    assert isl_values == sorted(set(isl_values))


def test_prefill_ttft_curve_two_steps_is_minimum_valid_input():
    # `_logspace_int_steps` requires steps >= 2; pin the boundary.
    out = PrefillTTFTCurve().expand(make_ctx(isl_steps=2))
    isl_values = out.sweep_parameters["datasets.main.prompts.isl"]
    assert isl_values == [256, 32768]

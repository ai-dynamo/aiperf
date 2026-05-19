# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DecodeITLCurve grid recipe.

Pins the concurrency x OSL grid defaults (concurrency in [1, 200], 6 steps;
osl in [64, 1024], 4 steps), the ``itl_surface_fit`` post-process spec, and
the streaming-required guard so the user-facing contract cannot drift.

Ported from ``ajc/k8s`` to the BenchmarkConfig-based ``SearchRecipeContext``.
Sweep keys are envelope-prefixed (``phases.profiling.concurrency``
and ``datasets.main.prompts.osl``).
"""

from __future__ import annotations

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import DecodeITLCurve
from tests.unit.search_recipes.conftest import make_ctx


def test_decode_itl_curve_default_grid_uses_concurrency_and_osl_endpoints():
    out = DecodeITLCurve().expand(make_ctx())
    assert out.sweep_parameters is not None
    concurrency_values = out.sweep_parameters["phases.profiling.concurrency"]
    osl_values = out.sweep_parameters["datasets.main.prompts.osl"]
    assert concurrency_values[0] == 1
    assert concurrency_values[-1] == 200
    assert osl_values[0] == 64
    assert osl_values[-1] == 1024


def test_decode_itl_curve_emits_itl_surface_fit_post_process():
    out = DecodeITLCurve().expand(make_ctx())
    assert out.post_process is not None
    assert out.post_process.handler == "itl_surface_fit"
    assert out.post_process.params["metric_tag"] == "inter_token_latency"
    assert out.post_process.params["stat"] == "avg"
    assert (
        out.post_process.params["concurrency_param"] == "phases.profiling.concurrency"
    )
    assert out.post_process.params["osl_param"] == "datasets.main.prompts.osl"
    assert out.post_process.output_filename == "decode_itl_surface.json"


def test_decode_itl_curve_overrides_concurrency_and_osl_ranges():
    out = DecodeITLCurve().expand(
        make_ctx(
            concurrency_min=4,
            concurrency_max=64,
            concurrency_steps=3,
            osl_min=128,
            osl_max=512,
            osl_steps=2,
        )
    )
    concurrency_values = out.sweep_parameters["phases.profiling.concurrency"]
    osl_values = out.sweep_parameters["datasets.main.prompts.osl"]
    assert concurrency_values[0] == 4
    assert concurrency_values[-1] == 64
    assert osl_values == [128, 512]


def test_decode_itl_curve_rejects_no_streaming():
    with pytest.raises(ValueError, match="streaming-only"):
        DecodeITLCurve().expand(make_ctx(streaming=False))


def test_decode_itl_curve_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "decode-itl-curve")
    assert resolved is DecodeITLCurve


def test_decode_itl_curve_default_step_counts_match_spec():
    out = DecodeITLCurve().expand(make_ctx())
    concurrency_values = out.sweep_parameters["phases.profiling.concurrency"]
    osl_values = out.sweep_parameters["datasets.main.prompts.osl"]
    assert len(concurrency_values) == 6
    assert len(osl_values) == 4


# ---- Adversarial cases ----


def test_decode_itl_curve_concurrency_lo_eq_hi_raises():
    with pytest.raises(ValueError, match=r"concurrency-min.*must be <"):
        DecodeITLCurve().expand(make_ctx(concurrency_min=10, concurrency_max=10))


def test_decode_itl_curve_osl_lo_eq_hi_raises():
    with pytest.raises(ValueError, match=r"hi .* must be > lo"):
        DecodeITLCurve().expand(make_ctx(osl_min=128, osl_max=128))


def test_decode_itl_curve_concurrency_lo_gt_hi_raises():
    with pytest.raises(ValueError, match=r"concurrency-min.*must be <"):
        DecodeITLCurve().expand(make_ctx(concurrency_min=200, concurrency_max=4))


def test_decode_itl_curve_zero_concurrency_steps_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        DecodeITLCurve().expand(make_ctx(concurrency_steps=0))


def test_decode_itl_curve_zero_osl_steps_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        DecodeITLCurve().expand(make_ctx(osl_steps=1))


def test_decode_itl_curve_unknown_override_keys_silently_ignored():
    out = DecodeITLCurve().expand(make_ctx(unrecognized=42, concurrency_max=128))
    values = out.sweep_parameters["phases.profiling.concurrency"]
    assert values[-1] == 128


def test_decode_itl_curve_ignores_sla_targets():
    # Grid recipe; sla_targets is irrelevant input.
    out = DecodeITLCurve().expand(make_ctx(sla_targets={"itl_sla_ms": 50.0}))
    assert out.sla_filters == []


def test_decode_itl_curve_string_overrides_coerce_via_int():
    out = DecodeITLCurve().expand(
        make_ctx(
            concurrency_min="2",
            concurrency_max="32",
            concurrency_steps="3",
            osl_min="64",
            osl_max="256",
            osl_steps="2",
        )
    )
    cvals = out.sweep_parameters["phases.profiling.concurrency"]
    ovals = out.sweep_parameters["datasets.main.prompts.osl"]
    assert cvals[0] == 2 and cvals[-1] == 32
    assert ovals == [64, 256]


def test_decode_itl_curve_unparseable_string_override_raises():
    with pytest.raises(ValueError, match="invalid literal"):
        DecodeITLCurve().expand(make_ctx(osl_min="abc"))


def test_decode_itl_curve_output_is_deterministic():
    a = DecodeITLCurve().expand(make_ctx())
    b = DecodeITLCurve().expand(make_ctx())
    assert a.sweep_parameters == b.sweep_parameters
    assert a.post_process == b.post_process


def test_decode_itl_curve_sweep_parameters_only_no_adaptive_search():
    out = DecodeITLCurve().expand(make_ctx())
    assert out.adaptive_search is None
    assert set(out.sweep_parameters.keys()) == {
        "phases.profiling.concurrency",
        "datasets.main.prompts.osl",
    }


def test_decode_itl_curve_grid_values_strictly_ascending_and_unique():
    out = DecodeITLCurve().expand(make_ctx())
    for path, values in out.sweep_parameters.items():
        assert values == sorted(set(values)), f"{path}: not strictly ascending"


def test_decode_itl_curve_streaming_error_message_names_recipe_and_metric():
    with pytest.raises(ValueError) as exc:
        DecodeITLCurve().expand(make_ctx(streaming=False))
    msg = str(exc.value)
    assert "decode-itl-curve" in msg
    assert "ITL is a streaming-only metric" in msg


def test_decode_itl_curve_only_concurrency_overridden_osl_keeps_default():
    # Mixing override with a default-kept dimension; default OSL endpoints stay
    # pinned at [64, 1024] when osl_* keys are absent from sweep_overrides.
    out = DecodeITLCurve().expand(make_ctx(concurrency_max=32))
    osl_values = out.sweep_parameters["datasets.main.prompts.osl"]
    assert osl_values[0] == 64
    assert osl_values[-1] == 1024

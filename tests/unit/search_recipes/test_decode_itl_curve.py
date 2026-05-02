# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DecodeITLCurve grid recipe."""

from __future__ import annotations

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes._base import SearchRecipeContext
from aiperf.search_recipes.builtins import DecodeITLCurve


def _ctx(*, streaming: bool = True, **overrides) -> SearchRecipeContext:
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=streaming))
    return SearchRecipeContext(user_config=user, sweep_overrides=overrides)


def test_decode_itl_curve_default_grid_uses_concurrency_and_osl_endpoints():
    out = DecodeITLCurve().expand(_ctx())
    assert out.sweep_variables is not None
    concurrency_path = "phases.profiling.concurrency"
    osl_path = "phases.profiling.synthetic_output_tokens.mean"
    concurrency_values = out.sweep_variables[concurrency_path]
    osl_values = out.sweep_variables[osl_path]
    # Defaults: concurrency in [1, 200] (6 steps), osl in [64, 1024] (4 steps).
    assert concurrency_values[0] == 1
    assert concurrency_values[-1] == 200
    assert osl_values[0] == 64
    assert osl_values[-1] == 1024


def test_decode_itl_curve_emits_itl_surface_fit_post_process():
    out = DecodeITLCurve().expand(_ctx())
    assert out.post_process is not None
    assert out.post_process.handler == "itl_surface_fit"
    assert out.post_process.params["metric_tag"] == "inter_token_latency"
    assert out.post_process.params["stat"] == "avg"
    assert (
        out.post_process.params["concurrency_param"] == "phases.profiling.concurrency"
    )
    assert (
        out.post_process.params["osl_param"]
        == "phases.profiling.synthetic_output_tokens.mean"
    )
    assert out.post_process.output_filename == "decode_itl_surface.json"


def test_decode_itl_curve_overrides_concurrency_and_osl_ranges():
    out = DecodeITLCurve().expand(
        _ctx(
            concurrency_min=4,
            concurrency_max=64,
            concurrency_steps=3,
            osl_min=128,
            osl_max=512,
            osl_steps=2,
        )
    )
    concurrency_values = out.sweep_variables["phases.profiling.concurrency"]
    osl_values = out.sweep_variables["phases.profiling.synthetic_output_tokens.mean"]
    assert concurrency_values[0] == 4
    assert concurrency_values[-1] == 64
    assert osl_values == [128, 512]


def test_decode_itl_curve_rejects_no_streaming():
    with pytest.raises(ValueError, match="streaming-only"):
        DecodeITLCurve().expand(_ctx(streaming=False))


def test_decode_itl_curve_resolves_through_plugin_registry():
    resolved = plugins.get_class(PluginType.SEARCH_RECIPE, "decode-itl-curve")
    assert resolved is DecodeITLCurve


def test_decode_itl_curve_default_step_counts_match_spec():
    out = DecodeITLCurve().expand(_ctx())
    concurrency_values = out.sweep_variables["phases.profiling.concurrency"]
    osl_values = out.sweep_variables["phases.profiling.synthetic_output_tokens.mean"]
    assert len(concurrency_values) == 6
    assert len(osl_values) == 4

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PrefillTTFTCurve grid recipe."""

from __future__ import annotations

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.search_recipes._base import SearchRecipeContext
from aiperf.search_recipes.builtins import PrefillTTFTCurve


def _ctx(*, streaming: bool = True, **overrides) -> SearchRecipeContext:
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=streaming))
    return SearchRecipeContext(user_config=user, sweep_overrides=overrides)


def test_prefill_ttft_curve_default_grid_uses_isl_min_max_endpoints():
    out = PrefillTTFTCurve().expand(_ctx())
    assert out.sweep_variables is not None
    isl_path = "datasets.profiling.prompts.isl"
    isl_values = out.sweep_variables[isl_path]
    assert isl_values[0] == 256
    assert isl_values[-1] == 32768


def test_prefill_ttft_curve_pins_concurrency_to_one():
    out = PrefillTTFTCurve().expand(_ctx())
    concurrency_values = out.sweep_variables["phases.profiling.concurrency"]
    assert concurrency_values == [1]


def test_prefill_ttft_curve_emits_ttft_curve_fit_post_process():
    out = PrefillTTFTCurve().expand(_ctx())
    assert out.post_process is not None
    assert out.post_process.handler == "ttft_curve_fit"
    assert out.post_process.params["metric_tag"] == "time_to_first_token"
    assert out.post_process.params["stat"] == "avg"
    assert out.post_process.output_filename == "prefill_curve.json"


def test_prefill_ttft_curve_overrides_isl_range():
    out = PrefillTTFTCurve().expand(_ctx(isl_min=128, isl_max=4096, isl_steps=4))
    isl_path = "datasets.profiling.prompts.isl"
    isl_values = out.sweep_variables[isl_path]
    assert isl_values[0] == 128
    assert isl_values[-1] == 4096


def test_prefill_ttft_curve_rejects_no_streaming():
    with pytest.raises(ValueError, match="streaming-only"):
        PrefillTTFTCurve().expand(_ctx(streaming=False))


def test_prefill_ttft_curve_allows_unset_streaming():
    # Default streaming=True path; recipe accepts and produces sweep_variables.
    out = PrefillTTFTCurve().expand(_ctx(streaming=True))
    assert out.sweep_variables is not None

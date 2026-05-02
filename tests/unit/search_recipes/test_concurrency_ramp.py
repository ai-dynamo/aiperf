# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ConcurrencyRamp grid recipe."""

from __future__ import annotations

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.search_recipes._base import SearchRecipeContext
from aiperf.search_recipes.builtins import ConcurrencyRamp


def _ctx(*, streaming: bool = True, **overrides) -> SearchRecipeContext:
    user = UserConfig(endpoint=EndpointConfig(model_names=["m"], streaming=streaming))
    return SearchRecipeContext(user_config=user, sweep_overrides=overrides)


def test_concurrency_ramp_default_grid_uses_endpoints_1_and_1000():
    out = ConcurrencyRamp().expand(_ctx())
    assert out.adaptive_search is None
    assert out.sweep_variables is not None
    values = out.sweep_variables["phases.profiling.concurrency"]
    assert values[0] == 1
    assert values[-1] == 1000
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_concurrency_ramp_emits_post_process_with_threshold():
    out = ConcurrencyRamp().expand(_ctx(degradation_threshold=0.30))
    assert out.post_process is not None
    assert out.post_process.handler == "degradation_knee_detect"
    assert out.post_process.params["threshold_pct"] == 0.30
    assert out.post_process.params["metric_tag"] == "request_latency"
    assert out.post_process.params["stat"] == "p99"
    assert out.post_process.output_filename == "degradation_knee.json"


def test_concurrency_ramp_default_threshold_is_20_percent():
    out = ConcurrencyRamp().expand(_ctx())
    assert out.post_process.params["threshold_pct"] == 0.20


def test_concurrency_ramp_overrides_extend_grid_range():
    out = ConcurrencyRamp().expand(
        _ctx(concurrency_min=4, concurrency_max=64, concurrency_steps=4)
    )
    values = out.sweep_variables["phases.profiling.concurrency"]
    assert values[0] == 4
    assert values[-1] == 64


def test_concurrency_ramp_invalid_step_count_raises():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        ConcurrencyRamp().expand(_ctx(concurrency_steps=1))


def test_concurrency_ramp_does_not_require_streaming():
    # request_latency is end-to-end, available without streaming.
    out = ConcurrencyRamp().expand(_ctx(streaming=False))
    assert out.sweep_variables is not None

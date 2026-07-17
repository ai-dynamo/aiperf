# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weka trace-model NaN/Inf discipline.

A bare ``ge`` bound on ``api_time`` / ``think_time`` /
``ttft`` would accept ``+inf`` (``inf >= 0`` satisfies it). ``inf``
ingress is real via HuggingFace ``datasets`` rows (Arrow floats bypass orjson's
``Infinity`` rejection) and flows into ``raw_end = t + api_time`` edge-delay
math, stamping non-finite microsecond delays into ``StaticEdge`` floats that
cross the msgpack boundary. Every duration field must reject non-finite values
at parse time.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.graph.adapters.weka.trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
)


def _request(type_: str, **overrides: object) -> dict:
    base: dict = {
        "t": 0.0,
        "type": type_,
        "model": "m",
        "in": 64,
        "out": 8,
        "hash_ids": [1],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "model_cls,type_,field",
    [
        param(WekaNormalRequest, "n", "api_time", id="n-api_time"),
        param(WekaNormalRequest, "n", "think_time", id="n-think_time"),
        param(WekaStreamingRequest, "s", "api_time", id="s-api_time"),
        param(WekaStreamingRequest, "s", "think_time", id="s-think_time"),
        param(WekaStreamingRequest, "s", "ttft", id="s-ttft"),
    ],
)  # fmt: skip
@pytest.mark.parametrize(
    "bad",
    [
        param(float("inf"), id="inf"),
        param(float("nan"), id="nan"),
    ],
)  # fmt: skip
def test_duration_fields_reject_non_finite(
    model_cls: type, type_: str, field: str, bad: float
) -> None:
    with pytest.raises(ValidationError):
        model_cls.model_validate(_request(type_, **{field: bad}))


@pytest.mark.parametrize(
    "model_cls,type_",
    [
        param(WekaNormalRequest, "n", id="normal"),
        param(WekaStreamingRequest, "s", id="streaming"),
    ],
)  # fmt: skip
def test_finite_duration_fields_still_accepted(model_cls: type, type_: str) -> None:
    req = model_cls.model_validate(_request(type_, api_time=0.5, think_time=1.5))
    assert req.api_time == 0.5
    assert req.think_time == 1.5

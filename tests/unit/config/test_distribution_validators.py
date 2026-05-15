# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution parameter validators: reject non-positive / non-finite values.

Round-2 R2-M14 reproduced ``isl: {mean: -512, stddev: 0}`` validating
silently. NormalDistribution truncates samples below 0, so a non-positive
mean produces a degenerate (or impossible) distribution that crashes
later in synthesis. The fix adds ``gt=0`` to the mean field plus a
finite-value validator covering NaN/inf.
"""

from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.config.distributions import SamplingDistribution

_DIST = TypeAdapter(SamplingDistribution)


@pytest.mark.parametrize(
    "mean",
    [
        pytest.param(-512, id="negative_int"),
        pytest.param(-1.0, id="negative_float"),
        pytest.param(-0.0001, id="just_below_zero"),
    ],
)
def test_normal_distribution_rejects_negative_mean(mean: float) -> None:
    """NormalDistribution rejects negative means at validation time.

    Zero is allowed (OSL=0 disables output, turn_delay.mean=0 disables
    inter-turn delay), but negatives are nonsensical because the
    distribution truncates at 0.
    """
    with pytest.raises(ValidationError):
        _DIST.validate_python({"mean": mean, "stddev": 0})


@pytest.mark.parametrize(
    "mean",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive_inf"),
        pytest.param(float("-inf"), id="negative_inf"),
    ],
)
def test_normal_distribution_rejects_non_finite_mean(mean: float) -> None:
    """NormalDistribution rejects NaN/inf means.

    NaN/inf would silently propagate through synthesis and surface only
    as a mid-flight crash. Validate at config time instead.
    """
    with pytest.raises(ValidationError):
        _DIST.validate_python({"mean": mean, "stddev": 0})


def test_normal_distribution_rejects_non_finite_stddev() -> None:
    with pytest.raises(ValidationError):
        _DIST.validate_python({"mean": 100, "stddev": math.inf})


def test_normal_distribution_accepts_positive_mean() -> None:
    """Sanity check: legitimate inputs still validate."""
    dist = _DIST.validate_python({"mean": 512, "stddev": 50})
    assert dist.mean == 512.0
    assert dist.stddev == 50.0

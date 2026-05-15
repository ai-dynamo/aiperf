# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_replicate_budget`` (Hyperband-style allocator + bootstrap CI)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import param

from aiperf.orchestrator.search_planner._replicate_budget import (
    boundary_ci,
    replicate_count,
)


def test_count_caps_at_20() -> None:
    assert replicate_count(sigma_margin=1000.0, threshold=1.0) == 20


def test_count_floors_at_3() -> None:
    assert replicate_count(sigma_margin=0.0, threshold=100.0) == 3
    assert replicate_count(sigma_margin=0.001, threshold=100.0) == 3


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [
        param(0.5, 3, id="sigma_half_threshold_floors"),
        param(1.0, 4, id="sigma_equal_threshold_gives_4"),
        param(2.0, 16, id="sigma_double_threshold_gives_16"),
        param(3.0, 20, id="sigma_triple_threshold_caps"),
    ],
)  # fmt: skip
def test_count_scales_with_sigma_squared(sigma: float, expected: int) -> None:
    assert replicate_count(sigma_margin=sigma, threshold=1.0) == expected


def test_override_bypasses_formula() -> None:
    assert replicate_count(sigma_margin=1000.0, threshold=1.0, override=5) == 5
    assert replicate_count(sigma_margin=0.0, threshold=100.0, override=7) == 7


def test_override_zero_uses_formula() -> None:
    assert replicate_count(sigma_margin=2.0, threshold=1.0, override=0) == 16


def test_count_handles_zero_threshold() -> None:
    result = replicate_count(sigma_margin=1.0, threshold=0.0)
    assert result == 20


def test_boundary_ci_singleton_returns_point() -> None:
    lo, hi = boundary_ci([42.0])
    assert lo == 42.0
    assert hi == 42.0


def test_boundary_ci_brackets_mean() -> None:
    rng = np.random.default_rng(42)
    margins = [float(x) for x in rng.normal(loc=10.0, scale=1.0, size=50)]
    lo, hi = boundary_ci(margins, n_resamples=2000)
    mean = float(np.mean(margins))
    assert lo <= mean <= hi
    assert hi - lo < 2.0


def test_boundary_ci_returns_floats() -> None:
    lo, hi = boundary_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=1000)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi

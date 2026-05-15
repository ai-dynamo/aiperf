# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_cliff_detect`` (PAVA-residual cliff guard)."""

from __future__ import annotations

import numpy as np

from aiperf.orchestrator.search_planner._cliff_detect import detect_cliff
from aiperf.orchestrator.search_planner._smooth_isotonic_fit import smooth_isotonic_fit


def test_smooth_curve_returns_false() -> None:
    rng = np.random.default_rng(42)
    xs = list(range(1, 11))
    truth = [0.5 * (x - 5.0) for x in xs]
    raw = [t + float(rng.normal(scale=0.05)) for t in truth]
    points = list(zip(xs, raw, strict=True))
    curve, _ = smooth_isotonic_fit(xs, raw)
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=4,
        infeasible_min=6,
        x_hi=10,
        precision=0.05,
    )
    assert is_cliff is False


def test_cliff_returns_true() -> None:
    # Mid-series spike forces PAVA to pool every point, so the smooth fit at
    # x_last (~18.6) sits far above the raw last-probe value (5.1). Last-3 raw
    # margins are tight (sigma ~0.04), so 3-sigma is dwarfed by the residual.
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    base = [1.0, 100.0, 5.0, 5.05, 5.1, 5.0, 5.05, 5.1]
    points = list(zip(xs, base, strict=True))
    curve, _ = smooth_isotonic_fit(xs, base)
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=2,
        infeasible_min=8,
        x_hi=10,
        precision=0.05,
    )
    assert is_cliff is True


def test_insufficient_points_returns_false() -> None:
    points = [(1, -1.0), (2, 1.0)]
    curve, _ = smooth_isotonic_fit([1, 2, 3], [-1.0, 0.0, 1.0])
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=1,
        infeasible_min=2,
        x_hi=10,
        precision=0.05,
    )
    assert is_cliff is False


def test_zero_sigma_returns_false() -> None:
    points = [(1, 0.0), (2, 0.0), (3, 0.0)]
    curve, _ = smooth_isotonic_fit([1, 2, 3], [0.0, 0.0, 0.0])
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=1,
        infeasible_min=3,
        x_hi=10,
        precision=0.05,
    )
    assert is_cliff is False


def test_cliff_residual_too_large_but_bracket_too_narrow_returns_false() -> None:
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    base = [1.0, 100.0, 5.0, 5.05, 5.1, 5.0, 5.05, 5.1]
    points = list(zip(xs, base, strict=True))
    curve, _ = smooth_isotonic_fit(xs, base)
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=7,
        infeasible_min=8,
        x_hi=10,
        precision=0.5,
    )
    assert is_cliff is False


def test_no_bracket_returns_false() -> None:
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    base = [1.0, 100.0, 5.0, 5.05, 5.1, 5.0, 5.05, 5.1]
    points = list(zip(xs, base, strict=True))
    curve, _ = smooth_isotonic_fit(xs, base)
    is_cliff = detect_cliff(
        raw_points=points,
        fitted_curve=curve,
        feasible_max=None,
        infeasible_min=None,
        x_hi=1000,
        precision=0.05,
    )
    assert is_cliff is False

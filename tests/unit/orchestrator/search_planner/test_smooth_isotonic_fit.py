# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_smooth_isotonic_fit`` (PAVA + PCHIP)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import param

from aiperf.orchestrator.search_planner._smooth_isotonic_fit import (
    _strictify,
    find_root,
    smooth_isotonic_fit,
)


def test_fit_monotone_data_recovers_root() -> None:
    xs = [1, 2, 3, 4, 5]
    ys = [-2.0, -1.0, 0.0, 1.0, 2.0]
    curve, y_hat = smooth_isotonic_fit(xs, ys)
    root = find_root(curve, 1.0, 5.0)
    assert root is not None
    assert root == pytest.approx(3.0, abs=0.05)
    assert len(y_hat) == 5


def test_fit_with_noise_denoises() -> None:
    rng = np.random.default_rng(42)
    xs = list(range(1, 21))
    truth = [0.5 * (x - 10.0) for x in xs]
    noisy = [t + float(rng.normal(scale=0.5)) for t in truth]
    curve, _ = smooth_isotonic_fit(xs, noisy)
    root = find_root(curve, 1.0, 20.0)
    assert root is not None
    assert root == pytest.approx(10.0, abs=1.0)


def test_strictify_breaks_flat_runs() -> None:
    y_hat = np.array([1.0, 2.5, 2.5, 2.5, 4.0])
    strict = _strictify(y_hat)
    diffs = np.diff(strict)
    assert np.all(diffs > 0)
    assert strict[0] == pytest.approx(1.0, abs=1e-6)
    assert strict[-1] == pytest.approx(4.0, abs=1e-6)


@pytest.mark.parametrize(
    ("ys", "expected_root_present"),
    [
        param([1.0, 2.0, 3.0, 4.0, 5.0], False, id="all_positive_no_crossing"),
        param([-5.0, -4.0, -3.0, -2.0, -1.0], False, id="all_negative_no_crossing"),
        param([-2.0, -1.0, 0.5, 1.0, 2.0], True, id="crossing_present"),
    ],
)  # fmt: skip
def test_find_root_returns_none_when_no_crossing(
    ys: list[float], expected_root_present: bool
) -> None:
    xs = [1, 2, 3, 4, 5]
    curve, _ = smooth_isotonic_fit(xs, ys)
    root = find_root(curve, 1.0, 5.0)
    assert (root is not None) == expected_root_present


def test_find_root_brentq_fallback_for_plain_callable() -> None:
    def linear(x: float) -> float:
        return x - 7.0

    root = find_root(linear, 0.0, 10.0)
    assert root == pytest.approx(7.0, abs=1e-6)


def test_find_root_brentq_fallback_returns_none_no_sign_change() -> None:
    def linear(x: float) -> float:
        return x + 5.0

    root = find_root(linear, 0.0, 10.0)
    assert root is None


def test_strictify_handles_all_equal_input() -> None:
    y_hat = np.array([2.0, 2.0, 2.0, 2.0])
    strict = _strictify(y_hat)
    assert np.all(np.diff(strict) > 0)

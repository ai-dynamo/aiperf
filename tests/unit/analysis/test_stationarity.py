# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for stationarity validation."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.stationarity import (
    batch_means_trend_test,
    spearman_rank_correlation,
)


class TestSpearmanRankCorrelation:
    def test_perfect_positive(self) -> None:
        x = np.arange(10, dtype=np.float64)
        y = np.arange(10, dtype=np.float64)
        rho, p = spearman_rank_correlation(x, y)
        assert rho == pytest.approx(1.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_perfect_negative(self) -> None:
        x = np.arange(10, dtype=np.float64)
        y = np.arange(10, dtype=np.float64)[::-1].copy()
        rho, p = spearman_rank_correlation(x, y)
        assert rho == pytest.approx(-1.0)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_no_correlation(self) -> None:
        """Uncorrelated data → rho near 0, p > 0.05."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        rho, p = spearman_rank_correlation(x, y)
        assert abs(rho) < 0.3
        assert p > 0.05

    def test_short_input(self) -> None:
        """Fewer than 3 elements → returns (0.0, 1.0)."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        rho, p = spearman_rank_correlation(x, y)
        assert rho == 0.0
        assert p == 1.0

    def test_constant_series_no_trend(self) -> None:
        """Perfectly flat (zero-variance) series → rho=0.0, p=1.0, no trend.

        Regression: positional ranks assigned a constant series rho≈1.0 with
        p≈0, misreading a stationary window as a strong upward trend.
        """
        x = np.arange(100, dtype=np.float64)
        y = np.full(100, 5.0)
        rho, p = spearman_rank_correlation(x, y)
        assert rho == 0.0
        assert p == 1.0

    def test_partially_tied_flat_series_no_spurious_trend(self) -> None:
        """Tied clusters with no directional drift → |rho| small, p not significant."""
        x = np.arange(30, dtype=np.float64)
        # Quantized latency oscillating between two levels — heavy ties, no trend.
        y = np.array(([5.0, 5.0, 6.0, 6.0] * 8)[:30], dtype=np.float64)
        rho, p = spearman_rank_correlation(x, y)
        assert abs(rho) < 0.3  # no-tie-handling path would bias this upward
        assert p > 0.05

    def test_some_ties_with_trend_matches_scipy(self) -> None:
        """Ties + genuine trend → average-rank rho matches scipy.stats.spearmanr."""
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        x = np.arange(60, dtype=np.float64)
        # Quantized upward series: real trend plus many tied ranks.
        y = np.round(x * 0.3 + rng.normal(0, 1, 60))
        rho, _ = spearman_rank_correlation(x, y)
        expected, _ = scipy_stats.spearmanr(x, y)
        assert rho == pytest.approx(float(expected), abs=1e-9)


class TestBatchMeansTrendTest:
    def test_stationary_input(self) -> None:
        """Stationary noise → small |rho|, large p."""
        rng = np.random.default_rng(42)
        values = rng.normal(100, 5, 200)
        rho, p = batch_means_trend_test(values)
        assert abs(rho) < 0.6
        assert p > 0.05

    def test_trending_input(self) -> None:
        """Monotonically increasing → large |rho|, small p."""
        values = np.linspace(0, 100, 200)
        rho, p = batch_means_trend_test(values)
        assert abs(rho) > 0.8
        assert p < 0.01

    def test_short_input(self) -> None:
        """Fewer than k elements → returns (0.0, 1.0)."""
        values = np.array([1.0, 2.0, 3.0])
        rho, p = batch_means_trend_test(values, k=10)
        assert rho == 0.0
        assert p == 1.0

    def test_constant_input_no_stationarity_warning(self) -> None:
        """Perfectly flat window → rho=0, p=1, no stationarity_warning.

        Regression: constant latency previously yielded rho≈1.0, p≈0 and a
        spurious stationarity_warning=True for a genuinely stable benchmark.
        """
        values = np.full(200, 5.0)
        rho, p = batch_means_trend_test(values)
        assert rho == 0.0
        assert p == 1.0
        assert (abs(rho) > 0.65 and p < 0.05) is False

    def test_trending_input_still_detected(self) -> None:
        """Genuine monotonic upward series still flags a trend (no regression)."""
        values = np.linspace(0, 100, 200)
        rho, p = batch_means_trend_test(values)
        assert abs(rho) > 0.65
        assert p < 0.05
        assert (abs(rho) > 0.65 and p < 0.05) is True

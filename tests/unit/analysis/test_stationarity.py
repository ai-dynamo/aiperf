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

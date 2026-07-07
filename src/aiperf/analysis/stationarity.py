# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc stationarity validation for steady-state windows."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def spearman_rank_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient and two-sided p-value.

    Pure numpy implementation — no scipy dependency.

    Args:
        x: First array of observations.
        y: Second array of observations (same length as x).

    Returns:
        (rho, p_value). rho in [-1, 1], p_value in [0, 1].
        Returns (0.0, 1.0) for arrays shorter than 3 elements, or when either
        input has zero variance (all-constant) — rho is undefined (0/0) there,
        so we report the documented "no trend" result rather than a spurious 1.0.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    # A constant series has no rank ordering: correlation is undefined (0/0).
    # Report "no trend" so a flat/stationary window is never flagged as trending.
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0, 1.0

    # Average-rank tie handling (scipy.stats.rankdata "average" convention):
    # equal values share the mean of the ranks they would otherwise occupy.
    # Positional ranks would give a constant/tied series a spurious rho of ~1.
    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)

    # Pearson correlation of ranks
    corr_matrix = np.corrcoef(rank_x, rank_y)
    rho = float(corr_matrix[0, 1])

    # Clamp to avoid numerical issues
    rho = max(-1.0, min(1.0, rho))

    if abs(rho) >= 1.0:
        return rho, 0.0

    # t-statistic: t = rho * sqrt((n-2) / (1 - rho^2))
    t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))

    # Two-sided p-value from t-distribution with n-2 degrees of freedom
    # using the regularized incomplete beta function
    p_value = _t_distribution_two_sided_p(t_stat, n - 2)
    return rho, p_value


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank values 1..n, assigning tied values the average of their ranks.

    Matches ``scipy.stats.rankdata(values, method="average")`` without the
    scipy dependency. A run of equal values receives the mean of the positional
    ranks it spans, so a constant series ranks all-equal (mean rank) rather than
    monotonically — the correction that keeps a flat window from reading as a trend.
    """
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    positional = np.empty(n, dtype=np.float64)
    positional[order] = np.arange(1, n + 1, dtype=np.float64)

    sorted_values = values[order]
    average = positional.copy()
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                mean_rank = (start + 1 + i) / 2.0
                average[order[start:i]] = mean_rank
            start = i
    return average


def _t_distribution_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value from the t-distribution using incomplete beta.

    p = 1 - I_{x}(a, b) where x = df/(df + t^2), a = df/2, b = 0.5.
    """
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    p = _regularized_incomplete_beta(x, a, b)
    return max(0.0, min(1.0, p))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Uses the Lentz algorithm for the continued fraction expansion.
    Sufficient precision for p-value computation in trend tests.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    log_prefactor = (
        a * math.log(x) + b * math.log(1.0 - x) - math.log(a) - _log_beta(a, b)
    )
    prefactor = math.exp(log_prefactor)

    # Continued fraction (Lentz's method)
    cf = _beta_continued_fraction(x, a, b)
    return prefactor * cf


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function: log(B(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Evaluate the continued fraction for the incomplete beta function."""
    max_iter = 200
    eps = 1e-14
    tiny = 1e-30

    # Modified Lentz's algorithm
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step: d_{2m}
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        f *= d * c

        # Odd step: d_{2m+1}
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return f


def batch_means_trend_test(
    values: NDArray[np.float64],
    k: int = 10,
) -> tuple[float, float]:
    """Test for trend using Spearman rank correlation of batch means.

    Splits time-ordered observations into k batches, computes their means,
    then tests whether the means correlate with their index (trend).

    Args:
        values: Time-ordered observations (NaN-free, len >= k).
        k: Number of batches (default 10).

    Returns:
        (correlation, p_value). |correlation| > 0.65 with p < 0.05
        suggests a statistically significant trend.
    """
    if len(values) < k:
        return 0.0, 1.0

    batch_size = len(values) // k
    trimmed = values[: batch_size * k]
    batch_means = trimmed.reshape(k, batch_size).mean(axis=1)

    return spearman_rank_correlation(np.arange(k, dtype=np.float64), batch_means)

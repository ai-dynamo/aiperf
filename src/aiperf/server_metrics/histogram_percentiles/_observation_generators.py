# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-bucket observation-generation strategies.

Three strategies based on learned variance/mean:
- F3 two-point mass (tiny variance)
- Variance-aware truncated normal (moderate variance)
- Blended variance-aware + shifted uniform (tight variance near bucket center)
"""

import numpy as np


def _generate_f3_observations(
    count: int,
    *,
    lower: float,
    upper: float,
    mean: float,
    variance: float,
) -> np.ndarray:
    """Generate F3 two-point mass distribution for tight variance.

    When variance is extremely tight (< 1% of bucket width), observations are
    highly concentrated. F3 (HistogramTools, arXiv 2504.00001) places mass at
    two carefully chosen points to exactly match the first two moments.

    F3 distribution: mass at {x, a} where:
        - x = lower bound (one point mass)
        - a = mean + variance / (mean - x) (second point mass)
        - p_x = variance / (variance + (mean - x)^2) (probability at x)
    """
    if count <= 0:
        return np.array([], dtype=np.float64)

    x = lower
    a = mean + variance / (mean - x) if mean - x > 0 else upper
    a = float(np.clip(a, lower, upper))

    denominator = variance + (mean - x) ** 2
    p_x = variance / denominator if denominator > 0 else 0.5
    p_x = float(np.clip(p_x, 0.0, 1.0))

    n_x = int(count * p_x)

    observations = np.empty(count, dtype=np.float64)
    observations[:n_x] = x
    observations[n_x:] = a

    return observations


def _generate_variance_aware_observations(
    count: int,
    *,
    lower: float,
    upper: float,
    mean: float,
    std: float,
) -> np.ndarray:
    """Generate observations shaped by learned variance.

    Uses linear interpolation from mean toward bucket edges, scaled by
    learned standard deviation. Below-mean fractions interpolate toward
    the lower bound; above-mean fractions toward the upper bound. More
    accurate than uniform distribution for moderate variance (20-50% of
    bucket width).
    """
    if count <= 0:
        return np.array([], dtype=np.float64)

    fractions = (np.arange(count) + 0.5) / count

    # How many stds from mean to bucket edges? Clamped to +/- 3 sigma for stability.
    stds_to_lower = min((mean - lower) / std if std > 0 else 3.0, 3.0)
    stds_to_upper = min((upper - mean) / std if std > 0 else 3.0, 3.0)

    # Below mean (f < 0.5): pos = mean - stds_to_lower * std * (1 - 2*f)
    # Above mean (f >= 0.5): pos = mean + stds_to_upper * std * (2*f - 1)
    positions = np.where(
        fractions < 0.5,
        mean - stds_to_lower * std * (1 - 2 * fractions),
        mean + stds_to_upper * std * (2 * fractions - 1),
    )

    return np.clip(positions, lower, upper)


def _generate_blended_observations(
    count: int,
    *,
    lower: float,
    upper: float,
    mean: float,
    std: float,
    blend_factor: float = 0.5,
) -> np.ndarray:
    """Blend variance-aware and shifted-uniform distributions.

    Used when variance is tight (< 20% of bucket width) AND mean is near
    bucket center (< 30% offset). Provides robustness against variance
    estimation errors while still incorporating learned distribution shape.
    """
    if count <= 0:
        return np.array([], dtype=np.float64)

    bucket_width = upper - lower
    midpoint = (lower + upper) / 2.0

    shift = mean - midpoint
    fractions = (np.arange(count) + 0.5) / count
    uniform_obs = np.clip(lower + bucket_width * fractions + shift, lower, upper)

    variance_obs = _generate_variance_aware_observations(
        count, lower=lower, upper=upper, mean=mean, std=std
    )

    blended = (1 - blend_factor) * uniform_obs + blend_factor * variance_obs
    return np.clip(blended, lower, upper)

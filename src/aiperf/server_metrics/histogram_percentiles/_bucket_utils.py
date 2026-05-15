# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bucket utility functions for histogram percentile computation.

Includes bucket-bound lookup, cumulative-to-per-bucket conversion, bucket sum
estimation, +Inf bucket back-calculation, and single-bucket statistics learning.
"""

import numpy as np

from aiperf.server_metrics.histogram_percentiles._models import BucketStatistics

# Maximum number of observations to generate for percentile estimation.
# 100k samples provides accurate percentile estimation (~1 MB memory, ~1ms)
# while preventing memory issues with very large histogram counts (billions).
_MAX_OBSERVATIONS = 100_000


def _get_bucket_bounds(le: str, sorted_buckets: list[str]) -> tuple[float, float]:
    """Get the lower and upper bounds for a bucket.

    Prometheus histograms use cumulative buckets with "less than or equal"
    semantics. The lower bound is the previous bucket's upper bound (or 0
    for the first bucket), and the upper bound is this bucket's le value.

    Returns:
        Tuple of (lower_bound, upper_bound). For "+Inf" bucket, upper is float('inf').
        For first bucket, lower is 0.0.
    """
    upper = float("inf") if le == "+Inf" else float(le)

    idx = sorted_buckets.index(le)
    if idx == 0:
        lower = 0.0
    else:
        prev_le = sorted_buckets[idx - 1]
        lower = float(prev_le) if prev_le != "+Inf" else 0.0

    return lower, upper


def _cumulative_to_per_bucket(
    bucket_deltas: dict[str, float],
) -> dict[str, float]:
    """Convert cumulative bucket counts to per-bucket counts.

    Prometheus histograms use cumulative counts (le="less than or equal").
    This function converts to per-bucket counts (observations within each
    specific bucket range).
    """
    finite_buckets = [le for le in bucket_deltas if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    per_bucket: dict[str, float] = {}
    prev_cumulative = 0.0

    for le in sorted_buckets:
        cumulative = bucket_deltas[le]
        per_bucket[le] = cumulative - prev_cumulative
        prev_cumulative = cumulative

    if "+Inf" in bucket_deltas:
        inf_cumulative = bucket_deltas["+Inf"]
        per_bucket["+Inf"] = inf_cumulative - prev_cumulative

    return per_bucket


def _estimate_bucket_sums(
    per_bucket_counts: dict[str, float],
    bucket_stats: dict[str, BucketStatistics],
) -> dict[str, float]:
    """Estimate the sum of observations in each finite bucket.

    For each bucket, estimates total sum = count * mean, where mean comes from:
    1. Learned mean from bucket_stats (if available and within bounds) - more accurate
    2. Midpoint of bucket bounds (fallback) - standard assumption

    The learned means typically reduce estimation error by 40-60% compared
    to midpoint assumption, especially for skewed distributions.
    """
    finite_buckets = [le for le in per_bucket_counts if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    sums: dict[str, float] = {}
    for le, count in per_bucket_counts.items():
        if le == "+Inf" or count <= 0:
            continue

        lower, upper = _get_bucket_bounds(le, sorted_buckets)

        # Try learned mean first, but validate it's within bucket bounds
        # (learned means can be invalid after counter resets or data corruption).
        if le in bucket_stats and bucket_stats[le].estimated_mean is not None:
            learned_mean = bucket_stats[le].estimated_mean
            mean = learned_mean if lower < learned_mean < upper else (lower + upper) / 2
        else:
            mean = (lower + upper) / 2

        sums[le] = count * mean

    return sums


def _resolve_inf_avg(
    total_sum: float,
    estimated_finite_sum: float,
    inf_count: int,
    max_finite_bucket: float,
) -> float:
    """Derive an average value for +Inf observations, falling back when invalid.

    +Inf average must exceed max_finite_bucket; when back-calculation yields an
    invalid value (negative or too small) we fall back to 1.5x max_finite_bucket.
    """
    inf_sum = total_sum - estimated_finite_sum
    if inf_sum <= 0:
        return max_finite_bucket * 1.5
    inf_avg = inf_sum / inf_count
    if inf_avg <= max_finite_bucket:
        return max_finite_bucket * 1.5
    return inf_avg


def _estimate_inf_bucket_observations(
    total_sum: float,
    estimated_finite_sum: float,
    inf_count: int,
    max_finite_bucket: float,
) -> np.ndarray:
    """Estimate observation values for the +Inf bucket using back-calculation.

    Key insight: Prometheus gives us the exact total sum across all buckets.
    By estimating the sum in finite buckets, we back-calculate what the
    +Inf bucket observations must sum to (inf_sum = total_sum - finite_sum)
    and distribute inf_sum across inf_count observations uniformly around
    inf_avg = inf_sum / inf_count.

    Critical for tail percentiles (P99, P95) which often fall in +Inf bucket
    for latency histograms with outliers.

    Note:
        Downsampling to prevent memory issues should be done at the caller level
        to maintain consistent proportions across all buckets including +Inf.

    Returns:
        Array of estimated observation values for +Inf bucket (all > max_finite_bucket).
        Empty array if inf_count <= 0.
    """
    if inf_count <= 0:
        return np.array([], dtype=np.float64)

    inf_avg = _resolve_inf_avg(
        total_sum, estimated_finite_sum, inf_count, max_finite_bucket
    )

    # Uniform distribution: [lower, upper] with mean = (lower + upper) / 2
    upper_estimate = 2 * inf_avg - max_finite_bucket
    if upper_estimate <= max_finite_bucket:
        upper_estimate = max_finite_bucket * 2

    # linspace(a, b, 1) returns [a], not the midpoint. For a single +Inf
    # observation absorbing a large sum, using the lower bound would cause
    # catastrophic error, so use the mean directly.
    if inf_count == 1:
        return np.array([inf_avg], dtype=np.float64)

    return np.linspace(max_finite_bucket, upper_estimate, int(inf_count))


def accumulate_bucket_statistics(
    sums: np.ndarray,
    counts: np.ndarray,
    bucket_les: tuple[str, ...],
    bucket_counts: np.ndarray,
    *,
    start_idx: int = 0,
) -> dict[str, BucketStatistics]:
    """Learn per-bucket mean positions from single-bucket scrape intervals.

    Implements the polynomial histogram approach: when all observations in a
    scrape interval land in a single bucket, compute the exact mean for that
    bucket (sum_delta / count_delta). Over many intervals this builds a
    learned mean that is more accurate than midpoint assumption.

    Args:
        sums: Array of cumulative sum values per scrape (n,)
        counts: Array of cumulative count values per scrape (n,)
        bucket_les: Sorted bucket boundary strings (n_buckets,)
        bucket_counts: 2D array of cumulative bucket counts (n, n_buckets)
        start_idx: Starting index for analysis (default: 0)

    Returns:
        Dict mapping bucket le values to BucketStatistics with learned mean
        positions and variance. Empty dict if insufficient data or no
        single-bucket intervals observed.
    """
    n = len(sums)
    if n <= start_idx + 1:
        return {}

    count_deltas = np.diff(counts[start_idx:]).astype(np.int64)
    sum_deltas = np.diff(sums[start_idx:])
    bucket_deltas_2d = np.diff(bucket_counts[start_idx:], axis=0)
    bucket_deltas_2d = np.maximum(bucket_deltas_2d, 0)  # Handle counter resets

    # Cumulative-to-per-bucket delta conversion (vectorized)
    per_bucket_2d = np.zeros_like(bucket_deltas_2d)
    per_bucket_2d[:, 0] = bucket_deltas_2d[:, 0]
    per_bucket_2d[:, 1:] = bucket_deltas_2d[:, 1:] - bucket_deltas_2d[:, :-1]
    per_bucket_2d = np.maximum(per_bucket_2d, 0)

    bucket_stats: dict[str, BucketStatistics] = {}

    for i, (count_delta, sum_delta) in enumerate(
        zip(count_deltas, sum_deltas, strict=True)
    ):
        if count_delta <= 0:
            continue

        active_mask = per_bucket_2d[i] > 0
        active_indices = np.where(active_mask)[0]

        # When all observations landed in ONE bucket, we know the exact mean
        if len(active_indices) == 1:
            bucket_idx = active_indices[0]
            le = bucket_les[bucket_idx]
            bucket_mean = sum_delta / count_delta

            if le not in bucket_stats:
                bucket_stats[le] = BucketStatistics(bucket_le=le)
            bucket_stats[le].record(bucket_mean, int(count_delta))

    return bucket_stats

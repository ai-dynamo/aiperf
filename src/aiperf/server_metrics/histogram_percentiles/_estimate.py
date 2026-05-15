# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Polynomial histogram percentile estimation (accurate, variance-aware).

Main entry point: :func:`compute_estimated_percentiles`. Uses learned per-bucket
means, +Inf back-calculation, and sum-constrained observation generation.
"""

import numpy as np

from aiperf.server_metrics.histogram_percentiles._bucket_utils import (
    _MAX_OBSERVATIONS,
    _cumulative_to_per_bucket,
    _estimate_bucket_sums,
    _estimate_inf_bucket_observations,
)
from aiperf.server_metrics.histogram_percentiles._models import (
    BucketStatistics,
    EstimatedPercentiles,
)
from aiperf.server_metrics.histogram_percentiles._sum_constraint import (
    _generate_observations_with_sum_constraint,
)


def _zero_sum_percentiles() -> EstimatedPercentiles:
    """All-zero percentiles for the count>0, sum=0 case.

    When all observations were exactly 0 we skip bucket interpolation (which
    would give misleading non-zero estimates).
    """
    return EstimatedPercentiles(
        p1_estimate=0.0,
        p5_estimate=0.0,
        p10_estimate=0.0,
        p25_estimate=0.0,
        p50_estimate=0.0,
        p75_estimate=0.0,
        p90_estimate=0.0,
        p95_estimate=0.0,
        p99_estimate=0.0,
    )


def _downsample_counts_and_sum(
    per_bucket_counts: dict[str, float],
    total_sum: float,
) -> tuple[dict[str, float], float]:
    """Proportionally downsample all buckets (including +Inf) and scale sum.

    Both counts AND sum are scaled by the same ratio so averages within buckets
    are preserved.
    """
    total_obs_count = sum(per_bucket_counts.values())
    if total_obs_count <= _MAX_OBSERVATIONS:
        return per_bucket_counts, total_sum

    sample_ratio = _MAX_OBSERVATIONS / total_obs_count
    scaled = {le: count * sample_ratio for le, count in per_bucket_counts.items()}
    return scaled, total_sum * sample_ratio


def _percentiles_from_observations(observations: np.ndarray) -> EstimatedPercentiles:
    """Compute EstimatedPercentiles from a raw observation array."""
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        observations, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    return EstimatedPercentiles(
        p1_estimate=float(p1),
        p5_estimate=float(p5),
        p10_estimate=float(p10),
        p25_estimate=float(p25),
        p50_estimate=float(p50),
        p75_estimate=float(p75),
        p90_estimate=float(p90),
        p95_estimate=float(p95),
        p99_estimate=float(p99),
    )


def _combine_finite_and_inf_observations(
    total_sum: float,
    per_bucket_counts: dict[str, float],
    bucket_stats: dict[str, BucketStatistics],
    max_finite_bucket: float,
) -> np.ndarray:
    """Run phases 2-4 of the algorithm and return the combined observation array."""
    # +Inf per-bucket count. Ceiling preserves at least 1 observation when
    # the original (unscaled) count was > 0.
    raw_inf_count = per_bucket_counts.get("+Inf", 0)
    inf_count = int(np.ceil(raw_inf_count)) if raw_inf_count > 0 else 0

    estimated_sums = _estimate_bucket_sums(per_bucket_counts, bucket_stats)
    estimated_finite_sum = sum(estimated_sums.values())

    inf_observations = _estimate_inf_bucket_observations(
        total_sum, estimated_finite_sum, inf_count, max_finite_bucket
    )

    # Actual finite sum = total minus what goes to +Inf
    inf_sum_estimate = (
        float(inf_observations.sum()) if len(inf_observations) > 0 else 0.0
    )
    actual_finite_sum = total_sum - inf_sum_estimate

    finite_obs_generated = _generate_observations_with_sum_constraint(
        per_bucket_counts, actual_finite_sum, bucket_stats
    )

    if inf_observations.size > 0:
        return np.concatenate([finite_obs_generated, inf_observations])
    return finite_obs_generated


def compute_estimated_percentiles(
    bucket_deltas: dict[str, float],
    bucket_stats: dict[str, BucketStatistics],
    total_sum: float,
    total_count: int,
) -> EstimatedPercentiles | None:
    """Compute percentiles including estimated +Inf bucket observations.

    Four-phase polynomial histogram approach:

    1. Learn per-bucket means from single-bucket intervals (done upstream in
       accumulate_bucket_statistics).
    2. Estimate bucket sums using learned means (or midpoint fallback).
    3. Back-calculate +Inf bucket: inf_sum = total_sum - estimated_finite_sum,
       generate +Inf observations around inf_avg = inf_sum / inf_count.
    4. Generate finite observations with sum constraint, adjusting positions
       proportionally to match the target sum.

    Largest gains over standard Prometheus interpolation are for tail
    percentiles where observations may fall in the +Inf bucket.

    Args:
        bucket_deltas: Cumulative bucket counts (Prometheus format) where
                      bucket_deltas[le] = count of observations <= le
        bucket_stats: Learned per-bucket statistics from
                     accumulate_bucket_statistics()
        total_sum: Exact total sum from histogram (sum_delta from Prometheus)
        total_count: Total observation count (count_delta from Prometheus)

    Returns:
        EstimatedPercentiles with p1 through p99 estimates, or None if
        insufficient data (total_count <= 0, no buckets, or invalid total_sum).
    """
    if total_count <= 0 or not bucket_deltas:
        return None
    # Reject NaN, Inf, or negative sums (data corruption)
    if not np.isfinite(total_sum) or total_sum < 0:
        return None
    if total_sum == 0:
        return _zero_sum_percentiles()

    finite_buckets = [le for le in bucket_deltas if le != "+Inf"]
    if not finite_buckets:
        return None
    max_finite_bucket = max(float(le) for le in finite_buckets)

    per_bucket_counts = _cumulative_to_per_bucket(bucket_deltas)
    per_bucket_counts, total_sum = _downsample_counts_and_sum(
        per_bucket_counts, total_sum
    )

    all_observations = _combine_finite_and_inf_observations(
        total_sum, per_bucket_counts, bucket_stats, max_finite_bucket
    )
    if len(all_observations) == 0:
        return None

    return _percentiles_from_observations(all_observations)

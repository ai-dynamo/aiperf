# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sum-constrained observation generation for polynomial histogram percentile estimation.

Generates observations per bucket using learned statistics, then adjusts positions
proportionally so the aggregate sum matches the histogram's exact sum_delta.
"""

import numpy as np

from aiperf.server_metrics.histogram_percentiles._bucket_utils import (
    _MAX_OBSERVATIONS,
    _get_bucket_bounds,
)
from aiperf.server_metrics.histogram_percentiles._models import BucketStatistics
from aiperf.server_metrics.histogram_percentiles._observation_generators import (
    _generate_blended_observations,
    _generate_f3_observations,
    _generate_variance_aware_observations,
)


def _downsample_if_needed(
    per_bucket_counts: dict[str, float],
    target_sum: float,
    finite_buckets: list[str],
) -> tuple[dict[str, float], float, float]:
    """Proportionally downsample counts and target_sum when over the cap.

    Both counts AND target_sum are scaled by the same ratio to preserve
    mean values within buckets. Returns (counts, target_sum, total_count).
    """
    total_count = sum(per_bucket_counts.get(le, 0) for le in finite_buckets)
    if total_count <= _MAX_OBSERVATIONS:
        return per_bucket_counts, target_sum, total_count

    sample_ratio = _MAX_OBSERVATIONS / total_count
    scaled = {le: count * sample_ratio for le, count in per_bucket_counts.items()}
    new_target_sum = target_sum * sample_ratio
    new_total = sum(scaled.get(le, 0) for le in finite_buckets)
    return scaled, new_target_sum, new_total


def _find_dominant_bucket(
    per_bucket_counts: dict[str, float],
    finite_buckets: list[str],
    total_count: float,
) -> str | None:
    """Return the bucket holding 95%+ of observations, if any.

    Dominant-bucket detection allows the generator to center observations on
    the overall average (more accurate than midpoint for narrow distributions
    where all data clusters in a single bucket).
    """
    if total_count <= 0:
        return None
    max_count = max(per_bucket_counts.get(le, 0) for le in finite_buckets)
    if max_count / total_count < 0.95:
        return None
    for le in finite_buckets:
        if per_bucket_counts.get(le, 0) == max_count:
            return le
    return None


def _extract_learned_stats(
    stats: BucketStatistics | None,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float | None, float | None]:
    """Pull (learned_mean, learned_variance) from stats, validating mean bounds."""
    if stats is None:
        return None, None
    learned_mean: float | None = None
    if stats.estimated_mean is not None:
        mean_val = stats.estimated_mean
        if lower_bound < mean_val < upper_bound:
            learned_mean = mean_val
    return learned_mean, stats.estimated_variance


def _generate_for_bucket(
    bucket_count: int,
    *,
    lower_bound: float,
    upper_bound: float,
    midpoint: float,
    bucket_width: float,
    learned_mean: float | None,
    learned_variance: float | None,
    is_dominant: bool,
    avg: float,
) -> np.ndarray:
    """Select a generation strategy for one bucket and produce its observations.

    Order of preference:
    1. F3 two-point mass if variance is extremely tight (< 1% of bucket width).
    2. Blended if variance is tight (< 20%) AND mean is near center (< 30%).
    3. Variance-aware for moderate variance.
    4. Shifted uniform centered on learned mean, avg (dominant bucket), or midpoint.
    """
    if (
        learned_mean is not None
        and learned_variance is not None
        and learned_variance > 0
    ):
        std = float(np.sqrt(learned_variance))
        spread_coverage = (4 * std) / bucket_width  # 2 std on each side
        mean_offset = abs(learned_mean - midpoint) / bucket_width

        if spread_coverage < 0.01:
            return _generate_f3_observations(
                bucket_count,
                lower=lower_bound,
                upper=upper_bound,
                mean=learned_mean,
                variance=learned_variance,
            )
        if spread_coverage < 0.2 and mean_offset < 0.3:
            return _generate_blended_observations(
                bucket_count,
                lower=lower_bound,
                upper=upper_bound,
                mean=learned_mean,
                std=std,
                blend_factor=0.5,
            )
        return _generate_variance_aware_observations(
            bucket_count,
            lower=lower_bound,
            upper=upper_bound,
            mean=learned_mean,
            std=std,
        )

    # Shifted uniform fallback
    if is_dominant and lower_bound < avg < upper_bound:
        center = avg
    elif learned_mean is not None:
        center = learned_mean
    else:
        center = midpoint

    shift = center - midpoint
    fractions = (np.arange(bucket_count) + 0.5) / bucket_count
    base_values = lower_bound + bucket_width * fractions
    return np.clip(base_values + shift, lower_bound, upper_bound)


def _place_initial_observations(
    sorted_buckets: list[str],
    bucket_int_counts: dict[str, int],
    total_observations: int,
    bucket_stats: dict[str, BucketStatistics],
    *,
    dominant_bucket: str | None,
    avg: float,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Pass 1: write per-bucket observations into a single array.

    Returns the observations array and a per-bucket (start, count, lower, upper)
    range list used by the sum-adjustment pass.
    """
    observations = np.empty(total_observations, dtype=np.float64)
    bucket_ranges: list[tuple[int, int, float, float]] = []
    write_idx = 0

    for bucket_le in sorted_buckets:
        bucket_count = bucket_int_counts[bucket_le]
        if bucket_count <= 0:
            continue

        # Clamp to remaining space to prevent array overflow
        remaining_space = total_observations - write_idx
        if remaining_space <= 0:
            break
        bucket_count = min(bucket_count, remaining_space)

        lower_bound, upper_bound = _get_bucket_bounds(bucket_le, sorted_buckets)
        bucket_width = upper_bound - lower_bound
        midpoint = (lower_bound + upper_bound) / 2

        learned_mean, learned_variance = _extract_learned_stats(
            bucket_stats.get(bucket_le), lower_bound, upper_bound
        )

        bucket_obs = _generate_for_bucket(
            bucket_count,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            midpoint=midpoint,
            bucket_width=bucket_width,
            learned_mean=learned_mean,
            learned_variance=learned_variance,
            is_dominant=(bucket_le == dominant_bucket),
            avg=avg,
        )

        observations[write_idx : write_idx + bucket_count] = bucket_obs
        bucket_ranges.append((write_idx, bucket_count, lower_bound, upper_bound))
        write_idx += bucket_count

    return observations, bucket_ranges


def _adjust_for_sum_constraint(
    observations: np.ndarray,
    bucket_ranges: list[tuple[int, int, float, float]],
    target_sum: float,
) -> None:
    """Pass 2: shift observations in-place so the aggregate sum matches target_sum.

    Per-bucket shifts are proportional to bucket sum contribution and capped at
    +/-40% of bucket width so no observation crosses its bucket boundary.
    """
    generated_sum = observations.sum()
    if generated_sum <= 0 or target_sum <= 0:
        return

    sum_discrepancy = target_sum - generated_sum
    if abs(sum_discrepancy) / target_sum < 0.001:
        return  # Close enough

    for slice_start, bucket_count, lower_bound, upper_bound in bucket_ranges:
        if bucket_count == 0:
            continue

        bucket_width = upper_bound - lower_bound
        bucket_slice = observations[slice_start : slice_start + bucket_count]
        bucket_sum = bucket_slice.sum()
        bucket_weight = (
            bucket_sum / generated_sum
            if generated_sum > 0
            else 1.0 / len(bucket_ranges)
        )

        bucket_adjustment = sum_discrepancy * bucket_weight
        per_obs_shift = bucket_adjustment / bucket_count

        max_shift = bucket_width * 0.4
        shift = np.clip(per_obs_shift, -max_shift, max_shift)

        observations[slice_start : slice_start + bucket_count] = np.clip(
            bucket_slice + shift, lower_bound, upper_bound
        )


def _generate_observations_with_sum_constraint(
    per_bucket_counts: dict[str, float],
    target_sum: float,
    bucket_stats: dict[str, BucketStatistics] | None = None,
) -> np.ndarray:
    """Generate observations constrained to match the exact histogram sum.

    Core of the polynomial histogram percentile estimation algorithm. Standard
    Prometheus bucket interpolation assumes uniform distribution within each
    bucket, which over/underestimates percentiles when observations cluster
    near bucket boundaries.

    Algorithm:
        1. Detect single-bucket dominance (>=95% in one bucket): use overall
           average as the center for narrow distributions.
        2. For each bucket, pick a placement strategy (F3, blended,
           variance-aware, or shifted uniform) based on learned statistics.
        3. Adjust positions proportionally across all buckets to match the
           exact target sum (capped at +/-40% of bucket width per observation).

    Note:
        When total_count exceeds _MAX_OBSERVATIONS, bucket counts are proportionally
        downsampled to prevent memory issues while maintaining distribution shape.

    Args:
        per_bucket_counts: Per-bucket counts
        target_sum: The exact sum of observations (from histogram sum_delta)
        bucket_stats: Optional learned per-bucket statistics from
                      accumulate_bucket_statistics()

    Returns:
        Array of generated observation values for finite buckets (excludes +Inf)
    """
    finite_buckets = [le for le in per_bucket_counts if le != "+Inf"]
    sorted_buckets = sorted(finite_buckets, key=lambda x: float(x))

    bucket_stats = bucket_stats or {}

    per_bucket_counts, target_sum, total_count = _downsample_if_needed(
        per_bucket_counts, target_sum, finite_buckets
    )

    avg = target_sum / total_count if total_count > 0 else 0.0
    dominant_bucket = _find_dominant_bucket(
        per_bucket_counts, finite_buckets, total_count
    )

    # Integer bucket counts ensure consistency between array sizing and
    # observation generation (avoids shape mismatches).
    bucket_int_counts = {le: int(per_bucket_counts.get(le, 0)) for le in sorted_buckets}
    total_observations = sum(bucket_int_counts.values())
    if total_observations <= 0:
        return np.array([], dtype=np.float64)

    observations, bucket_ranges = _place_initial_observations(
        sorted_buckets,
        bucket_int_counts,
        total_observations,
        bucket_stats,
        dominant_bucket=dominant_bucket,
        avg=avg,
    )

    _adjust_for_sum_constraint(observations, bucket_ranges, target_sum)

    return observations

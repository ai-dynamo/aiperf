# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard Prometheus linear-interpolation percentile estimation.

Faster baseline used when learned bucket statistics are unavailable or when
~15-40% error is acceptable. See compute_estimated_percentiles for the more
accurate polynomial histogram approach.
"""

from aiperf.server_metrics.histogram_percentiles._models import EstimatedPercentiles


def _sort_bucket_keys(bucket_cumulative: dict[str, float]) -> list[str]:
    """Sort bucket keys numerically, with +Inf last."""

    def sort_key(le: str) -> float:
        if le == "+Inf":
            return float("inf")
        return float(le)

    return sorted(bucket_cumulative.keys(), key=sort_key)


def _prometheus_quantile(
    quantile: float,
    bucket_cumulative: dict[str, float],
    sorted_keys: list[str],
    total_count: float,
) -> float | None:
    """Compute a single quantile using Prometheus's histogram_quantile algorithm.

    The algorithm:
    1. Find the bucket where the quantile rank falls
    2. Linear interpolate within that bucket assuming uniform distribution

    Returns:
        The estimated quantile value, or None if cannot be computed.
    """
    if total_count == 0:
        return None

    target_rank = quantile * total_count

    prev_bound = 0.0
    prev_count = 0.0

    for key in sorted_keys:
        current_count = bucket_cumulative[key]

        if key == "+Inf":
            # Can't interpolate within +Inf bucket; return last finite upper bound
            return prev_bound

        current_bound = float(key)

        if current_count >= target_rank:
            bucket_count = current_count - prev_count
            if bucket_count == 0:
                return prev_bound

            bucket_fraction = (target_rank - prev_count) / bucket_count
            return prev_bound + (current_bound - prev_bound) * bucket_fraction

        prev_bound = current_bound
        prev_count = current_count

    return prev_bound


_PERCENTILES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def compute_prometheus_percentiles(
    bucket_cumulative: dict[str, float],
    total_count: float | None = None,
) -> EstimatedPercentiles:
    """Compute percentiles using standard Prometheus histogram_quantile algorithm.

    Assumes uniform distribution within each bucket and uses linear interpolation.
    Faster but less accurate than the polynomial histogram algorithm, especially for
    skewed distributions and tail percentiles when data falls in the +Inf bucket.

    Reference: https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile

    Args:
        bucket_cumulative: Cumulative bucket counts in Prometheus format where
                          bucket_cumulative[le] = count of observations <= le.
                          Must include "+Inf" bucket for proper handling.
        total_count: Optional total observation count. If not provided, uses
                    the +Inf bucket count or the last bucket count.

    Returns:
        EstimatedPercentiles with P1, P5, P10, P25, P50, P75, P90, P95, P99 estimates.
        Returns empty EstimatedPercentiles if input is invalid.
    """
    if not bucket_cumulative:
        return EstimatedPercentiles()

    sorted_keys = _sort_bucket_keys(bucket_cumulative)
    if not sorted_keys:
        return EstimatedPercentiles()

    if total_count is None:
        total_count = bucket_cumulative.get("+Inf", bucket_cumulative[sorted_keys[-1]])

    if total_count == 0:
        return EstimatedPercentiles()

    values = [
        _prometheus_quantile(q, bucket_cumulative, sorted_keys, total_count)
        for q in _PERCENTILES
    ]
    return EstimatedPercentiles(
        p1_estimate=values[0],
        p5_estimate=values[1],
        p10_estimate=values[2],
        p25_estimate=values[3],
        p50_estimate=values[4],
        p75_estimate=values[5],
        p90_estimate=values[6],
        p95_estimate=values[7],
        p99_estimate=values[8],
    )

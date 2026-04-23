# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models for histogram percentile computation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class EstimatedPercentiles:
    """Estimated percentiles from histogram data using polynomial histogram algorithm.

    Contains percentile estimates (P1, P5, P10, P25, P50, P75, P90, P95, P99)
    computed from Prometheus histogram bucket data using the polynomial histogram
    approach (arXiv 2504.00001).

    Uses learned per-bucket means and +Inf bucket back-calculation for
    significantly more accurate estimates than standard Prometheus linear
    interpolation (typically 2.5x improvement, up to 47x for tail percentiles).

    All percentile values are in the same units as the histogram sum
    (e.g., seconds for latency histograms, bytes for size histograms).
    """

    p1_estimate: float | None = None
    """Estimated 1st percentile value."""

    p5_estimate: float | None = None
    """Estimated 5th percentile value."""

    p10_estimate: float | None = None
    """Estimated 10th percentile value."""

    p25_estimate: float | None = None
    """Estimated 25th percentile value."""

    p50_estimate: float | None = None
    """Estimated 50th percentile (median) value."""

    p75_estimate: float | None = None
    """Estimated 75th percentile value."""

    p90_estimate: float | None = None
    """Estimated 90th percentile value."""

    p95_estimate: float | None = None
    """Estimated 95th percentile value."""

    p99_estimate: float | None = None
    """Estimated 99th percentile value."""


@dataclass(slots=True)
class BucketStatistics:
    """Statistics for a single histogram bucket learned from single-bucket scrape intervals.

    When all observations in a scrape interval land in ONE bucket, we can compute the
    exact mean for that bucket in that interval: mean = sum_delta / count_delta. Over many
    such intervals, we learn the typical position of observations within each bucket.

    This is a core component of the "polynomial histogram" approach (arXiv 2504.00001)
    which improves percentile estimation accuracy by 2.5x compared to simple linear
    interpolation (which assumes uniform distribution within each bucket).

    Additionally tracks individual observed means to compute variance, enabling optimal
    observation generation strategies (Section II.1.1 "Second Moment"):
    - F3 two-point mass when 4 sigma spread is < 1% of bucket width
    - Blended distribution for tight variance (< 20% spread) near bucket center (< 30% offset)
    - Variance-aware distribution for wider spreads or off-center means
    """

    bucket_le: str
    """Bucket upper bound (le value)."""

    observation_count: int = 0
    """Total observations used to learn this bucket's mean."""

    weighted_mean_sum: float = 0.0
    """Sum of (mean * count) for weighted average calculation."""

    sample_count: int = 0
    """Number of single-bucket intervals observed."""

    observed_means: list[float] = field(default_factory=list)
    """Individual mean values from each single-bucket interval."""

    MIN_VARIANCE_OBSERVATIONS: int = 3
    """Minimum observations required to trust variance estimate."""

    @property
    def estimated_mean(self) -> float | None:
        """Compute the weighted average position within this bucket.

        Aggregates all single-bucket intervals observed for this bucket,
        weighting each interval's mean by its observation count. This provides
        a more accurate mean estimate than simple midpoint assumption.

        Returns:
            Weighted average mean position, or None if no single-bucket intervals
            have been observed for this bucket.
        """
        if self.observation_count == 0:
            return None
        return self.weighted_mean_sum / self.observation_count

    @property
    def estimated_variance(self) -> float | None:
        """Compute variance from observed means across intervals.

        Uses sample variance (ddof=1) of the observed means across multiple
        single-bucket intervals. Requires at least MIN_VARIANCE_OBSERVATIONS (3)
        intervals to produce a reliable variance estimate.

        Returns:
            Sample variance of observed means, or None if fewer than
            MIN_VARIANCE_OBSERVATIONS intervals observed.
        """
        if len(self.observed_means) < self.MIN_VARIANCE_OBSERVATIONS:
            return None
        return float(np.var(self.observed_means, ddof=1))

    def record(self, mean: float, count: int) -> None:
        """Record statistics from a single-bucket scrape interval.

        Called when all observations in a scrape interval land in this bucket,
        allowing us to compute an exact mean position within the bucket.

        Args:
            mean: Exact mean value for observations in this interval (sum_delta/count_delta)
            count: Number of observations in this interval (used for weighted averaging)
        """
        self.observation_count += count
        self.weighted_mean_sum += mean * count
        self.sample_count += 1
        self.observed_means.append(mean)

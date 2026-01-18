# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bucket statistics (polynomial histogram approach)."""

import numpy as np
import pytest

from aiperf.server_metrics.histogram_percentiles import (
    BucketStatistics,
    accumulate_bucket_statistics,
)
from aiperf.server_metrics.histogram_percentiles import (
    _estimate_bucket_sums as estimate_bucket_sums,
)
from aiperf.server_metrics.histogram_percentiles import (
    _estimate_inf_bucket_observations as estimate_inf_bucket_observations,
)
from aiperf.server_metrics.histogram_percentiles import (
    _get_bucket_bounds as get_bucket_bounds,
)

from .helpers import convert_bucket_snapshots

# =============================================================================
# Bucket Statistics Tests (Polynomial Histogram Approach)
# =============================================================================


class TestBucketStatistics:
    """Test BucketStatistics model for polynomial histogram approach."""

    def test_bucket_statistics_record_single(self):
        """Test recording a single observation."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.25, count=1)

        assert stats.observation_count == 1
        assert stats.sample_count == 1
        assert stats.estimated_mean == pytest.approx(0.25)

    def test_bucket_statistics_record_multiple(self):
        """Test recording multiple observations."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.2, count=10)
        stats.record(mean=0.3, count=10)

        assert stats.observation_count == 20
        assert stats.sample_count == 2
        # Weighted mean: (0.2*10 + 0.3*10) / 20 = 5/20 = 0.25
        assert stats.estimated_mean == pytest.approx(0.25)

    def test_bucket_statistics_weighted_average(self):
        """Test that estimated_mean is weighted by count."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.record(mean=0.1, count=1)  # weight 1
        stats.record(mean=0.5, count=9)  # weight 9

        # Weighted: (0.1*1 + 0.5*9) / 10 = 4.6/10 = 0.46
        assert stats.estimated_mean == pytest.approx(0.46)

    def test_bucket_statistics_empty_returns_none(self):
        """Test empty bucket returns None for estimated_mean."""
        stats = BucketStatistics(bucket_le="0.5")
        assert stats.estimated_mean is None


class TestAccumulateBucketStatistics:
    """Test accumulate_bucket_statistics function."""

    def test_single_bucket_intervals_learn_means(self):
        """Test that single-bucket intervals are used to learn means."""
        # All observations in same bucket each interval
        counts = np.array([0.0, 5.0, 10.0])  # +5, +5
        sums = np.array([0.0, 1.5, 3.5])  # mean=0.3 first, mean=0.4 second
        bucket_snapshots = [
            {"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0},
            {"0.5": 5.0, "1.0": 5.0, "+Inf": 5.0},  # all in 0.5 bucket
            {"0.5": 10.0, "1.0": 10.0, "+Inf": 10.0},  # all in 0.5 bucket
        ]
        bucket_les, bucket_counts = convert_bucket_snapshots(bucket_snapshots)

        stats = accumulate_bucket_statistics(sums, counts, bucket_les, bucket_counts)

        assert "0.5" in stats
        assert stats["0.5"].observation_count == 10
        # Weighted: (0.3*5 + 0.4*5) / 10 = 0.35
        assert stats["0.5"].estimated_mean == pytest.approx(0.35)

    def test_multi_bucket_intervals_not_learned(self):
        """Test that multi-bucket intervals are NOT used to learn means."""
        counts = np.array([0.0, 10.0])
        sums = np.array([0.0, 3.0])
        # Observations split across buckets
        bucket_snapshots = [
            {"0.1": 0.0, "0.5": 0.0, "+Inf": 0.0},
            {"0.1": 3.0, "0.5": 10.0, "+Inf": 10.0},  # 3 in 0.1, 7 in 0.5
        ]
        bucket_les, bucket_counts = convert_bucket_snapshots(bucket_snapshots)

        stats = accumulate_bucket_statistics(sums, counts, bucket_les, bucket_counts)

        # No single-bucket intervals, so no stats learned
        assert len(stats) == 0

    def test_inf_bucket_learns_mean(self):
        """Test that +Inf bucket can learn means when all observations land there."""
        counts = np.array([0.0, 3.0])
        sums = np.array([0.0, 45.0])  # mean = 15.0
        bucket_snapshots = [
            {"1.0": 0.0, "+Inf": 0.0},
            {"1.0": 0.0, "+Inf": 3.0},  # all in +Inf
        ]
        bucket_les, bucket_counts = convert_bucket_snapshots(bucket_snapshots)

        stats = accumulate_bucket_statistics(sums, counts, bucket_les, bucket_counts)

        assert "+Inf" in stats
        assert stats["+Inf"].estimated_mean == pytest.approx(15.0)


class TestGetBucketBounds:
    """Test get_bucket_bounds function."""

    def test_first_bucket_has_zero_lower_bound(self):
        """First bucket has lower bound of 0."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("0.1", sorted_buckets)

        assert lower == 0.0
        assert upper == 0.1

    def test_middle_bucket_has_previous_as_lower(self):
        """Middle bucket has previous bucket as lower bound."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("0.5", sorted_buckets)

        assert lower == 0.1
        assert upper == 0.5

    def test_last_finite_bucket(self):
        """Last finite bucket has previous bucket as lower bound."""
        sorted_buckets = ["0.1", "0.5", "1.0"]
        lower, upper = get_bucket_bounds("1.0", sorted_buckets)

        assert lower == 0.5
        assert upper == 1.0


class TestEstimateBucketSums:
    """Test estimate_bucket_sums function."""

    def test_uses_learned_means_when_available(self):
        """Test that learned means are used when available."""
        # Per-bucket counts (not cumulative): 10 in [0,0.1], 10 in (0.1,0.5], 10 in +Inf
        per_bucket_counts = {"0.1": 10.0, "0.5": 10.0, "+Inf": 10.0}
        bucket_stats = {
            "0.1": BucketStatistics(
                bucket_le="0.1",
                observation_count=100,
                weighted_mean_sum=5.0,  # mean = 0.05
                sample_count=10,
            ),
        }

        sums = estimate_bucket_sums(per_bucket_counts, bucket_stats)

        # 0.1 bucket uses learned mean: 10 * 0.05 = 0.5
        assert sums["0.1"] == pytest.approx(0.5)
        # 0.5 bucket uses midpoint: 10 * (0.1 + 0.5) / 2 = 10 * 0.3 = 3.0
        assert sums["0.5"] == pytest.approx(3.0)
        # +Inf bucket is not included
        assert "+Inf" not in sums

    def test_falls_back_to_midpoint(self):
        """Test fallback to midpoint when no learned means."""
        # Per-bucket counts (not cumulative): 10 in [0,0.2], 5 in (0.2,1.0]
        per_bucket_counts = {"0.2": 10.0, "1.0": 5.0, "+Inf": 0.0}
        bucket_stats = {}  # No learned stats

        sums = estimate_bucket_sums(per_bucket_counts, bucket_stats)

        # 0.2 bucket: 10 * (0 + 0.2) / 2 = 10 * 0.1 = 1.0
        assert sums["0.2"] == pytest.approx(1.0)
        # 1.0 bucket: 5 * (0.2 + 1.0) / 2 = 5 * 0.6 = 3.0
        assert sums["1.0"] == pytest.approx(3.0)


class TestEstimateInfBucketObservations:
    """Test estimate_inf_bucket_observations function."""

    def test_back_calculates_inf_observations(self):
        """Test back-calculation of +Inf bucket observations."""
        total_sum = 100.0
        estimated_finite_sum = 70.0
        inf_count = 3
        max_finite_bucket = 10.0

        observations = estimate_inf_bucket_observations(
            total_sum, estimated_finite_sum, inf_count, max_finite_bucket
        )

        # Back-calculated inf_sum = 100 - 70 = 30
        # inf_avg = 30 / 3 = 10.0
        # But 10.0 is not > max_finite_bucket, so fallback to 1.5x
        # inf_avg = 10.0 * 1.5 = 15.0
        # Observations spread from 10.0 to 20.0 (mean = 15.0)
        assert len(observations) == 3
        assert all(obs >= max_finite_bucket for obs in observations)
        assert pytest.approx(sum(observations) / len(observations), rel=0.01) == 15.0

    def test_inf_observations_above_max_finite(self):
        """Test all +Inf observations are above max finite bucket."""
        total_sum = 200.0
        estimated_finite_sum = 50.0
        inf_count = 10
        max_finite_bucket = 5.0

        observations = estimate_inf_bucket_observations(
            total_sum, estimated_finite_sum, inf_count, max_finite_bucket
        )

        # inf_avg = 150 / 10 = 15.0 (> 5.0, valid)
        assert len(observations) == 10
        assert all(obs >= max_finite_bucket for obs in observations)

    def test_zero_inf_count_returns_empty(self):
        """Test zero inf_count returns empty array."""
        observations = estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=100.0,
            inf_count=0,
            max_finite_bucket=10.0,
        )

        assert len(observations) == 0

    def test_negative_inf_sum_falls_back(self):
        """Test negative back-calculated sum falls back to 1.5x max."""
        # When estimated_finite_sum > total_sum (estimation error)
        observations = estimate_inf_bucket_observations(
            total_sum=50.0,
            estimated_finite_sum=100.0,  # Error: greater than total
            inf_count=5,
            max_finite_bucket=10.0,
        )

        # Should fall back to 1.5x max_finite_bucket = 15.0
        assert len(observations) == 5
        assert pytest.approx(sum(observations) / len(observations), rel=0.01) == 15.0

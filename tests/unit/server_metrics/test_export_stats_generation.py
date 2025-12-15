# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for observation generation with sum constraint (core polynomial histogram algorithm)."""

import numpy as np
import pytest

from aiperf.server_metrics.histogram_percentiles import (
    BucketStatistics,
    _generate_observations_with_sum_constraint,
)

# =============================================================================
# Generate Observations With Sum Constraint Tests (Core Algorithm)
# =============================================================================


class TestGenerateObservationsWithSumConstraint:
    """Test generate_observations_with_sum_constraint function.

    This is the core of the polynomial histogram percentile estimation algorithm.
    It generates observation values from bucket counts, constrained to match
    the exact histogram sum.
    """

    def test_basic_uniform_distribution_no_learned_means(self):
        """Test basic uniform distribution without learned means."""
        # Per-bucket: 10 in [0,0.1], 10 in (0.1,0.5] = 20 total
        per_bucket_counts = {"0.1": 10.0, "0.5": 10.0, "+Inf": 0.0}
        # With uniform distribution:
        # [0,0.1]: mean=0.05, 10 obs -> sum=0.5
        # (0.1,0.5]: mean=0.3, 10 obs -> sum=3.0
        # Total expected = 3.5
        target_sum = 3.5

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum, bucket_stats=None
        )

        assert len(observations) == 20
        assert np.isclose(observations.sum(), target_sum, rtol=0.01)
        # First 10 should be in [0,0.1]
        assert all(0 <= obs <= 0.1 for obs in observations[:10])
        # Next 10 should be in (0.1,0.5]
        assert all(0.1 <= obs <= 0.5 for obs in observations[10:])

    def test_observations_within_bucket_bounds(self):
        """Test all observations stay within their bucket bounds."""
        # Per-bucket counts: 50 in [0,0.05], 50 in (0.05,0.1], 50 in (0.1,0.5]
        per_bucket_counts = {"0.05": 50.0, "0.1": 50.0, "0.5": 50.0, "+Inf": 0.0}
        target_sum = 15.0  # Arbitrary

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum, bucket_stats=None
        )

        # Sort observations to check bounds
        sorted_obs = np.sort(observations)

        # First 50 in [0,0.05]
        assert all(0 <= obs <= 0.05 for obs in sorted_obs[:50])
        # Next 50 in (0.05,0.1]
        assert all(0.05 <= obs <= 0.1 for obs in sorted_obs[50:100])
        # Last 50 in (0.1,0.5]
        assert all(0.1 <= obs <= 0.5 for obs in sorted_obs[100:])

    def test_shifted_distribution_with_learned_mean(self):
        """Test distribution shifts toward learned mean."""
        # Single bucket for simplicity - per-bucket counts
        per_bucket_counts = {"0.5": 100.0, "+Inf": 0.0}

        # Without learned mean, midpoint is 0.25, sum would be ~25
        obs_no_mean = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=25.0, bucket_stats=None
        )

        # With learned mean of 0.1 (observations cluster near lower bound)
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=50,
                weighted_mean_sum=5.0,  # mean = 0.1
                sample_count=10,
            )
        }
        obs_with_mean = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=10.0, bucket_stats=bucket_stats
        )

        # With learned mean of 0.1, observations should be shifted lower
        assert np.mean(obs_with_mean) < np.mean(obs_no_mean)
        # All should still be in [0, 0.5]
        assert all(0 <= obs <= 0.5 for obs in obs_with_mean)

    def test_learned_mean_ignored_if_outside_bounds(self):
        """Test learned mean is ignored if outside bucket bounds."""
        per_bucket_counts = {"0.5": 100.0, "+Inf": 0.0}

        # Invalid learned mean (outside bucket bounds)
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=50,
                weighted_mean_sum=35.0,  # mean = 0.7 > 0.5 (invalid!)
                sample_count=10,
            )
        }

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=25.0, bucket_stats=bucket_stats
        )

        # Should fall back to midpoint (0.25)
        # Mean should be around 0.25, not 0.7
        assert np.mean(observations) == pytest.approx(0.25, rel=0.1)

    def test_sum_constraint_adjusts_positions(self):
        """Test sum constraint adjusts observation positions."""
        per_bucket_counts = {"1.0": 100.0, "+Inf": 0.0}

        # Midpoint-based sum would be ~50 (100 * 0.5)
        # Test that different target sums result in different distributions
        # Note: adjustment is limited to 40% of bucket width to stay within bounds
        high_sum = 65.0  # Above midpoint
        obs_high = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=high_sum, bucket_stats=None
        )

        # Target lower sum means observations shift toward lower bound
        low_sum = 35.0  # Below midpoint
        obs_low = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=low_sum, bucket_stats=None
        )

        # High sum observations should be shifted higher on average
        assert np.mean(obs_high) > np.mean(obs_low)
        # High sum target should result in higher actual sum than low sum target
        assert obs_high.sum() > obs_low.sum()

    def test_proportional_adjustment_across_buckets(self):
        """Test adjustment is distributed proportionally across buckets."""
        # Per-bucket: 10 in [0,0.1], 100 in (0.1,1.0]
        per_bucket_counts = {"0.1": 10.0, "1.0": 100.0, "+Inf": 0.0}

        # Midpoint sums: 10*0.05=0.5, 100*0.55=55, total=55.5
        # Target 60 means +4.5 needs to be distributed
        target_sum = 60.0

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        # The larger bucket (100 obs) should absorb more of the adjustment
        assert len(observations) == 110
        # Sum should be close to target
        assert np.isclose(observations.sum(), target_sum, rtol=0.05)

    def test_empty_buckets_returns_empty(self):
        """Test empty bucket deltas returns empty array."""
        result = _generate_observations_with_sum_constraint({}, target_sum=0.0)

        assert len(result) == 0

    def test_only_inf_bucket_returns_empty(self):
        """Test only +Inf bucket returns empty (no finite observations)."""
        bucket_deltas = {"+Inf": 100.0}

        result = _generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=100.0
        )

        # Only +Inf bucket, no finite observations to generate
        assert len(result) == 0

    def test_zero_target_sum_returns_observations(self):
        """Test zero target sum still generates observations."""
        per_bucket_counts = {"0.5": 10.0, "+Inf": 0.0}

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=0.0
        )

        # Should still generate observations (no adjustment possible)
        assert len(result) == 10

    def test_single_observation_per_bucket(self):
        """Test single observation per bucket."""
        # Per-bucket: 1 in [0,0.2], 1 in (0.2,0.4], 1 in (0.4,0.6]
        per_bucket_counts = {"0.2": 1.0, "0.4": 1.0, "0.6": 1.0, "+Inf": 0.0}

        target_sum = 0.6  # Approx midpoints: 0.1 + 0.3 + 0.5 = 0.9

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 3
        # Each in respective bucket
        assert 0 <= observations[0] <= 0.2
        assert 0.2 <= observations[1] <= 0.4
        assert 0.4 <= observations[2] <= 0.6

    def test_multiple_learned_means_used(self):
        """Test multiple learned means are used for respective buckets."""
        # Per-bucket: 50 in [0,0.1], 50 in (0.1,0.5]
        per_bucket_counts = {"0.1": 50.0, "0.5": 50.0, "+Inf": 0.0}

        # Learn means for both buckets
        bucket_stats = {
            "0.1": BucketStatistics(
                bucket_le="0.1",
                observation_count=100,
                weighted_mean_sum=3.0,  # mean = 0.03 (near lower bound)
                sample_count=20,
            ),
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=100,
                weighted_mean_sum=40.0,  # mean = 0.4 (near upper bound)
                sample_count=20,
            ),
        }

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts,
            target_sum=22.0,  # 50*0.03 + 50*0.4 = 1.5 + 20 = 21.5
            bucket_stats=bucket_stats,
        )

        # First bucket should cluster near 0.03
        first_bucket = observations[:50]
        assert np.mean(first_bucket) < 0.05  # Near 0.03

        # Second bucket should cluster near 0.4
        second_bucket = observations[50:]
        assert np.mean(second_bucket) > 0.3  # Near 0.4

    def test_single_bucket_dominance_uses_avg_as_center(self):
        """Test that when 95%+ observations are in one bucket, avg is used as center.

        This handles narrow distributions (e.g., decode-only worker metrics) where
        all data clusters in a single bucket but the actual mean is far from midpoint.
        """
        # Scenario: 100% of observations in [0, 0.3] bucket - per-bucket counts
        # Midpoint = 0.15, but actual avg = 0.01 (like decode worker E2E)
        per_bucket_counts = {"0.3": 5000.0, "+Inf": 0.0}
        actual_avg = 0.01  # 10ms average in a [0, 300ms] bucket
        target_sum = 5000 * actual_avg  # 50.0

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 5000

        # Without the fix, midpoint=0.15 would give p50 around 0.15
        # With the fix, using avg=0.01 as center should give p50 around 0.01
        p50 = np.percentile(observations, 50)

        # p50 should be close to avg (within 2x), not close to midpoint (0.15)
        assert p50 < 0.05, f"p50={p50} should be < 0.05 (near avg=0.01), not ~0.15"
        # More specifically, should be reasonably close to avg
        assert abs(p50 - actual_avg) < 0.02, (
            f"p50={p50} should be close to avg={actual_avg}"
        )


class TestGenerateObservationsAccuracy:
    """Test accuracy of polynomial histogram approach vs standard interpolation."""

    def test_accuracy_improvement_clustered_distribution(self):
        """Test accuracy improvement when observations cluster near bucket edge.

        Standard Prometheus interpolation assumes uniform distribution (midpoint).
        When observations actually cluster near one edge, our sum-constrained
        approach should produce more accurate percentiles.
        """
        # Scenario: 100 observations all in [0, 0.5] bucket - per-bucket counts
        # True distribution: clustered near 0.4 (mean=0.4, not midpoint 0.25)
        per_bucket_counts = {"0.5": 100.0, "+Inf": 0.0}
        true_mean = 0.4
        target_sum = 100 * true_mean  # 40.0

        # Standard midpoint approach would estimate mean=0.25, sum=25
        midpoint_estimate = 0.25

        # Our approach
        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        actual_mean = np.mean(observations)

        # Our estimate should be closer to true mean than midpoint
        our_error = abs(actual_mean - true_mean)
        midpoint_error = abs(midpoint_estimate - true_mean)

        assert our_error < midpoint_error

    def test_accuracy_with_learned_means(self):
        """Test accuracy improvement when learned means are available."""
        # Scenario: We learned that observations in bucket cluster at specific position
        per_bucket_counts = {"0.5": 100.0, "+Inf": 0.0}
        true_mean = 0.35

        # Learned mean from previous single-bucket intervals
        bucket_stats = {
            "0.5": BucketStatistics(
                bucket_le="0.5",
                observation_count=500,
                weighted_mean_sum=175.0,  # mean = 0.35
                sample_count=50,
            )
        }

        target_sum = 100 * true_mean  # 35.0

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=bucket_stats
        )

        actual_mean = np.mean(observations)

        # With learned means, should be very close to true mean
        assert np.isclose(actual_mean, true_mean, rtol=0.05)

    def test_p99_accuracy_tail_distribution(self):
        """Test p99 accuracy for heavy-tailed distribution.

        This is a key use case: when high percentiles matter, the standard
        midpoint assumption can significantly underestimate latencies.
        """
        # Realistic latency scenario - per-bucket counts:
        # 900 requests < 100ms, 80 requests 100-500ms, 20 requests 500ms-1s
        per_bucket_counts = {"0.1": 900.0, "0.5": 80.0, "1.0": 20.0, "+Inf": 0.0}

        # p99 = 990th observation (index 989 in 0-indexed array)
        # Bucket 1: indices 0-899, Bucket 2: indices 900-979, Bucket 3: indices 980-999
        # So p99 falls in the (0.5, 1.0] bucket
        target_sum = 900 * 0.05 + 80 * 0.3 + 20 * 0.8  # 45 + 24 + 16 = 85

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 1000
        p99 = np.percentile(observations, 99)

        # p99 (990th observation) should be in the last finite bucket
        assert 0.5 < p99 <= 1.0

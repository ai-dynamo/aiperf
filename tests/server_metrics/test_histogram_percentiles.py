# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for histogram percentile estimation."""

from __future__ import annotations

import numpy as np

from aiperf.server_metrics.histogram_percentiles import (
    _MAX_OBSERVATIONS,
    BucketStatistics,
    _cumulative_to_per_bucket,
    _estimate_bucket_sums,
    _estimate_inf_bucket_observations,
    _generate_blended_observations,
    _generate_f3_observations,
    _generate_observations_with_sum_constraint,
    _generate_variance_aware_observations,
    _get_bucket_bounds,
    accumulate_bucket_statistics,
    compute_estimated_percentiles,
    compute_prometheus_percentiles,
)


class TestComputeEstimatedPercentiles:
    """Tests for compute_estimated_percentiles function."""

    def test_single_inf_observation_absorbs_sum_correctly(self) -> None:
        """Test that single +Inf observation correctly absorbs large sum.

        This tests the linspace bug fix: np.linspace(lower, upper, 1) returns
        [lower] not the mean, causing catastrophic errors when a single +Inf
        observation should absorb a large sum.
        """
        # Setup: 1 observation in finite bucket with tiny value, 1 in +Inf with huge value
        # Finite bucket [0, 0.001]: 1 observation, contributes ~0.0005 to sum
        # +Inf bucket: 1 observation, should absorb ~100000 to match total_sum
        bucket_cumulative = {
            "0.001": 1.0,  # 1 observation in [0, 0.001]
            "+Inf": 2.0,  # 1 observation in +Inf (cumulative)
        }

        bucket_stats = {
            "0.001": BucketStatistics(bucket_le="0.001"),
            "+Inf": BucketStatistics(bucket_le="+Inf"),
        }

        total_sum = 100000.0  # Almost all from the +Inf observation
        total_count = 2

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=total_sum,
            total_count=total_count,
        )

        assert result is not None
        # P99 should be close to the +Inf observation value (~100000)
        # Before fix: P99 would be ~0.004 (40000% error)
        # After fix: P99 should be close to 100000
        assert result.p99_estimate > 1000, (
            f"P99 should be large (near +Inf obs value), got {result.p99_estimate}"
        )

    def test_multiple_inf_observations_distributed_correctly(self) -> None:
        """Test that multiple +Inf observations are distributed via linspace."""
        bucket_cumulative = {
            "1.0": 10.0,  # 10 observations in [0, 1]
            "+Inf": 20.0,  # 10 observations in +Inf (cumulative)
        }

        bucket_stats = {
            "1.0": BucketStatistics(bucket_le="1.0"),
            "+Inf": BucketStatistics(bucket_le="+Inf"),
        }

        total_sum = 150.0  # ~5 from finite (avg 0.5), ~145 from +Inf (avg 14.5)
        total_count = 20

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=total_sum,
            total_count=total_count,
        )

        assert result is not None
        # P99 should be in the upper range of +Inf observations
        assert result.p99_estimate > 1.0, "P99 should be above max finite bucket"

    def test_no_inf_observations(self) -> None:
        """Test handling when no +Inf observations exist."""
        bucket_cumulative = {
            "1.0": 5.0,
            "10.0": 10.0,
            "+Inf": 10.0,  # No +Inf obs (same as previous cumulative)
        }

        bucket_stats = {
            "1.0": BucketStatistics(bucket_le="1.0"),
            "10.0": BucketStatistics(bucket_le="10.0"),
            "+Inf": BucketStatistics(bucket_le="+Inf"),
        }

        total_sum = 30.0
        total_count = 10

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=total_sum,
            total_count=total_count,
        )

        assert result is not None
        assert result.p99_estimate <= 10.0, "P99 should be within finite bucket bounds"

    def test_zero_count_returns_none(self) -> None:
        """Test that zero total_count returns None."""
        bucket_cumulative = {"1.0": 0.0, "+Inf": 0.0}
        bucket_stats = {
            "1.0": BucketStatistics(bucket_le="1.0"),
            "+Inf": BucketStatistics(bucket_le="+Inf"),
        }

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=0.0,
            total_count=0,
        )

        assert result is None

    def test_empty_bucket_deltas_returns_none(self) -> None:
        """Test that empty bucket_deltas returns None."""
        result = compute_estimated_percentiles(
            bucket_deltas={},
            bucket_stats={},
            total_sum=100.0,
            total_count=10,
        )

        assert result is None


class TestAccumulateBucketStatistics:
    """Tests for accumulate_bucket_statistics function."""

    def test_single_bucket_learning(self) -> None:
        """Test learning statistics from single-bucket intervals."""
        # Two snapshots where all new observations go to one bucket
        sums = np.array([0.0, 10.0], dtype=np.float64)
        counts = np.array([0.0, 2.0], dtype=np.float64)
        bucket_les = ("1.0", "10.0", "+Inf")
        bucket_counts = np.array(
            [
                [0.0, 0.0, 0.0],  # Initial
                [2.0, 2.0, 2.0],  # 2 obs in [0, 1], cumulative
            ],
            dtype=np.float64,
        )

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        # Should have learned mean for bucket "1.0"
        assert "1.0" in stats
        assert stats["1.0"].estimated_mean is not None
        # Mean should be 10/2 = 5.0... but wait, that's outside [0, 1]
        # Actually the mean would be sum_delta/count_delta = 10/2 = 5
        # But the bucket bounds are [0, 1], so this tests edge cases

    def test_multi_bucket_interval_no_learning(self) -> None:
        """Test that multi-bucket intervals don't update statistics."""
        sums = np.array([0.0, 10.0], dtype=np.float64)
        counts = np.array([0.0, 4.0], dtype=np.float64)
        bucket_les = ("1.0", "10.0", "+Inf")
        bucket_counts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 4.0, 4.0],  # 2 obs in [0,1], 2 obs in [1,10]
            ],
            dtype=np.float64,
        )

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        # Multi-bucket interval - function only returns stats for buckets it found
        # With observations split across buckets, no single-bucket learning occurs
        # The returned dict may be empty or contain unlearned stats
        for _, bucket_stats in stats.items():
            # Any stats present should have no learned mean (multi-bucket interval)
            assert (
                bucket_stats.estimated_mean is None
                or bucket_stats.observation_count == 0
            )


class TestBucketStatistics:
    """Tests for BucketStatistics dataclass."""

    def test_estimated_mean_calculation(self) -> None:
        """Test that estimated_mean is calculated correctly."""
        stats = BucketStatistics(bucket_le="1.0")
        assert stats.estimated_mean is None

        # Simulate learning: estimated_mean = weighted_mean_sum / observation_count
        stats.weighted_mean_sum = 15.0
        stats.observation_count = 3

        assert stats.estimated_mean == 5.0

    def test_estimated_mean_zero_observation_count(self) -> None:
        """Test that estimated_mean returns None when observation_count is 0."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.weighted_mean_sum = 10.0
        stats.observation_count = 0

        assert stats.estimated_mean is None

    def test_estimated_variance_insufficient_observations(self) -> None:
        """Test that estimated_variance returns None with < 3 observations."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.observed_means = [1.0, 2.0]  # Only 2 observations

        assert stats.estimated_variance is None

    def test_estimated_variance_sufficient_observations(self) -> None:
        """Test that estimated_variance is computed with >= 3 observations."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.observed_means = [1.0, 2.0, 3.0]

        variance = stats.estimated_variance
        assert variance is not None
        # Sample variance of [1, 2, 3] with ddof=1 = 1.0
        assert abs(variance - 1.0) < 1e-10

    def test_record_updates_all_fields(self) -> None:
        """Test that record() updates all statistics fields."""
        stats = BucketStatistics(bucket_le="1.0")

        stats.record(mean=5.0, count=10)
        assert stats.observation_count == 10
        assert stats.weighted_mean_sum == 50.0
        assert stats.sample_count == 1
        assert stats.observed_means == [5.0]

        stats.record(mean=3.0, count=5)
        assert stats.observation_count == 15
        assert stats.weighted_mean_sum == 65.0
        assert stats.sample_count == 2
        assert stats.observed_means == [5.0, 3.0]


class TestGetBucketBounds:
    """Tests for _get_bucket_bounds function."""

    def test_first_bucket_lower_bound_is_zero(self) -> None:
        """Test that first bucket has lower bound of 0."""
        sorted_buckets = ["0.1", "1.0", "10.0"]
        lower, upper = _get_bucket_bounds("0.1", sorted_buckets)

        assert lower == 0.0
        assert upper == 0.1

    def test_middle_bucket_bounds(self) -> None:
        """Test middle bucket gets correct bounds."""
        sorted_buckets = ["0.1", "1.0", "10.0"]
        lower, upper = _get_bucket_bounds("1.0", sorted_buckets)

        assert lower == 0.1
        assert upper == 1.0

    def test_last_finite_bucket_bounds(self) -> None:
        """Test last finite bucket bounds."""
        sorted_buckets = ["0.1", "1.0", "10.0"]
        lower, upper = _get_bucket_bounds("10.0", sorted_buckets)

        assert lower == 1.0
        assert upper == 10.0

    def test_inf_bucket_has_inf_upper_bound(self) -> None:
        """Test +Inf bucket has infinity upper bound."""
        sorted_buckets = ["1.0", "10.0", "+Inf"]
        lower, upper = _get_bucket_bounds("+Inf", sorted_buckets)

        assert lower == 10.0
        assert upper == float("inf")

    def test_single_bucket(self) -> None:
        """Test single bucket list."""
        sorted_buckets = ["1.0"]
        lower, upper = _get_bucket_bounds("1.0", sorted_buckets)

        assert lower == 0.0
        assert upper == 1.0


class TestCumulativeToPerBucket:
    """Tests for _cumulative_to_per_bucket function."""

    def test_simple_conversion(self) -> None:
        """Test basic cumulative to per-bucket conversion."""
        cumulative = {"1.0": 5.0, "10.0": 8.0, "+Inf": 10.0}
        per_bucket = _cumulative_to_per_bucket(cumulative)

        assert per_bucket["1.0"] == 5.0
        assert per_bucket["10.0"] == 3.0
        assert per_bucket["+Inf"] == 2.0

    def test_all_in_first_bucket(self) -> None:
        """Test when all observations are in first bucket."""
        cumulative = {"0.1": 10.0, "1.0": 10.0, "+Inf": 10.0}
        per_bucket = _cumulative_to_per_bucket(cumulative)

        assert per_bucket["0.1"] == 10.0
        assert per_bucket["1.0"] == 0.0
        assert per_bucket["+Inf"] == 0.0

    def test_all_in_inf_bucket(self) -> None:
        """Test when all observations are in +Inf bucket."""
        cumulative = {"1.0": 0.0, "10.0": 0.0, "+Inf": 10.0}
        per_bucket = _cumulative_to_per_bucket(cumulative)

        assert per_bucket["1.0"] == 0.0
        assert per_bucket["10.0"] == 0.0
        assert per_bucket["+Inf"] == 10.0

    def test_empty_buckets(self) -> None:
        """Test empty bucket dict."""
        per_bucket = _cumulative_to_per_bucket({})
        assert per_bucket == {}

    def test_unsorted_input_keys(self) -> None:
        """Test that unsorted input keys are handled correctly."""
        # Keys are intentionally out of order
        cumulative = {"+Inf": 10.0, "1.0": 3.0, "0.1": 1.0, "10.0": 7.0}
        per_bucket = _cumulative_to_per_bucket(cumulative)

        assert per_bucket["0.1"] == 1.0
        assert per_bucket["1.0"] == 2.0
        assert per_bucket["10.0"] == 4.0
        assert per_bucket["+Inf"] == 3.0


class TestEstimateBucketSums:
    """Tests for _estimate_bucket_sums function."""

    def test_uses_learned_mean(self) -> None:
        """Test that learned mean is used when available."""
        per_bucket = {"1.0": 10.0, "10.0": 5.0}
        bucket_stats = {
            "1.0": BucketStatistics(bucket_le="1.0"),
            "10.0": BucketStatistics(bucket_le="10.0"),
        }
        # Set learned mean for first bucket
        bucket_stats["1.0"].weighted_mean_sum = 7.0
        bucket_stats["1.0"].observation_count = 10

        sums = _estimate_bucket_sums(per_bucket, bucket_stats)

        # First bucket uses learned mean (0.7)
        assert abs(sums["1.0"] - 7.0) < 1e-10
        # Second bucket uses midpoint ((1 + 10) / 2 = 5.5)
        assert abs(sums["10.0"] - 27.5) < 1e-10

    def test_fallback_to_midpoint(self) -> None:
        """Test fallback to midpoint when no learned mean."""
        per_bucket = {"1.0": 10.0}
        bucket_stats = {}  # No learned stats

        sums = _estimate_bucket_sums(per_bucket, bucket_stats)

        # Midpoint of [0, 1] = 0.5, so sum = 10 * 0.5 = 5
        assert abs(sums["1.0"] - 5.0) < 1e-10

    def test_skips_inf_bucket(self) -> None:
        """Test that +Inf bucket is skipped."""
        per_bucket = {"1.0": 5.0, "+Inf": 10.0}
        bucket_stats = {}

        sums = _estimate_bucket_sums(per_bucket, bucket_stats)

        assert "+Inf" not in sums
        assert "1.0" in sums

    def test_skips_zero_count_buckets(self) -> None:
        """Test that zero-count buckets are skipped."""
        per_bucket = {"1.0": 0.0, "10.0": 5.0}
        bucket_stats = {}

        sums = _estimate_bucket_sums(per_bucket, bucket_stats)

        assert "1.0" not in sums
        assert "10.0" in sums


class TestEstimateInfBucketObservations:
    """Tests for _estimate_inf_bucket_observations function."""

    def test_zero_count_returns_empty(self) -> None:
        """Test that zero inf_count returns empty array."""
        result = _estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=50.0,
            inf_count=0,
            max_finite_bucket=10.0,
        )
        assert len(result) == 0

    def test_negative_count_returns_empty(self) -> None:
        """Test that negative inf_count returns empty array."""
        result = _estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=50.0,
            inf_count=-1,
            max_finite_bucket=10.0,
        )
        assert len(result) == 0

    def test_single_observation_uses_mean_directly(self) -> None:
        """Test that single observation is placed at mean, not linspace lower bound."""
        result = _estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=10.0,
            inf_count=1,
            max_finite_bucket=5.0,
        )

        assert len(result) == 1
        # inf_sum = 100 - 10 = 90, inf_avg = 90 / 1 = 90
        assert result[0] == 90.0

    def test_multiple_observations_uses_linspace(self) -> None:
        """Test that multiple observations are spread via linspace."""
        result = _estimate_inf_bucket_observations(
            total_sum=100.0,
            estimated_finite_sum=10.0,
            inf_count=5,
            max_finite_bucket=5.0,
        )

        assert len(result) == 5
        # All observations should be > max_finite_bucket
        assert all(obs >= 5.0 for obs in result)
        # Observations should be sorted (linspace produces sorted output)
        assert np.array_equal(result, np.sort(result))

    def test_negative_inf_sum_fallback(self) -> None:
        """Test fallback when back-calculated inf_sum is negative."""
        result = _estimate_inf_bucket_observations(
            total_sum=10.0,
            estimated_finite_sum=100.0,  # Greater than total_sum!
            inf_count=2,
            max_finite_bucket=5.0,
        )

        assert len(result) == 2
        # Should fall back to 1.5x max_finite_bucket = 7.5
        assert all(obs >= 5.0 for obs in result)

    def test_inf_avg_below_max_bucket_fallback(self) -> None:
        """Test fallback when inf_avg <= max_finite_bucket."""
        result = _estimate_inf_bucket_observations(
            total_sum=20.0,
            estimated_finite_sum=15.0,
            inf_count=10,  # inf_avg = 5/10 = 0.5, which is < max_finite_bucket
            max_finite_bucket=5.0,
        )

        assert len(result) == 10
        # Should fall back to 1.5x max_finite_bucket
        assert all(obs >= 5.0 for obs in result)


class TestGenerateF3Observations:
    """Tests for _generate_f3_observations function."""

    def test_zero_count_returns_empty(self) -> None:
        """Test that zero count returns empty array."""
        result = _generate_f3_observations(
            count=0, lower=0.0, upper=1.0, mean=0.5, variance=0.01
        )
        assert len(result) == 0

    def test_negative_count_returns_empty(self) -> None:
        """Test that negative count returns empty array."""
        result = _generate_f3_observations(
            count=-1, lower=0.0, upper=1.0, mean=0.5, variance=0.01
        )
        assert len(result) == 0

    def test_observations_within_bounds(self) -> None:
        """Test that all observations are within bucket bounds."""
        result = _generate_f3_observations(
            count=100, lower=0.0, upper=1.0, mean=0.5, variance=0.01
        )

        assert len(result) == 100
        assert all(0.0 <= obs <= 1.0 for obs in result)

    def test_mean_at_lower_bound_edge_case(self) -> None:
        """Test edge case where mean equals lower bound."""
        result = _generate_f3_observations(
            count=10, lower=0.0, upper=1.0, mean=0.0, variance=0.01
        )

        assert len(result) == 10
        assert all(0.0 <= obs <= 1.0 for obs in result)

    def test_zero_variance(self) -> None:
        """Test handling of zero variance."""
        result = _generate_f3_observations(
            count=10, lower=0.0, upper=1.0, mean=0.5, variance=0.0
        )

        assert len(result) == 10
        # With zero variance, denominator = 0 + (mean-x)^2, p_x = 0, all obs at 'a'
        assert all(0.0 <= obs <= 1.0 for obs in result)


class TestGenerateVarianceAwareObservations:
    """Tests for _generate_variance_aware_observations function."""

    def test_zero_count_returns_empty(self) -> None:
        """Test that zero count returns empty array."""
        result = _generate_variance_aware_observations(
            count=0, lower=0.0, upper=1.0, mean=0.5, std=0.1
        )
        assert len(result) == 0

    def test_observations_within_bounds(self) -> None:
        """Test that all observations are within bucket bounds."""
        result = _generate_variance_aware_observations(
            count=100, lower=0.0, upper=1.0, mean=0.5, std=0.1
        )

        assert len(result) == 100
        assert all(0.0 <= obs <= 1.0 for obs in result)

    def test_zero_std_uses_fallback(self) -> None:
        """Test that zero std uses fallback (3 stds)."""
        result = _generate_variance_aware_observations(
            count=10, lower=0.0, upper=1.0, mean=0.5, std=0.0
        )

        assert len(result) == 10
        assert all(0.0 <= obs <= 1.0 for obs in result)

    def test_asymmetric_mean_position(self) -> None:
        """Test with mean closer to one edge."""
        result = _generate_variance_aware_observations(
            count=100, lower=0.0, upper=1.0, mean=0.1, std=0.05
        )

        assert len(result) == 100
        assert all(0.0 <= obs <= 1.0 for obs in result)
        # Mean of observations should be near 0.1
        assert abs(np.mean(result) - 0.1) < 0.2


class TestGenerateBlendedObservations:
    """Tests for _generate_blended_observations function."""

    def test_zero_count_returns_empty(self) -> None:
        """Test that zero count returns empty array."""
        result = _generate_blended_observations(
            count=0, lower=0.0, upper=1.0, mean=0.5, std=0.1
        )
        assert len(result) == 0

    def test_observations_within_bounds(self) -> None:
        """Test that all observations are within bucket bounds."""
        result = _generate_blended_observations(
            count=100, lower=0.0, upper=1.0, mean=0.5, std=0.1
        )

        assert len(result) == 100
        assert all(0.0 <= obs <= 1.0 for obs in result)

    def test_blend_factor_zero_is_uniform(self) -> None:
        """Test that blend_factor=0 produces shifted uniform distribution."""
        result = _generate_blended_observations(
            count=100, lower=0.0, upper=1.0, mean=0.5, std=0.1, blend_factor=0.0
        )

        assert len(result) == 100
        # Should be evenly distributed
        assert all(0.0 <= obs <= 1.0 for obs in result)


class TestGenerateObservationsWithSumConstraint:
    """Tests for _generate_observations_with_sum_constraint function."""

    def test_empty_buckets_returns_empty(self) -> None:
        """Test that empty per_bucket_counts returns empty array."""
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={}, target_sum=100.0
        )
        assert len(result) == 0

    def test_only_inf_bucket_returns_empty(self) -> None:
        """Test that only +Inf bucket returns empty (finite obs only)."""
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"+Inf": 10.0}, target_sum=100.0
        )
        assert len(result) == 0

    def test_zero_count_buckets_skipped(self) -> None:
        """Test that zero-count buckets are skipped."""
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 0.0, "10.0": 5.0}, target_sum=25.0
        )
        assert len(result) == 5

    def test_observations_within_bucket_bounds(self) -> None:
        """Test that observations stay within bucket bounds."""
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 10.0, "10.0": 10.0, "+Inf": 0.0},
            target_sum=60.0,
        )

        assert len(result) == 20
        # First 10 should be in [0, 1], next 10 in [1, 10]
        bucket1_obs = result[:10]
        bucket2_obs = result[10:]
        assert all(0.0 <= obs <= 1.0 for obs in bucket1_obs)
        assert all(1.0 <= obs <= 10.0 for obs in bucket2_obs)

    def test_dominant_bucket_uses_avg(self) -> None:
        """Test that dominant bucket (95%+) uses average as center."""
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 100.0, "10.0": 2.0},  # 98% in first bucket
            target_sum=70.0,  # avg = 70/102 ≈ 0.69
        )

        assert len(result) == 102
        # Most observations should be centered around avg (0.69)
        bucket1_obs = result[:100]
        mean_bucket1 = np.mean(bucket1_obs)
        # Mean should be closer to 0.69 than midpoint (0.5)
        assert abs(mean_bucket1 - 0.69) < 0.3

    def test_sum_constraint_adjustment(self) -> None:
        """Test that sum constraint adjusts observations."""
        # Without learned stats, uses midpoint. For [0, 1], midpoint = 0.5
        # 10 observations * 0.5 = 5.0 expected sum
        # With target_sum = 8.0, should shift observations up
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 10.0},
            target_sum=8.0,
        )

        assert len(result) == 10
        actual_sum = sum(result)
        # Should be close to target (within adjustment limits)
        assert abs(actual_sum - 8.0) < 2.0

    def test_uses_learned_statistics(self) -> None:
        """Test that learned bucket statistics are used."""
        bucket_stats = {"1.0": BucketStatistics(bucket_le="1.0")}
        bucket_stats["1.0"].weighted_mean_sum = 8.0
        bucket_stats["1.0"].observation_count = 10
        bucket_stats["1.0"].observed_means = [0.78, 0.82, 0.80]  # For variance

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 10.0},
            target_sum=8.0,
            bucket_stats=bucket_stats,
        )

        assert len(result) == 10
        # Observations should be centered around learned mean (0.8)
        assert abs(np.mean(result) - 0.8) < 0.2

    def test_large_count_is_downsampled(self) -> None:
        """Test that very large counts are downsampled to prevent memory issues."""
        large_count = _MAX_OBSERVATIONS * 10  # Way over limit
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": large_count / 2, "10.0": large_count / 2},
            target_sum=large_count * 5.0,  # avg ~5.0
        )

        # Should be capped at approximately _MAX_OBSERVATIONS
        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0

        # Distribution proportions should be preserved (50/50)
        bucket1_obs = [obs for obs in result if obs <= 1.0]
        bucket2_obs = [obs for obs in result if obs > 1.0]
        # Each bucket should have roughly half the observations
        assert 0.4 < len(bucket1_obs) / len(result) < 0.6
        assert 0.4 < len(bucket2_obs) / len(result) < 0.6

    def test_downsampling_preserves_percentiles(self) -> None:
        """Test that downsampling preserves percentile accuracy."""
        # Create a distribution with known characteristics
        large_count = _MAX_OBSERVATIONS * 5
        per_bucket_counts = {
            "1.0": large_count * 0.1,  # 10% in [0, 1]
            "10.0": large_count * 0.8,  # 80% in [1, 10]
            "100.0": large_count * 0.1,  # 10% in [10, 100]
        }

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=large_count * 10.0,  # avg = 10.0
        )

        # Should be downsampled
        assert len(result) <= _MAX_OBSERVATIONS

        # P50 should be in the middle bucket [1, 10] since 80% is there
        sorted_result = np.sort(result)
        p50 = np.percentile(sorted_result, 50)
        assert 1.0 <= p50 <= 10.0

        # P10 should be in first bucket, P90 should be in middle/last
        p10 = np.percentile(sorted_result, 10)
        p90 = np.percentile(sorted_result, 90)
        assert p10 < p50 < p90

    def test_downsampling_fractional_counts_no_overflow(self) -> None:
        """Test that fractional counts after downsampling don't cause array overflow.

        When total_count > _MAX_OBSERVATIONS, bucket counts are scaled by a ratio.
        This creates fractional values. The sum of int(count) for each bucket may
        differ from int(sum(counts)) due to truncation. This must not cause
        array index overflow.
        """
        # Create counts that will produce fractional values after downsampling
        # Use prime-ish numbers to maximize truncation differences
        large_multiplier = _MAX_OBSERVATIONS * 3
        per_bucket_counts = {
            "0.001": large_multiplier * 0.333,  # Fractional after scaling
            "0.01": large_multiplier * 0.167,
            "0.1": large_multiplier * 0.234,
            "1.0": large_multiplier * 0.123,
            "10.0": large_multiplier * 0.143,
        }

        # This should not raise ValueError about shape mismatch
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=large_multiplier * 1.5,
        )

        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0

    def test_many_buckets_cumulative_truncation_no_overflow(self) -> None:
        """Test many buckets with small fractional counts don't overflow.

        With many buckets, the cumulative truncation error from int(count)
        could be significant. This tests that we handle this gracefully.
        """
        # Create 50 buckets with counts that will be fractional after downsampling
        large_multiplier = _MAX_OBSERVATIONS * 5
        per_bucket_counts = {}
        for i in range(1, 51):
            bucket_le = str(float(i))
            # Each bucket gets 2% of total, but with slight variations
            per_bucket_counts[bucket_le] = large_multiplier * (0.02 + 0.001 * (i % 7))

        # This should not raise ValueError
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=large_multiplier * 25.0,
        )

        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0

    def test_variance_aware_paths_no_overflow(self) -> None:
        """Test that all variance-aware paths handle bounds correctly.

        Tests F3, blended, and variance-aware observation generation paths
        with downsampling to ensure no array overflow occurs.
        """
        large_multiplier = _MAX_OBSERVATIONS * 2

        # Create bucket statistics that will trigger each variance-aware path
        bucket_stats = {}

        # Bucket 1: F3 path (extremely tight variance < 1% of bucket width)
        stats_f3 = BucketStatistics(bucket_le="0.1")
        stats_f3.weighted_mean_sum = 0.05 * (large_multiplier * 0.3)
        stats_f3.observation_count = int(large_multiplier * 0.3)
        stats_f3.observed_means = [0.0501, 0.0499, 0.05, 0.05, 0.05]  # Very tight
        bucket_stats["0.1"] = stats_f3

        # Bucket 2: Blended path (tight variance < 20%, mean near center < 30%)
        stats_blended = BucketStatistics(bucket_le="1.0")
        stats_blended.weighted_mean_sum = 0.55 * (large_multiplier * 0.3)
        stats_blended.observation_count = int(large_multiplier * 0.3)
        stats_blended.observed_means = [
            0.54,
            0.55,
            0.56,
            0.55,
            0.55,
        ]  # Moderate variance
        bucket_stats["1.0"] = stats_blended

        # Bucket 3: Variance-aware path (moderate variance)
        stats_var = BucketStatistics(bucket_le="10.0")
        stats_var.weighted_mean_sum = 7.0 * (large_multiplier * 0.4)
        stats_var.observation_count = int(large_multiplier * 0.4)
        stats_var.observed_means = [6.0, 7.0, 8.0, 5.5, 8.5]  # Higher variance
        bucket_stats["10.0"] = stats_var

        per_bucket_counts = {
            "0.1": large_multiplier * 0.3,
            "1.0": large_multiplier * 0.3,
            "10.0": large_multiplier * 0.4,
        }

        # This should not raise ValueError
        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=large_multiplier * 5.0,
            bucket_stats=bucket_stats,
        )

        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0

    def test_edge_case_counts_near_one(self) -> None:
        """Test buckets with counts that truncate to 0 or 1.

        Edge case where fractional counts after downsampling are close to 1.0,
        which could cause off-by-one errors in array sizing.
        """
        large_multiplier = _MAX_OBSERVATIONS * 10
        # Create many buckets where each will have count ~1.x after downsampling
        per_bucket_counts = {}
        num_buckets = 100
        count_per_bucket = (
            large_multiplier / num_buckets
        )  # Each ~1000 after downsampling

        for i in range(1, num_buckets + 1):
            bucket_le = str(float(i))
            # Slight variation to create different truncation behavior
            per_bucket_counts[bucket_le] = count_per_bucket + (i % 10) * 0.1

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=large_multiplier * 50.0,
        )

        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0

    def test_exact_max_observations_boundary(self) -> None:
        """Test behavior at exactly _MAX_OBSERVATIONS threshold.

        When total count is exactly at or just over the threshold,
        floating point precision in the ratio calculation could cause issues.
        """
        # Just over the limit to trigger downsampling
        per_bucket_counts = {
            "1.0": _MAX_OBSERVATIONS * 0.5 + 1,
            "10.0": _MAX_OBSERVATIONS * 0.5 + 1,
        }

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts=per_bucket_counts,
            target_sum=(_MAX_OBSERVATIONS + 2) * 5.0,
        )

        # Should be exactly at or just under _MAX_OBSERVATIONS
        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > _MAX_OBSERVATIONS * 0.9  # Should be close to max

    def test_single_bucket_large_count_no_overflow(self) -> None:
        """Test single large bucket doesn't overflow after downsampling."""
        large_count = _MAX_OBSERVATIONS * 7.777  # Fractional multiplier

        result = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": large_count},
            target_sum=large_count * 0.5,
        )

        assert len(result) <= _MAX_OBSERVATIONS
        assert len(result) > 0
        # All observations should be in [0, 1]
        assert all(0.0 <= obs <= 1.0 for obs in result)


class TestComputeEstimatedPercentilesEdgeCases:
    """Additional edge case tests for compute_estimated_percentiles."""

    def test_zero_sum_returns_all_zeros(self) -> None:
        """Test that zero sum returns all-zero percentiles."""
        bucket_cumulative = {"1.0": 10.0, "+Inf": 10.0}
        bucket_stats = {
            "1.0": BucketStatistics(bucket_le="1.0"),
            "+Inf": BucketStatistics(bucket_le="+Inf"),
        }

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=0.0,
            total_count=10,
        )

        assert result is not None
        assert result.p50_estimate == 0.0
        assert result.p99_estimate == 0.0

    def test_negative_count_returns_none(self) -> None:
        """Test that negative count returns None."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1.0": 5.0, "+Inf": 5.0},
            bucket_stats={},
            total_sum=10.0,
            total_count=-1,
        )

        assert result is None

    def test_only_finite_buckets_no_inf(self) -> None:
        """Test with only finite buckets (no +Inf in dict)."""
        bucket_cumulative = {"0.1": 5.0, "1.0": 10.0}
        bucket_stats = {}

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=5.0,
            total_count=10,
        )

        assert result is not None
        # All percentiles should be within [0, 1]
        assert 0.0 <= result.p50_estimate <= 1.0
        assert 0.0 <= result.p99_estimate <= 1.0

    def test_all_observations_in_single_bucket(self) -> None:
        """Test when all observations are in a single bucket."""
        bucket_cumulative = {"0.1": 0.0, "1.0": 100.0, "+Inf": 100.0}
        bucket_stats = {}

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=50.0,
            total_count=100,
        )

        assert result is not None
        # All percentiles should be in [0.1, 1.0]
        assert 0.1 <= result.p50_estimate <= 1.0
        assert 0.1 <= result.p99_estimate <= 1.0

    def test_very_small_bucket_width(self) -> None:
        """Test with very small bucket widths."""
        bucket_cumulative = {
            "0.001": 50.0,
            "0.002": 100.0,
            "+Inf": 100.0,
        }
        bucket_stats = {}

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=0.15,  # avg = 0.0015
            total_count=100,
        )

        assert result is not None
        assert result.p50_estimate > 0.0
        assert result.p50_estimate < 0.01

    def test_large_bucket_values(self) -> None:
        """Test with large bucket boundary values."""
        bucket_cumulative = {
            "1000.0": 50.0,
            "10000.0": 100.0,
            "+Inf": 100.0,
        }
        bucket_stats = {}

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats=bucket_stats,
            total_sum=500000.0,
            total_count=100,
        )

        assert result is not None
        assert result.p50_estimate > 100.0

    def test_large_count_downsamples_all_buckets_proportionally(self) -> None:
        """Test that large counts downsample ALL buckets including +Inf proportionally.

        This is critical: if finite buckets are downsampled but +Inf is not,
        the percentile estimates would be skewed toward +Inf values.
        """
        # Create a distribution with 10% in +Inf
        large_count = _MAX_OBSERVATIONS * 10
        bucket_cumulative = {
            "1.0": large_count * 0.5,  # 50% in [0, 1]
            "10.0": large_count * 0.9,  # 40% in [1, 10] (cumulative)
            "+Inf": large_count,  # 10% in +Inf (cumulative)
        }

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats={},
            total_sum=large_count * 5.0,  # avg = 5.0
            total_count=large_count,
        )

        assert result is not None
        # P50 should be in the finite buckets (since 90% is finite)
        # If +Inf wasn't downsampled proportionally, P50 would incorrectly
        # be in the +Inf range (> 10)
        assert result.p50_estimate <= 10.0
        # P90 should be at the boundary of +Inf
        assert result.p90_estimate >= 1.0

    def test_single_inf_observation_survives_downsampling(self) -> None:
        """Test that a single +Inf observation is preserved even with heavy downsampling.

        When billions of finite observations are downsampled, a single +Inf observation
        would get a fractional count (e.g., 0.00001). Using int() would lose it entirely.
        The fix uses ceiling to ensure at least 1 observation is preserved.
        """
        # 10 billion finite observations, 1 +Inf observation
        large_count = 10_000_000_000
        bucket_cumulative = {
            "1.0": large_count,  # 10B in [0, 1]
            "+Inf": large_count + 1,  # 1 observation in +Inf (cumulative)
        }

        # The +Inf observation has a value of 1000 (huge outlier)
        # Finite sum ≈ 5B (avg 0.5), +Inf contributes 1000
        total_sum = large_count * 0.5 + 1000.0

        result = compute_estimated_percentiles(
            bucket_deltas=bucket_cumulative,
            bucket_stats={},
            total_sum=total_sum,
            total_count=large_count + 1,
        )

        assert result is not None
        # P50 and P90 should be in finite bucket
        assert result.p50_estimate <= 1.0
        assert result.p90_estimate <= 1.0
        # P99 should still be in finite bucket (only 1 out of 10B is in +Inf)
        # The +Inf observation is at the 99.99999999% percentile
        assert result.p99_estimate <= 1.0


class TestAccumulateBucketStatisticsEdgeCases:
    """Additional edge case tests for accumulate_bucket_statistics."""

    def test_empty_arrays_returns_empty(self) -> None:
        """Test that empty arrays return empty dict."""
        sums = np.array([], dtype=np.float64)
        counts = np.array([], dtype=np.float64)
        bucket_les = ("1.0", "+Inf")
        bucket_counts = np.empty((0, 2), dtype=np.float64)

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        assert stats == {}

    def test_single_snapshot_returns_empty(self) -> None:
        """Test that single snapshot (no deltas) returns empty dict."""
        sums = np.array([10.0], dtype=np.float64)
        counts = np.array([5.0], dtype=np.float64)
        bucket_les = ("1.0", "+Inf")
        bucket_counts = np.array([[5.0, 5.0]], dtype=np.float64)

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        assert stats == {}

    def test_start_idx_beyond_data_returns_empty(self) -> None:
        """Test that start_idx beyond data returns empty dict."""
        sums = np.array([0.0, 10.0], dtype=np.float64)
        counts = np.array([0.0, 5.0], dtype=np.float64)
        bucket_les = ("1.0", "+Inf")
        bucket_counts = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=10,  # Beyond data
        )

        assert stats == {}

    def test_zero_count_delta_skipped(self) -> None:
        """Test that intervals with zero count delta are skipped."""
        sums = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        counts = np.array([0.0, 0.0, 5.0], dtype=np.float64)  # First delta is 0
        bucket_les = ("1.0", "+Inf")
        bucket_counts = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]], dtype=np.float64)

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        # First interval has count_delta=0, should be skipped
        # Second interval has count_delta=5, all in bucket "1.0"
        assert "1.0" in stats
        assert stats["1.0"].estimated_mean == 2.0  # 10/5

    def test_counter_reset_handled(self) -> None:
        """Test that counter resets (negative deltas) are handled."""
        # Simulate counter reset: values go down then back up
        sums = np.array([100.0, 50.0, 60.0], dtype=np.float64)
        counts = np.array([10.0, 5.0, 7.0], dtype=np.float64)
        bucket_les = ("1.0", "+Inf")
        bucket_counts = np.array(
            [[10.0, 10.0], [5.0, 5.0], [7.0, 7.0]], dtype=np.float64
        )

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )
        assert stats is not None

        # First delta has negative count (-5), should be skipped
        # Second delta has positive count (2), should be processed
        # bucket_deltas are clipped to 0, so np.maximum handles negative

    def test_multiple_single_bucket_intervals(self) -> None:
        """Test accumulation across multiple single-bucket intervals."""
        sums = np.array([0.0, 10.0, 25.0, 45.0], dtype=np.float64)
        counts = np.array([0.0, 2.0, 5.0, 9.0], dtype=np.float64)
        bucket_les = ("1.0", "10.0", "+Inf")
        bucket_counts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],  # 2 in bucket "1.0"
                [2.0, 5.0, 5.0],  # 3 in bucket "10.0"
                [2.0, 9.0, 9.0],  # 4 in bucket "10.0"
            ],
            dtype=np.float64,
        )

        stats = accumulate_bucket_statistics(
            sums=sums,
            counts=counts,
            bucket_les=bucket_les,
            bucket_counts=bucket_counts,
            start_idx=0,
        )

        # Bucket "1.0": one interval, mean = 10/2 = 5
        assert "1.0" in stats
        assert stats["1.0"].estimated_mean == 5.0

        # Bucket "10.0": two intervals
        # Interval 1: sum_delta=15, count_delta=3, mean=5
        # Interval 2: sum_delta=20, count_delta=4, mean=5
        # Weighted mean = (5*3 + 5*4) / 7 = 5
        assert "10.0" in stats
        assert abs(stats["10.0"].estimated_mean - 5.0) < 0.01


class TestComputePrometheusPercentiles:
    """Tests for compute_prometheus_percentiles function."""

    def test_empty_buckets_returns_empty(self) -> None:
        """Test that empty buckets return empty result."""
        result = compute_prometheus_percentiles({})
        assert result.p50_estimate is None
        assert result.p99_estimate is None

    def test_zero_count_returns_empty(self) -> None:
        """Test that zero total count returns empty result."""
        buckets = {"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}
        result = compute_prometheus_percentiles(buckets)
        assert result.p50_estimate is None

    def test_simple_linear_interpolation(self) -> None:
        """Test basic linear interpolation within buckets."""
        # 100 observations: 20 in [0, 0.1], 40 in (0.1, 0.5], 30 in (0.5, 1.0], 10 in +Inf
        buckets = {"0.1": 20.0, "0.5": 60.0, "1.0": 90.0, "+Inf": 100.0}
        result = compute_prometheus_percentiles(buckets)

        # P50 = 50th observation should be in (0.1, 0.5] bucket
        # Position in bucket: (50 - 20) / 40 = 0.75
        # Value: 0.1 + (0.5 - 0.1) * 0.75 = 0.4
        assert result.p50_estimate is not None
        assert abs(result.p50_estimate - 0.4) < 0.01

    def test_p99_capped_at_last_finite_bucket(self) -> None:
        """Test that P99 in +Inf bucket returns last finite bound."""
        # P99 falls in +Inf bucket
        buckets = {"0.5": 50.0, "1.0": 90.0, "+Inf": 100.0}
        result = compute_prometheus_percentiles(buckets)

        # P99 = 99th observation, which is in +Inf bucket
        # Prometheus returns the last finite bucket bound (1.0)
        assert result.p99_estimate == 1.0

    def test_all_percentiles_computed(self) -> None:
        """Test that all standard percentiles are computed."""
        buckets = {"0.1": 10.0, "0.5": 50.0, "1.0": 80.0, "5.0": 95.0, "+Inf": 100.0}
        result = compute_prometheus_percentiles(buckets)

        assert result.p1_estimate is not None
        assert result.p5_estimate is not None
        assert result.p10_estimate is not None
        assert result.p25_estimate is not None
        assert result.p50_estimate is not None
        assert result.p75_estimate is not None
        assert result.p90_estimate is not None
        assert result.p95_estimate is not None
        assert result.p99_estimate is not None

    def test_unsorted_bucket_keys(self) -> None:
        """Test that unsorted bucket keys are handled correctly."""
        # Keys intentionally unsorted
        buckets = {"+Inf": 100.0, "0.1": 20.0, "1.0": 90.0, "0.5": 60.0}
        result = compute_prometheus_percentiles(buckets)

        # Should still compute correctly
        assert result.p50_estimate is not None
        assert abs(result.p50_estimate - 0.4) < 0.01

    def test_explicit_total_count(self) -> None:
        """Test that explicit total_count parameter is used."""
        buckets = {"0.5": 50.0, "1.0": 100.0, "+Inf": 100.0}

        # Without explicit count, uses +Inf bucket (100)
        result1 = compute_prometheus_percentiles(buckets)

        # With explicit count of 200 (as if +Inf had 100 more)
        result2 = compute_prometheus_percentiles(buckets, total_count=200.0)

        # P50 should be different
        assert result1.p50_estimate != result2.p50_estimate

    def test_single_bucket(self) -> None:
        """Test histogram with single bucket."""
        buckets = {"+Inf": 100.0}
        result = compute_prometheus_percentiles(buckets)

        # Can't interpolate, should return 0 (prev_bound)
        assert result.p50_estimate == 0.0

    def test_all_in_first_bucket(self) -> None:
        """Test when all observations are in the first bucket."""
        buckets = {"0.1": 100.0, "0.5": 100.0, "1.0": 100.0, "+Inf": 100.0}
        result = compute_prometheus_percentiles(buckets)

        # All percentiles should be within first bucket [0, 0.1]
        assert result.p50_estimate is not None
        assert 0 <= result.p50_estimate <= 0.1
        assert result.p99_estimate is not None
        assert 0 <= result.p99_estimate <= 0.1

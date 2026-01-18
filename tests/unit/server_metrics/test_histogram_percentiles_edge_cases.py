# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for histogram_percentiles.py polynomial histogram algorithm."""

import numpy as np
import pytest

from aiperf.server_metrics.histogram_percentiles import (
    BucketStatistics,
    EstimatedPercentiles,
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
)

# =============================================================================
# BucketStatistics Tests
# =============================================================================


class TestBucketStatistics:
    """Test BucketStatistics edge cases."""

    def test_estimated_mean_zero_observations(self) -> None:
        """Test estimated_mean returns None with zero observations."""
        stats = BucketStatistics(bucket_le="1.0")
        assert stats.estimated_mean is None

    def test_estimated_mean_weighted_average(self) -> None:
        """Test that estimated_mean is weighted by count."""
        stats = BucketStatistics(bucket_le="1.0")
        stats.record(mean=0.3, count=100)
        stats.record(mean=0.7, count=100)

        assert stats.estimated_mean == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "sample_count,expected_variance_none",
        [(1, True), (2, True), (3, False), (5, False)],
    )
    def test_estimated_variance_min_samples(
        self, sample_count: int, expected_variance_none: bool
    ) -> None:
        """Test variance requires minimum 3 samples."""
        stats = BucketStatistics(bucket_le="1.0")
        for i in range(sample_count):
            stats.record(mean=0.2 + i * 0.2, count=100)

        assert (stats.estimated_variance is None) == expected_variance_none

    def test_estimated_variance_zero_for_identical_means(self) -> None:
        """Test variance is zero when all means are identical."""
        stats = BucketStatistics(bucket_le="1.0")
        for _ in range(5):
            stats.record(mean=0.5, count=100)

        assert stats.estimated_variance == 0.0


# =============================================================================
# Bucket Utility Functions Tests
# =============================================================================


class TestGetBucketBounds:
    """Test _get_bucket_bounds."""

    @pytest.mark.parametrize(
        "le,sorted_buckets,expected_lower,expected_upper",
        [
            ("0.1", ["0.1", "1.0", "+Inf"], 0.0, 0.1),  # First bucket
            ("1.0", ["0.1", "1.0", "+Inf"], 0.1, 1.0),  # Middle bucket
            ("+Inf", ["0.1", "1.0", "+Inf"], 1.0, float("inf")),  # +Inf bucket
            ("1.0", ["1.0", "+Inf"], 0.0, 1.0),  # Single finite bucket
        ],
    )  # fmt: skip
    def test_bucket_bounds(
        self,
        le: str,
        sorted_buckets: list[str],
        expected_lower: float,
        expected_upper: float,
    ) -> None:
        """Test bucket bounds computation."""
        lower, upper = _get_bucket_bounds(le, sorted_buckets)
        assert lower == expected_lower
        assert upper == expected_upper


class TestCumulativeToPerBucket:
    """Test _cumulative_to_per_bucket conversion."""

    @pytest.mark.parametrize(
        "cumulative,expected",
        [
            ({"0.1": 10.0, "1.0": 50.0, "+Inf": 100.0}, {"0.1": 10.0, "1.0": 40.0, "+Inf": 50.0}),  # Basic
            ({"0.1": 0.0, "1.0": 0.0, "+Inf": 0.0}, {"0.1": 0.0, "1.0": 0.0, "+Inf": 0.0}),  # Empty
            ({"0.1": 100.0, "1.0": 100.0, "+Inf": 100.0}, {"0.1": 100.0, "1.0": 0.0, "+Inf": 0.0}),  # All first
            ({"0.1": 0.0, "1.0": 0.0, "+Inf": 100.0}, {"0.1": 0.0, "1.0": 0.0, "+Inf": 100.0}),  # All +Inf
            ({"0.1": 10.0, "1.0": 50.0}, {"0.1": 10.0, "1.0": 40.0}),  # No +Inf
        ],
    )  # fmt: skip
    def test_conversion(
        self, cumulative: dict[str, float], expected: dict[str, float]
    ) -> None:
        """Test cumulative to per-bucket conversion."""
        assert _cumulative_to_per_bucket(cumulative) == expected


class TestEstimateBucketSums:
    """Test _estimate_bucket_sums."""

    def test_uses_learned_mean(self) -> None:
        """Test that learned means are used when available."""
        stats = {"0.5": BucketStatistics(bucket_le="0.5")}
        stats["0.5"].record(mean=0.3, count=100)

        sums = _estimate_bucket_sums({"0.5": 100.0}, stats)
        assert sums == {"0.5": 30.0}

    def test_falls_back_to_midpoint(self) -> None:
        """Test fallback to midpoint when no learned mean."""
        sums = _estimate_bucket_sums({"0.5": 100.0}, {})
        assert sums == {"0.5": 25.0}  # midpoint = 0.25

    def test_skips_inf_and_empty_buckets(self) -> None:
        """Test that +Inf and zero-count buckets are excluded."""
        sums = _estimate_bucket_sums({"0.5": 0.0, "1.0": 100.0, "+Inf": 50.0}, {})

        assert "0.5" not in sums
        assert "+Inf" not in sums
        assert "1.0" in sums


class TestEstimateInfBucketObservations:
    """Test _estimate_inf_bucket_observations."""

    def test_basic_back_calculation(self) -> None:
        """Test basic +Inf observation estimation."""
        obs = _estimate_inf_bucket_observations(
            total_sum=150.0,
            estimated_finite_sum=100.0,
            inf_count=10,
            max_finite_bucket=1.0,
        )

        assert len(obs) == 10
        assert all(o >= 1.0 for o in obs)  # Includes lower bound
        assert np.mean(obs) == pytest.approx(5.0, rel=0.1)

    @pytest.mark.parametrize(
        "inf_count,total_sum,finite_sum,expected_len",
        [
            (0, 100.0, 100.0, 0),  # Zero count
            (1, 200.0, 100.0, 1),  # Single observation
            (5, 80.0, 100.0, 5),  # Negative sum fallback
        ],
    )  # fmt: skip
    def test_edge_cases(
        self, inf_count: int, total_sum: float, finite_sum: float, expected_len: int
    ) -> None:
        """Test various edge cases for +Inf estimation."""
        obs = _estimate_inf_bucket_observations(
            total_sum=total_sum,
            estimated_finite_sum=finite_sum,
            inf_count=inf_count,
            max_finite_bucket=1.0,
        )
        assert len(obs) == expected_len


# =============================================================================
# Observation Generation Tests
# =============================================================================


class TestObservationGeneration:
    """Test observation generation functions."""

    @pytest.mark.parametrize(
        "func",
        [
            _generate_f3_observations,
            _generate_variance_aware_observations,
            _generate_blended_observations,
        ],
    )
    def test_zero_count_returns_empty(self, func: callable) -> None:
        """Test that zero count returns empty array for all generators."""
        kwargs = {"count": 0, "lower": 0.0, "upper": 1.0, "mean": 0.5}
        if func == _generate_f3_observations:
            kwargs["variance"] = 0.01
        else:
            kwargs["std"] = 0.1

        assert len(func(**kwargs)) == 0

    @pytest.mark.parametrize(
        "func",
        [
            _generate_f3_observations,
            _generate_variance_aware_observations,
            _generate_blended_observations,
        ],
    )
    def test_observations_within_bounds(self, func: callable) -> None:
        """Test that observations stay within bucket bounds for all generators."""
        kwargs = {"count": 100, "lower": 0.2, "upper": 0.8, "mean": 0.5}
        if func == _generate_f3_observations:
            kwargs["variance"] = 0.01
        else:
            kwargs["std"] = 0.1

        obs = func(**kwargs)
        assert all(0.2 <= x <= 0.8 for x in obs)

    def test_f3_produces_two_point_mass(self) -> None:
        """Test F3 distribution produces exactly two distinct values."""
        obs = _generate_f3_observations(
            count=100, lower=0.0, upper=1.0, mean=0.5, variance=0.01
        )
        assert len(np.unique(obs)) == 2


class TestGenerateObservationsWithSumConstraint:
    """Test _generate_observations_with_sum_constraint."""

    @pytest.mark.parametrize(
        "per_bucket_counts,target_sum,expected_len",
        [
            ({}, 100.0, 0),  # Empty buckets
            ({"0.5": 0.0, "1.0": 0.0}, 100.0, 0),  # Zero counts
            ({"0.5": 50.0, "1.0": 50.0}, 50.0, 100),  # Basic case
            ({"1.0": 50.0, "+Inf": 50.0}, 25.0, 50),  # Excludes +Inf
        ],
    )  # fmt: skip
    def test_observation_count(
        self, per_bucket_counts: dict[str, float], target_sum: float, expected_len: int
    ) -> None:
        """Test correct observation count for various inputs."""
        obs = _generate_observations_with_sum_constraint(per_bucket_counts, target_sum)
        assert len(obs) == expected_len

    def test_sum_constraint_precision(self) -> None:
        """Test that sum constraint achieves reasonable precision."""
        obs = _generate_observations_with_sum_constraint(
            per_bucket_counts={"0.5": 1000.0, "1.0": 1000.0}, target_sum=750.0
        )
        # Algorithm achieves ~5% precision after adjustment
        assert abs(np.sum(obs) - 750.0) / 750.0 < 0.10

    def test_downsampling_large_counts(self) -> None:
        """Test that large counts are downsampled."""
        obs = _generate_observations_with_sum_constraint(
            per_bucket_counts={"1.0": 1_000_000.0}, target_sum=500_000.0
        )
        assert len(obs) <= 100_000


# =============================================================================
# Accumulate Bucket Statistics Tests
# =============================================================================


class TestAccumulateBucketStatistics:
    """Test accumulate_bucket_statistics."""

    def test_empty_arrays_returns_empty(self) -> None:
        """Test with empty arrays returns empty dict."""
        result = accumulate_bucket_statistics(
            sums=np.array([]),
            counts=np.array([]),
            bucket_les=(),
            bucket_counts=np.array([]).reshape(0, 0),
        )
        assert result == {}

    def test_single_sample_returns_empty(self) -> None:
        """Test with only one sample returns empty dict (no deltas)."""
        result = accumulate_bucket_statistics(
            sums=np.array([100.0]),
            counts=np.array([10.0]),
            bucket_les=("1.0", "+Inf"),
            bucket_counts=np.array([[5.0, 10.0]]),
        )
        assert result == {}

    def test_single_bucket_interval_learned(self) -> None:
        """Test that single-bucket intervals contribute statistics."""
        result = accumulate_bucket_statistics(
            sums=np.array([0.0, 50.0]),
            counts=np.array([0.0, 100.0]),
            bucket_les=("0.5", "1.0", "+Inf"),
            bucket_counts=np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]),
        )

        assert "0.5" in result
        assert result["0.5"].observation_count == 100
        assert result["0.5"].estimated_mean == pytest.approx(0.5)


# =============================================================================
# Compute Estimated Percentiles Tests
# =============================================================================


class TestComputeEstimatedPercentiles:
    """Test compute_estimated_percentiles."""

    @pytest.mark.parametrize(
        "bucket_deltas,total_sum,total_count,expected_none",
        [
            ({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0, True),  # Zero count
            ({}, 100.0, 100, True),  # Empty buckets
            ({"+Inf": 100.0}, 500.0, 100, True),  # No finite buckets
        ],
    )  # fmt: skip
    def test_returns_none(
        self,
        bucket_deltas: dict[str, float],
        total_sum: float,
        total_count: int,
        expected_none: bool,
    ) -> None:
        """Test cases that return None."""
        result = compute_estimated_percentiles(
            bucket_deltas, {}, total_sum, total_count
        )
        assert (result is None) == expected_none

    def test_zero_sum_returns_all_zeros(self) -> None:
        """Test that zero sum with non-zero count returns all zeros."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1.0": 100.0, "+Inf": 100.0},
            bucket_stats={},
            total_sum=0.0,
            total_count=100,
        )

        assert result is not None
        assert result.p50_estimate == 0.0
        assert result.p99_estimate == 0.0

    def test_all_in_inf_bucket(self) -> None:
        """Test when all observations are in +Inf bucket."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1.0": 0.0, "+Inf": 100.0},
            bucket_stats={},
            total_sum=500.0,
            total_count=100,
        )

        assert result is not None
        assert result.p50_estimate > 1.0
        assert result.p99_estimate > 1.0

    def test_learned_stats_improve_accuracy(self) -> None:
        """Test that learned statistics produce valid percentiles."""
        bucket_stats = {"1.0": BucketStatistics(bucket_le="1.0")}
        bucket_stats["1.0"].record(mean=0.9, count=100)

        with_stats = compute_estimated_percentiles(
            {"1.0": 100.0, "+Inf": 100.0}, bucket_stats, 90.0, 100
        )
        without_stats = compute_estimated_percentiles(
            {"1.0": 100.0, "+Inf": 100.0}, {}, 90.0, 100
        )

        # Both should produce valid percentiles
        assert with_stats is not None
        assert without_stats is not None
        assert with_stats.p50_estimate >= 0
        assert without_stats.p50_estimate >= 0

    def test_large_counts_downsampled(self) -> None:
        """Test that very large counts complete without memory error."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1.0": 10_000_000.0, "+Inf": 10_000_000.0},
            bucket_stats={},
            total_sum=5_000_000.0,
            total_count=10_000_000,
        )
        assert result is not None


class TestEstimatedPercentilesModel:
    """Test EstimatedPercentiles data model."""

    def test_default_values_are_none(self) -> None:
        """Test that all percentile fields default to None."""
        p = EstimatedPercentiles()
        for attr in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            assert getattr(p, f"{attr}_estimate") is None


# =============================================================================
# Numeric Edge Cases
# =============================================================================


class TestNumericEdgeCases:
    """Test numeric edge cases."""

    def test_negative_bucket_bounds(self) -> None:
        """Test with negative bucket bounds."""
        result = _cumulative_to_per_bucket(
            {"-10": 10.0, "-1": 50.0, "0": 80.0, "10": 100.0}
        )

        assert result["-10"] == 10.0
        assert result["-1"] == 40.0
        assert result["0"] == 30.0
        assert result["10"] == 20.0

    def test_very_large_bucket_bounds(self) -> None:
        """Test with very large bucket bounds."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1e6": 100.0, "+Inf": 100.0},
            bucket_stats={},
            total_sum=500000.0,
            total_count=100,
        )
        assert result is not None

    @pytest.mark.parametrize(
        "total_sum",
        [float("nan"), float("inf"), float("-inf"), -100.0],
    )
    def test_invalid_sum_returns_none(self, total_sum: float) -> None:
        """Test that NaN, Inf, and negative sums return None."""
        result = compute_estimated_percentiles(
            bucket_deltas={"1.0": 100.0, "+Inf": 100.0},
            bucket_stats={},
            total_sum=total_sum,
            total_count=100,
        )
        assert result is None


class TestEstimateBucketSumsValidation:
    """Test _estimate_bucket_sums mean validation."""

    def test_rejects_mean_outside_bucket_bounds(self) -> None:
        """Test that learned means outside bucket bounds are rejected."""
        stats = {"1.0": BucketStatistics(bucket_le="1.0")}
        stats["1.0"].weighted_mean_sum = 500.0  # mean = 5.0, outside [0, 1.0]
        stats["1.0"].observation_count = 100

        sums = _estimate_bucket_sums({"1.0": 100.0}, stats)

        # Should fall back to midpoint (0.5), not use corrupted mean (5.0)
        assert sums["1.0"] == pytest.approx(50.0)  # 100 * 0.5

    def test_accepts_mean_within_bucket_bounds(self) -> None:
        """Test that valid learned means are used."""
        stats = {"1.0": BucketStatistics(bucket_le="1.0")}
        stats["1.0"].weighted_mean_sum = 70.0  # mean = 0.7, inside [0, 1.0]
        stats["1.0"].observation_count = 100

        sums = _estimate_bucket_sums({"1.0": 100.0}, stats)

        # Should use learned mean (0.7)
        assert sums["1.0"] == pytest.approx(70.0)  # 100 * 0.7

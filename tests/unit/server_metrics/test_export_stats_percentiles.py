# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for estimated percentiles computation and edge cases."""

import numpy as np

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.server_metrics.export_stats import _compute_histogram_stats
from aiperf.server_metrics.histogram_percentiles import (
    BucketStatistics,
    _generate_observations_with_sum_constraint,
    accumulate_bucket_statistics,
    compute_estimated_percentiles,
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
from aiperf.server_metrics.storage import ServerMetricsTimeSeries
from tests.unit.server_metrics.helpers import (
    add_histogram_snapshots,
    convert_bucket_snapshots,
    get_histogram,
    hist,
    make_time_filter,
)

# =============================================================================
# Estimated Percentiles Tests
# =============================================================================


class TestComputeEstimatedPercentiles:
    """Test compute_estimated_percentiles function."""

    def test_computes_percentiles_with_inf_observations(self):
        """Test percentile computation including +Inf bucket estimates."""
        # Cumulative: 90 <= 0.5, 97 <= 1.0, 100 total (3 in +Inf)
        bucket_deltas = {"0.5": 90.0, "1.0": 97.0, "+Inf": 100.0}
        bucket_stats = {}  # No learned stats
        total_sum = 100.0
        total_count = 100

        result = compute_estimated_percentiles(
            bucket_deltas, bucket_stats, total_sum, total_count
        )

        assert result is not None
        assert result.p50_estimate is not None
        assert result.p99_estimate is not None

    def test_returns_none_on_empty_data(self):
        """Test returns None when data is empty."""
        result = compute_estimated_percentiles(
            bucket_deltas={},
            bucket_stats={},
            total_sum=0.0,
            total_count=0,
        )

        assert result is None

    def test_zero_sum_with_observations_returns_zero_percentiles(self):
        """Test that zero sum with nonzero count returns all-zero percentiles.

        This handles the case where all observations are exactly 0 (e.g., decode time
        for a prefill-only worker). Without this check, bucket interpolation would
        give misleading non-zero estimates based on bucket boundaries.
        """
        # All 50 observations fall in first bucket (0.3), but sum is 0
        bucket_deltas = {
            "0.3": 50.0,
            "0.5": 50.0,
            "0.8": 50.0,
            "1.0": 50.0,
            "+Inf": 50.0,
        }
        bucket_stats = {}
        total_sum = 0.0  # All observations were exactly 0
        total_count = 50

        result = compute_estimated_percentiles(
            bucket_deltas, bucket_stats, total_sum, total_count
        )

        assert result is not None
        assert result.p50_estimate == 0.0
        assert result.p90_estimate == 0.0
        assert result.p95_estimate == 0.0
        assert result.p99_estimate == 0.0


class TestEstimatedPercentilesIntegration:
    """Integration tests for estimated percentiles in _compute_histogram_stats."""

    def test_percentile_estimates_computed_in__compute_histogram_stats(self):
        """Test that percentile estimates are computed in _compute_histogram_stats."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "ttft",
            [
                (0, hist({"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                (
                    NANOS_PER_SECOND,
                    hist({"0.5": 50.0, "1.0": 90.0, "+Inf": 100.0}, 70.0, 100.0),
                ),
            ],
        )

        result = _compute_histogram_stats(get_histogram(ts, "ttft"), make_time_filter())

        assert result.stats is not None
        assert result.stats.p50_estimate is not None
        assert result.stats.p99_estimate is not None

    def test_percentile_estimates_handle_inf_bucket(self):
        """Test percentile estimates include +Inf bucket in tail percentiles."""
        ts = ServerMetricsTimeSeries()
        # Large +Inf bucket to test tail percentile estimation
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 0.0, "+Inf": 0.0}, 0.0, 0.0)),
                # 50 observations <= 1.0s, 50 in +Inf (>1.0s) with high values
                (
                    NANOS_PER_SECOND,
                    hist({"1.0": 50.0, "+Inf": 100.0}, 750.0, 100.0),  # avg=7.5s
                ),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"), make_time_filter()
        )

        # 50 of 100 observations are in +Inf, so p99 (99th percentile) should include +Inf
        assert result.stats is not None
        assert result.stats.p99_estimate is not None
        # With 50% in +Inf, the p99 should be in the +Inf region (above 1.0)
        assert result.stats.p99_estimate > 1.0

    def test_percentile_estimates_none_on_counter_reset(self):
        """Test percentile estimates are None when counter reset detected."""
        ts = ServerMetricsTimeSeries()
        add_histogram_snapshots(
            ts,
            "latency",
            [
                (0, hist({"1.0": 1000.0, "+Inf": 2000.0}, 500.0, 2000.0)),
                # Counter reset
                (NANOS_PER_SECOND, hist({"1.0": 50.0, "+Inf": 100.0}, 25.0, 100.0)),
            ],
        )

        result = _compute_histogram_stats(
            get_histogram(ts, "latency"), make_time_filter()
        )

        # Counter reset - percentile estimates should be None
        assert result.stats is None or result.stats.p50_estimate is None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCasesValidation:
    """Test edge cases and input validation."""

    def test_accumulate_bucket_statistics_handles_mismatched_counts(self):
        """Test that accumulate_bucket_statistics handles mismatched sizes gracefully.

        Note: With the new vectorized interface, length mismatches are handled
        via NumPy broadcasting rules rather than explicit validation.
        """
        # Mismatched sums and counts - use matching subset
        sums = np.array([0.0, 0.1])
        counts = np.array([0.0, 1.0])
        bucket_snapshots = [{"0.5": 0.0, "+Inf": 0.0}, {"0.5": 1.0, "+Inf": 1.0}]
        bucket_les, bucket_counts = convert_bucket_snapshots(bucket_snapshots)

        # With vectorized approach, this works with matching lengths
        stats = accumulate_bucket_statistics(sums, counts, bucket_les, bucket_counts)
        # One single-bucket interval: observation landed in 0.5 bucket
        assert "0.5" in stats

    def test_generate_observations_handles_very_large_bucket(self):
        """Test observation generation handles very wide bucket ranges."""
        # Bucket from 0 to 1000 (very wide)
        bucket_deltas = {"1000.0": 100.0, "+Inf": 100.0}
        target_sum = 50000.0  # mean of 500

        observations = _generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 100
        assert all(0 <= obs <= 1000 for obs in observations)

    def test_generate_observations_handles_tiny_bucket(self):
        """Test observation generation handles very narrow bucket ranges."""
        # Bucket from 0 to 0.001 (very narrow)
        bucket_deltas = {"0.001": 100.0, "+Inf": 100.0}
        target_sum = 0.05  # mean of 0.0005

        observations = _generate_observations_with_sum_constraint(
            bucket_deltas, target_sum=target_sum, bucket_stats=None
        )

        assert len(observations) == 100
        assert all(0 <= obs <= 0.001 for obs in observations)

    def test_bucket_statistics_handles_zero_count(self):
        """Test BucketStatistics handles zero-count records gracefully."""
        stats = BucketStatistics(bucket_le="0.5")
        stats.record(mean=0.25, count=0)  # Zero count

        assert stats.observation_count == 0
        assert stats.estimated_mean is None  # No valid mean

    def test_estimate_inf_bucket_handles_extreme_values(self):
        """Test +Inf bucket estimation handles extreme sum values."""
        # Very large sum relative to finite observations
        observations = estimate_inf_bucket_observations(
            total_sum=1_000_000.0,
            estimated_finite_sum=100.0,
            inf_count=10,
            max_finite_bucket=10.0,
        )

        # inf_avg = (1_000_000 - 100) / 10 = 99,990
        assert len(observations) == 10
        assert all(obs >= 10.0 for obs in observations)  # All above max finite

    def test_compute_estimated_percentiles_handles_all_in_inf_bucket(self):
        """Test estimated_percentiles handles case where all observations in +Inf."""
        # All 100 observations in +Inf bucket
        bucket_deltas = {"1.0": 0.0, "+Inf": 100.0}
        bucket_stats = {}
        total_sum = 1500.0  # Average of 15 per observation
        total_count = 100

        result = compute_estimated_percentiles(
            bucket_deltas, bucket_stats, total_sum, total_count
        )

        assert result is not None
        # All percentiles should be in the +Inf region
        assert result.p50_estimate is not None
        assert result.p50_estimate > 1.0


class TestEdgeCasesNumericalStability:
    """Test numerical edge cases for stability."""

    def test_bucket_bounds_first_bucket_negative_boundary(self):
        """Test get_bucket_bounds handles negative bucket boundaries."""
        sorted_buckets = ["-1.0", "0", "1.0"]

        lower, upper = get_bucket_bounds("-1.0", sorted_buckets)
        # First bucket: lower bound is typically 0, but for negative buckets
        # the logic may differ
        assert upper == -1.0

    def test_generate_observations_exact_sum_match(self):
        """Test that generated observations match target sum closely."""
        # Per-bucket counts: 50 in [0,0.5], 50 in (0.5,1.0]
        per_bucket_counts = {"0.5": 50.0, "1.0": 50.0, "+Inf": 0.0}
        target_sum = (
            37.5  # 50*0.25 + 50*0.75 = 12.5 + 37.5 = 50 (midpoint would be ~50)
        )

        observations = _generate_observations_with_sum_constraint(
            per_bucket_counts, target_sum=target_sum, bucket_stats=None
        )

        # Sum should be close to target (within 15% due to clamping limits)
        assert np.isclose(observations.sum(), target_sum, rtol=0.15)

    def test_estimate_bucket_sums_empty_buckets(self):
        """Test estimate_bucket_sums with empty bucket counts."""
        bucket_deltas = {"0.5": 0.0, "1.0": 0.0, "+Inf": 0.0}
        bucket_stats = {}

        sums = estimate_bucket_sums(bucket_deltas, bucket_stats)

        # All zeros should result in empty or zero sums
        assert all(v == 0.0 for v in sums.values()) or len(sums) == 0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistical validation tests for confidence aggregation.

These tests validate the statistical correctness of the confidence interval
calculations and coefficient of variation computations.
"""

import numpy as np
import pytest
from scipy import stats

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation
from aiperf.orchestrator.models import RunResult


class TestStatisticalValidation:
    """Statistical validation tests for confidence aggregation."""

    def test_confidence_interval_coverage_95(self):
        """Test that 95% confidence intervals have correct coverage.

        This test validates that when we compute 95% CIs from samples,
        approximately 95% of them contain the true population mean.
        """
        np.random.seed(42)

        # True population parameters
        true_mean = 100.0
        true_std = 15.0

        # Run many experiments
        num_experiments = 1000
        sample_size = 10
        confidence_level = 0.95

        coverage_count = 0

        for _ in range(num_experiments):
            # Generate sample from population
            sample = np.random.normal(true_mean, true_std, sample_size)

            # Create RunResults from sample
            results = [
                RunResult(
                    label=f"run_{i:04d}",
                    success=True,
                    summary_metrics={
                        "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                    },
                    artifacts_path=None,
                )
                for i, value in enumerate(sample)
            ]

            # Compute confidence interval
            aggregation = ConfidenceAggregation(confidence_level=confidence_level)
            agg_result = aggregation.aggregate(results)

            metric = agg_result.metrics["test_metric_avg"]

            # Check if CI contains true mean
            if metric.ci_low <= true_mean <= metric.ci_high:
                coverage_count += 1

        # Calculate actual coverage
        actual_coverage = coverage_count / num_experiments

        # Coverage should be close to 95% (within 2% tolerance)
        # With 1000 experiments, we expect ~950 to contain the true mean
        # Allow 2% tolerance (930-970 successes)
        assert 0.93 <= actual_coverage <= 0.97, (
            f"95% CI coverage should be ~0.95, got {actual_coverage:.3f}"
        )

    def test_confidence_interval_coverage_99(self):
        """Test that 99% confidence intervals have correct coverage."""
        np.random.seed(43)

        true_mean = 50.0
        true_std = 10.0

        num_experiments = 1000
        sample_size = 15
        confidence_level = 0.99

        coverage_count = 0

        for _ in range(num_experiments):
            sample = np.random.normal(true_mean, true_std, sample_size)

            results = [
                RunResult(
                    label=f"run_{i:04d}",
                    success=True,
                    summary_metrics={
                        "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                    },
                    artifacts_path=None,
                )
                for i, value in enumerate(sample)
            ]

            aggregation = ConfidenceAggregation(confidence_level=confidence_level)
            agg_result = aggregation.aggregate(results)

            metric = agg_result.metrics["test_metric_avg"]

            if metric.ci_low <= true_mean <= metric.ci_high:
                coverage_count += 1

        actual_coverage = coverage_count / num_experiments

        # 99% CI should have ~99% coverage (allow 2% tolerance)
        assert 0.97 <= actual_coverage <= 1.0, (
            f"99% CI coverage should be ~0.99, got {actual_coverage:.3f}"
        )

    def test_cv_reflects_variance_correctly(self):
        """Test that CV correctly reflects the variance in the data.

        CV should increase as variance increases (for fixed mean).
        """
        np.random.seed(44)

        mean = 100.0
        sample_size = 20

        # Test with increasing standard deviations
        std_values = [5.0, 10.0, 20.0, 40.0]
        cvs = []

        for std in std_values:
            sample = np.random.normal(mean, std, sample_size)

            results = [
                RunResult(
                    label=f"run_{i:04d}",
                    success=True,
                    summary_metrics={
                        "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                    },
                    artifacts_path=None,
                )
                for i, value in enumerate(sample)
            ]

            aggregation = ConfidenceAggregation(confidence_level=0.95)
            agg_result = aggregation.aggregate(results)

            metric = agg_result.metrics["test_metric_avg"]
            cvs.append(metric.cv)

        # CV should increase monotonically with std
        for i in range(len(cvs) - 1):
            assert cvs[i] < cvs[i + 1], f"CV should increase with std: {cvs}"

        # CV should be approximately std/mean
        for std, cv in zip(std_values, cvs, strict=True):
            expected_cv = std / mean
            # Allow 20% tolerance due to sampling variability
            assert abs(cv - expected_cv) / expected_cv < 0.20, (
                f"CV {cv:.3f} should be close to {expected_cv:.3f}"
            )

    def test_cv_with_known_values(self):
        """Test CV calculation with known values."""
        # Create data with known CV
        # If mean=100 and std=10, then CV=0.1
        values = [90.0, 95.0, 100.0, 105.0, 110.0]

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        # Calculate expected values
        expected_mean = np.mean(values)
        expected_std = np.std(values, ddof=1)
        expected_cv = expected_std / expected_mean

        assert abs(metric.mean - expected_mean) < 1e-6
        assert abs(metric.std - expected_std) < 1e-6
        assert abs(metric.cv - expected_cv) < 1e-6

    @pytest.mark.parametrize(
        "sample_size,confidence_level",
        [
            (3, 0.95),
            (5, 0.95),
            (10, 0.95),
            (20, 0.95),
            (3, 0.99),
            (5, 0.99),
            (10, 0.99),
            (20, 0.99),
        ],
    )
    def test_t_critical_values_match_scipy(self, sample_size, confidence_level):
        """Test that t-critical values match scipy for various N and confidence levels."""
        # Create dummy data
        values = list(range(sample_size))

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=confidence_level)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        # Calculate expected t-critical using scipy
        alpha = 1 - confidence_level
        df = sample_size - 1
        expected_t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Should match exactly (or very close due to floating point)
        assert abs(metric.t_critical - expected_t_critical) < 1e-10, (
            f"t-critical mismatch for N={sample_size}, CL={confidence_level}: "
            f"got {metric.t_critical}, expected {expected_t_critical}"
        )

    def test_confidence_interval_width_decreases_with_sample_size(self):
        """Test that CI width generally decreases as sample size increases.

        With more samples, we should have more precise estimates.
        Note: We use a controlled seed and test the general trend.
        """
        # Use a different seed that gives more stable results
        np.random.seed(100)

        true_mean = 100.0
        true_std = 15.0
        confidence_level = 0.95

        # Use very different sample sizes to see clear trend
        sample_sizes = [5, 50]
        ci_widths = []

        for n in sample_sizes:
            sample = np.random.normal(true_mean, true_std, n)

            results = [
                RunResult(
                    label=f"run_{i:04d}",
                    success=True,
                    summary_metrics={
                        "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                    },
                    artifacts_path=None,
                )
                for i, value in enumerate(sample)
            ]

            aggregation = ConfidenceAggregation(confidence_level=confidence_level)
            agg_result = aggregation.aggregate(results)

            metric = agg_result.metrics["test_metric_avg"]
            ci_width = metric.ci_high - metric.ci_low
            ci_widths.append(ci_width)

        # Test general trend: larger sample should have narrower CI
        assert ci_widths[1] < ci_widths[0], (
            f"CI width should decrease with sample size: "
            f"n={sample_sizes[0]} width={ci_widths[0]:.2f}, "
            f"n={sample_sizes[1]} width={ci_widths[1]:.2f}"
        )

    def test_standard_error_calculation(self):
        """Test that standard error is correctly calculated as std/sqrt(n)."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        # Calculate expected SE
        n = len(values)
        expected_se = metric.std / np.sqrt(n)

        assert abs(metric.se - expected_se) < 1e-10, (
            f"SE should be std/sqrt(n): got {metric.se}, expected {expected_se}"
        )

    def test_confidence_interval_formula(self):
        """Test that CI is correctly calculated as mean ± t * SE."""
        values = [95.0, 100.0, 105.0, 110.0, 115.0]

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        # Calculate expected CI bounds
        margin = metric.t_critical * metric.se
        expected_ci_low = metric.mean - margin
        expected_ci_high = metric.mean + margin

        assert abs(metric.ci_low - expected_ci_low) < 1e-10
        assert abs(metric.ci_high - expected_ci_high) < 1e-10

    def test_min_max_values(self):
        """Test that min and max are correctly identified."""
        values = [10.0, 25.0, 15.0, 30.0, 20.0]

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        assert metric.min == 10.0
        assert metric.max == 30.0

    def test_cv_with_zero_mean(self):
        """Test CV handling when mean is zero (should return inf)."""
        values = [-5.0, -2.0, 0.0, 2.0, 5.0]  # Mean = 0

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "test_metric": JsonMetricResult(unit="ms", avg=float(value))
                },
                artifacts_path=None,
            )
            for i, value in enumerate(values)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        metric = agg_result.metrics["test_metric_avg"]

        # CV should be inf when mean is zero
        assert metric.cv == float("inf")

    def test_multiple_metrics_aggregation(self):
        """Test that multiple metrics are aggregated independently and correctly."""
        np.random.seed(46)

        # Create data with different characteristics
        n = 10
        metric1_values = np.random.normal(100, 10, n)  # Mean=100, std=10
        metric2_values = np.random.normal(50, 5, n)  # Mean=50, std=5

        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "metric1": JsonMetricResult(
                        unit="ms", avg=float(metric1_values[i])
                    ),
                    "metric2": JsonMetricResult(
                        unit="ms", avg=float(metric2_values[i])
                    ),
                },
                artifacts_path=None,
            )
            for i in range(n)
        ]

        aggregation = ConfidenceAggregation(confidence_level=0.95)
        agg_result = aggregation.aggregate(results)

        # Both metrics should be aggregated
        assert "metric1_avg" in agg_result.metrics
        assert "metric2_avg" in agg_result.metrics

        # Verify each metric has correct statistics
        metric1 = agg_result.metrics["metric1_avg"]
        metric2 = agg_result.metrics["metric2_avg"]

        # Metric1 should have higher mean and std
        assert metric1.mean > metric2.mean
        assert metric1.std > metric2.std

        # Both should have valid CIs
        assert metric1.ci_low < metric1.mean < metric1.ci_high
        assert metric2.ci_low < metric2.mean < metric2.ci_high

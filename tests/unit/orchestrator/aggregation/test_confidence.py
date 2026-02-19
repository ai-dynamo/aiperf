# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for confidence aggregation strategy."""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.aggregation.confidence import (
    ConfidenceAggregation,
    ConfidenceMetric,
)
from aiperf.orchestrator.models import RunResult


class TestConfidenceAggregation:
    """Tests for ConfidenceAggregation strategy."""

    def test_get_aggregation_type(self):
        """Test get_aggregation_type returns 'confidence'."""
        strategy = ConfidenceAggregation()
        assert strategy.get_aggregation_type() == "confidence"

    def test_aggregate_with_known_values(self, tmp_path):
        """Test aggregation with known values."""
        strategy = ConfidenceAggregation(confidence_level=0.95)

        # Create results with known values
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={
                    "ttft": JsonMetricResult(unit="ms", avg=100.0),
                    "tpot": JsonMetricResult(unit="ms", avg=10.0),
                },
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={
                    "ttft": JsonMetricResult(unit="ms", avg=110.0),
                    "tpot": JsonMetricResult(unit="ms", avg=12.0),
                },
                artifacts_path=tmp_path / "run_0002",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={
                    "ttft": JsonMetricResult(unit="ms", avg=105.0),
                    "tpot": JsonMetricResult(unit="ms", avg=11.0),
                },
                artifacts_path=tmp_path / "run_0003",
            ),
        ]

        aggregate = strategy.aggregate(results)

        # Verify basic structure
        assert aggregate.aggregation_type == "confidence"
        assert aggregate.num_runs == 3
        assert aggregate.num_successful_runs == 3
        assert len(aggregate.failed_runs) == 0

        # Verify ttft_avg metrics
        ttft_metric = aggregate.metrics["ttft_avg"]
        assert ttft_metric.mean == pytest.approx(105.0)
        assert ttft_metric.std == pytest.approx(5.0)
        assert ttft_metric.min == 100.0
        assert ttft_metric.max == 110.0

        # Verify tpot_avg metrics
        tpot_metric = aggregate.metrics["tpot_avg"]
        assert tpot_metric.mean == pytest.approx(11.0)
        assert tpot_metric.std == pytest.approx(1.0)

    def test_aggregate_with_failed_runs(self):
        """Test aggregation excludes failed runs."""
        strategy = ConfidenceAggregation()

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=False,
                error="Connection timeout",
                artifacts_path=Path("/tmp/run_0002"),
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=110.0)},
                artifacts_path=Path("/tmp/run_0003"),
            ),
        ]

        aggregate = strategy.aggregate(results)

        # Verify counts
        assert aggregate.num_runs == 3
        assert aggregate.num_successful_runs == 2
        assert len(aggregate.failed_runs) == 1
        assert aggregate.failed_runs[0]["label"] == "run_0002"
        assert aggregate.failed_runs[0]["error"] == "Connection timeout"

        # Verify metrics computed from successful runs only
        ttft_metric = aggregate.metrics["ttft_avg"]
        assert ttft_metric.mean == pytest.approx(105.0)

    def test_aggregate_error_with_insufficient_runs(self):
        """Test aggregation raises error with < 2 successful runs."""
        strategy = ConfidenceAggregation()

        # Only one successful run
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            ),
        ]

        with pytest.raises(ValueError, match="Insufficient successful runs"):
            strategy.aggregate(results)

    def test_aggregate_error_with_all_failed_runs(self):
        """Test aggregation raises error when all runs failed."""
        strategy = ConfidenceAggregation()

        results = [
            RunResult(
                label="run_0001",
                success=False,
                error="Error 1",
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=False,
                error="Error 2",
                artifacts_path=Path("/tmp/run_0002"),
            ),
        ]

        with pytest.raises(ValueError, match="All runs failed"):
            strategy.aggregate(results)

    def test_t_critical_value_computation(self):
        """Test t-critical value matches scipy for various N and confidence levels."""
        # Test with N=3, confidence=0.95
        strategy = ConfidenceAggregation(confidence_level=0.95)
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=float(i))},
                artifacts_path=Path(f"/tmp/run_{i:04d}"),
            )
            for i in range(1, 4)
        ]

        aggregate = strategy.aggregate(results)
        metric = aggregate.metrics["metric_avg"]

        # Compute expected t-critical
        n = 3
        df = n - 1
        alpha = 1 - 0.95
        expected_t_critical = stats.t.ppf(1 - alpha / 2, df)

        assert metric.t_critical == pytest.approx(expected_t_critical)

        # Test with N=10, confidence=0.99
        strategy = ConfidenceAggregation(confidence_level=0.99)
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=float(i))},
                artifacts_path=Path(f"/tmp/run_{i:04d}"),
            )
            for i in range(1, 11)
        ]

        aggregate = strategy.aggregate(results)
        metric = aggregate.metrics["metric_avg"]

        n = 10
        df = n - 1
        alpha = 1 - 0.99
        expected_t_critical = stats.t.ppf(1 - alpha / 2, df)

        assert metric.t_critical == pytest.approx(expected_t_critical)

    def test_cv_computation(self):
        """Test coefficient of variation computation."""
        strategy = ConfidenceAggregation()

        # Test with known values
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=110.0)},
                artifacts_path=Path("/tmp/run_0002"),
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=105.0)},
                artifacts_path=Path("/tmp/run_0003"),
            ),
        ]

        aggregate = strategy.aggregate(results)
        metric = aggregate.metrics["metric_avg"]

        # CV = std / mean (as a ratio, not percentage)
        expected_cv = metric.std / metric.mean
        assert metric.cv == pytest.approx(expected_cv)

    def test_cv_division_by_zero_handling(self):
        """Test CV handles division by zero when mean is zero."""
        strategy = ConfidenceAggregation()

        # Create results with zero mean
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=0.0)},
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=0.0)},
                artifacts_path=Path("/tmp/run_0002"),
            ),
        ]

        aggregate = strategy.aggregate(results)
        metric = aggregate.metrics["metric_avg"]

        # CV should be inf when mean is 0 (division by zero)
        assert metric.cv == float("inf")

    def test_confidence_interval_bounds(self):
        """Test confidence interval bounds are computed correctly."""
        strategy = ConfidenceAggregation(confidence_level=0.95)

        values = [100.0, 110.0, 105.0, 95.0, 108.0]
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=val)},
                artifacts_path=Path(f"/tmp/run_{i:04d}"),
            )
            for i, val in enumerate(values, 1)
        ]

        aggregate = strategy.aggregate(results)
        metric = aggregate.metrics["metric_avg"]

        # Manually compute expected values
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        se = std / np.sqrt(n)
        df = n - 1
        alpha = 1 - 0.95
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        margin = t_critical * se

        expected_ci_low = mean - margin
        expected_ci_high = mean + margin

        assert metric.mean == pytest.approx(mean)
        assert metric.std == pytest.approx(std)
        assert metric.se == pytest.approx(se)
        assert metric.ci_low == pytest.approx(expected_ci_low)
        assert metric.ci_high == pytest.approx(expected_ci_high)

    def test_unit_preservation(self):
        """Test that units are preserved from extraction."""
        strategy = ConfidenceAggregation()

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=100.0),
                    "inter_token_latency": JsonMetricResult(unit="ms", avg=10.0),
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=25.0
                    ),
                    "output_token_throughput": JsonMetricResult(
                        unit="tokens/sec", avg=500.0
                    ),
                },
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=110.0),
                    "inter_token_latency": JsonMetricResult(unit="ms", avg=12.0),
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=27.0
                    ),
                    "output_token_throughput": JsonMetricResult(
                        unit="tokens/sec", avg=520.0
                    ),
                },
                artifacts_path=Path("/tmp/run_0002"),
            ),
        ]

        aggregate = strategy.aggregate(results)

        # Verify units are preserved from JsonMetricResult objects
        assert aggregate.metrics["time_to_first_token_avg"].unit == "ms"
        assert aggregate.metrics["inter_token_latency_avg"].unit == "ms"
        assert aggregate.metrics["request_throughput_avg"].unit == "requests/sec"
        assert aggregate.metrics["output_token_throughput_avg"].unit == "tokens/sec"

    def test_non_standard_percentiles_preserve_units(self):
        """Test that non-standard percentiles (p1, p5, p10, p25, p75) preserve units correctly."""
        strategy = ConfidenceAggregation()

        # Include non-standard percentiles that were previously causing warnings
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={
                    "request_latency": JsonMetricResult(
                        unit="ms",
                        p1=50.0,
                        p5=75.0,
                        p10=90.0,
                        p25=110.0,
                        p75=180.0,
                        p90=220.0,
                    ),
                },
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={
                    "request_latency": JsonMetricResult(
                        unit="ms",
                        p1=52.0,
                        p5=77.0,
                        p10=92.0,
                        p25=112.0,
                        p75=182.0,
                        p90=222.0,
                    ),
                },
                artifacts_path=Path("/tmp/run_0002"),
            ),
        ]

        aggregate = strategy.aggregate(results)

        # Verify all percentiles preserve units correctly without warnings
        assert aggregate.metrics["request_latency_p1"].unit == "ms"
        assert aggregate.metrics["request_latency_p5"].unit == "ms"
        assert aggregate.metrics["request_latency_p10"].unit == "ms"
        assert aggregate.metrics["request_latency_p25"].unit == "ms"
        assert aggregate.metrics["request_latency_p75"].unit == "ms"
        assert aggregate.metrics["request_latency_p90"].unit == "ms"

    def test_metadata_includes_confidence_level(self):
        """Test metadata includes confidence level."""
        strategy = ConfidenceAggregation(confidence_level=0.99)

        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"metric": JsonMetricResult(unit="ms", avg=110.0)},
                artifacts_path=Path("/tmp/run_0002"),
            ),
        ]

        aggregate = strategy.aggregate(results)

        assert "confidence_level" in aggregate.metadata
        assert aggregate.metadata["confidence_level"] == 0.99

    def test_invalid_confidence_level_too_low(self):
        """Test that confidence level <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence level"):
            ConfidenceAggregation(confidence_level=0.0)

        with pytest.raises(ValueError, match="Invalid confidence level"):
            ConfidenceAggregation(confidence_level=-0.1)

    def test_invalid_confidence_level_too_high(self):
        """Test that confidence level >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence level"):
            ConfidenceAggregation(confidence_level=1.0)

        with pytest.raises(ValueError, match="Invalid confidence level"):
            ConfidenceAggregation(confidence_level=1.5)


class TestConfidenceMetric:
    """Tests for ConfidenceMetric data model."""

    def test_create_confidence_metric(self):
        """Test creating a ConfidenceMetric."""
        metric = ConfidenceMetric(
            mean=105.0,
            std=5.0,
            min=100.0,
            max=110.0,
            cv=4.76,
            se=2.89,
            ci_low=98.5,
            ci_high=111.5,
            t_critical=2.262,
            unit="ms",
        )

        assert metric.mean == 105.0
        assert metric.std == 5.0
        assert metric.min == 100.0
        assert metric.max == 110.0
        assert metric.cv == 4.76
        assert metric.se == 2.89
        assert metric.ci_low == 98.5
        assert metric.ci_high == 111.5
        assert metric.t_critical == 2.262
        assert metric.unit == "ms"

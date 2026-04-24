# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CIWidthConvergence criterion.

Feature: adaptive-sweep-and-detailed-aggregation
Property 3: CIWidthConvergence decision matches CI width ratio vs threshold
"""

import math

import numpy as np
import pytest
from scipy.stats import t as t_dist

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.convergence.ci_width import CIWidthConvergence
from aiperf.orchestrator.models import RunResult


class TestCIWidthConvergence:
    """Tests for CIWidthConvergence.is_converged."""

    def test_known_values_converged(self, make_results) -> None:
        # [100, 102, 98, 101, 99] at 95%: ratio ~0.0393, well below 0.10
        criterion = CIWidthConvergence(metric="time_to_first_token", threshold=0.10)
        results = make_results([100, 102, 98, 101, 99])
        assert criterion.is_converged(results) is True

    def test_known_values_hand_computed_ci(self, make_results) -> None:
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / math.sqrt(n)
        t_crit = t_dist.ppf(0.975, df=n - 1)
        expected_ratio = (2 * t_crit * se) / mean

        # Verify the ratio is what we expect (~0.0393)
        assert expected_ratio == pytest.approx(0.03926, rel=1e-3)

        # Threshold just above the ratio -> converged
        criterion = CIWidthConvergence(
            metric="time_to_first_token", threshold=expected_ratio + 0.001
        )
        results = make_results(values)
        assert criterion.is_converged(results) is True

    def test_high_variance_not_converged(self, make_results) -> None:
        # [80, 90, 100, 110, 120]: ratio ~0.393, above 0.10
        criterion = CIWidthConvergence(metric="time_to_first_token", threshold=0.10)
        results = make_results([80, 90, 100, 110, 120])
        assert criterion.is_converged(results) is False

    def test_tight_values_converged(self, make_results) -> None:
        criterion = CIWidthConvergence(metric="time_to_first_token", threshold=0.10)
        results = make_results([100.0, 100.1, 99.9, 100.0, 100.05])
        assert criterion.is_converged(results) is True

    def test_fewer_than_min_runs_returns_false(self, make_results) -> None:
        criterion = CIWidthConvergence(
            metric="time_to_first_token", min_runs=3, threshold=1.0
        )
        results = make_results([100, 100])
        assert criterion.is_converged(results) is False

    def test_mean_zero_returns_false(self, make_results) -> None:
        criterion = CIWidthConvergence(metric="time_to_first_token", threshold=1.0)
        results = make_results([0.0, 0.0, 0.0])
        assert criterion.is_converged(results) is False

    def test_missing_metric_in_some_runs_excluded(self) -> None:
        criterion = CIWidthConvergence(
            metric="time_to_first_token", threshold=0.10, min_runs=3
        )
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=100.0)
                },
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={
                    "request_latency": JsonMetricResult(unit="ms", avg=500.0)
                },
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=100.5)
                },
            ),
            RunResult(
                label="run_0004",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=99.5)
                },
            ),
        ]
        # 3 runs have the metric (run_0002 excluded), tight values -> converged
        assert criterion.is_converged(results) is True

    def test_single_run_with_metric_returns_false(self) -> None:
        criterion = CIWidthConvergence(
            metric="time_to_first_token", threshold=1.0, min_runs=1
        )
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=100.0)
                },
            ),
        ]
        assert criterion.is_converged(results) is False

    def test_all_failed_runs_returns_false(self) -> None:
        criterion = CIWidthConvergence(metric="time_to_first_token", threshold=1.0)
        results = [RunResult(label=f"run_{i:04d}", success=False) for i in range(5)]
        assert criterion.is_converged(results) is False

    def test_exactly_at_threshold_not_converged(self, make_results) -> None:
        # Compute the exact ratio for [100, 102, 98, 101, 99] and use it as threshold
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / math.sqrt(n)
        t_crit = t_dist.ppf(0.975, df=n - 1)
        exact_ratio = (2 * t_crit * se) / mean

        # threshold == ratio -> not converged (strict <)
        criterion = CIWidthConvergence(
            metric="time_to_first_token", threshold=exact_ratio
        )
        results = make_results(values)
        assert criterion.is_converged(results) is False

    def test_stat_p99_reads_correct_field(self) -> None:
        criterion = CIWidthConvergence(
            metric="request_latency", stat="p99", threshold=0.10, min_runs=3
        )
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "request_latency": JsonMetricResult(unit="ms", p99=val)
                },
            )
            for i, val in enumerate([200.0, 201.0, 199.0])
        ]
        assert criterion.is_converged(results) is True

    def test_stat_missing_from_metric_result_excluded(self) -> None:
        criterion = CIWidthConvergence(
            metric="time_to_first_token", stat="p99", threshold=0.10, min_runs=3
        )
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(unit="ms", avg=100.0)
                },
            )
            for i in range(5)
        ]
        # p99 is None on all runs -> 0 values extracted -> not converged
        assert criterion.is_converged(results) is False

    def test_custom_confidence_level(self, make_results) -> None:
        # 99% confidence -> wider CI -> harder to converge
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        criterion_95 = CIWidthConvergence(
            metric="time_to_first_token", confidence_level=0.95, threshold=0.05
        )
        criterion_99 = CIWidthConvergence(
            metric="time_to_first_token", confidence_level=0.99, threshold=0.05
        )
        results = make_results(values)
        # 95% converges at 0.05 (ratio ~0.039), 99% may not
        assert criterion_95.is_converged(results) is True
        assert criterion_99.is_converged(results) is False

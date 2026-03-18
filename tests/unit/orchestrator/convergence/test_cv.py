# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CVConvergence criterion.

Feature: adaptive-sweep-and-detailed-aggregation
Property 4: CVConvergence decision matches CV vs threshold
"""

import numpy as np
import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.convergence.cv import CVConvergence
from aiperf.orchestrator.models import RunResult


class TestCVConvergence:
    """Tests for CVConvergence.is_converged."""

    def test_known_values_hand_computed_cv(self, make_results) -> None:
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        expected_cv = float(np.std(values, ddof=1) / np.mean(values))
        # ~0.0158, well below default 0.05
        assert expected_cv == pytest.approx(0.01581, rel=1e-3)

        criterion = CVConvergence(
            metric="time_to_first_token", threshold=expected_cv + 0.001
        )
        results = make_results(values)
        assert criterion.is_converged(results) is True

    def test_cv_above_threshold_not_converged(self, make_results) -> None:
        # [80, 90, 100, 110, 120]: CV ~0.158, above 0.05
        criterion = CVConvergence(metric="time_to_first_token", threshold=0.05)
        results = make_results([80, 90, 100, 110, 120])
        assert criterion.is_converged(results) is False

    def test_cv_below_threshold_converged(self, make_results) -> None:
        criterion = CVConvergence(metric="time_to_first_token", threshold=0.05)
        results = make_results([100.0, 100.1, 99.9, 100.0, 100.05])
        assert criterion.is_converged(results) is True

    def test_mean_zero_returns_false(self, make_results) -> None:
        criterion = CVConvergence(metric="time_to_first_token", threshold=1.0)
        results = make_results([0.0, 0.0, 0.0])
        assert criterion.is_converged(results) is False

    def test_fewer_than_min_runs_returns_false(self, make_results) -> None:
        criterion = CVConvergence(
            metric="time_to_first_token", min_runs=3, threshold=1.0
        )
        results = make_results([100, 100])
        assert criterion.is_converged(results) is False

    def test_missing_metric_excluded(self) -> None:
        criterion = CVConvergence(
            metric="time_to_first_token", threshold=0.05, min_runs=3
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
        # 3 runs have the metric, tight values -> converged
        assert criterion.is_converged(results) is True

    def test_all_failed_runs_returns_false(self) -> None:
        criterion = CVConvergence(metric="time_to_first_token", threshold=1.0)
        results = [RunResult(label=f"run_{i:04d}", success=False) for i in range(5)]
        assert criterion.is_converged(results) is False

    def test_exactly_at_threshold_not_converged(self, make_results) -> None:
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        exact_cv = float(np.std(values, ddof=1) / np.mean(values))

        # threshold == cv -> not converged (strict <)
        criterion = CVConvergence(metric="time_to_first_token", threshold=exact_cv)
        results = make_results(values)
        assert criterion.is_converged(results) is False

    def test_avg_none_excluded(self) -> None:
        criterion = CVConvergence(
            metric="time_to_first_token", threshold=0.05, min_runs=3
        )
        results = [
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={"time_to_first_token": JsonMetricResult(unit="ms")},
            )
            for i in range(5)
        ]
        # avg is None on all runs -> 0 values extracted -> not converged
        assert criterion.is_converged(results) is False

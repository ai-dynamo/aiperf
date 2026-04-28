# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strategy-level aggregation (confidence, sweep, sweep+confidence).

These tests validate that each strategy's aggregate() method produces correct
results. Previously, aggregation was tested via orchestrator.aggregate_results()
which has been removed in favor of strategy-owned polymorphism.
"""

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import (
    FixedTrialsStrategy,
    ParameterSweepStrategy,
    SweepConfidenceStrategy,
    SweepMode,
)


class TestFixedTrialsAggregation:
    """Test FixedTrialsStrategy.aggregate() for confidence-only mode."""

    @pytest.fixture
    def config(self) -> UserConfig:
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.confidence_level = 0.95
        config.loadgen.profile_run_cooldown_seconds = 10
        config.loadgen.num_profile_runs = 3
        return config

    @pytest.fixture
    def strategy(self) -> FixedTrialsStrategy:
        return FixedTrialsStrategy(num_trials=3, cooldown_seconds=10.0)

    @pytest.fixture
    def successful_results(self) -> list[RunResult]:
        return [
            RunResult(
                label=f"trial_{i + 1:04d}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=100.0 + i,
                        std=5.0,
                        min=95.0,
                        max=105.0,
                    ),
                },
                metadata={"trial_index": i},
            )
            for i in range(3)
        ]

    def test_aggregate_returns_confidence_result(
        self, strategy, successful_results, config
    ):
        aggregate = strategy.aggregate(successful_results, config)

        assert aggregate is not None
        assert aggregate.aggregation_type == "confidence"
        assert aggregate.num_runs == 3
        assert aggregate.num_successful_runs == 3

    def test_aggregate_includes_cooldown_metadata(
        self, strategy, successful_results, config
    ):
        aggregate = strategy.aggregate(successful_results, config)

        assert aggregate is not None
        assert "cooldown_seconds" in aggregate.metadata
        assert aggregate.metadata["cooldown_seconds"] == 10

    def test_aggregate_returns_none_with_insufficient_runs(self, strategy, config):
        results = [
            RunResult(
                label="trial_0001",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                },
            )
        ]
        aggregate = strategy.aggregate(results, config)
        assert aggregate is None

    def test_aggregate_filters_failed_runs(self, strategy, config):
        results = [
            RunResult(
                label="trial_0001",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0, std=5.0
                    ),
                },
            ),
            RunResult(label="trial_0002", success=False, error="timeout"),
            RunResult(
                label="trial_0003",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=102.0, std=5.0
                    ),
                },
            ),
        ]
        aggregate = strategy.aggregate(results, config)
        # 2 successful runs is enough for confidence
        assert aggregate is not None
        assert aggregate.num_successful_runs == 2


class TestParameterSweepAnalyzer:
    """Test ParameterSweepStrategy.aggregate() for sweep-only mode."""

    @pytest.fixture
    def config(self) -> UserConfig:
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]
        return config

    @pytest.fixture
    def strategy(self) -> ParameterSweepStrategy:
        return ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

    @pytest.fixture
    def sweep_results(self) -> list[RunResult]:
        return [
            RunResult(
                label=f"concurrency_{c}",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=float(c * 10),
                    ),
                    "time_to_first_token": JsonMetricResult(
                        unit="ms",
                        avg=50.0 + c,
                    ),
                },
                metadata={"concurrency": c, "value_index": i},
            )
            for i, c in enumerate([10, 20, 30])
        ]

    def test_aggregate_returns_sweep_result(self, strategy, sweep_results, config):
        aggregate = strategy.aggregate(sweep_results, config)

        assert aggregate is not None
        assert aggregate.aggregation_type == "sweep"
        assert aggregate.num_runs == 3
        assert aggregate.num_successful_runs == 3

    def test_aggregate_includes_best_configurations(
        self, strategy, sweep_results, config
    ):
        aggregate = strategy.aggregate(sweep_results, config)

        assert aggregate is not None
        assert "best_configurations" in aggregate.metadata
        assert "pareto_optimal" in aggregate.metadata

    def test_aggregate_returns_none_with_no_successful_runs(self, strategy, config):
        results = [
            RunResult(
                label="concurrency_10",
                success=False,
                error="timeout",
                metadata={"concurrency": 10},
            ),
        ]
        aggregate = strategy.aggregate(results, config)
        assert aggregate is None

    def test_aggregate_skips_failed_runs(self, strategy, config):
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                },
                metadata={"concurrency": 10, "value_index": 0},
            ),
            RunResult(
                label="concurrency_20",
                success=False,
                error="timeout",
                metadata={"concurrency": 20, "value_index": 1},
            ),
            RunResult(
                label="concurrency_30",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=300.0
                    ),
                },
                metadata={"concurrency": 30, "value_index": 2},
            ),
        ]
        aggregate = strategy.aggregate(results, config)
        assert aggregate is not None
        assert aggregate.num_runs == 3
        assert aggregate.num_successful_runs == 2


class TestSweepConfidenceAggregation:
    """Test SweepConfidenceStrategy.aggregate() for sweep + confidence mode."""

    @pytest.fixture
    def config(self) -> UserConfig:
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.confidence_level = 0.95
        config.loadgen.num_profile_runs = 2
        return config

    @pytest.fixture
    def strategy(self) -> SweepConfidenceStrategy:
        return SweepConfidenceStrategy(
            sweep=ParameterSweepStrategy(
                parameter_name="concurrency", parameter_values=[10, 20, 30]
            ),
            confidence=FixedTrialsStrategy(num_trials=2),
            mode=SweepMode.REPEATED,
        )

    @pytest.fixture
    def sweep_confidence_results(self) -> list[RunResult]:
        results = []
        for c in [10, 20, 30]:
            for trial in range(2):
                results.append(
                    RunResult(
                        label=f"concurrency_{c}_trial_{trial + 1:04d}",
                        success=True,
                        summary_metrics={
                            "request_throughput": JsonMetricResult(
                                unit="requests/sec",
                                avg=100.0 + c + trial,
                                std=5.0,
                            ),
                        },
                        metadata={
                            "concurrency": c,
                            "trial_index": trial,
                            "value_index": [10, 20, 30].index(c),
                            "sweep_mode": "repeated",
                        },
                    )
                )
        return results

    def test_aggregate_returns_sweep_result(
        self, strategy, sweep_confidence_results, config
    ):
        aggregate = strategy.aggregate(sweep_confidence_results, config)

        assert aggregate is not None
        assert aggregate.aggregation_type == "sweep"

    def test_aggregate_includes_per_value_aggregates(
        self, strategy, sweep_confidence_results, config
    ):
        aggregate = strategy.aggregate(sweep_confidence_results, config)

        assert aggregate is not None
        per_value = aggregate.metadata.get("per_value_aggregates", {})
        assert len(per_value) == 3
        assert 10 in per_value
        assert 20 in per_value
        assert 30 in per_value

    def test_aggregate_includes_best_configurations(
        self, strategy, sweep_confidence_results, config
    ):
        aggregate = strategy.aggregate(sweep_confidence_results, config)

        assert aggregate is not None
        assert "best_configurations" in aggregate.metadata
        assert "pareto_optimal" in aggregate.metadata

    def test_aggregate_skips_values_with_insufficient_trials(self, strategy, config):
        """Values with < 2 successful trials are skipped."""
        results = []
        # concurrency=10: 2 successful trials
        for trial in range(2):
            results.append(
                RunResult(
                    label=f"concurrency_10_trial_{trial + 1:04d}",
                    success=True,
                    summary_metrics={
                        "request_throughput": JsonMetricResult(
                            unit="requests/sec",
                            avg=110.0 + trial,
                            std=5.0,
                        ),
                    },
                    metadata={
                        "concurrency": 10,
                        "trial_index": trial,
                        "sweep_mode": "repeated",
                    },
                )
            )
        # concurrency=20: 1 successful + 1 failed
        results.append(
            RunResult(
                label="concurrency_20_trial_0001",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=120.0, std=5.0
                    ),
                },
                metadata={
                    "concurrency": 20,
                    "trial_index": 0,
                    "sweep_mode": "repeated",
                },
            )
        )
        results.append(
            RunResult(
                label="concurrency_20_trial_0002",
                success=False,
                error="timeout",
                metadata={
                    "concurrency": 20,
                    "trial_index": 1,
                    "sweep_mode": "repeated",
                },
            )
        )
        # concurrency=30: 2 successful trials
        for trial in range(2):
            results.append(
                RunResult(
                    label=f"concurrency_30_trial_{trial + 1:04d}",
                    success=True,
                    summary_metrics={
                        "request_throughput": JsonMetricResult(
                            unit="requests/sec",
                            avg=130.0 + trial,
                            std=5.0,
                        ),
                    },
                    metadata={
                        "concurrency": 30,
                        "trial_index": trial,
                        "sweep_mode": "repeated",
                    },
                )
            )

        aggregate = strategy.aggregate(results, config)
        assert aggregate is not None
        per_value = aggregate.metadata.get("per_value_aggregates", {})
        # Only 10 and 30 should have aggregates (20 had only 1 successful)
        assert len(per_value) == 2
        assert 10 in per_value
        assert 30 in per_value
        assert 20 not in per_value

    def test_aggregate_returns_none_when_no_values_have_enough_trials(
        self, strategy, config
    ):
        results = [
            RunResult(
                label="concurrency_10_trial_0001",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                },
                metadata={
                    "concurrency": 10,
                    "trial_index": 0,
                    "sweep_mode": "repeated",
                },
            ),
        ]
        aggregate = strategy.aggregate(results, config)
        assert aggregate is None


class TestCollectFailedValues:
    """Test collect_failed_values on sweep strategies."""

    def test_parameter_sweep_collects_failed_values(self):
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )
        results = [
            RunResult(
                label="concurrency_10", success=True, metadata={"concurrency": 10}
            ),
            RunResult(
                label="concurrency_20",
                success=False,
                error="Connection timeout",
                metadata={"concurrency": 20},
            ),
            RunResult(
                label="concurrency_30", success=True, metadata={"concurrency": 30}
            ),
        ]
        failed = strategy.collect_failed_values(results)
        assert len(failed) == 1
        assert failed[0]["value"] == 20
        assert failed[0]["parameter_name"] == "concurrency"

    def test_parameter_sweep_deduplicates_failures(self):
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20]
        )
        results = [
            RunResult(
                label="r1", success=False, error="err1", metadata={"concurrency": 20}
            ),
            RunResult(
                label="r2", success=False, error="err2", metadata={"concurrency": 20}
            ),
        ]
        failed = strategy.collect_failed_values(results)
        assert len(failed) == 1

    def test_sweep_confidence_collects_failed_values(self):
        strategy = SweepConfidenceStrategy(
            sweep=ParameterSweepStrategy(
                parameter_name="concurrency", parameter_values=[10, 20]
            ),
            confidence=FixedTrialsStrategy(num_trials=2),
            mode=SweepMode.REPEATED,
        )
        results = [
            RunResult(label="r1", success=True, metadata={"concurrency": 10}),
            RunResult(
                label="r2", success=False, error="timeout", metadata={"concurrency": 20}
            ),
        ]
        failed = strategy.collect_failed_values(results)
        assert len(failed) == 1
        assert failed[0]["value"] == 20

    def test_fixed_trials_returns_empty(self):
        strategy = FixedTrialsStrategy(num_trials=3)
        results = [
            RunResult(label="r1", success=False, error="err"),
        ]
        failed = strategy.collect_failed_values(results)
        assert failed == []

    def test_no_failures_returns_empty(self):
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20]
        )
        results = [
            RunResult(label="r1", success=True, metadata={"concurrency": 10}),
            RunResult(label="r2", success=True, metadata={"concurrency": 20}),
        ]
        failed = strategy.collect_failed_values(results)
        assert failed == []

    def test_ignores_results_without_sweep_metadata(self):
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20]
        )
        results = [
            RunResult(
                label="r1", success=False, error="err", metadata={"concurrency": 20}
            ),
            RunResult(label="r2", success=False, error="err", metadata={}),
        ]
        failed = strategy.collect_failed_values(results)
        assert len(failed) == 1
        assert failed[0]["value"] == 20

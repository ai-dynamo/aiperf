# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator aggregation and export delegation.

The orchestrator delegates aggregation and export to strategies. These tests
verify the delegation flow: orchestrator calls strategy.aggregate() and
strategy.export_aggregates() with correct arguments.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import (
    FixedTrialsStrategy,
    ParameterSweepStrategy,
    SweepConfidenceStrategy,
    SweepMode,
)


class TestOrchestratorAggregationDelegation:
    """Tests that execute_and_export delegates aggregation to strategy."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    @pytest.fixture
    def mock_results(self, tmp_path):
        return [
            RunResult(
                label="trial_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "trial_0001",
            ),
            RunResult(
                label="trial_0002",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=105.0)},
                artifacts_path=tmp_path / "trial_0002",
            ),
        ]

    def test_execute_and_export_calls_strategy_aggregate(
        self, orchestrator, mock_results
    ):
        """Verify orchestrator calls strategy.aggregate() with results and config."""
        mock_strategy = Mock(spec=FixedTrialsStrategy)
        mock_strategy.execute.return_value = None
        # _execute_loop calls should_continue: while(True), run, cooldown(True), while(True), run, cooldown(False), while(False)
        mock_strategy.should_continue.side_effect = [True, True, True, False, False]
        mock_strategy.get_next_config.side_effect = lambda c, r: c
        mock_strategy.get_run_label.side_effect = ["trial_0001", "trial_0002"]
        mock_strategy.get_run_path.side_effect = [Path("/tmp/r1"), Path("/tmp/r2")]
        mock_strategy.tag_result.side_effect = lambda r, i: r
        mock_strategy.collect_failed_values.return_value = []
        mock_strategy.get_cooldown_seconds.return_value = 0.0
        mock_strategy.aggregate.return_value = None

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            orchestrator.execute_and_export(config, strategy=mock_strategy)

        mock_strategy.aggregate.assert_called_once()
        call_args = mock_strategy.aggregate.call_args
        assert len(call_args[0][0]) == 2  # results list
        assert call_args[0][1] is config  # config

    def test_execute_and_export_calls_export_when_aggregate_exists(
        self, orchestrator, tmp_path
    ):
        """Verify orchestrator calls strategy.export_aggregates() when aggregate is not None."""
        mock_aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=2,
            num_successful_runs=2,
            failed_runs=[],
            metadata={},
            metrics={},
        )

        mock_strategy = Mock(spec=FixedTrialsStrategy)
        mock_strategy.execute.return_value = None
        mock_strategy.should_continue.side_effect = [True, False, False]
        mock_strategy.get_next_config.side_effect = lambda c, r: c
        mock_strategy.get_run_label.return_value = "trial_0001"
        mock_strategy.get_run_path.return_value = Path("/tmp/r1")
        mock_strategy.tag_result.side_effect = lambda r, i: r
        mock_strategy.collect_failed_values.return_value = []
        mock_strategy.get_cooldown_seconds.return_value = 0.0
        mock_strategy.aggregate.return_value = mock_aggregate

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        mock_result = RunResult(
            label="trial_0001",
            success=True,
            summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
            artifacts_path=tmp_path / "trial_0001",
        )

        with patch.object(
            orchestrator, "_execute_single_run", return_value=mock_result
        ):
            orchestrator.execute_and_export(config, strategy=mock_strategy)

        mock_strategy.export_aggregates.assert_called_once_with(
            mock_aggregate, orchestrator.base_dir
        )

    def test_execute_and_export_skips_export_when_aggregate_is_none(
        self, orchestrator, tmp_path
    ):
        """Verify orchestrator skips export when strategy.aggregate() returns None."""
        mock_strategy = Mock(spec=FixedTrialsStrategy)
        mock_strategy.execute.return_value = None
        mock_strategy.should_continue.side_effect = [True, False, False]
        mock_strategy.get_next_config.side_effect = lambda c, r: c
        mock_strategy.get_run_label.return_value = "trial_0001"
        mock_strategy.get_run_path.return_value = Path("/tmp/r1")
        mock_strategy.tag_result.side_effect = lambda r, i: r
        mock_strategy.collect_failed_values.return_value = []
        mock_strategy.get_cooldown_seconds.return_value = 0.0
        mock_strategy.aggregate.return_value = None

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        mock_result = RunResult(
            label="trial_0001",
            success=True,
            summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
            artifacts_path=tmp_path / "trial_0001",
        )

        with patch.object(
            orchestrator, "_execute_single_run", return_value=mock_result
        ):
            orchestrator.execute_and_export(config, strategy=mock_strategy)

        mock_strategy.export_aggregates.assert_not_called()


class TestConfidenceExport:
    """Tests for FixedTrialsStrategy.export_aggregates()."""

    def test_export_calls_export_confidence_helper(self, tmp_path):
        strategy = FixedTrialsStrategy(num_trials=3)
        aggregate = AggregateResult(
            aggregation_type="confidence",
            num_runs=3,
            num_successful_runs=3,
            failed_runs=[],
            metadata={},
            metrics={},
        )

        with patch(
            "aiperf.orchestrator.export_helpers.export_confidence"
        ) as mock_export:
            mock_export.return_value = (tmp_path / "a.json", tmp_path / "a.csv")
            strategy.export_aggregates(aggregate, tmp_path)

        mock_export.assert_called_once()
        call_args = mock_export.call_args[0]
        assert call_args[0] is aggregate
        assert call_args[1] == tmp_path / "aggregate"


class TestSweepExport:
    """Tests for ParameterSweepStrategy.export_aggregates()."""

    def test_export_calls_export_sweep_helper(self, tmp_path):
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )
        aggregate = AggregateResult(
            aggregation_type="sweep",
            num_runs=3,
            num_successful_runs=3,
            failed_runs=[],
            metadata={"best_configurations": {}, "pareto_optimal": []},
            metrics=[],
        )

        with patch("aiperf.orchestrator.export_helpers.export_sweep") as mock_export:
            mock_export.return_value = (tmp_path / "s.json", tmp_path / "s.csv")
            strategy.export_aggregates(aggregate, tmp_path)

        mock_export.assert_called_once()
        call_args = mock_export.call_args[0]
        assert call_args[0] is aggregate
        assert call_args[1] == tmp_path / "sweep_aggregate"


class TestSweepConfidenceExport:
    """Tests for SweepConfidenceStrategy.export_aggregates()."""

    @pytest.fixture
    def strategy_repeated(self):
        return SweepConfidenceStrategy(
            sweep=ParameterSweepStrategy(
                parameter_name="concurrency", parameter_values=[10, 20]
            ),
            confidence=FixedTrialsStrategy(num_trials=2),
            mode=SweepMode.REPEATED,
        )

    @pytest.fixture
    def strategy_independent(self):
        return SweepConfidenceStrategy(
            sweep=ParameterSweepStrategy(
                parameter_name="concurrency", parameter_values=[10, 20]
            ),
            confidence=FixedTrialsStrategy(num_trials=2),
            mode=SweepMode.INDEPENDENT,
        )

    def _make_aggregate(self):
        per_value = {
            10: AggregateResult(
                aggregation_type="confidence",
                num_runs=2,
                num_successful_runs=2,
                failed_runs=[],
                metadata={"concurrency": 10},
                metrics={},
            ),
            20: AggregateResult(
                aggregation_type="confidence",
                num_runs=2,
                num_successful_runs=2,
                failed_runs=[],
                metadata={"concurrency": 20},
                metrics={},
            ),
        }
        return AggregateResult(
            aggregation_type="sweep",
            num_runs=4,
            num_successful_runs=4,
            failed_runs=[],
            metadata={
                "per_value_aggregates": per_value,
                "best_configurations": {},
                "pareto_optimal": [],
            },
            metrics=[],
        )

    def test_repeated_mode_exports_per_value_and_sweep(
        self, strategy_repeated, tmp_path
    ):
        aggregate = self._make_aggregate()

        with (
            patch("aiperf.orchestrator.export_helpers.export_confidence") as mock_conf,
            patch("aiperf.orchestrator.export_helpers.export_sweep") as mock_sweep,
        ):
            mock_conf.return_value = (tmp_path / "c.json", tmp_path / "c.csv")
            mock_sweep.return_value = (tmp_path / "s.json", tmp_path / "s.csv")
            strategy_repeated.export_aggregates(aggregate, tmp_path)

        # 2 per-value confidence exports
        assert mock_conf.call_count == 2
        # 1 sweep export
        mock_sweep.assert_called_once()

    def test_repeated_mode_path_structure(self, strategy_repeated, tmp_path):
        aggregate = self._make_aggregate()
        exported_paths = []

        with (
            patch("aiperf.orchestrator.export_helpers.export_confidence") as mock_conf,
            patch("aiperf.orchestrator.export_helpers.export_sweep") as mock_sweep,
        ):
            mock_conf.side_effect = lambda agg, path: exported_paths.append(
                ("conf", path)
            ) or (path / "a.json", path / "a.csv")
            mock_sweep.side_effect = lambda agg, path: exported_paths.append(
                ("sweep", path)
            ) or (path / "s.json", path / "s.csv")
            strategy_repeated.export_aggregates(aggregate, tmp_path)

        # Repeated mode: base_dir/aggregate/concurrency_10, base_dir/aggregate/concurrency_20
        conf_paths = [p for t, p in exported_paths if t == "conf"]
        assert tmp_path / "aggregate" / "concurrency_10" in conf_paths
        assert tmp_path / "aggregate" / "concurrency_20" in conf_paths

        # Sweep: base_dir/aggregate/sweep_aggregate
        sweep_paths = [p for t, p in exported_paths if t == "sweep"]
        assert tmp_path / "aggregate" / "sweep_aggregate" in sweep_paths

    def test_independent_mode_path_structure(self, strategy_independent, tmp_path):
        aggregate = self._make_aggregate()
        exported_paths = []

        with (
            patch("aiperf.orchestrator.export_helpers.export_confidence") as mock_conf,
            patch("aiperf.orchestrator.export_helpers.export_sweep") as mock_sweep,
        ):
            mock_conf.side_effect = lambda agg, path: exported_paths.append(
                ("conf", path)
            ) or (path / "a.json", path / "a.csv")
            mock_sweep.side_effect = lambda agg, path: exported_paths.append(
                ("sweep", path)
            ) or (path / "s.json", path / "s.csv")
            strategy_independent.export_aggregates(aggregate, tmp_path)

        # Independent mode: base_dir/concurrency_10/aggregate, base_dir/concurrency_20/aggregate
        conf_paths = [p for t, p in exported_paths if t == "conf"]
        assert tmp_path / "concurrency_10" / "aggregate" in conf_paths
        assert tmp_path / "concurrency_20" / "aggregate" in conf_paths

        # Sweep: base_dir/sweep_aggregate
        sweep_paths = [p for t, p in exported_paths if t == "sweep"]
        assert tmp_path / "sweep_aggregate" in sweep_paths

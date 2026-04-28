# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator execution methods.

Tests cover:
- _resolve_strategy() auto-detection from config
- _execute() delegation to strategy.execute() (Option B) or generic loop
- _execute_loop() generic loop with tag_result and collect_failed_values
- SweepConfidenceStrategy.execute() for both modes
- _create_sweep_strategy() and _create_confidence_strategy() factory methods
"""

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import (
    FixedTrialsStrategy,
    ParameterSweepStrategy,
    SweepConfidenceStrategy,
    SweepMode,
)


class TestResolveStrategy:
    """Tests for _resolve_strategy() auto-detection."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    def test_resolves_sweep_only(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]
        config.loadgen.num_profile_runs = 1

        strategy = orchestrator._resolve_strategy(config)
        assert isinstance(strategy, ParameterSweepStrategy)
        assert strategy.parameter_name == "concurrency"
        assert strategy.parameter_values == [10, 20, 30]

    def test_resolves_confidence_only(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 10
        config.loadgen.num_profile_runs = 5

        strategy = orchestrator._resolve_strategy(config)
        assert isinstance(strategy, FixedTrialsStrategy)
        assert strategy.num_trials == 5

    def test_resolves_sweep_and_confidence_repeated(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 3
        config.loadgen.parameter_sweep_mode = "repeated"

        strategy = orchestrator._resolve_strategy(config)
        assert isinstance(strategy, SweepConfidenceStrategy)
        assert strategy.mode == SweepMode.REPEATED
        assert isinstance(strategy.sweep, ParameterSweepStrategy)
        assert isinstance(strategy.confidence, FixedTrialsStrategy)

    def test_resolves_sweep_and_confidence_independent(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 3
        config.loadgen.parameter_sweep_mode = "independent"

        strategy = orchestrator._resolve_strategy(config)
        assert isinstance(strategy, SweepConfidenceStrategy)
        assert strategy.mode == SweepMode.INDEPENDENT

    def test_raises_for_single_run(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 10
        config.loadgen.num_profile_runs = 1

        with pytest.raises(ValueError, match="requires sweep or confidence mode"):
            orchestrator._resolve_strategy(config)


class TestExecuteDelegation:
    """Tests for _execute() delegation to strategy.execute() (Option B)."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    def test_uses_custom_loop_when_strategy_returns_results(
        self, orchestrator, tmp_path
    ):
        """When strategy.execute() returns a list, orchestrator uses it."""
        custom_results = [
            RunResult(label="custom_1", success=True, artifacts_path=tmp_path / "r1"),
            RunResult(label="custom_2", success=True, artifacts_path=tmp_path / "r2"),
        ]

        mock_strategy = Mock()
        mock_strategy.execute.return_value = custom_results

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        results = orchestrator._execute(config, mock_strategy)

        assert results == custom_results
        mock_strategy.execute.assert_called_once()

    def test_falls_back_to_generic_loop_when_strategy_returns_none(
        self, orchestrator, tmp_path
    ):
        """When strategy.execute() returns None, orchestrator uses _execute_loop."""
        mock_strategy = Mock(spec=FixedTrialsStrategy)
        mock_strategy.execute.return_value = None
        mock_strategy.should_continue.side_effect = [True, False, False]
        mock_strategy.get_next_config.side_effect = lambda c, r: c
        mock_strategy.get_run_label.return_value = "trial_0001"
        mock_strategy.get_run_path.return_value = tmp_path / "trial_0001"
        mock_strategy.tag_result.side_effect = lambda r, i: r
        mock_strategy.collect_failed_values.return_value = []
        mock_strategy.get_cooldown_seconds.return_value = 0.0

        mock_result = RunResult(
            label="trial_0001",
            success=True,
            summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
            artifacts_path=tmp_path / "trial_0001",
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        with patch.object(
            orchestrator, "_execute_single_run", return_value=mock_result
        ):
            results = orchestrator._execute(config, mock_strategy)

        assert len(results) == 1
        assert results[0].label == "trial_0001"


class TestExecuteLoop:
    """Tests for _execute_loop() generic loop."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    def test_sweep_execution(self, orchestrator, tmp_path):
        """Test generic loop with ParameterSweepStrategy."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
        )

        mock_results = [
            RunResult(
                label=f"concurrency_{val}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"concurrency_{val}",
            )
            for val in [10, 20, 30]
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator._execute_loop(config, strategy)

        assert len(results) == 3
        # tag_result should have stamped metadata
        assert all("concurrency" in r.metadata for r in results)

    def test_confidence_execution(self, orchestrator, tmp_path):
        """Test generic loop with FixedTrialsStrategy."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.num_profile_runs = 3

        strategy = FixedTrialsStrategy(num_trials=3)

        mock_results = [
            RunResult(
                label=f"trial_{i + 1:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"trial_{i + 1:04d}",
            )
            for i in range(3)
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator._execute_loop(config, strategy)

        assert len(results) == 3

    def test_logs_failed_sweep_values(self, orchestrator, tmp_path):
        """Test that _execute_loop logs failed sweep values."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
        )

        mock_results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "concurrency_10",
            ),
            RunResult(
                label="concurrency_20",
                success=False,
                error="Connection timeout",
                artifacts_path=tmp_path / "concurrency_20",
            ),
            RunResult(
                label="concurrency_30",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "concurrency_30",
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.logger") as mock_logger,
        ):
            results = orchestrator._execute_loop(config, strategy)

        assert len(results) == 3
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("sweep values failed" in str(call) for call in warning_calls)

    def test_no_sweep_warning_for_confidence_failures(self, orchestrator, tmp_path):
        """Test that confidence-only failures don't trigger sweep warnings."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

        strategy = FixedTrialsStrategy(num_trials=2)

        mock_results = [
            RunResult(
                label="trial_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "trial_0001",
            ),
            RunResult(
                label="trial_0002",
                success=False,
                error="Some error",
                artifacts_path=tmp_path / "trial_0002",
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.logger") as mock_logger,
        ):
            results = orchestrator._execute_loop(config, strategy)

        assert len(results) == 2
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("sweep values failed" in str(call) for call in warning_calls)

    def test_applies_cooldown_between_runs(self, orchestrator, tmp_path):
        """Test that cooldown is applied between runs."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))

        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=5.0)

        mock_results = [
            RunResult(
                label=f"trial_{i + 1:04d}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / f"trial_{i + 1:04d}",
            )
            for i in range(3)
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.time.sleep") as mock_sleep,
        ):
            orchestrator._execute_loop(config, strategy)

        # 3 runs, cooldown between runs 1-2 and 2-3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5.0)


class TestSweepConfidenceExecution:
    """Tests for SweepConfidenceStrategy.execute() (Option B custom loop)."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    def test_repeated_mode_execution_order(self, orchestrator, tmp_path):
        """Test repeated mode: for trial in trials: for value in values."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "repeated"

        executed_concurrencies = []

        def capture_run(run_config, label, path):
            executed_concurrencies.append(run_config.loadgen.concurrency)
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with patch.object(orchestrator, "_execute_single_run", side_effect=capture_run):
            results = orchestrator.execute(config)

        # Repeated: trial 1 [10, 20], trial 2 [10, 20]
        assert executed_concurrencies == [10, 20, 10, 20]
        assert len(results) == 4

    def test_independent_mode_execution_order(self, orchestrator, tmp_path):
        """Test independent mode: for value in values: for trial in trials."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "independent"

        executed_concurrencies = []

        def capture_run(run_config, label, path):
            executed_concurrencies.append(run_config.loadgen.concurrency)
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with patch.object(orchestrator, "_execute_single_run", side_effect=capture_run):
            results = orchestrator.execute(config)

        # Independent: value 10 [trial1, trial2], value 20 [trial1, trial2]
        assert executed_concurrencies == [10, 10, 20, 20]
        assert len(results) == 4

    def test_repeated_mode_metadata_tagging(self, orchestrator, tmp_path):
        """Test that results are tagged with correct metadata in repeated mode."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "repeated"

        def mock_run(run_config, label, path):
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with patch.object(orchestrator, "_execute_single_run", side_effect=mock_run):
            results = orchestrator.execute(config)

        assert all("trial_index" in r.metadata for r in results)
        assert all("value_index" in r.metadata for r in results)
        assert all("concurrency" in r.metadata for r in results)
        assert all(r.metadata["sweep_mode"] == "repeated" for r in results)

    def test_independent_mode_metadata_tagging(self, orchestrator, tmp_path):
        """Test that results are tagged with correct metadata in independent mode."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "independent"

        def mock_run(run_config, label, path):
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with patch.object(orchestrator, "_execute_single_run", side_effect=mock_run):
            results = orchestrator.execute(config)

        assert all("trial_index" in r.metadata for r in results)
        assert all("value_index" in r.metadata for r in results)
        assert all("concurrency" in r.metadata for r in results)
        assert all(r.metadata["sweep_mode"] == "independent" for r in results)

    def test_repeated_mode_cooldowns(self, orchestrator, tmp_path):
        """Test cooldown application in repeated mode."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "repeated"
        config.loadgen.parameter_sweep_cooldown_seconds = 0.1
        config.loadgen.profile_run_cooldown_seconds = 0.2

        def mock_run(run_config, label, path):
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_run),
            patch("aiperf.orchestrator.strategies.time.sleep") as mock_sleep,
        ):
            orchestrator.execute(config)

        # Repeated: trial 1 [10, sweep_cooldown, 20], trial_cooldown, trial 2 [10, sweep_cooldown, 20]
        # = 1 sweep + 1 trial + 1 sweep = 3 sleeps
        assert mock_sleep.call_count == 3

    def test_independent_mode_cooldowns(self, orchestrator, tmp_path):
        """Test cooldown application in independent mode."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.num_profile_runs = 2
        config.loadgen.parameter_sweep_mode = "independent"
        config.loadgen.parameter_sweep_cooldown_seconds = 0.1
        config.loadgen.profile_run_cooldown_seconds = 0.2

        def mock_run(run_config, label, path):
            return RunResult(
                label=label,
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=path,
            )

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_run),
            patch("aiperf.orchestrator.strategies.time.sleep") as mock_sleep,
        ):
            orchestrator.execute(config)

        # Independent: value 10 [trial1, trial_cooldown, trial2], sweep_cooldown, value 20 [trial1, trial_cooldown, trial2]
        # = 1 trial + 1 sweep + 1 trial = 3 sleeps
        assert mock_sleep.call_count == 3


class TestFactoryMethods:
    """Tests for _create_sweep_strategy and _create_confidence_strategy."""

    @pytest.fixture
    def mock_service_config(self):
        mock_config = Mock(spec=ServiceConfig)
        mock_config.model_dump.return_value = {}
        return mock_config

    @pytest.fixture
    def orchestrator(self, mock_service_config, tmp_path):
        return MultiRunOrchestrator(tmp_path, mock_service_config)

    def test_create_sweep_strategy(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20, 30]

        strategy = orchestrator._create_sweep_strategy(config)

        assert isinstance(strategy, ParameterSweepStrategy)
        assert strategy.parameter_name == "concurrency"
        assert strategy.parameter_values == [10, 20, 30]

    def test_create_sweep_strategy_raises_without_sweep_param(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 10

        with pytest.raises(ValueError, match="No sweep parameter detected"):
            orchestrator._create_sweep_strategy(config)

    def test_create_confidence_strategy(self, orchestrator):
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.num_profile_runs = 5

        strategy = orchestrator._create_confidence_strategy(config)

        assert isinstance(strategy, FixedTrialsStrategy)
        assert strategy.num_trials == 5

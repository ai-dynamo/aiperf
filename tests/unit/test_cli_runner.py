# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cli_runner.py"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.plugin.enums import UIType


class TestRunSystemController:
    """Test the run_system_controller routing logic."""

    @pytest.fixture
    def user_config_single_run(self, user_config: UserConfig) -> UserConfig:
        """Create a UserConfig for single run (num_profile_runs=1)."""
        user_config.loadgen.num_profile_runs = 1
        return user_config

    @pytest.fixture
    def user_config_multi_run(self, user_config: UserConfig) -> UserConfig:
        """Create a UserConfig for multi-run (num_profile_runs>1)."""
        user_config.loadgen.num_profile_runs = 3
        user_config.loadgen.confidence_level = 0.95
        user_config.loadgen.profile_run_cooldown_seconds = 5
        return user_config

    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_routes_to_single_benchmark_when_num_runs_is_one(
        self,
        mock_single: Mock,
        user_config_single_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that single run is called when num_profile_runs=1."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(user_config_single_run, service_config)

        mock_single.assert_called_once_with(user_config_single_run, service_config)

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_routes_to_multi_benchmark_when_num_runs_greater_than_one(
        self,
        mock_multi: Mock,
        user_config_multi_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that multi-run is called when num_profile_runs>1."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(user_config_multi_run, service_config)

        mock_multi.assert_called_once_with(user_config_multi_run, service_config)

    def test_raises_error_when_using_dashboard_ui_with_multi_run(
        self,
        user_config_multi_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that an error is raised when explicitly using dashboard UI with multi-run."""
        from aiperf.cli_runner import _run_multi_benchmark

        # Set dashboard UI explicitly (simulate user setting it)
        service_config.ui_type = UIType.DASHBOARD
        service_config.model_fields_set.add("ui_type")

        # Should raise ValueError when _run_multi_benchmark is called
        with pytest.raises(
            ValueError, match="Dashboard UI is not supported with multi-run mode"
        ):
            _run_multi_benchmark(user_config_multi_run, service_config)

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_no_warning_when_using_simple_ui_with_multi_run(
        self,
        mock_multi: Mock,
        user_config_multi_run: UserConfig,
        service_config: ServiceConfig,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that no warning is logged when using simple UI with multi-run."""
        from aiperf.cli_runner import run_system_controller

        # Set simple UI
        service_config.ui_type = UIType.SIMPLE

        run_system_controller(user_config_multi_run, service_config)

        # Check that no dashboard warning was logged
        assert not any(
            "Dashboard UI does not show live updates" in record.message
            for record in caplog.records
        )

    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_no_warning_when_using_dashboard_ui_with_single_run(
        self,
        mock_single: Mock,
        user_config_single_run: UserConfig,
        service_config: ServiceConfig,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that no warning is logged when using dashboard UI with single run."""
        from aiperf.cli_runner import run_system_controller

        # Set dashboard UI
        service_config.ui_type = UIType.DASHBOARD

        run_system_controller(user_config_single_run, service_config)

        # Check that no dashboard warning was logged
        assert not any(
            "Dashboard UI does not show live updates" in record.message
            for record in caplog.records
        )


class TestRunSingleBenchmark:
    """Test the _run_single_benchmark function."""

    @pytest.fixture
    def service_config_simple(self) -> ServiceConfig:
        """Create a ServiceConfig with Simple UI type."""
        config = ServiceConfig()
        config.ui_type = UIType.SIMPLE
        return config

    @patch("aiperf.cli_runner.MetricsConfigLoader")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_simple_ui_uses_rich_logging(
        self,
        mock_setup_rich: Mock,
        mock_bootstrap: Mock,
        mock_loader_cls: Mock,
        service_config_simple: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that simple UI uses rich logging instead of log queue."""
        from aiperf.cli_runner import _run_single_benchmark

        _run_single_benchmark(user_config, service_config_simple)

        # Verify rich logging was set up
        mock_setup_rich.assert_called_once_with(user_config, service_config_simple)

        # Verify bootstrap was called without log_queue
        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs.get("log_queue") is None

    @patch("aiperf.cli_runner.MetricsConfigLoader")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_bootstrap_exception_is_raised(
        self,
        mock_bootstrap: Mock,
        mock_loader_cls: Mock,
        service_config: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that exceptions from bootstrap are raised."""
        from aiperf.cli_runner import _run_single_benchmark

        # Make bootstrap raise an exception
        mock_bootstrap.side_effect = RuntimeError("Bootstrap failed")

        with pytest.raises(RuntimeError, match="Bootstrap failed"):
            _run_single_benchmark(user_config, service_config)


class TestRunMultiBenchmark:
    """Test the _run_multi_benchmark function."""

    @pytest.fixture
    def user_config_multi(self, user_config: UserConfig) -> UserConfig:
        """Create a UserConfig for multi-run."""
        user_config.loadgen.num_profile_runs = 3
        user_config.loadgen.confidence_level = 0.95
        user_config.loadgen.profile_run_cooldown_seconds = 5
        user_config.loadgen.profile_run_disable_warmup_after_first = True
        return user_config

    @pytest.fixture
    def mock_run_result(self):
        """Create a mock run result."""
        result = MagicMock()
        result.success = True
        result.label = "run_1"
        result.metrics_file = Path("/tmp/metrics.json")
        return result

    @pytest.fixture
    def mock_aggregate_result(self):
        """Create a mock aggregate result."""
        result = MagicMock()
        result.aggregation_type = "confidence"
        result.num_runs = 3
        result.num_successful_runs = 3
        result.failed_runs = []
        result.metadata = {"confidence_level": 0.95}
        result.metrics = {}
        return result

    @patch("aiperf.orchestrator.strategies.FixedTrialsStrategy")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation")
    @patch("aiperf.exporters.aggregate.AggregateConfidenceJsonExporter")
    @patch("aiperf.exporters.aggregate.AggregateConfidenceCsvExporter")
    def test_multi_run_success_with_aggregation(
        self,
        mock_csv_exporter_cls: Mock,
        mock_json_exporter_cls: Mock,
        mock_aggregation_cls: Mock,
        mock_orchestrator_cls: Mock,
        mock_strategy_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        mock_run_result: MagicMock,
        mock_aggregate_result: MagicMock,
        tmp_path: Path,
    ):
        """Test successful multi-run with aggregation."""
        from aiperf.cli_runner import _run_multi_benchmark

        # Set up artifact directory
        user_config_multi.output.artifact_directory = tmp_path

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy.get_aggregate_path.return_value = tmp_path / "aggregate"
        mock_strategy_cls.return_value = mock_strategy

        # Mock orchestrator to return 3 successful results
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute.return_value = [
            mock_run_result,
            mock_run_result,
            mock_run_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        # Mock aggregation
        mock_aggregation = MagicMock()
        mock_aggregation.aggregate.return_value = mock_aggregate_result
        mock_aggregation_cls.return_value = mock_aggregation

        # Mock exporters with async export() method
        mock_json_exporter = MagicMock()
        mock_json_exporter.export = AsyncMock(return_value=tmp_path / "aggregate.json")
        mock_json_exporter_cls.return_value = mock_json_exporter

        mock_csv_exporter = MagicMock()
        mock_csv_exporter.export = AsyncMock(return_value=tmp_path / "aggregate.csv")
        mock_csv_exporter_cls.return_value = mock_csv_exporter

        _run_multi_benchmark(user_config_multi, service_config)

        # Verify strategy was created with correct parameters
        mock_strategy_cls.assert_called_once_with(
            num_trials=3,
            cooldown_seconds=5,
            auto_set_seed=True,
            disable_warmup_after_first=True,
        )

        # Verify orchestrator was created and executed
        mock_orchestrator_cls.assert_called_once_with(
            base_dir=tmp_path, service_config=service_config
        )
        mock_orchestrator.execute.assert_called_once_with(
            user_config_multi, mock_strategy
        )

        # Verify aggregation was performed
        mock_aggregation_cls.assert_called_once_with(confidence_level=0.95)
        mock_aggregation.aggregate.assert_called_once()

        # Verify exporters were called
        mock_json_exporter.export.assert_called_once()
        mock_csv_exporter.export.assert_called_once()

    @patch("aiperf.orchestrator.strategies.FixedTrialsStrategy")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_orchestrator_exception(
        self,
        mock_orchestrator_cls: Mock,
        mock_strategy_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        tmp_path: Path,
    ):
        """Test that orchestrator exceptions are raised."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy_cls.return_value = mock_strategy

        # Mock orchestrator to raise exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute.side_effect = RuntimeError("Orchestrator failed")
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(RuntimeError, match="Orchestrator failed"):
            _run_multi_benchmark(user_config_multi, service_config)

    @patch("aiperf.orchestrator.strategies.FixedTrialsStrategy")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_only_one_successful_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        mock_strategy_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        mock_run_result: MagicMock,
        tmp_path: Path,
    ):
        """Test that only 1 successful run exits with error code 1."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy_cls.return_value = mock_strategy

        # Mock orchestrator to return 1 successful and 2 failed results
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_2"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute.return_value = [
            mock_run_result,
            failed_result,
            failed_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(user_config_multi, service_config)

        assert exc_info.value.code == 1

    @patch("aiperf.orchestrator.strategies.FixedTrialsStrategy")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_all_failed_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        mock_strategy_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        tmp_path: Path,
    ):
        """Test that all failed runs exit with error code 1."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy_cls.return_value = mock_strategy

        # Mock orchestrator to return all failed results
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_1"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute.return_value = [
            failed_result,
            failed_result,
            failed_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(user_config_multi, service_config)

        assert exc_info.value.code == 1


class TestPrintAggregateSummary:
    """Test the _print_aggregate_summary function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def aggregate_result_with_metrics(self):
        """Create an aggregate result with metrics."""
        result = MagicMock()
        result.aggregation_type = "confidence"
        result.num_runs = 3
        result.num_successful_runs = 3
        result.failed_runs = []
        result.metadata = {"confidence_level": 0.95}

        # Create mock metrics
        mock_metric = MagicMock()
        mock_metric.mean = 100.5
        mock_metric.std = 5.2
        mock_metric.min = 95.0
        mock_metric.max = 105.0
        mock_metric.cv = 0.052
        mock_metric.ci_low = 98.0
        mock_metric.ci_high = 103.0
        mock_metric.unit = "req/s"

        # Use actual flattened metric keys that match the aggregation output
        result.metrics = {
            "request_throughput_avg": mock_metric,
            "time_to_first_token_avg": mock_metric,
        }
        return result

    @pytest.fixture
    def aggregate_result_with_failures(self):
        """Create an aggregate result with failed runs."""
        result = MagicMock()
        result.aggregation_type = "confidence"
        result.num_runs = 3
        result.num_successful_runs = 2
        result.failed_runs = [
            {"label": "run_3", "error": "Connection timeout"},
        ]
        result.metadata = {"confidence_level": 0.95}
        result.metrics = {}
        return result

    def test_prints_basic_summary_info(
        self, aggregate_result_with_metrics: MagicMock, mock_logger: MagicMock
    ):
        """Test that basic summary information is printed."""
        from aiperf.cli_runner import _print_aggregate_summary

        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

        # Verify logger.info was called with expected content
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        assert any("AGGREGATE STATISTICS SUMMARY" in call for call in info_calls)
        assert any("Aggregation Type: confidence" in call for call in info_calls)
        assert any("Total Runs: 3" in call for call in info_calls)
        assert any("Successful Runs: 3" in call for call in info_calls)
        assert any("Confidence Level: 95%" in call for call in info_calls)

    def test_prints_failed_runs_warning(
        self, aggregate_result_with_failures: MagicMock, mock_logger: MagicMock
    ):
        """Test that failed runs are printed as warnings."""
        from aiperf.cli_runner import _print_aggregate_summary

        _print_aggregate_summary(aggregate_result_with_failures, mock_logger)

        # Verify logger.warning was called for failed runs
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]

        assert any("Failed Runs (1):" in call for call in warning_calls)
        assert any("run_3: Connection timeout" in call for call in warning_calls)

    def test_prints_key_metrics(
        self, aggregate_result_with_metrics: MagicMock, mock_logger: MagicMock
    ):
        """Test that key metrics are printed with correct formatting."""
        from aiperf.cli_runner import _print_aggregate_summary

        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

        # Verify metrics were printed
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Check for the dynamic display name format (e.g., "Request Throughput (Avg)")
        assert any("Request Throughput" in call for call in info_calls)
        assert any("Mean:" in call and "100.5000" in call for call in info_calls)
        assert any("Std Dev:" in call and "5.2000" in call for call in info_calls)
        assert any("CV:" in call and "5.20%" in call for call in info_calls)

    def test_prints_interpretation_guide(
        self, aggregate_result_with_metrics: MagicMock, mock_logger: MagicMock
    ):
        """Test that interpretation guide is printed."""
        from aiperf.cli_runner import _print_aggregate_summary

        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

        # Verify interpretation guide was printed
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        assert any(
            "Coefficient of Variation (CV) Interpretation Guide:" in call
            for call in info_calls
        )
        assert any("CV < 5%:" in call for call in info_calls)
        assert any(
            "Confidence Interval (CI) Interpretation:" in call for call in info_calls
        )

    def test_handles_empty_metrics(self, mock_logger: MagicMock):
        """Test that empty metrics are handled gracefully."""
        from aiperf.cli_runner import _print_aggregate_summary

        result = MagicMock()
        result.aggregation_type = "confidence"
        result.num_runs = 3
        result.num_successful_runs = 3
        result.failed_runs = []
        result.metadata = {"confidence_level": 0.95}
        result.metrics = {}

        _print_aggregate_summary(result, mock_logger)

        # Verify warning was logged for no metrics
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("No key metrics found" in call for call in warning_calls)

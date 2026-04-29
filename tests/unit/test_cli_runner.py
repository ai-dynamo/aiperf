# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cli_runner.py"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.orchestrator.strategies import FixedTrialsStrategy
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
        from aiperf.cli_runner import run_system_controller

        # Set dashboard UI explicitly (simulate user setting it)
        service_config.ui_type = UIType.DASHBOARD
        service_config.model_fields_set.add("ui_type")

        # Should raise ValueError when run_system_controller validates UI compatibility
        with pytest.raises(
            ValueError, match="Dashboard UI.*is not supported with multi-run mode"
        ):
            run_system_controller(user_config_multi_run, service_config)

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

    def test_raises_error_when_using_dashboard_ui_with_parameter_sweep(
        self,
        user_config_single_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that an error is raised when explicitly using dashboard UI with parameter sweep."""
        from aiperf.cli_runner import run_system_controller

        # Set concurrency as a list (parameter sweep)
        user_config_single_run.loadgen.concurrency = [10, 20, 30]

        # Set dashboard UI explicitly (simulate user setting it)
        service_config.ui_type = UIType.DASHBOARD
        service_config.model_fields_set.add("ui_type")

        # Should raise ValueError
        with pytest.raises(
            ValueError,
            match="Dashboard UI.*is not supported with multi-run mode",
        ):
            run_system_controller(user_config_single_run, service_config)

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_no_error_when_using_simple_ui_with_parameter_sweep(
        self,
        mock_multi: Mock,
        user_config_single_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that no error is raised when using simple UI with parameter sweep."""
        from aiperf.cli_runner import run_system_controller

        # Set concurrency as a list (parameter sweep)
        user_config_single_run.loadgen.concurrency = [10, 20, 30]

        # Set simple UI
        service_config.ui_type = UIType.SIMPLE
        service_config.model_fields_set.add("ui_type")

        # Should not raise an error
        run_system_controller(user_config_single_run, service_config)
        mock_multi.assert_called_once()

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_no_error_when_dashboard_ui_not_explicitly_set_with_parameter_sweep(
        self,
        mock_multi: Mock,
        user_config_single_run: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that no error is raised when dashboard UI is default (not explicitly set) with parameter sweep."""
        from aiperf.cli_runner import run_system_controller

        # Set concurrency as a list (parameter sweep)
        user_config_single_run.loadgen.concurrency = [10, 20, 30]

        # Dashboard UI is default but not explicitly set by user
        service_config.ui_type = UIType.DASHBOARD
        # Explicitly remove from model_fields_set to simulate default value
        service_config.model_fields_set.discard("ui_type")

        # Should not raise an error (validation only checks explicitly set values)
        run_system_controller(user_config_single_run, service_config)
        mock_multi.assert_called_once()


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

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_success_with_aggregation(
        self,
        mock_orchestrator_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        mock_run_result: MagicMock,
        tmp_path: Path,
    ):
        """Test successful multi-run with aggregation."""
        from aiperf.cli_runner import _run_multi_benchmark

        # Set up artifact directory
        user_config_multi.output.artifact_directory = tmp_path

        # Mock orchestrator to return 3 successful results
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_and_export.return_value = [
            mock_run_result,
            mock_run_result,
            mock_run_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        _run_multi_benchmark(user_config_multi, service_config)

        # Verify orchestrator was created and execute_and_export was called with a FixedTrialsStrategy
        mock_orchestrator_cls.assert_called_once_with(
            base_dir=tmp_path, service_config=service_config
        )
        mock_orchestrator.execute_and_export.assert_called_once()
        call_kwargs = mock_orchestrator.execute_and_export.call_args
        assert call_kwargs[0][0] is user_config_multi
        assert isinstance(call_kwargs[1]["strategy"], FixedTrialsStrategy)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_orchestrator_exception(
        self,
        mock_orchestrator_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        tmp_path: Path,
    ):
        """Test that orchestrator exceptions are raised."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock orchestrator to raise exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_and_export.side_effect = RuntimeError(
            "Orchestrator failed"
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(RuntimeError, match="Orchestrator failed"):
            _run_multi_benchmark(user_config_multi, service_config)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_only_one_successful_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        mock_run_result: MagicMock,
        tmp_path: Path,
    ):
        """Test that only 1 successful run exits with error code 1."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock orchestrator to return 1 successful and 2 failed results
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_2"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_and_export.return_value = [
            mock_run_result,
            failed_result,
            failed_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(user_config_multi, service_config)

        assert exc_info.value.code == 1

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_all_failed_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        user_config_multi: UserConfig,
        service_config: ServiceConfig,
        tmp_path: Path,
    ):
        """Test that all failed runs exit with error code 1."""
        from aiperf.cli_runner import _run_multi_benchmark

        user_config_multi.output.artifact_directory = tmp_path

        # Mock orchestrator to return all failed results
        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_1"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_and_export.return_value = [
            failed_result,
            failed_result,
            failed_result,
        ]
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(user_config_multi, service_config)

        assert exc_info.value.code == 1

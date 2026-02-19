# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator."""

import json
from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy


class TestMultiRunOrchestrator:
    """Tests for MultiRunOrchestrator."""

    @pytest.fixture
    def mock_service_config(self):
        """Create a mock service config."""
        mock_config = Mock(spec=ServiceConfig)
        # Make model_dump return a serializable dict
        mock_config.model_dump.return_value = {
            "workers_max": 4,
            "workers_min": 1,
        }
        return mock_config

    @pytest.fixture
    def mock_user_config(self):
        """Create a mock user config."""
        from aiperf.common.config import EndpointConfig

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_num_sessions = None
        config.loadgen.warmup_duration = None
        config.loadgen.warmup_concurrency = None
        config.loadgen.warmup_request_rate = None
        config.loadgen.warmup_prefill_concurrency = None
        config.input.random_seed = 42
        return config

    def test_execute_with_fixed_trials_strategy(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test execute with FixedTrialsStrategy."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=0.0)

        # Mock the _execute_single_run method
        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=105.0)},
                artifacts_path=tmp_path / "run_0002",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=102.0)},
                artifacts_path=tmp_path / "run_0003",
            ),
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify we got 3 results
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].label == "run_0001"
        assert results[1].label == "run_0002"
        assert results[2].label == "run_0003"

    def test_execute_handles_failures(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that execute handles run failures gracefully."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=0.0)

        # Mock results with one failure
        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=False,
                error="Connection timeout",
                artifacts_path=tmp_path / "run_0002",
            ),
            RunResult(
                label="run_0003",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=102.0)},
                artifacts_path=tmp_path / "run_0003",
            ),
        ]

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_results
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify we got 3 results
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Connection timeout"
        assert results[2].success is True

    def test_execute_with_cooldown(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that cooldown is applied between runs."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.5)

        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
            ),
            RunResult(
                label="run_0002",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=105.0)},
                artifacts_path=tmp_path / "run_0002",
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.time.sleep") as mock_sleep,
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify cooldown was called once (between run 1 and run 2)
        mock_sleep.assert_called_once_with(0.5)
        assert len(results) == 2

    def test_execute_no_cooldown_after_last_run(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that cooldown is NOT applied after the last run."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1, cooldown_seconds=1.0)

        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch("aiperf.orchestrator.orchestrator.time.sleep") as mock_sleep,
        ):
            results = orchestrator.execute(mock_user_config, strategy)

        # Verify cooldown was NOT called (only 1 run)
        mock_sleep.assert_not_called()
        assert len(results) == 1

    def test_execute_single_run_success(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _execute_single_run with successful subprocess execution."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Create mock artifacts
        artifacts_path = tmp_path / "profile_runs" / "run_0001"
        artifacts_path.mkdir(parents=True)

        # Create mock profile export JSON
        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "p99": 195.0,
            },
            "request_count": {
                "unit": "requests",
                "avg": 10.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        assert result.success is True
        assert result.label == "run_0001"
        assert "time_to_first_token" in result.summary_metrics
        assert result.summary_metrics["time_to_first_token"].avg == 150.5
        assert result.summary_metrics["time_to_first_token"].unit == "ms"

    def test_execute_single_run_subprocess_failure(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _execute_single_run when subprocess fails."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Mock subprocess.run to return failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Connection refused"

        with patch("subprocess.run", return_value=mock_result):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        assert result.success is False
        assert result.label == "run_0001"
        assert "exit code 1" in result.error
        assert "Connection refused" in result.error

    def test_execute_single_run_no_metrics(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _execute_single_run when no metrics are found."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Mock subprocess.run to return success but no metrics file
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        assert result.success is False
        assert "No metrics found" in result.error

    def test_execute_single_run_zero_requests(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _execute_single_run when request_count is zero."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Create mock artifacts with zero request count
        artifacts_path = tmp_path / "profile_runs" / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "request_count": {
                "unit": "requests",
                "avg": 0.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        assert result.success is False
        assert "No requests completed" in result.error

    def test_execute_single_run_exception(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test _execute_single_run when an exception occurs."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Mock subprocess.run to raise an exception
        with patch("subprocess.run", side_effect=Exception("Unexpected error")):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        assert result.success is False
        assert "Unexpected error" in result.error

    def test_execute_single_run_creates_config_file(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that _execute_single_run creates run_config.json."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Create mock artifacts
        artifacts_path = tmp_path / "profile_runs" / "run_0001"
        artifacts_path.mkdir(parents=True)

        # Create mock profile export JSON
        json_content = {
            "request_count": {
                "unit": "requests",
                "avg": 10.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        # Mock subprocess.run to return success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            orchestrator._execute_single_run(mock_user_config, strategy, 0)

        # Verify run_config.json was created
        config_file = artifacts_path / "run_config.json"
        assert config_file.exists()

        # Verify it contains valid JSON
        with open(config_file) as f:
            config_data = json.load(f)
            assert "user_config" in config_data
            assert "service_config" in config_data

    def test_extract_summary_metrics(self, mock_service_config, tmp_path):
        """Test extracting summary metrics from JSON file."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        # Create a mock profile_export_aiperf.json file with the correct structure
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "min": 100.0,
                "max": 200.0,
                "p50": 145.0,
                "p99": 195.0,
            },
            "request_throughput": {
                "unit": "requests/sec",
                "avg": 25.4,
                "min": 20.0,
                "max": 30.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        # Extract metrics
        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Verify metrics were extracted with full JsonMetricResult structure
        assert "time_to_first_token" in metrics
        assert metrics["time_to_first_token"].avg == 150.5
        assert metrics["time_to_first_token"].p99 == 195.0
        assert metrics["time_to_first_token"].unit == "ms"
        assert "request_throughput" in metrics
        assert metrics["request_throughput"].avg == 25.4
        assert metrics["request_throughput"].unit == "requests/sec"

    def test_extract_summary_metrics_missing_file(self, mock_service_config, tmp_path):
        """Test extracting metrics when file doesn't exist."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        # Extract metrics (file doesn't exist)
        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Should return empty dict
        assert metrics == {}

    def test_extract_summary_metrics_invalid_json(self, mock_service_config, tmp_path):
        """Test extracting metrics when JSON is invalid."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        # Create invalid JSON file
        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            f.write("{ invalid json }")

        # Extract metrics (invalid JSON)
        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Should return empty dict
        assert metrics == {}

    def test_extract_summary_metrics_preserves_structure(
        self, mock_service_config, tmp_path
    ):
        """Test that the full JsonMetricResult structure is preserved."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "p50": 145.0,
                "p99": 195.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        # Verify the full structure is preserved
        assert "time_to_first_token" in metrics
        assert isinstance(metrics["time_to_first_token"], JsonMetricResult)
        assert metrics["time_to_first_token"].unit == "ms"
        assert metrics["time_to_first_token"].avg == 150.5
        assert metrics["time_to_first_token"].p50 == 145.0
        assert metrics["time_to_first_token"].p99 == 195.0

    def test_warmup_disabled_after_first_run(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that warmup is disabled after the first run."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.0)

        configs_used = []

        def mock_execute(config, strategy, run_index):
            # Capture the config used
            configs_used.append(config.model_copy(deep=True))
            return RunResult(
                label=strategy.get_run_label(run_index),
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / strategy.get_run_label(run_index),
            )

        with patch.object(
            orchestrator, "_execute_single_run", side_effect=mock_execute
        ):
            orchestrator.execute(mock_user_config, strategy)

        # Verify first run has warmup, second run doesn't
        assert len(configs_used) == 2
        # First run should have warmup (original config)
        assert configs_used[0].loadgen.warmup_request_count == 10
        # Second run should have warmup disabled
        assert configs_used[1].loadgen.warmup_request_count is None

    def test_strategy_validation_called(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that strategy.validate_config is called before execution."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1, cooldown_seconds=0.0)

        mock_results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=tmp_path / "run_0001",
            ),
        ]

        with (
            patch.object(orchestrator, "_execute_single_run", side_effect=mock_results),
            patch.object(strategy, "validate_config") as mock_validate,
        ):
            orchestrator.execute(mock_user_config, strategy)

        # Verify validate_config was called
        mock_validate.assert_called_once_with(mock_user_config)

    def test_execute_single_run_handles_early_exception(
        self, mock_service_config, mock_user_config, tmp_path
    ):
        """Test that early exceptions (before label is set) are handled gracefully."""
        orchestrator = MultiRunOrchestrator(tmp_path, mock_service_config)

        strategy = FixedTrialsStrategy(num_trials=1)

        # Mock get_run_path to raise an exception before label is set
        with patch.object(
            strategy, "get_run_path", side_effect=Exception("Path creation failed")
        ):
            result = orchestrator._execute_single_run(mock_user_config, strategy, 0)

        # Should return a RunResult with error, not crash with UnboundLocalError
        assert result.success is False
        assert "Path creation failed" in result.error
        # Label should be the fallback value (run_index=0 -> run_0000) since get_run_label was never called
        assert result.label == "run_0000"
        assert result.artifacts_path is None

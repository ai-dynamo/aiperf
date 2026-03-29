# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cli_runner.py"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.cli_runner import (
    _print_aggregate_summary,
    _run_multi_benchmark,
    _run_single_benchmark,
    run_benchmark,
)
from aiperf.config import AIPerfConfig, BenchmarkPlan, BenchmarkRun
from aiperf.config.loader import build_benchmark_plan
from aiperf.plugin.enums import UIType

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def _make_config(**overrides) -> AIPerfConfig:
    """Create a minimal AIPerfConfig for testing."""
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return AIPerfConfig(**kwargs)


def _make_plan(**overrides) -> BenchmarkPlan:
    """Create a BenchmarkPlan from config overrides."""
    config = _make_config(**overrides)
    return build_benchmark_plan(config)


def _make_run(**overrides) -> BenchmarkRun:
    """Create a BenchmarkRun from config overrides."""
    config = _make_config(**overrides)
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=config.artifacts.dir,
    )


class TestRunBenchmark:
    """Test the run_benchmark routing logic."""

    @pytest.fixture
    def plan_single_run(self) -> BenchmarkPlan:
        """Create a BenchmarkPlan for single run."""
        return _make_plan()

    @pytest.fixture
    def plan_multi_run(self) -> BenchmarkPlan:
        """Create a BenchmarkPlan for multi-run (num_runs>1)."""
        return _make_plan(
            multi_run={
                "num_runs": 3,
                "confidence_level": 0.95,
                "cooldown_seconds": 5,
            }
        )

    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_routes_to_single_benchmark_when_single_run(
        self,
        mock_single: Mock,
        plan_single_run: BenchmarkPlan,
    ):
        """Test that single run is called for single-run plans."""
        run_benchmark(plan_single_run)

        mock_single.assert_called_once()

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_routes_to_multi_benchmark_when_multi_run(
        self,
        mock_multi: Mock,
        plan_multi_run: BenchmarkPlan,
    ):
        """Test that multi-run is called for multi-run plans."""
        run_benchmark(plan_multi_run)

        mock_multi.assert_called_once_with(plan_multi_run)

    def test_raises_error_when_using_dashboard_ui_with_multi_run(
        self,
        plan_multi_run: BenchmarkPlan,
    ):
        """Test that an error is raised when explicitly using dashboard UI with multi-run."""
        for cfg in plan_multi_run.configs:
            cfg.runtime.ui = UIType.DASHBOARD

        with pytest.raises(
            ValueError, match="Dashboard UI is not supported with sweep/multi-run mode"
        ):
            _run_multi_benchmark(plan_multi_run)

    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_no_warning_when_using_simple_ui_with_multi_run(
        self,
        mock_multi: Mock,
        plan_multi_run: BenchmarkPlan,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that no warning is logged when using simple UI with multi-run."""
        for cfg in plan_multi_run.configs:
            cfg.runtime.ui = UIType.SIMPLE

        run_benchmark(plan_multi_run)

        assert not any(
            "Dashboard UI does not show live updates" in record.message
            for record in caplog.records
        )

    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_no_warning_when_using_dashboard_ui_with_single_run(
        self,
        mock_single: Mock,
        plan_single_run: BenchmarkPlan,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that no warning is logged when using dashboard UI with single run."""
        plan_single_run.configs[0].runtime.ui = UIType.DASHBOARD

        run_benchmark(plan_single_run)

        assert not any(
            "Dashboard UI does not show live updates" in record.message
            for record in caplog.records
        )


class TestRunSingleBenchmark:
    """Test the _run_single_benchmark function."""

    @pytest.fixture(autouse=True)
    def _mock_tokenizer_validation(self, mock_get_global_error_queue):
        """Prevent real HF network calls and forkserver init during single benchmark tests."""
        with patch(
            "aiperf.common.tokenizer_validator.validate_tokenizer_early",
            return_value={"test-model": "test-model"},
        ):
            yield

    @pytest.fixture
    def run_simple(self) -> BenchmarkRun:
        """Create a BenchmarkRun with Simple UI type."""
        return _make_run(runtime={"ui": "simple"})

    @patch("aiperf.config.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_simple_ui_uses_rich_logging(
        self,
        mock_setup_rich: Mock,
        mock_bootstrap: Mock,
        mock_chain: Mock,
        run_simple: BenchmarkRun,
    ):
        """Test that simple UI uses rich logging instead of log queue."""
        _run_single_benchmark(run_simple)

        mock_setup_rich.assert_called_once_with(run_simple.cfg)

        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs.get("log_queue") is None

    @patch("aiperf.config.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_bootstrap_exception_is_raised(
        self,
        mock_bootstrap: Mock,
        mock_chain: Mock,
        run: BenchmarkRun,
    ):
        """Test that exceptions from bootstrap are raised."""
        mock_bootstrap.side_effect = RuntimeError("Bootstrap failed")

        with pytest.raises(RuntimeError, match="Bootstrap failed"):
            _run_single_benchmark(run)


class TestRunMultiBenchmark:
    """Test the _run_multi_benchmark function."""

    @pytest.fixture
    def config_multi(self) -> BenchmarkPlan:
        """Create a BenchmarkPlan for multi-run."""
        return _make_plan(
            runtime={"ui": "simple"},
            multi_run={
                "num_runs": 3,
                "confidence_level": 0.95,
                "cooldown_seconds": 5,
                "disable_warmup_after_first": True,
            },
        )

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
    @patch("aiperf.orchestrator.aggregation.confidence.ConfidenceAggregation")
    @patch("aiperf.exporters.aggregate.AggregateConfidenceJsonExporter")
    @patch("aiperf.exporters.aggregate.AggregateConfidenceCsvExporter")
    def test_multi_run_success_with_aggregation(
        self,
        mock_csv_exporter_cls: Mock,
        mock_json_exporter_cls: Mock,
        mock_aggregation_cls: Mock,
        mock_orchestrator_cls: Mock,
        config_multi: BenchmarkPlan,
        mock_run_result: MagicMock,
        mock_aggregate_result: MagicMock,
        tmp_path: Path,
    ):
        """Test successful multi-run with aggregation."""
        config_multi.configs[0].artifacts.dir = tmp_path

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = MagicMock(
            return_value=[
                mock_run_result,
                mock_run_result,
                mock_run_result,
            ]
        )
        mock_orchestrator.get_aggregate_path.return_value = tmp_path / "aggregate"
        mock_orchestrator_cls.return_value = mock_orchestrator

        mock_aggregation = MagicMock()
        mock_aggregation.aggregate.return_value = mock_aggregate_result
        mock_aggregation_cls.return_value = mock_aggregation

        mock_json_exporter = MagicMock()
        mock_json_exporter.export = AsyncMock(return_value=tmp_path / "aggregate.json")
        mock_json_exporter_cls.return_value = mock_json_exporter

        mock_csv_exporter = MagicMock()
        mock_csv_exporter.export = AsyncMock(return_value=tmp_path / "aggregate.csv")
        mock_csv_exporter_cls.return_value = mock_csv_exporter

        _run_multi_benchmark(config_multi)

        mock_orchestrator_cls.assert_called_once()
        mock_orchestrator.execute.assert_called_once()

        mock_aggregation_cls.assert_called_once_with(confidence_level=0.95)
        mock_aggregation.aggregate.assert_called_once()

        mock_json_exporter.export.assert_called_once()
        mock_csv_exporter.export.assert_called_once()

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_orchestrator_exception(
        self,
        mock_orchestrator_cls: Mock,
        config_multi: BenchmarkPlan,
        tmp_path: Path,
    ):
        """Test that orchestrator exceptions are raised."""
        config_multi.configs[0].artifacts.dir = tmp_path

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = MagicMock(
            side_effect=RuntimeError("Orchestrator failed")
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(RuntimeError, match="Orchestrator failed"):
            _run_multi_benchmark(config_multi)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_only_one_successful_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        config_multi: BenchmarkPlan,
        mock_run_result: MagicMock,
        tmp_path: Path,
    ):
        """Test that only 1 successful run exits with error code 1."""
        config_multi.configs[0].artifacts.dir = tmp_path

        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_2"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = MagicMock(
            return_value=[
                mock_run_result,
                failed_result,
                failed_result,
            ]
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(config_multi)

        assert exc_info.value.code == 1

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_all_failed_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        config_multi: BenchmarkPlan,
        tmp_path: Path,
    ):
        """Test that all failed runs exit with error code 1."""
        config_multi.configs[0].artifacts.dir = tmp_path

        failed_result = MagicMock()
        failed_result.success = False
        failed_result.label = "run_1"

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = MagicMock(
            return_value=[
                failed_result,
                failed_result,
                failed_result,
            ]
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(config_multi)

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

        mock_metric = MagicMock()
        mock_metric.mean = 100.5
        mock_metric.std = 5.2
        mock_metric.min = 95.0
        mock_metric.max = 105.0
        mock_metric.cv = 0.052
        mock_metric.ci_low = 98.0
        mock_metric.ci_high = 103.0
        mock_metric.unit = "req/s"

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
        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

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
        _print_aggregate_summary(aggregate_result_with_failures, mock_logger)

        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]

        assert any("Failed Runs (1):" in call for call in warning_calls)
        assert any("run_3: Connection timeout" in call for call in warning_calls)

    def test_prints_key_metrics(
        self, aggregate_result_with_metrics: MagicMock, mock_logger: MagicMock
    ):
        """Test that key metrics are printed with correct formatting."""
        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        assert any("Request Throughput" in call for call in info_calls)
        assert any("Mean:" in call and "100.5000" in call for call in info_calls)
        assert any("Std Dev:" in call and "5.2000" in call for call in info_calls)
        assert any("CV:" in call and "5.20%" in call for call in info_calls)

    def test_prints_interpretation_guide(
        self, aggregate_result_with_metrics: MagicMock, mock_logger: MagicMock
    ):
        """Test that interpretation guide is printed."""
        _print_aggregate_summary(aggregate_result_with_metrics, mock_logger)

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
        result = MagicMock()
        result.aggregation_type = "confidence"
        result.num_runs = 3
        result.num_successful_runs = 3
        result.failed_runs = []
        result.metadata = {"confidence_level": 0.95}
        result.metrics = {}

        _print_aggregate_summary(result, mock_logger)

        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("No key metrics found" in call for call in warning_calls)

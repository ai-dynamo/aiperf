# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI convergence wiring in _run_multi_benchmark."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import ExportLevel
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.convergence.ci_width import CIWidthConvergence
from aiperf.orchestrator.convergence.cv import CVConvergence
from aiperf.orchestrator.convergence.distribution import DistributionConvergence
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import AdaptiveStrategy, FixedTrialsStrategy


def _make_user_config(
    num_profile_runs: int = 5,
    convergence_metric: str | None = None,
    convergence_mode: str = "ci_width",
    convergence_stat: str = "avg",
    convergence_threshold: float = 0.10,
    export_level: ExportLevel = ExportLevel.RECORDS,
    artifact_directory: Path | None = None,
) -> UserConfig:
    """Build a UserConfig with multi-run and convergence settings."""
    config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
    config.loadgen.num_profile_runs = num_profile_runs
    config.loadgen.convergence_metric = convergence_metric
    config.loadgen.convergence_mode = convergence_mode
    config.loadgen.convergence_stat = convergence_stat
    config.loadgen.convergence_threshold = convergence_threshold
    config.output.export_level = export_level
    if artifact_directory is not None:
        config.output.artifact_directory = artifact_directory
    return config


def _make_service_config() -> ServiceConfig:
    return ServiceConfig()


def _make_successful_results(count: int = 3) -> list[RunResult]:
    """Build a list of successful RunResult with minimal summary metrics."""
    results = []
    for i in range(count):
        results.append(
            RunResult(
                label=f"run_{i:04d}",
                success=True,
                summary_metrics={
                    "time_to_first_token": JsonMetricResult(
                        unit="ms",
                        avg=100.0 + i,
                        p50=99.0,
                        p90=110.0,
                        p95=115.0,
                        p99=120.0,
                    )
                },
                artifacts_path=None,
            )
        )
    return results


class TestCliConvergenceValidation:
    """Tests for convergence validation errors."""

    def test_convergence_metric_with_single_run_raises(self):
        config = _make_user_config(
            num_profile_runs=5, convergence_metric="time_to_first_token"
        )
        config.loadgen.num_profile_runs = 1

        with pytest.raises(
            ValueError, match="--convergence-metric requires --num-profile-runs > 1"
        ):
            from aiperf.cli_runner import _run_multi_benchmark

            _run_multi_benchmark(config, _make_service_config())

    def test_distribution_mode_with_summary_export_raises(self):
        config = _make_user_config(
            num_profile_runs=5,
            convergence_metric="time_to_first_token",
            convergence_mode="distribution",
            export_level=ExportLevel.SUMMARY,
        )

        with pytest.raises(
            ValueError,
            match="--convergence-mode distribution requires per-request JSONL",
        ):
            from aiperf.cli_runner import _run_multi_benchmark

            _run_multi_benchmark(config, _make_service_config())


class TestCliConvergenceStrategyWiring:
    """Tests for strategy and criterion creation based on convergence flags."""

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_no_convergence_flags_uses_fixed_trials(self, mock_orch_cls, tmp_path):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        config = _make_user_config(
            num_profile_runs=3, convergence_metric=None, artifact_directory=tmp_path
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(config, _make_service_config())

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, FixedTrialsStrategy)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_ci_width_mode_creates_adaptive_with_ci_width(
        self, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        config = _make_user_config(
            num_profile_runs=5,
            convergence_metric="time_to_first_token",
            convergence_mode="ci_width",
            convergence_stat="p99",
            convergence_threshold=0.05,
            artifact_directory=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(config, _make_service_config())

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CIWidthConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._stat == "p99"
        assert strategy.criterion._threshold == 0.05
        assert strategy.max_runs == 5

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_cv_mode_creates_adaptive_with_cv(self, mock_orch_cls, tmp_path):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        config = _make_user_config(
            num_profile_runs=5,
            convergence_metric="request_latency",
            convergence_mode="cv",
            convergence_threshold=0.08,
            artifact_directory=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(config, _make_service_config())

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CVConvergence)
        assert strategy.criterion._metric == "request_latency"
        assert strategy.criterion._threshold == 0.08

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_distribution_mode_creates_adaptive_with_distribution(
        self, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        config = _make_user_config(
            num_profile_runs=5,
            convergence_metric="time_to_first_token",
            convergence_mode="distribution",
            convergence_threshold=0.05,
            export_level=ExportLevel.RECORDS,
            artifact_directory=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(config, _make_service_config())

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, DistributionConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._p_value_threshold == 0.05


class TestCliConvergenceDefaults:
    """Tests for default convergence field values."""

    def test_default_convergence_metric_is_none(self):
        from aiperf.common.config import LoadGeneratorConfig

        config = LoadGeneratorConfig()
        assert config.convergence_metric is None

    def test_default_convergence_stat(self):
        from aiperf.common.config import LoadGeneratorConfig

        config = LoadGeneratorConfig()
        assert config.convergence_stat == "avg"

    def test_default_convergence_threshold(self):
        from aiperf.common.config import LoadGeneratorConfig

        config = LoadGeneratorConfig()
        assert config.convergence_threshold == 0.10

    def test_default_convergence_mode(self):
        from aiperf.common.config import LoadGeneratorConfig

        config = LoadGeneratorConfig()
        assert config.convergence_mode == "ci_width"

    def test_invalid_convergence_mode_raises(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(ValidationError, match="Input should be"):
            LoadGeneratorConfig(convergence_mode="invalid")

    def test_convergence_metric_set_with_single_run_raises_in_validator(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(
            ValueError,
            match="--convergence-metric only applies when --num-profile-runs > 1",
        ):
            LoadGeneratorConfig(
                convergence_metric="time_to_first_token", num_profile_runs=1
            )

    def test_convergence_mode_without_metric_raises(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(
            ValueError,
            match="--convergence-mode requires --convergence-metric to be set",
        ):
            LoadGeneratorConfig(convergence_mode="cv", num_profile_runs=5)

    def test_convergence_threshold_without_metric_raises(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(
            ValueError,
            match="--convergence-threshold requires --convergence-metric to be set",
        ):
            LoadGeneratorConfig(convergence_threshold=0.05, num_profile_runs=5)

    def test_convergence_stat_without_metric_raises(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(
            ValueError,
            match="--convergence-stat requires --convergence-metric to be set",
        ):
            LoadGeneratorConfig(convergence_stat="p99", num_profile_runs=5)

    def test_convergence_stat_with_distribution_mode_raises(self):
        from aiperf.common.config import LoadGeneratorConfig

        with pytest.raises(
            ValueError,
            match="--convergence-stat is not applicable with --convergence-mode distribution",
        ):
            LoadGeneratorConfig(
                convergence_metric="time_to_first_token",
                convergence_mode="distribution",
                convergence_stat="p99",
                num_profile_runs=5,
            )

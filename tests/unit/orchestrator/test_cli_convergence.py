# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI convergence wiring in _run_multi_benchmark."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import ConvergenceMode, ConvergenceStat
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.orchestrator.convergence.ci_width import CIWidthConvergence
from aiperf.orchestrator.convergence.cv import CVConvergence
from aiperf.orchestrator.convergence.distribution import DistributionConvergence
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import AdaptiveStrategy, FixedTrialsStrategy

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
    "phases": {
        "default": {"type": "concurrency", "requests": 100, "concurrency": 1},
    },
    "runtime": {"ui": "simple"},
    "random_seed": 42,
}


def _make_config(**overrides) -> BenchmarkConfig:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _make_plan(
    trials: int = 5,
    convergence_metric: str | None = None,
    convergence_mode: ConvergenceMode = ConvergenceMode.CI_WIDTH,
    convergence_stat: ConvergenceStat = ConvergenceStat.AVG,
    convergence_threshold: float = 0.10,
    export_level: str = "records",
    artifact_dir: Path | None = None,
    **overrides,
) -> BenchmarkPlan:
    """Build a BenchmarkPlan with convergence settings."""
    cfg = _make_config(
        artifacts={"dir": artifact_dir} if artifact_dir is not None else {},
    )
    return BenchmarkPlan(
        configs=[cfg],
        trials=trials,
        convergence_metric=convergence_metric,
        convergence_mode=convergence_mode,
        convergence_stat=convergence_stat,
        convergence_threshold=convergence_threshold,
        export_level=export_level,
        **overrides,
    )


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

    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_convergence_metric_with_single_run_raises(self, _tr, _adr, _log):
        plan = _make_plan(
            trials=1,
            convergence_metric="time_to_first_token",
        )

        with pytest.raises(
            ValueError, match="--convergence-metric requires --num-profile-runs > 1"
        ):
            from aiperf.cli_runner import _run_multi_benchmark

            _run_multi_benchmark(plan)

    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_distribution_mode_with_summary_export_raises(self, _tr, _adr, _log):
        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceMode.DISTRIBUTION,
            export_level="summary",
        )

        with pytest.raises(
            ValueError,
            match="--convergence-mode distribution requires per-request JSONL",
        ):
            from aiperf.cli_runner import _run_multi_benchmark

            _run_multi_benchmark(plan)


class TestCliConvergenceStrategyWiring:
    """Tests for strategy and criterion creation based on convergence flags."""

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_no_convergence_flags_uses_fixed_trials(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=3,
            convergence_metric=None,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(plan)

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, FixedTrialsStrategy)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_ci_width_mode_creates_adaptive_with_ci_width(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceMode.CI_WIDTH,
            convergence_stat=ConvergenceStat.P99,
            convergence_threshold=0.05,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(plan)

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CIWidthConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._stat == "p99"
        assert strategy.criterion._threshold == 0.05
        assert strategy.max_runs == 5

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_cv_mode_creates_adaptive_with_cv(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="request_latency",
            convergence_mode=ConvergenceMode.CV,
            convergence_threshold=0.08,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(plan)

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CVConvergence)
        assert strategy.criterion._metric == "request_latency"
        assert strategy.criterion._threshold == 0.08

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolvers.TimingResolver")
    def test_distribution_mode_creates_adaptive_with_distribution(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute.return_value = _make_successful_results(3)
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceMode.DISTRIBUTION,
            convergence_threshold=0.05,
            export_level="records",
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark

        _run_multi_benchmark(plan)

        strategy = mock_orch.execute.call_args[0][1]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, DistributionConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._p_value_threshold == 0.05


class TestCliConvergenceDefaults:
    """Tests for default convergence field values on BenchmarkPlan."""

    def test_default_convergence_metric_is_none(self):
        plan = _make_plan()
        assert plan.convergence_metric is None

    def test_default_convergence_stat(self):
        plan = _make_plan()
        assert plan.convergence_stat == ConvergenceStat.AVG

    def test_default_convergence_threshold(self):
        plan = _make_plan()
        assert plan.convergence_threshold == 0.10

    def test_default_convergence_mode(self):
        plan = _make_plan()
        assert plan.convergence_mode == ConvergenceMode.CI_WIDTH

    def test_invalid_convergence_mode_raises(self):
        with pytest.raises(ValueError, match="Input should be"):
            _make_plan(convergence_mode="invalid")

    def test_use_adaptive_false_when_no_metric(self):
        plan = _make_plan(convergence_metric=None)
        assert plan.use_adaptive is False

    def test_use_adaptive_true_when_metric_set(self):
        plan = _make_plan(convergence_metric="time_to_first_token")
        assert plan.use_adaptive is True

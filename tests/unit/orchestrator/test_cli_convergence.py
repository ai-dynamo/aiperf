# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI convergence wiring in _run_multi_benchmark."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ConvergenceStat, SweepMode
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.config.sweep import GridSweep
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig
from aiperf.orchestrator.convergence.ci_width import CIWidthConvergence
from aiperf.orchestrator.convergence.cv import CVConvergence
from aiperf.orchestrator.convergence.distribution import DistributionConvergence
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import AdaptiveStrategy, FixedTrialsStrategy
from aiperf.plugin.enums import ConvergenceCriterionType

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 100, "concurrency": 1}
    ],
    "runtime": {"ui": "simple"},
}


def _make_config(**overrides) -> BenchmarkConfig:
    overrides.pop("random_seed", None)
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _make_plan(
    trials: int = 5,
    convergence_metric: str | None = None,
    convergence_mode: ConvergenceCriterionType = ConvergenceCriterionType.CI_WIDTH,
    convergence_stat: ConvergenceStat = ConvergenceStat.AVG,
    convergence_threshold: float | None = None,
    convergence_min_runs: int = 2,
    export_level: str = "records",
    artifact_dir: Path | None = None,
    **overrides,
) -> BenchmarkPlan:
    """Build a BenchmarkPlan with convergence settings."""
    cfg = _make_config(
        artifacts={"dir": artifact_dir} if artifact_dir is not None else {},
    )
    convergence = (
        ConvergenceConfig(
            metric=convergence_metric,
            mode=convergence_mode,
            stat=convergence_stat,
            threshold=convergence_threshold,
            min_runs=convergence_min_runs,
        )
        if convergence_metric is not None
        else None
    )
    multi_run = MultiRunConfig(num_runs=trials, convergence=convergence)
    return BenchmarkPlan(
        configs=[cfg],
        trials=trials,
        multi_run=multi_run,
        sweep=GridSweep(
            parameters={"phases.profiling.concurrency": [1]},
            iteration_order=SweepMode.INDEPENDENT,
        ),
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
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_convergence_metric_with_single_run_raises(self, _tr, _adr, _log):
        # MultiRunConfig validator rejects min_runs > num_runs at construction
        # time, so a single-run plan with a convergence block fails before the
        # CLI runner's check. Verifies the same constraint, just earlier.
        with pytest.raises(
            ValueError,
            match="convergence.min_runs",
        ):
            _make_plan(
                trials=1,
                convergence_metric="time_to_first_token",
            )

    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_distribution_mode_with_summary_export_raises(self, _tr, _adr, _log):
        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.DISTRIBUTION,
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

    @pytest.fixture(autouse=True)
    def mock_os_exit(self):
        """Mock ``os._exit`` so the multi-run hang-protection terminator
        is a no-op under the test harness."""
        with patch("os._exit") as mock:
            yield mock

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_no_convergence_flags_uses_fixed_trials(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute = AsyncMock(return_value=_make_successful_results(3))
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=3,
            convergence_metric=None,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark
        from aiperf.cli_runner._strategy import build_strategy as real_build_strategy

        captured: list = []

        def spy(p, lg):
            s = real_build_strategy(p, lg)
            captured.append(s)
            return s

        with patch("aiperf.cli_runner._multi_run.build_strategy", side_effect=spy):
            _run_multi_benchmark(plan)

        assert isinstance(captured[0], FixedTrialsStrategy)

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_ci_width_mode_creates_adaptive_with_ci_width(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute = AsyncMock(return_value=_make_successful_results(3))
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.CI_WIDTH,
            convergence_stat=ConvergenceStat.P99,
            convergence_threshold=0.05,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark
        from aiperf.cli_runner._strategy import build_strategy as real_build_strategy

        captured: list = []

        def spy(p, lg):
            s = real_build_strategy(p, lg)
            captured.append(s)
            return s

        with patch("aiperf.cli_runner._multi_run.build_strategy", side_effect=spy):
            _run_multi_benchmark(plan)

        strategy = captured[0]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CIWidthConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._stat == "p99"
        assert strategy.criterion._threshold == 0.05
        assert strategy.max_runs == 5

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_cv_mode_creates_adaptive_with_cv(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute = AsyncMock(return_value=_make_successful_results(3))
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="request_latency",
            convergence_mode=ConvergenceCriterionType.CV,
            convergence_threshold=0.08,
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark
        from aiperf.cli_runner._strategy import build_strategy as real_build_strategy

        captured: list = []

        def spy(p, lg):
            s = real_build_strategy(p, lg)
            captured.append(s)
            return s

        with patch("aiperf.cli_runner._multi_run.build_strategy", side_effect=spy):
            _run_multi_benchmark(plan)

        strategy = captured[0]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CVConvergence)
        assert strategy.criterion._metric == "request_latency"
        assert strategy.criterion._threshold == 0.08

    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    @patch("aiperf.common.logging.setup_rich_logging")
    @patch("aiperf.config.resolution.resolvers.ArtifactDirResolver")
    @patch("aiperf.config.resolution.resolvers.TimingResolver")
    def test_distribution_mode_creates_adaptive_with_distribution(
        self, _tr, _adr, _log, mock_orch_cls, tmp_path
    ):
        mock_orch = MagicMock()
        mock_orch.execute = AsyncMock(return_value=_make_successful_results(3))
        mock_orch_cls.return_value = mock_orch

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.DISTRIBUTION,
            convergence_threshold=0.05,
            export_level="records",
            artifact_dir=tmp_path,
        )

        from aiperf.cli_runner import _run_multi_benchmark
        from aiperf.cli_runner._strategy import build_strategy as real_build_strategy

        captured: list = []

        def spy(p, lg):
            s = real_build_strategy(p, lg)
            captured.append(s)
            return s

        with patch("aiperf.cli_runner._multi_run.build_strategy", side_effect=spy):
            _run_multi_benchmark(plan)

        strategy = captured[0]
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, DistributionConvergence)
        assert strategy.criterion._metric == "time_to_first_token"
        assert strategy.criterion._p_value_threshold == 0.05


class TestCliConvergenceDefaults:
    """Tests for default convergence field values on BenchmarkPlan."""

    def test_default_convergence_metric_is_none(self):
        plan = _make_plan()
        assert plan.multi_run.convergence is None

    def test_default_convergence_stat(self):
        plan = _make_plan(convergence_metric="time_to_first_token")
        assert plan.multi_run.convergence is not None
        assert plan.multi_run.convergence.stat == ConvergenceStat.AVG

    def test_default_convergence_threshold(self):
        plan = _make_plan(convergence_metric="time_to_first_token")
        assert plan.multi_run.convergence is not None
        assert plan.multi_run.convergence.threshold is None

    def test_default_convergence_mode(self):
        plan = _make_plan(convergence_metric="time_to_first_token")
        assert plan.multi_run.convergence is not None
        assert plan.multi_run.convergence.mode == ConvergenceCriterionType.CI_WIDTH

    def test_invalid_convergence_mode_raises(self):
        with pytest.raises(ValueError, match="Input should be"):
            _make_plan(
                convergence_metric="time_to_first_token", convergence_mode="invalid"
            )

    def test_use_adaptive_false_when_no_metric(self):
        plan = _make_plan(convergence_metric=None)
        assert plan.use_adaptive is False

    def test_use_adaptive_true_when_metric_set(self):
        plan = _make_plan(convergence_metric="time_to_first_token")
        assert plan.use_adaptive is True


class TestBuildStrategyMinRunsPropagation:
    """`build_strategy` must thread `convergence.min_runs` into AdaptiveStrategy.

    Pre-fix, AdaptiveStrategy.min_runs was hardcoded to `min(3, plan.trials)`,
    silently ignoring the user's `convergence.min_runs` setting. These tests
    pin the new contract: the strategy's pre-criterion gate and the criterion's
    own internal floor are both driven by `ConvergenceConfig.min_runs`.
    """

    def test_propagates_min_runs_into_adaptive_strategy(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(
            trials=10,
            convergence_metric="time_to_first_token",
            convergence_min_runs=7,
        )
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.min_runs == 7

    def test_propagates_min_runs_into_criterion(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(
            trials=10,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.CV,
            convergence_min_runs=6,
        )
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, AdaptiveStrategy)
        assert isinstance(strategy.criterion, CVConvergence)
        assert strategy.criterion._min_runs == 6

    def test_min_runs_below_three_emits_warning(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_min_runs=2,
        )
        logger = MagicMock()
        build_strategy(plan, logger)
        assert logger.warning.called
        msg = logger.warning.call_args[0][0]
        assert "convergence.min_runs=2" in msg
        assert "recommended minimum of 3" in msg

    def test_min_runs_at_least_three_does_not_warn(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_min_runs=3,
        )
        logger = MagicMock()
        build_strategy(plan, logger)
        assert not logger.warning.called

    def test_strategy_and_criterion_min_runs_agree(self):
        """The two `min_runs` knobs must be sourced from the same config field.

        Prevents regression where one is fixed but not the other, leaving an
        invisible mismatch (criterion permits convergence at N samples but
        strategy still won't ask, or vice versa).
        """
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(
            trials=10,
            convergence_metric="time_to_first_token",
            convergence_min_runs=5,
        )
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.min_runs == strategy.criterion._min_runs == 5


class TestBuildStrategyRouting:
    """`build_strategy` routes to FixedTrialsStrategy or AdaptiveStrategy.

    Pins the routing contract independently of the heavier
    `_run_multi_benchmark` end-to-end tests: the routing decision is owned
    by `plan.use_adaptive`, which in turn is owned by
    `multi_run.convergence is not None`. A third call site that bypasses
    `build_strategy` would silently desync from this contract.
    """

    def test_returns_fixed_trials_when_not_adaptive(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(trials=3, convergence_metric=None)
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, FixedTrialsStrategy)

    def test_returns_adaptive_when_convergence_set(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(trials=5, convergence_metric="time_to_first_token")
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, AdaptiveStrategy)

    def test_adaptive_max_runs_equals_plan_trials(self):
        """`AdaptiveStrategy.max_runs` is the user's hard ceiling on trials."""
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(trials=8, convergence_metric="time_to_first_token")
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.max_runs == 8

    def test_fixed_trials_propagates_cooldown(self):
        from aiperf.cli_runner._strategy import build_strategy

        plan = _make_plan(trials=3, convergence_metric=None)
        # `_make_plan` doesn't expose cooldown; assert on default to confirm
        # the propagation path is wired (not synthesized to a different value).
        strategy = build_strategy(plan, MagicMock())
        assert isinstance(strategy, FixedTrialsStrategy)
        assert strategy.cooldown_seconds == plan.cooldown_seconds


class TestValidateConvergenceConfig:
    """Focused tests for `validate_convergence_config`.

    Existing distribution+summary coverage routes through the heavyweight
    `_run_multi_benchmark` path; these tests pin the validator's contract
    directly so its branches stay testable as the surrounding CLI changes.
    """

    def test_no_op_when_not_adaptive(self):
        from aiperf.cli_runner._strategy import validate_convergence_config

        plan = _make_plan(trials=3, convergence_metric=None)
        # Must not raise even with export_level=summary etc.
        validate_convergence_config(plan)

    def test_distribution_with_summary_export_raises(self):
        from aiperf.cli_runner._strategy import validate_convergence_config

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.DISTRIBUTION,
            export_level="summary",
        )
        with pytest.raises(
            ValueError,
            match="--convergence-mode distribution requires per-request JSONL",
        ):
            validate_convergence_config(plan)

    def test_distribution_with_records_export_passes(self):
        from aiperf.cli_runner._strategy import validate_convergence_config

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.DISTRIBUTION,
            export_level="records",
        )
        validate_convergence_config(plan)

    def test_ci_width_with_summary_export_passes(self):
        """Only DISTRIBUTION mode requires per-request data; CI-width is fine on summary."""
        from aiperf.cli_runner._strategy import validate_convergence_config

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.CI_WIDTH,
            export_level="summary",
        )
        validate_convergence_config(plan)

    def test_cv_with_summary_export_passes(self):
        from aiperf.cli_runner._strategy import validate_convergence_config

        plan = _make_plan(
            trials=5,
            convergence_metric="time_to_first_token",
            convergence_mode=ConvergenceCriterionType.CV,
            export_level="summary",
        )
        validate_convergence_config(plan)

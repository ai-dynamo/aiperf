# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plan-driven cli_runner.py entry points."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.sweep import GridSweep, SweepVariation
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import FixedTrialsStrategy
from aiperf.plugin.enums import UIType

_MINIMAL_CONFIG = {
    "models": ["test-model"],
    "endpoint": {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "wait_for_model_timeout": 0,
    },
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        }
    ],
    "runtime": {"ui": UIType.SIMPLE},
}


def _make_config(**overrides) -> BenchmarkConfig:
    data = {**_MINIMAL_CONFIG, **overrides}
    return BenchmarkConfig.model_validate(data)


def _make_run(
    config: BenchmarkConfig | None = None, artifact_dir: Path | None = None
) -> BenchmarkRun:
    cfg = config or _make_config()
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=artifact_dir or cfg.artifacts.dir,
    )


def _make_plan(
    *,
    config: BenchmarkConfig | None = None,
    trials: int = 1,
    configs: list[BenchmarkConfig] | None = None,
    sweep: GridSweep | None = None,
    variations: list[SweepVariation] | None = None,
    cooldown_seconds: float = 0.0,
) -> BenchmarkPlan:
    plan_configs = configs or [config or _make_config()]
    if variations is None:
        variations = [
            SweepVariation(index=i, label=f"variation_{i:04d}", values={})
            for i in range(len(plan_configs))
        ]
    return BenchmarkPlan(
        configs=plan_configs,
        variations=variations,
        trials=trials,
        cooldown_seconds=cooldown_seconds,
        sweep=sweep,
    )


class TestRunBenchmark:
    """Test run_benchmark routing for BenchmarkPlan inputs."""

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_routes_single_run_plan_to_single_benchmark(
        self,
        mock_single: Mock,
        mock_artifact: Mock,
        mock_fd: Mock,
        mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        plan = _make_plan()

        run_benchmark(plan)

        mock_single.assert_called_once()
        run = mock_single.call_args.args[0]
        assert isinstance(run, BenchmarkRun)
        assert run.cfg is plan.configs[0]
        mock_artifact.assert_called_once_with(plan)
        mock_fd.assert_called_once_with()
        mock_endpoint.assert_called_once_with(plan)

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_routes_multi_trial_plan_to_multi_benchmark(
        self,
        mock_multi: Mock,
        mock_artifact: Mock,
        mock_fd: Mock,
        mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        plan = _make_plan(trials=3)

        run_benchmark(plan)

        mock_multi.assert_called_once_with(plan, on_complete=[])
        mock_artifact.assert_called_once_with(plan)
        mock_fd.assert_called_once_with()
        mock_endpoint.assert_called_once_with(plan)

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_routes_sweep_plan_to_multi_benchmark(
        self,
        mock_multi: Mock,
        mock_artifact: Mock,
        mock_fd: Mock,
        mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        cfg = _make_config()
        plan = _make_plan(
            configs=[cfg, cfg.model_copy(deep=True)],
            variations=[
                SweepVariation(
                    index=0, label="concurrency=1", values={"concurrency": 1}
                ),
                SweepVariation(
                    index=1, label="concurrency=2", values={"concurrency": 2}
                ),
            ],
            sweep=GridSweep(parameters={"phases.profiling.concurrency": [1, 2]}),
        )

        run_benchmark(plan)

        mock_multi.assert_called_once_with(plan, on_complete=[])
        mock_artifact.assert_called_once_with(plan)
        mock_fd.assert_called_once_with()
        mock_endpoint.assert_called_once_with(plan)


class TestRunSingleBenchmark:
    """Test the _run_single_benchmark function."""

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_simple_ui_uses_rich_logging(
        self,
        mock_setup_rich: Mock,
        mock_bootstrap: Mock,
        mock_resolver_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        config = _make_config(runtime={"ui": UIType.SIMPLE})
        run = _make_run(config)
        chain = MagicMock()
        mock_resolver_chain.return_value = chain

        _run_single_benchmark(run)

        mock_setup_rich.assert_called_once_with(run)
        chain.resolve_all.assert_called_once_with(run)
        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["run"] is run
        assert call_kwargs.get("log_queue") is None
        mock_exit.assert_called_once_with(0)

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_bootstrap_exception_exits_nonzero(
        self,
        mock_bootstrap: Mock,
        mock_resolver_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_resolver_chain.return_value = MagicMock()
        mock_bootstrap.side_effect = RuntimeError("Bootstrap failed")

        # Production: os._exit terminates the process. With os._exit mocked
        # to a no-op the runner falls through to sys.exit(exit_code) so the
        # failure still surfaces — assert both paths fire.
        with pytest.raises(SystemExit) as excinfo:
            _run_single_benchmark(run)

        assert excinfo.value.code == 1
        mock_exit.assert_called_once_with(1)


class TestRunMultiBenchmark:
    """Test the _run_multi_benchmark function."""

    @pytest.fixture(autouse=True)
    def mock_rich_logging(self):
        with patch("aiperf.common.logging.setup_rich_logging") as mock:
            yield mock

    @pytest.fixture(autouse=True)
    def mock_os_exit(self):
        """Mock ``os._exit`` so the multi-run hang-protection terminator is
        a no-op under the test harness; the runner falls through to
        ``sys.exit`` which pytest catches as ``SystemExit``."""
        with patch("os._exit") as mock:
            yield mock

    @pytest.fixture
    def successful_result(self, tmp_path: Path) -> RunResult:
        return RunResult(label="run_0001", success=True, artifacts_path=tmp_path)

    @patch("aiperf.cli_runner._multi_run.aggregate_and_export", new_callable=AsyncMock)
    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_success_with_aggregation(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        mock_aggregate: AsyncMock,
        successful_result: RunResult,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=[successful_result] * 3)
        mock_orchestrator_cls.return_value = mock_orchestrator

        _run_multi_benchmark(plan)

        mock_orchestrator_cls.assert_called_once()
        call_kwargs = mock_orchestrator_cls.call_args.kwargs
        assert call_kwargs["base_dir"] == tmp_path
        # cell_callback is either None or a SweepTableLogger; not asserting which here.
        assert "cell_callback" in call_kwargs
        mock_orchestrator.execute.assert_awaited_once()
        execute_args = mock_orchestrator.execute.await_args.args
        assert execute_args[0] is plan
        assert mock_orchestrator.execute.await_args.kwargs["search_planner"] is None
        mock_aggregate.assert_awaited_once()
        assert isinstance(
            mock_aggregate.await_args.kwargs["strategy"], FixedTrialsStrategy
        )

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_orchestrator_exception(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(
            side_effect=RuntimeError("Orchestrator failed")
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(RuntimeError, match="Orchestrator failed"):
            _run_multi_benchmark(plan)

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_only_one_successful_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        successful_result: RunResult,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        failed_result = RunResult(label="run_0002", success=False, error="timeout")
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(
            return_value=[successful_result, failed_result, failed_result]
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(plan)

        assert exc_info.value.code == 1

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_multi_run_all_failed_exits_with_error(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        failed_result = RunResult(label="run_0001", success=False, error="timeout")
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=[failed_result] * 3)
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            _run_multi_benchmark(plan)

        assert exc_info.value.code == 1

    def test_dashboard_ui_rejected_for_multi_run(self):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(
            trials=2, config=_make_config(runtime={"ui": UIType.DASHBOARD})
        )

        with pytest.raises(ValueError, match="Dashboard UI is not supported"):
            _run_multi_benchmark(plan)

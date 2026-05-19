# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OnComplete callback contract on cli_runner.

Covers:
  * CompletedRun dataclass shape (frozen, slots).
  * ``run_benchmark`` builds an auto-plot callback only when
    ``BenchmarkConfig.artifacts.auto_plot`` is True, and threads it through
    to the dispatch path.
  * ``_run_single_benchmark`` invokes callbacks in list order on success and
    skips them on bootstrap failure.
  * ``_run_multi_benchmark`` invokes callbacks in list order on successful
    orchestrator run and skips them when the orchestrator raises.
  * Callback failures are isolated: subsequent callbacks still run, the
    process exits non-zero, and the traceback is logged. Strict mode is
    opt-in via ``AIPERF_RAISE_ON_CALLBACK_ERROR``.
  * The 1-successful-run partial-failure path runs callbacks against the
    surviving trial's artifacts before propagating the non-zero exit.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.cli_runner import CompletedRun, OnComplete
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.orchestrator.models import RunResult
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
) -> BenchmarkPlan:
    from aiperf.config.sweep import SweepVariation

    plan_configs = [config or _make_config()]
    return BenchmarkPlan(
        configs=plan_configs,
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=trials,
        cooldown_seconds=0.0,
        sweep=None,
    )


class TestCompletedRunDataclass:
    """CompletedRun is the public payload type passed to OnComplete."""

    def test_completed_run_is_frozen_and_has_artifact_dir(self, tmp_path: Path):
        completed = CompletedRun(artifact_dir=tmp_path)
        assert completed.artifact_dir == tmp_path
        with pytest.raises((AttributeError, Exception)):
            completed.artifact_dir = Path("/other")  # type: ignore[misc]

    def test_completed_run_uses_slots(self, tmp_path: Path):
        completed = CompletedRun(artifact_dir=tmp_path)
        assert hasattr(type(completed), "__slots__")
        assert "artifact_dir" in type(completed).__slots__

    def test_oncomplete_alias_callable_signature(self):
        # Smoke-test: any callable taking CompletedRun -> None satisfies the
        # alias. We don't enforce isinstance at runtime; this keeps the test
        # honest for mypy users.
        def _hook(run: CompletedRun) -> None:
            return None

        cb: OnComplete = _hook
        assert callable(cb)


class TestRunBenchmarkAutoPlotWiring:
    """run_benchmark builds the auto-plot callback only when configured."""

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_no_auto_plot_passes_empty_callback_list(
        self,
        mock_single: Mock,
        _mock_artifact: Mock,
        _mock_fd: Mock,
        _mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        plan = _make_plan()
        # default for ArtifactsConfig.auto_plot is False
        assert plan.configs[0].artifacts.auto_plot is False

        run_benchmark(plan)

        on_complete = mock_single.call_args.kwargs["on_complete"]
        assert on_complete == []

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_single_benchmark")
    def test_auto_plot_on_appends_one_callback(
        self,
        mock_single: Mock,
        _mock_artifact: Mock,
        _mock_fd: Mock,
        _mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        plan = _make_plan()
        # Mutate post-construction; ArtifactsConfig is a plain BaseConfig
        # (Pydantic) and these fields are not frozen.
        plan.configs[0].artifacts.auto_plot = True
        plan.configs[0].artifacts.plot_required = False

        sentinel = object()
        with patch(
            "aiperf.plot.auto_plot.build_auto_plot_callback",
            return_value=sentinel,
        ) as mock_build:
            run_benchmark(plan)

        mock_build.assert_called_once_with(plot_required=False, plot_envelope=None)
        on_complete = mock_single.call_args.kwargs["on_complete"]
        assert on_complete == [sentinel]

    @patch("aiperf.cli_runner._preflight_endpoint_ready")
    @patch("aiperf.cli_runner._preflight_fd_limit")
    @patch("aiperf.cli_runner._preflight_artifact_dir")
    @patch("aiperf.cli_runner._run_multi_benchmark")
    def test_auto_plot_multi_run_path_passes_callback(
        self,
        mock_multi: Mock,
        _mock_artifact: Mock,
        _mock_fd: Mock,
        _mock_endpoint: Mock,
    ):
        from aiperf.cli_runner import run_benchmark

        plan = _make_plan(trials=3)
        plan.configs[0].artifacts.auto_plot = True
        plan.configs[0].artifacts.plot_required = True

        sentinel = object()
        with patch(
            "aiperf.plot.auto_plot.build_auto_plot_callback",
            return_value=sentinel,
        ) as mock_build:
            run_benchmark(plan)

        mock_build.assert_called_once_with(plot_required=True, plot_envelope=None)
        on_complete = mock_multi.call_args.kwargs["on_complete"]
        assert on_complete == [sentinel]


class TestRunSingleBenchmarkCallbacks:
    """_run_single_benchmark invokes callbacks on success, skips on failure."""

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_callbacks_invoked_in_order_on_success(
        self,
        _mock_setup: Mock,
        _mock_bootstrap: Mock,
        mock_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()

        order: list[str] = []
        cb_a = Mock(side_effect=lambda r: order.append("a"))
        cb_b = Mock(side_effect=lambda r: order.append("b"))

        _run_single_benchmark(run, on_complete=[cb_a, cb_b])

        assert order == ["a", "b"]
        for cb in (cb_a, cb_b):
            cb.assert_called_once()
            payload = cb.call_args.args[0]
            assert isinstance(payload, CompletedRun)
            assert payload.artifact_dir == run.artifact_dir
        mock_exit.assert_called_once_with(0)

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_callbacks_skipped_when_bootstrap_raises(
        self,
        _mock_setup: Mock,
        mock_bootstrap: Mock,
        mock_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()
        mock_bootstrap.side_effect = RuntimeError("Bootstrap failed")

        cb = Mock()

        # os._exit is mocked to a no-op; runner falls through to sys.exit.
        with pytest.raises(SystemExit) as excinfo:
            _run_single_benchmark(run, on_complete=[cb])

        assert excinfo.value.code == 1
        cb.assert_not_called()
        mock_exit.assert_called_once_with(1)

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_no_callbacks_provided_runs_normally(
        self,
        _mock_setup: Mock,
        _mock_bootstrap: Mock,
        mock_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()

        # Default on_complete=None, no callbacks to invoke; should still exit 0.
        _run_single_benchmark(run)

        mock_exit.assert_called_once_with(0)


class TestRunMultiBenchmarkCallbacks:
    """_run_multi_benchmark invokes callbacks on success, skips on failure."""

    @pytest.fixture(autouse=True)
    def mock_rich_logging(self):
        with patch("aiperf.common.logging.setup_rich_logging") as mock:
            yield mock

    @pytest.fixture(autouse=True)
    def mock_os_exit(self):
        """Mock ``os._exit`` so the multi-run hang-protection terminator
        is a no-op under the test harness; the runner falls through to
        ``sys.exit`` which pytest catches as ``SystemExit``."""
        with patch("os._exit") as mock:
            yield mock

    @pytest.fixture
    def successful_result(self, tmp_path: Path) -> RunResult:
        return RunResult(label="run_0001", success=True, artifacts_path=tmp_path)

    @patch("aiperf.cli_runner._multi_run.aggregate_and_export", new_callable=AsyncMock)
    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_callbacks_invoked_in_order_after_success(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        _mock_aggregate: AsyncMock,
        successful_result: RunResult,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=[successful_result] * 3)
        mock_orchestrator_cls.return_value = mock_orchestrator

        order: list[str] = []
        cb_a = Mock(side_effect=lambda r: order.append("a"))
        cb_b = Mock(side_effect=lambda r: order.append("b"))

        _run_multi_benchmark(plan, on_complete=[cb_a, cb_b])

        assert order == ["a", "b"]
        for cb in (cb_a, cb_b):
            cb.assert_called_once()
            payload = cb.call_args.args[0]
            assert isinstance(payload, CompletedRun)
            assert payload.artifact_dir == plan.configs[0].artifacts.dir

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_callbacks_skipped_when_orchestrator_raises(
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

        cb = Mock()

        with pytest.raises(RuntimeError, match="Orchestrator failed"):
            _run_multi_benchmark(plan, on_complete=[cb])

        cb.assert_not_called()

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_callbacks_skipped_when_summarize_exits_on_all_failures(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        failed = RunResult(
            label="run_0001", success=False, error="boom", artifacts_path=tmp_path
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=[failed, failed, failed])
        mock_orchestrator_cls.return_value = mock_orchestrator

        cb = Mock()

        # No successful runs => the runner exits non-zero before any callback
        # could be invoked. Callbacks are still skipped in this branch (no
        # artifacts to plot/export), but the SystemExit now comes from
        # _run_multi_benchmark's tail propagation rather than from
        # _summarize_and_export directly.
        with pytest.raises(SystemExit) as excinfo:
            _run_multi_benchmark(plan, on_complete=[cb])

        assert excinfo.value.code == 1
        cb.assert_not_called()


class TestCallbackFailureIsolation:
    """OnComplete callback exceptions must not bypass ``os._exit``.

    The hang-protection ``os._exit`` exists because Python teardown can hang
    on multiprocessing atexit handlers, leftover ZMQ contexts, and daemon
    threads. A callback raise (e.g. auto-plot in strict mode) used to
    propagate out of the runner BEFORE that ``os._exit``, dropping the run
    into the very teardown path the comment says is unsafe — and silently
    skipping subsequent callbacks. These tests pin the new contract:
    failures are logged, exit code is forced non-zero, every callback runs.
    """

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_callback_exception_does_not_skip_subsequent_callbacks(
        self,
        _mock_setup: Mock,
        _mock_bootstrap: Mock,
        mock_chain: Mock,
        mock_exit: Mock,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()

        order: list[str] = []

        def cb_a(_completed: CompletedRun) -> None:
            order.append("a")

        def cb_b(_completed: CompletedRun) -> None:
            order.append("b")
            raise RuntimeError("auto-plot failed strictly")

        def cb_c(_completed: CompletedRun) -> None:
            order.append("c")

        # Production: os._exit terminates the process. Under the test harness
        # os._exit is mocked to a no-op, so the runner falls through to
        # sys.exit(exit_code) — exactly what the comment in cli_runner.py
        # documents — and the SystemExit surfaces here.
        with pytest.raises(SystemExit) as excinfo:
            _run_single_benchmark(run, on_complete=[cb_a, cb_b, cb_c])

        # All three callbacks ran in order despite the middle one raising.
        assert order == ["a", "b", "c"]
        # Exit code is non-zero because a callback failed.
        assert excinfo.value.code != 0
        # os._exit was still called exactly once with that non-zero code.
        mock_exit.assert_called_once()
        (code,) = mock_exit.call_args.args
        assert code == excinfo.value.code
        assert code != 0

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_callback_exception_logs_traceback(
        self,
        _mock_setup: Mock,
        _mock_bootstrap: Mock,
        mock_chain: Mock,
        _mock_exit: Mock,
        caplog: pytest.LogCaptureFixture,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()

        def boom(_completed: CompletedRun) -> None:
            raise RuntimeError("kaleido missing")

        with (
            caplog.at_level(logging.ERROR, logger="aiperf.cli_runner._single_run"),
            pytest.raises(SystemExit),
        ):
            _run_single_benchmark(run, on_complete=[boom])

        # logger.exception emits an ERROR record with exc_info attached so
        # the traceback is rendered in the log file. We assert both the
        # diagnostic header and the underlying exception text.
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, "expected at least one ERROR-level record"
        rendered = "\n".join(r.getMessage() for r in error_records)
        assert "OnComplete callback" in rendered
        assert "continuing with remaining callbacks" in rendered
        # exc_info must be set so the traceback is captured.
        assert any(r.exc_info is not None for r in error_records), (
            "expected at least one record with exc_info set (traceback)"
        )

    @patch("os._exit")
    @patch("aiperf.config.resolution.resolvers.build_default_resolver_chain")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.setup_rich_logging")
    def test_raise_on_callback_error_env_var(
        self,
        _mock_setup: Mock,
        _mock_bootstrap: Mock,
        mock_chain: Mock,
        mock_exit: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from aiperf.cli_runner import _run_single_benchmark

        run = _make_run(_make_config(runtime={"ui": UIType.SIMPLE}))
        mock_chain.return_value = MagicMock()

        order: list[str] = []

        def cb_a(_completed: CompletedRun) -> None:
            order.append("a")
            raise RuntimeError("strict-mode failure")

        def cb_b(_completed: CompletedRun) -> None:
            order.append("b")

        monkeypatch.setenv("AIPERF_RAISE_ON_CALLBACK_ERROR", "true")

        # Strict mode re-raises the FIRST captured exception, but only after
        # every remaining callback has been attempted (so cleanup hooks
        # downstream of the failing one still run).
        with pytest.raises(RuntimeError, match="strict-mode failure"):
            _run_single_benchmark(run, on_complete=[cb_a, cb_b])

        assert order == ["a", "b"]
        # os._exit is unreachable past the raise; the test harness mocks it
        # to a no-op anyway. The contract under strict mode is that the
        # exception surfaces — which it did.
        mock_exit.assert_not_called()


class TestMultiBenchmarkOneSuccessRunsCallbacks:
    """The 1-success/N-failure path must still run callbacks.

    A single trial's per-run JSONL/CSV/JSON exist on disk; downstream hooks
    (auto-plot, exporters) can render them. Previously the runner exited
    before reaching the callback loop — now we exit AFTER running them.
    """

    @pytest.fixture(autouse=True)
    def mock_rich_logging(self):
        with patch("aiperf.common.logging.setup_rich_logging") as mock:
            yield mock

    @pytest.fixture(autouse=True)
    def mock_os_exit(self):
        """Mock ``os._exit`` so the multi-run hang-protection terminator
        is a no-op under the test harness; the runner falls through to
        ``sys.exit`` which pytest catches as ``SystemExit``."""
        with patch("os._exit") as mock:
            yield mock

    @patch("aiperf.cli_runner._multi_run._estimate_and_log_duration")
    @patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator")
    def test_one_successful_run_still_runs_callbacks(
        self,
        mock_orchestrator_cls: Mock,
        mock_estimate: Mock,
        tmp_path: Path,
    ):
        from aiperf.cli_runner import _run_multi_benchmark

        plan = _make_plan(trials=3)
        mock_estimate.return_value = tmp_path
        success = RunResult(label="run_0001", success=True, artifacts_path=tmp_path)
        failed = RunResult(
            label="run_0002", success=False, error="timeout", artifacts_path=tmp_path
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=[success, failed, failed])
        mock_orchestrator_cls.return_value = mock_orchestrator

        invocations: list[CompletedRun] = []

        def real_callback(completed: CompletedRun) -> None:
            invocations.append(completed)

        # The 1-successful-run branch still raises SystemExit(1) (insufficient
        # data for confidence aggregation) but only AFTER the callback
        # observes the surviving artifact directory.
        with pytest.raises(SystemExit) as excinfo:
            _run_multi_benchmark(plan, on_complete=[real_callback])

        assert excinfo.value.code == 1
        assert len(invocations) == 1
        assert invocations[0].artifact_dir == plan.configs[0].artifacts.dir

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-benchmark execution through the pure-Python service mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.cli_runner._callbacks import (
    CompletedRun,
    OnComplete,
    _invoke_callbacks,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
    from aiperf.orchestrator.models import RunResult


def _execute_native_run(run: BenchmarkRun) -> RunResult:
    """Execute one resolved run on the legacy pure-Python service mesh.

    The Python frontend (``python -m aiperf.cli profile``) never bridges to the
    native ``aiperf`` binary: it runs entirely in-process on the service mesh
    (SystemController -> Worker/TimingManager/RecordsManager). The native
    execution engine is reached only by invoking the ``aiperf`` binary directly.
    """
    return _execute_legacy_python_run(run)


def _execute_legacy_python_run(run: BenchmarkRun) -> RunResult:
    """Run the legacy pure-Python service mesh in-process for this one run.

    Boots the ``SystemController`` (which spawns the Worker / TimingManager /
    RecordsManager children) against the same ``BenchmarkRun`` the native path
    consumes. ``bootstrap_and_run_service`` blocks until the mesh drains and
    raises ``SystemExit(1)`` on service failure; a normal return means the run
    completed and wrote its artifacts.
    """
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.orchestrator.models import RunResult
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.SYSTEM_CONTROLLER, run=run)

    return RunResult(
        label=run.label or f"run_{run.trial:04d}",
        success=True,
        artifacts_path=run.artifact_dir,
    )


def _run_single_benchmark(
    run: BenchmarkRun,
    *,
    on_complete: list[OnComplete] | None = None,
) -> None:
    """Run a single benchmark.

    Args:
        run: BenchmarkRun to execute.
        on_complete: Optional list of callbacks invoked in list order after a
            successful run (exit_code == 0). Skipped on failure. Each
            callback is isolated by ``_invoke_callbacks``: an exception is
            logged, the exit code is forced non-zero, and remaining callbacks
            still run. ``AIPERF_RAISE_ON_CALLBACK_ERROR=true`` opts into
            re-raising the first failure after all callbacks have run.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging

    setup_rich_logging(run)
    logger = AIPerfLogger(__name__)

    logger.info("Starting native AIPerf run")
    try:
        result = _execute_native_run(run)
    except Exception:
        logger.exception("Native AIPerf runner could not be started")
        exit_code = 1
    else:
        exit_code = 0 if result.success else 1
        if result.success:
            logger.info("Native AIPerf run completed")
        else:
            logger.error(f"Native AIPerf run failed: {result.error or 'unknown error'}")

    if exit_code == 0 and on_complete:
        completed = CompletedRun(artifact_dir=run.artifact_dir)
        exit_code = _invoke_callbacks(on_complete, completed, exit_code, logger)

    # Keep the established CLI termination contract. The benchmark itself has
    # already completed in an isolated Rust child and all callbacks are flushed.
    import os as _os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(exit_code)
    # Production never reaches here (``os._exit`` terminates the process).
    # The component-integration test harness mocks ``os._exit`` to a no-op,
    # so re-raise via ``sys.exit`` to surface the failure as a SystemExit
    # the harness can catch.
    if exit_code:
        sys.exit(exit_code)

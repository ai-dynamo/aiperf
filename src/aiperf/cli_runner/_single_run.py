# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-benchmark execution path for aiperf.cli_runner.

One ``_run_single_benchmark`` call runs one BenchmarkRun under a fresh
SystemController, then ``os._exit``-s to bypass Python's normal teardown
(multiprocessing atexit + leftover ZMQ contexts can otherwise hang the
interpreter under pytest-xdist).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from aiperf.cli_runner._callbacks import (
    CompletedRun,
    OnComplete,
    _invoke_callbacks,
)
from aiperf.cli_runner._process_setup import (
    _configure_multiprocessing_start_method,
    _configure_tokenizer_preload,
    _setup_ui_queues,
)
from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


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
    config = run.cfg
    using_dashboard = config.ui_type == UIType.DASHBOARD

    _configure_multiprocessing_start_method(using_dashboard)
    _configure_tokenizer_preload(run)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.config.resolution.resolvers import build_default_resolver_chain

    logger = AIPerfLogger(__name__)

    # Create queues before UI initialization to minimize FD inheritance issues.
    log_queue = _setup_ui_queues(using_dashboard, run, logger)

    logger.info("Starting AIPerf System")

    try:
        chain = build_default_resolver_chain()
        chain.resolve_all(run)
    except Exception as e:  # resolver chain wraps every user-input error type
        # ``logger.error`` over ``.exception``: user-input errors carry their
        # own context; tracebacks trip chaos-harness crash heuristics.
        logger.error(f"Configuration resolution failed: {e}")
        raise_startup_error_and_exit(
            f"Configuration resolution failed: {e}",
            title="Configuration Error",
        )

    exit_code = 0
    try:
        bootstrap_and_run_service(
            service_type=ServiceType.SYSTEM_CONTROLLER,
            run=run,
            log_queue=log_queue,
        )
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    except Exception:
        logger.exception("Error running AIPerf System")
        exit_code = 1
    finally:
        logger.debug("AIPerf System exited")

    if exit_code == 0 and on_complete:
        completed = CompletedRun(artifact_dir=run.artifact_dir)
        exit_code = _invoke_callbacks(on_complete, completed, exit_code, logger)

    # Bypass Python's normal teardown: multiprocessing atexit handlers,
    # leftover ZMQ contexts, and daemon threads can otherwise block the
    # interpreter from exiting — which is fatal under pytest-xdist where
    # the parent waits on communicate(). The controller already flushed
    # logs and wrote artifacts; killing the interpreter here is safe.
    import os as _os

    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(exit_code)
    # Production never reaches here (``os._exit`` terminates the process).
    # The component-integration test harness mocks ``os._exit`` to a no-op,
    # so re-raise via ``sys.exit`` to surface the failure as a SystemExit
    # the harness can catch.
    if exit_code:
        sys.exit(exit_code)

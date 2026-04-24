# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from aiperf._cli_runner_helpers import (
    aggregate_and_export,
    build_strategy,
    log_multi_run_banner,
    validate_convergence_config,
)
from aiperf._cli_runner_helpers import (
    print_aggregate_summary as _print_aggregate_summary,
)
from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkConfig, BenchmarkPlan, BenchmarkRun

__all__ = [
    "_print_aggregate_summary",
    "_run_multi_benchmark",
    "_run_single_benchmark",
    "run_benchmark",
]


def run_benchmark(plan: BenchmarkPlan) -> None:
    """Run benchmarks from a BenchmarkPlan.

    For single-config single-trial plans, runs directly (Dashboard works).
    For multi-config or multi-trial plans, uses the MultiRunOrchestrator.
    """
    if plan.use_adaptive and plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )

    _preflight_endpoint_ready(plan)

    if plan.is_single_run:
        run = _make_benchmark_run(plan.configs[0])
        _run_single_benchmark(run)
    else:
        _run_multi_benchmark(plan)


def _preflight_endpoint_ready(plan: BenchmarkPlan) -> None:
    """Block until the target endpoint is ready (see ready_checker).

    Runs before any service bootstrap so a slow/down server fails fast with
    a clear error instead of timing out inside the system controller. Uses
    the endpoint config of the first run in the plan — multi-run sweeps are
    assumed to share an endpoint.
    """
    import asyncio
    import logging

    cfg = plan.configs[0].endpoint
    if cfg.ready_check_timeout <= 0:
        return

    # Preflight runs before rich logging is installed; install a minimal
    # stderr handler so probe lines are visible.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    from aiperf.workers.ready_checker import wait_for_endpoint

    asyncio.run(
        wait_for_endpoint(
            urls=list(cfg.urls),
            model_names=plan.configs[0].get_model_names(),
            mode=cfg.ready_check_mode,
            endpoint_type=str(cfg.type),
            path=cfg.path,
            timeout=cfg.ready_check_timeout,
            interval=cfg.ready_check_interval,
            api_key=cfg.api_key,
            headers=cfg.headers or None,
        )
    )


def _make_benchmark_run(
    config: BenchmarkConfig,
    *,
    benchmark_id: str | None = None,
    trial: int = 0,
    artifact_dir: Path | None = None,
) -> BenchmarkRun:
    """Wrap a BenchmarkConfig into a BenchmarkRun."""
    from aiperf.config import BenchmarkRun

    return BenchmarkRun(
        benchmark_id=benchmark_id or uuid4().hex[:12],
        cfg=config,
        trial=trial,
        artifact_dir=artifact_dir or config.artifacts.dir,
    )


def _configure_multiprocessing_start_method(using_dashboard: bool) -> None:
    """Pick a multiprocessing start method compatible with the current UI.

    NOTE: On macOS, when using the Textual UI with multiprocessing, terminal
    corruption (ASCII garbage, freezing) can occur when mouse events interfere
    with child processes. We apply multiple layers of protection:
      1. Set spawn method early (before any multiprocessing operations)
      2. Create log_queue before any UI initialization
      3. Set FD_CLOEXEC on terminal file descriptors
      4. Close terminal FDs in child processes (done in bootstrap.py)
    Env override takes precedence for all platforms.
    """
    import multiprocessing
    import platform

    from aiperf.common.environment import Environment

    if Environment.SERVICE.MULTIPROCESSING_START_METHOD:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method(
                Environment.SERVICE.MULTIPROCESSING_START_METHOD, force=True
            )
        return

    if platform.system() == "Darwin" and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)


def _setup_ui_queues(
    using_dashboard: bool, config: BenchmarkConfig, logger: AIPerfLogger
):  # noqa: ANN202
    """Create the global error queue and (for Dashboard UI) the log queue.

    Returns the log_queue (or ``None`` when no Dashboard UI is active). When
    Dashboard UI is running on macOS, FD_CLOEXEC is set on terminal
    descriptors to prevent child processes corrupting the parent terminal.
    """
    import platform

    from aiperf.common.error_queue import get_global_error_queue

    get_global_error_queue()

    if not using_dashboard:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(config)
        return None

    from aiperf.common.logging import get_global_log_queue

    log_queue = get_global_log_queue()

    if platform.system() == "Darwin":
        _set_fd_cloexec_on_terminal(logger)
    return log_queue


def _set_fd_cloexec_on_terminal(logger: AIPerfLogger) -> None:
    """Mark stdio as close-on-exec (macOS terminal-corruption mitigation)."""
    import fcntl

    try:
        for fd in [sys.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()]:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        logger.debug("Set FD_CLOEXEC on terminal file descriptors for macOS")
    except (OSError, ValueError, AttributeError) as e:
        # Non-fatal if this fails, other layers will protect
        logger.debug(f"Could not set FD_CLOEXEC on terminal descriptors: {e}")


def _configure_tokenizer_preload(run: BenchmarkRun) -> None:
    """Surface tokenizer identities into env so the forkserver preload sees them.

    Read by :mod:`aiperf.records._tokenizer_preload` at forkserver-helper
    startup. Must be called before the first subprocess spawn (and
    therefore before queue creation in :func:`_setup_ui_queues`), since
    Python's forkserver starts on demand and snapshots the env once.

    Name selection mirrors :class:`~aiperf.records.inference_result_parser.InferenceResultParser`:
    an explicit ``tokenizer.name`` in config overrides per-model defaults
    for every model. Without it, each model name is used as its own
    tokenizer name.

    Uses raw (unresolved) names because the resolver chain hasn't run yet
    when this is called. In the common case of canonical HF IDs (e.g.
    ``Qwen/Qwen3-0.6B``) the raw name is the correct tokenizer name and
    CoW sharing works; aliased names (e.g. ``gpt2``) miss the preload
    cache and fall through to per-RP on-demand loading — same as without
    this feature.
    """
    import os

    cfg = run.cfg
    tokenizer_cfg = cfg.tokenizer
    if tokenizer_cfg is not None and tokenizer_cfg.name:
        names = [tokenizer_cfg.name]
    else:
        names = cfg.get_model_names()
    if not names:
        return
    os.environ.setdefault("AIPERF_PRELOAD_TOKENIZERS", ",".join(names))
    if tokenizer_cfg is not None:
        os.environ.setdefault(
            "AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE",
            "true" if tokenizer_cfg.trust_remote_code else "false",
        )
        os.environ.setdefault(
            "AIPERF_PRELOAD_TOKENIZER_REVISION",
            tokenizer_cfg.revision or "main",
        )


def _run_single_benchmark(run: BenchmarkRun) -> None:
    """Run a single benchmark."""
    config = run.cfg
    using_dashboard = config.ui_type == UIType.DASHBOARD

    _configure_multiprocessing_start_method(using_dashboard)
    _configure_tokenizer_preload(run)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.config.resolvers import build_default_resolver_chain

    logger = AIPerfLogger(__name__)

    # Create queues before UI initialization to minimize FD inheritance issues.
    log_queue = _setup_ui_queues(using_dashboard, config, logger)

    logger.info("Starting AIPerf System")

    try:
        chain = build_default_resolver_chain()
        chain.resolve_all(run)
    except Exception as e:
        logger.exception("Configuration resolution failed")
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

    # Bypass Python's normal teardown: multiprocessing atexit handlers,
    # leftover ZMQ contexts, and daemon threads can otherwise block the
    # interpreter from exiting — which is fatal under pytest-xdist where
    # the parent waits on communicate(). The controller already flushed
    # logs and wrote artifacts; killing the interpreter here is safe.
    import os as _os

    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(exit_code)


def _estimate_and_log_duration(
    plan: BenchmarkPlan,
    first_config: BenchmarkConfig,
    total_runs: int,
    logger: AIPerfLogger,
) -> Path:
    """Resolve artifact/timing for a probe run, log duration, return base_dir."""
    from aiperf.config import BenchmarkRun
    from aiperf.config.resolvers import ArtifactDirResolver, TimingResolver

    probe_run = BenchmarkRun(
        benchmark_id="probe",
        cfg=first_config,
        artifact_dir=first_config.artifacts.dir,
    )
    ArtifactDirResolver().resolve(probe_run)
    TimingResolver().resolve(probe_run)

    per_run_duration = probe_run.resolved.total_expected_duration
    if per_run_duration is not None:
        total_benchmark = per_run_duration * total_runs
        total_with_cooldown = total_benchmark + plan.cooldown_seconds * max(
            total_runs - 1, 0
        )
        logger.info(f"  Estimated duration: {total_with_cooldown:.0f}s")

    return probe_run.artifact_dir


def _run_multi_benchmark(plan: BenchmarkPlan) -> None:
    """Run multiple benchmarks from a BenchmarkPlan.

    Executes trials x configs benchmarks, then aggregates results and
    computes confidence statistics. When convergence flags are set, uses
    AdaptiveStrategy for early stopping and runs both ConfidenceAggregation
    and DetailedAggregation.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

    first_config = plan.configs[0]

    if first_config.ui_type == UIType.DASHBOARD:
        raise ValueError(
            "Dashboard UI is not supported with sweep/multi-run mode. "
            "Please use '--ui simple' or '--ui none' instead."
        )

    setup_rich_logging(first_config)
    logger = AIPerfLogger(__name__)

    total_runs = len(plan.configs) * plan.trials

    validate_convergence_config(plan)
    log_multi_run_banner(plan, total_runs, logger)

    base_dir = _estimate_and_log_duration(plan, first_config, total_runs, logger)

    strategy = build_strategy(plan, logger)

    orchestrator = MultiRunOrchestrator(base_dir=base_dir)

    try:
        results = orchestrator.execute(first_config, strategy)
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise

    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    logger.info(f"All runs complete: {len(successful_runs)}/{total_runs} successful")
    if failed_runs:
        logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")
        aggregate_and_export(
            results, plan, strategy=strategy, base_dir=base_dir, logger=logger
        )
    elif len(successful_runs) == 1:
        logger.warning(
            "Only 1 successful run - cannot compute confidence statistics. "
            "At least 2 successful runs are required."
        )
        sys.exit(1)
    else:
        logger.error(
            "All runs failed - cannot compute aggregate statistics. "
            "Please check the error messages above."
        )
        sys.exit(1)

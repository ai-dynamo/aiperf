# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aiperf._cli_runner_helpers import (
    aggregate_and_export,
    aggregate_per_variation_and_export,
    aggregate_sweep_and_export,
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
    import multiprocessing as _mp

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkConfig, BenchmarkPlan, BenchmarkRun
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


@dataclass(frozen=True, slots=True)
class CompletedRun:
    """Payload passed to post-run callbacks after a successful benchmark.

    Carries the resolved artifact directory so callbacks (e.g. auto-plot) can
    locate the run's outputs without re-deriving them from BenchmarkConfig.
    """

    artifact_dir: Path


# Post-run hook signature. Invoked once per successful run/sweep with a
# CompletedRun payload. Each callback runs in isolation: an exception is
# logged with a full traceback, the run is forced to exit non-zero, and
# subsequent callbacks still run. Set ``AIPERF_RAISE_ON_CALLBACK_ERROR=true``
# to re-raise the first failure (after running the remaining callbacks) for
# strict-mode pipelines that want the exception to surface.
OnComplete = Callable[[CompletedRun], None]


def _invoke_callbacks(
    callbacks: list[OnComplete],
    completed: CompletedRun,
    exit_code: int,
    logger: Any,
) -> int:
    """Run every OnComplete callback, isolating failures.

    Each callback is invoked even if a prior one raised. On any failure the
    traceback is logged and ``exit_code`` is forced non-zero (preserving an
    already non-zero ``exit_code``). When the opt-in env var
    ``AIPERF_RAISE_ON_CALLBACK_ERROR`` is true, the first captured exception
    is re-raised after every callback has been attempted, providing a
    strict-mode contract where callback failures propagate to the caller.

    Returns the (possibly elevated) exit code so the caller can pass it to
    ``os._exit``.
    """
    # Read through the Pydantic Settings registry so the env var goes
    # through the project-wide validation/coercion pipeline (booleans
    # accept ``on``/``true``/``1``/``yes`` consistently). Reading raw
    # ``os.environ`` here would diverge from Pydantic's bool coercion
    # and silently default-False values like ``=on``. Instantiated at
    # call-time (not import-time) so unit tests that set the env var
    # via ``monkeypatch.setenv`` see the updated value.
    from aiperf.common.environment import _CLIRunnerSettings

    raise_on_error = _CLIRunnerSettings().RAISE_ON_CALLBACK_ERROR
    first_exc: BaseException | None = None
    for callback in callbacks:
        try:
            callback(completed)
        except Exception as exc:
            logger.exception(
                f"OnComplete callback {getattr(callback, '__name__', callback)!r} "
                f"failed; continuing with remaining callbacks"
            )
            if first_exc is None:
                first_exc = exc
            if exit_code == 0:
                exit_code = 1
    if first_exc is not None and raise_on_error:
        raise first_exc
    return exit_code


__all__ = [
    "CompletedRun",
    "OnComplete",
    "_print_aggregate_summary",
    "_run_multi_benchmark",
    "_run_single_benchmark",
    "run_benchmark",
]


def run_benchmark(plan: BenchmarkPlan) -> None:
    """Run benchmarks from a BenchmarkPlan.

    For single-config single-trial plans, runs directly (Dashboard works).
    For multi-config or multi-trial plans, uses the MultiRunOrchestrator.

    Args:
        plan: BenchmarkPlan to execute.
    """
    if plan.use_adaptive and plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )

    _preflight_artifact_dir(plan)
    _preflight_fd_limit()
    _preflight_endpoint_ready(plan)

    callbacks: list[OnComplete] = []
    if plan.configs[0].artifacts.auto_plot:
        from aiperf.plot.auto_plot import build_auto_plot_callback

        callbacks.append(
            build_auto_plot_callback(
                plot_required=plan.configs[0].artifacts.plot_required,
                plot_envelope=plan.plot,
            )
        )

    if plan.is_single_run:
        from aiperf.orchestrator.orchestrator import _resolve_run_seed

        seed = _resolve_run_seed(plan, plan.variations[0])
        run = _make_benchmark_run(
            plan.configs[0], random_seed=seed, variables=plan.variables
        )
        _run_single_benchmark(run, on_complete=callbacks)
    else:
        _run_multi_benchmark(plan, on_complete=callbacks)


def _preflight_artifact_dir(plan: BenchmarkPlan) -> None:
    """Validate that the artifact directory is creatable and writable.

    Why: ``setup_rich_logging`` calls ``log_folder.mkdir(parents=True)`` deep
    inside the controller bootstrap; without this preflight, an existing-file
    artifact path or a read-only parent surfaces as a stack-trace-laden
    ``NotADirectoryError``/``PermissionError`` instead of a clean config error.
    Surfacing it here lets ``profile.py`` render a single actionable panel via
    ``exit_on_error(quiet_for=(ConfigurationError,))``.
    """
    from aiperf.config.loader.errors import ConfigurationError

    artifact_dir: Path = plan.configs[0].artifacts.dir
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ConfigurationError(
            f"artifact_dir '{artifact_dir}' exists but is not a directory. "
            f"Remove the file or pick a different --artifact-dir."
        )

    parent = artifact_dir if artifact_dir.exists() else artifact_dir.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        raise ConfigurationError(
            f"artifact_dir '{artifact_dir}' is not writable "
            f"(no write permission on existing parent '{parent}'). "
            f"Pick a different --artifact-dir or fix permissions."
        )


def _preflight_fd_limit() -> None:
    """Raise RLIMIT_NOFILE soft limit and bail early if hard limit is too low.

    Why: aiperf's multiprocess service mesh (ZMQ inproc/IPC + per-worker HTTP
    pools) needs hundreds of file descriptors. With the default soft limit of
    1024 on most distros it usually fits, but bumping to 8192 leaves headroom
    for higher concurrency. When the hard limit is below the working floor,
    the ZMQ ipc_listener SIGABRTs mid-startup (`Too many open files
    src/ipc_listener.cpp:297`) — surface a clean error here instead.
    """
    try:
        import resource
    except ImportError:
        return  # Windows / unsupported platform; nothing to do.

    from aiperf.config.loader.errors import ConfigurationError

    target_soft = 8192
    min_required = 256
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < min_required:
        raise ConfigurationError(
            f"file descriptor hard limit too low: {hard} (need at least "
            f"{min_required}). Raise it via `ulimit -n 4096` (or larger) "
            f"before running aiperf."
        )
    if soft >= target_soft or soft == resource.RLIM_INFINITY:
        return
    new_soft = target_soft if hard == resource.RLIM_INFINITY else min(target_soft, hard)
    if new_soft <= soft:
        return
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


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
    if cfg.wait_for_model_timeout <= 0:
        return

    # Preflight runs before rich logging is installed; install a minimal
    # stderr handler so probe lines are visible.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    from aiperf.common.readiness_probe import wait_for_endpoint

    headers = dict(cfg.headers or {})
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    asyncio.run(
        wait_for_endpoint(
            urls=list(cfg.urls),
            model_names=plan.configs[0].get_model_names(),
            mode=cfg.wait_for_model_mode,
            endpoint_type=str(cfg.type),
            custom_endpoint=cfg.path,
            timeout_s=cfg.wait_for_model_timeout,
            interval_s=cfg.wait_for_model_interval,
            headers=headers,
        )
    )


def _make_benchmark_run(
    config: BenchmarkConfig,
    *,
    benchmark_id: str | None = None,
    trial: int = 0,
    artifact_dir: Path | None = None,
    random_seed: int | None = None,
    variables: dict[str, Any] | None = None,
) -> BenchmarkRun:
    """Wrap a BenchmarkConfig into a BenchmarkRun."""
    from aiperf.config import BenchmarkRun

    return BenchmarkRun(
        benchmark_id=benchmark_id or uuid4().hex[:12],
        cfg=config,
        trial=trial,
        artifact_dir=artifact_dir or config.artifacts.dir,
        random_seed=random_seed,
        variables=dict(variables or {}),
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

    configured_start_method = getattr(
        Environment.SERVICE, "MULTIPROCESSING_START_METHOD", None
    )
    if configured_start_method:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method(configured_start_method, force=True)
        return

    if platform.system() == "Darwin" and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)


def _setup_ui_queues(
    using_dashboard: bool, run: BenchmarkRun, logger: AIPerfLogger
) -> _mp.Queue | None:
    """Create the Dashboard log queue when needed.

    Returns the log_queue (or ``None`` when no Dashboard UI is active). When
    Dashboard UI is running on macOS, FD_CLOEXEC is set on terminal
    descriptors to prevent child processes corrupting the parent terminal.
    """
    import platform

    if not using_dashboard:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(run)
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
    except Exception as e:  # noqa: BLE001 - resolver chain wraps every user-input error type
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


def _estimate_and_log_duration(
    plan: BenchmarkPlan,
    first_config: BenchmarkConfig,
    total_runs: int,
    logger: AIPerfLogger,
) -> Path:
    """Resolve artifact/timing for a probe run, log duration, return base_dir."""
    from aiperf.config import BenchmarkRun
    from aiperf.config.resolution.resolvers import ArtifactDirResolver, TimingResolver

    probe_run = BenchmarkRun(
        benchmark_id="probe",
        cfg=first_config,
        artifact_dir=first_config.artifacts.dir,
        variables=dict(plan.variables),
    )
    ArtifactDirResolver().resolve(probe_run, for_probe=True)
    TimingResolver().resolve(probe_run)

    per_run_duration = probe_run.resolved.total_expected_duration
    if per_run_duration is not None:
        total_benchmark = per_run_duration * total_runs
        total_with_cooldown = total_benchmark + plan.cooldown_seconds * max(
            total_runs - 1, 0
        )
        logger.info(f"  Estimated duration: {total_with_cooldown:.0f}s")

    return probe_run.artifact_dir


def _validate_multi_benchmark_plan(plan: BenchmarkPlan) -> None:
    """Reject configurations multi-run can't honor before any setup work."""
    _reject_in_process_sweep_under_operator(plan)

    first_config = plan.configs[0]

    if first_config.ui_type == UIType.DASHBOARD:
        raise ValueError(
            "Dashboard UI is not supported with sweep/multi-run mode. "
            "Please use '--ui simple' or '--ui none' instead."
        )


def _log_search_planner_active(
    plan: BenchmarkPlan, search_planner: Any, logger: Any
) -> None:
    """Log the adaptive-search banner when a planner was built."""
    if search_planner is None:
        return
    sweep = plan.sweep
    assert sweep is not None  # _build_search_planner returned non-None
    logger.info(
        f"Adaptive search active: planner={sweep.planner}, "
        f"max_iterations={sweep.max_iterations}, "
        f"search-space={[d.path for d in sweep.search_space]}, "
        f"objectives=[{','.join(f'{o.metric}:{o.stat}:{o.direction}' for o in sweep.objectives)}]"
    )


def _run_multi_benchmark(
    plan: BenchmarkPlan,
    *,
    on_complete: list[OnComplete] | None = None,
) -> None:
    """Run multiple benchmarks from a BenchmarkPlan.

    Executes trials x configs benchmarks, then aggregates results and
    computes confidence statistics. When convergence flags are set, uses
    AdaptiveStrategy for early stopping and runs both ConfidenceAggregation
    and DetailedAggregation.

    Args:
        plan: BenchmarkPlan describing the configs/trials to execute.
        on_complete: Optional list of callbacks invoked in list order after
            the orchestrator returns successfully. Skipped if execution
            raises. Each callback is isolated by ``_invoke_callbacks``:
            an exception is logged, the exit code is forced non-zero, and
            remaining callbacks still run. ``AIPERF_RAISE_ON_CALLBACK_ERROR=true``
            opts into re-raising the first failure.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging

    _validate_multi_benchmark_plan(plan)
    first_config = plan.configs[0]

    setup_rich_logging(_make_benchmark_run(first_config))
    logger = AIPerfLogger(__name__)

    total_runs = len(plan.configs) * plan.trials

    validate_convergence_config(plan)
    log_multi_run_banner(plan, total_runs, logger)

    base_dir = _estimate_and_log_duration(plan, first_config, total_runs, logger)

    # Strategy is rebuilt per-cell inside the orchestrator; this top-level
    # instance is kept solely so aggregate_and_export() can resolve aggregate
    # paths and seed/warmup helpers consistently with what the runs used.
    strategy = build_strategy(plan, logger)

    results = _execute_multi_benchmark(plan, base_dir, logger)

    exit_code = _summarize_and_export(
        plan,
        results,
        total_runs=total_runs,
        strategy=strategy,
        base_dir=base_dir,
        logger=logger,
    )

    # Run callbacks whenever ANY run produced artifacts, even on a partial
    # failure path: with one successful trial the per-run JSONL/CSV/JSON are
    # on disk and downstream hooks (auto-plot, exporters) can still consume
    # them. Only skip when zero runs succeeded.
    successful_runs = [r for r in results if r.success]
    if on_complete and successful_runs:
        completed = CompletedRun(artifact_dir=plan.configs[0].artifacts.dir)
        exit_code = _invoke_callbacks(on_complete, completed, exit_code, logger)

    # Match _run_single_benchmark's hang-protection: bypass Python's normal
    # teardown so multiprocessing atexit handlers and leftover ZMQ contexts
    # cannot block the interpreter from exiting (multi-run has MORE
    # subprocesses than single-run, so this is at least as critical here).
    # The orchestrator already flushed logs and wrote artifacts; killing
    # the interpreter is safe.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
    # Production never reaches here (``os._exit`` terminates the process).
    # The component-integration test harness mocks ``os._exit`` to a no-op,
    # so re-raise via ``sys.exit`` to surface the failure as a SystemExit
    # the harness can catch.
    if exit_code != 0:
        sys.exit(exit_code)


def _execute_multi_benchmark(
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> list[RunResult]:
    """Build the orchestrator + executor + planner and run the plan to completion.

    Split out from :func:`_run_multi_benchmark` so the outer function stays
    under the function-size ceiling. Wraps the asyncio run in a
    try/except that re-raises after logging — exceptions are owned by the
    caller's exit-code path.
    """
    import asyncio as _asyncio

    from aiperf._cli_runner_helpers import _build_search_planner
    from aiperf._sweep_table_logger import (
        SweepTableLogger,
        _should_emit_sweep_table,
    )
    from aiperf.orchestrator.local_executor import LocalSubprocessExecutor
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

    no_flag = plan.no_sweep_table
    table_logger = (
        SweepTableLogger(plan, logger)
        if _should_emit_sweep_table(plan, no_sweep_table=no_flag)
        else None
    )
    orchestrator = MultiRunOrchestrator(base_dir=base_dir, cell_callback=table_logger)
    executor = LocalSubprocessExecutor(base_dir=base_dir)
    search_planner = _build_search_planner(plan)
    _log_search_planner_active(plan, search_planner, logger)

    try:
        return _asyncio.run(
            orchestrator.execute(plan, executor, search_planner=search_planner)
        )
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise


def _reject_in_process_sweep_under_operator(plan: BenchmarkPlan) -> None:
    """Block in-process grid sweep when running inside an operator-managed pod.

    The k8s operator drives grid sweeps cluster-wide via the AIPerfSweep CR
    (one AIPerfJob per variation, controller pod sees a single-config plan).
    Adaptive outer loops, in contrast, run inside the controller pod itself
    via ``BayesianSearchPlanner`` — the controller proposes each variation
    one at a time, so the in-process adaptive path is allowed under the
    operator and is not blocked here.
    """
    if os.environ.get("AIPERF_OPERATOR_MANAGED") != "1":
        return
    if plan.is_sweep:
        swept_params = sorted(
            {
                k
                for variation in plan.variations
                if variation is not None
                for k in variation.values
            }
        )
        raise SystemExit(
            f"In-process parameter sweep ({len(plan.configs)} variations across "
            f"{swept_params or '<unknown>'}) is not supported in operator-managed "
            f"runs (AIPERF_OPERATOR_MANAGED=1). Use the AIPerfSweep CRD "
            f"(cluster-scope) for cross-job sweeps — see docs/kubernetes/sweeps.md "
            f"— or submit one AIPerfJob per variation. To run as a single point "
            f"benchmark, drop the comma in --concurrency / other magic-list flags."
        )


def _log_failed_sweep_variations(
    failed_runs: list[RunResult], logger: AIPerfLogger
) -> None:
    """Log per-variation failures for a sweep, grouped by (label, sorted values).

    Keying by label too is required so QMC cells with collision-prone integer
    values (Sobol/LHS) don't get pooled into one row of the summary; mirrors
    ``_cli_runner_sweep_helpers``.
    """
    by_variation: dict[tuple, list[RunResult]] = {}
    for r in failed_runs:
        key = (
            r.variation_label or "",
            tuple(sorted((r.variation_values or {}).items())),
        )
        by_variation.setdefault(key, []).append(r)

    def _format_key(label: str, params: tuple) -> str:
        kvs = ", ".join(f"{k}={v}" for k, v in params)
        return f"{label}: {kvs}" if label else kvs

    failed_values_str = [_format_key(label, params) for label, params in by_variation]
    logger.warning(f"Some sweep variations failed: {failed_values_str}")
    for (label, params), group in by_variation.items():
        params_str = _format_key(label, params)
        for r in group:
            error_msg = r.error or "(no error message)"
            logger.warning(f"  {params_str}: {error_msg}")


def _summarize_and_export(
    plan: BenchmarkPlan,
    results: list[RunResult],
    *,
    total_runs: int,
    strategy: ExecutionStrategy,
    base_dir: Path,
    logger: AIPerfLogger,
) -> int:
    """Log success/failure summary and run confidence + sweep aggregation.

    Returns an exit code (0 on full success, 1 when fewer than 2 runs
    succeeded). Does not call ``sys.exit`` — the caller is responsible for
    propagating the code so that registered ``on_complete`` callbacks still
    run on whatever per-run artifacts were produced.
    """
    import asyncio as _asyncio

    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    if not plan.is_sweep:
        logger.info(
            f"All runs complete: {len(successful_runs)}/{total_runs} successful"
        )
    if failed_runs:
        if plan.is_sweep:
            _log_failed_sweep_variations(failed_runs, logger)
        else:
            logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")
        if plan.is_sweep:
            # Per-variation confidence aggregates (one JSON+CSV per cell with
            # >=2 successful runs) and the cross-variation sweep aggregate
            # are independent; run concurrently.
            async def _aggregate_sweep() -> None:
                await _asyncio.gather(
                    aggregate_per_variation_and_export(results, plan, base_dir, logger),
                    aggregate_sweep_and_export(results, plan, base_dir, logger),
                )

            _asyncio.run(_aggregate_sweep())
        else:
            _asyncio.run(
                aggregate_and_export(
                    results, plan, strategy=strategy, base_dir=base_dir, logger=logger
                )
            )
        return 0
    if len(successful_runs) == 1:
        if plan.is_sweep:
            logger.warning(
                "Only 1 variation succeeded - cannot compute sweep aggregate "
                "statistics. At least 2 successful variations are required."
            )
        else:
            logger.warning(
                "Only 1 successful run - cannot compute confidence statistics. "
                "At least 2 successful runs are required."
            )
        return 1
    logger.error(
        "All runs failed - cannot compute aggregate statistics. "
        "Please check the error messages above."
    )
    return 1

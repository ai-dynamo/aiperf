# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
import contextlib
import glob
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import uuid
import warnings
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import IS_MACOS, IS_WINDOWS
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.logging import LogQueue
from aiperf.plugin.enums import ServiceType

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig, BenchmarkRun

# Suppress ZMQ RuntimeWarning about dropped messages during shutdown.
# This is expected behavior when async tasks are cancelled while ZMQ messages are in-flight.
warnings.filterwarnings(
    "ignore",
    message=".*Future.*completed while awaiting.*A message has been dropped.*",
    category=RuntimeWarning,
    module="zmq._future",
)


def _enable_hf_offline_mode() -> None:
    """Force HuggingFace libraries to use local cache only.

    The parent process warms the disk cache before spawning children
    (see ``tokenizer_validator._prefetch_tokenizers``). Setting these
    env vars ensures child processes never hit the network, avoiding
    the thundering-herd problem when many workers start concurrently.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _configure_child_process() -> None:
    """Prepare a child-process environment: signals and HF offline mode.

    Ignore SIGINT and SIGTERM in child processes. SIGINT is ignored so only
    the parent handles Ctrl+C. SIGTERM is ignored because graceful shutdown is
    handled via the message bus (ShutdownCommand); process.terminate() is only
    called after the message bus path has already timed out, and the manager
    falls through to SIGKILL after the join timeout anyway. Ignoring SIGTERM
    prevents SIGSEGV crashes that occur when SIGTERM arrives while C extension
    code (uvloop, zmq, aiohttp, orjson) is executing.
    """
    if multiprocessing.parent_process() is None:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # HF offline-mode gate: enable everywhere EXCEPT controller-pod containers.
    # The controller pod's api / dataset-manager containers need HF egress for
    # prewarming the shared cache and for synthetic-dataset prompt generation;
    # every other context (worker pods, local mode) defaults to offline so a
    # regression that re-introduces from_pretrained(name) blows up immediately
    # instead of silently re-establishing HF egress.
    if os.environ.get("AIPERF_CONTROLLER_POD") != "1":
        _enable_hf_offline_mode()


def _resolve_service_id(
    service_type: ServiceType,
    service_id: str | None,
    service_metadata: Any,
) -> str:
    """Resolve the final service_id for a bootstrapping service."""
    if service_id:
        return service_id
    # Use AIPERF_POD_INDEX (set via Downward API from the JobSet job-index
    # label) for deterministic pod-level IDs in Kubernetes.
    pod_index = os.environ.get("AIPERF_POD_INDEX")
    if pod_index is not None:
        return f"{service_type}_{pod_index}"
    if service_metadata.replicable:
        return f"{service_type}_{uuid.uuid4().hex[:8]}"
    return str(service_type)


def _build_run_if_missing(
    run: BenchmarkRun | None,
    config: AIPerfConfig | None,
) -> BenchmarkRun:
    """Return ``run`` or build a standalone one from ``config``/env vars."""
    if run is not None:
        return run

    from aiperf.config import BenchmarkRun as BenchmarkRunCls

    if config is None:
        from aiperf.config.loader import load_config_from_env

        config = load_config_from_env()
    return BenchmarkRunCls(
        benchmark_id="standalone",
        cfg=config.benchmark,
        artifact_dir=config.benchmark.artifacts.dir,
        random_seed=config.random_seed,
    )


def _disable_gc_for_latency() -> None:
    """Disable GC in child processes to prevent unpredictable latency spikes.

    Only required in timing critical services such as Worker and TimingManager.
    """
    import gc

    for _ in range(3):  # Run 3 times to ensure all objects are collected
        gc.collect()
    gc.freeze()
    gc.set_threshold(0)
    gc.disable()


def _apply_custom_gpu_metrics(run: BenchmarkRun) -> None:
    """Apply custom GPU metrics from resolver cache or re-parse if needed."""
    if not run.cfg.gpu_telemetry.metrics_file:
        return

    from aiperf.gpu_telemetry import constants

    if run.resolved.gpu_custom_metrics is not None:
        custom_metrics = run.resolved.gpu_custom_metrics
        new_dcgm_mappings = run.resolved.gpu_dcgm_mappings or {}
    else:
        from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

        loader = MetricsConfigLoader()
        custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
            custom_csv_path=run.cfg.gpu_telemetry.metrics_file
        )

    constants.GPU_TELEMETRY_METRICS_CONFIG.extend(custom_metrics)
    constants.DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)


async def _drive_service_lifecycle(
    service: Any,
    error_queue: ErrorQueue | None,
) -> bool:
    """Run the service's initialize/start/wait-for-stop lifecycle.

    Returns True if the service recorded any exit errors.
    """
    try:
        await service.initialize()
        await service.start()
        await service.stopped_event.wait()
    except Exception as e:  # noqa: BLE001 - top-level service entry must log and surface any unhandled exception
        service.exception(f"Unhandled exception in service: {e}")
    finally:
        if error_queue is not None and service._exit_errors:
            from aiperf.common.error_queue import report_errors

            report_errors(error_queue, service._exit_errors)
    return bool(service._exit_errors)


def _run_event_loop(coro: Any) -> None:
    """Run ``coro`` on uvloop or asyncio, suppressing CancelledError on shutdown.

    The Windows event-loop-policy switch and high-resolution-timer bump MUST
    run before the loop is constructed; both are no-ops on non-Windows.
    """
    _configure_event_loop_policy_for_platform()
    _request_high_resolution_timer_on_windows()
    with contextlib.suppress(asyncio.CancelledError):
        if not Environment.SERVICE.DISABLE_UVLOOP:
            import uvloop

            uvloop.run(coro)
        else:
            asyncio.run(coro)


async def _run_service(
    ServiceClass: Any,
    service_metadata: Any,
    *,
    run: BenchmarkRun,
    service_id: str,
    log_queue: LogQueue | None,
    error_queue: ErrorQueue | None,
    health_port: int | None,
    api_port: int | None,
    kwargs: dict[str, Any],
) -> bool:
    """Construct and drive the service lifecycle; return True on exit errors."""
    # Disable health server in child processes to prevent port conflicts.
    # Multiple child processes on the same host cannot bind to the same port.
    # The main process (SystemController) handles health probes for local mode.
    is_child_process = multiprocessing.parent_process() is not None
    if is_child_process:
        Environment.SERVICE.HEALTH_ENABLED = False

    if Environment.DEV.ENABLE_YAPPI:
        _start_yappi_profiling()

    if service_metadata.disable_gc:
        _disable_gc_for_latency()

    _apply_custom_gpu_metrics(run)

    service = ServiceClass(
        run=run,
        service_id=service_id,
        health_port=health_port,
        api_port=api_port,
        **kwargs,
    )

    from aiperf.common.logging import setup_child_process_logging

    setup_child_process_logging(log_queue, service.service_id, run)

    # Redirect child process stdio to /dev/null unconditionally.
    # - On macOS this fixes Textual UI terminal corruption.
    # - On Linux this is required so that when the parent aiperf
    #   process exits (e.g. after a startup failure) the inherited
    #   stdout/stderr pipes close promptly; otherwise a harness doing
    #   process.communicate() waits until every child also exits,
    #   which can hang indefinitely if a grandchild is stuck.
    if is_child_process:
        _redirect_stdio_to_devnull()

    # Initialize global RandomGenerator for reproducible random number generation.
    # Always reset and then initialize to ensure a clean state.
    from aiperf.common import random_generator as rng

    rng.reset()
    rng.init(run.random_seed)

    has_errors = await _drive_service_lifecycle(service, error_queue)

    if Environment.DEV.ENABLE_YAPPI:
        _stop_yappi_profiling(service.service_id, run)
    return has_errors


def bootstrap_and_run_service(
    service_type: ServiceType,
    *,
    run: BenchmarkRun | None = None,
    config: AIPerfConfig | None = None,
    service_id: str | None = None,
    log_queue: LogQueue | None = None,
    error_queue: ErrorQueue | None = None,
    health_port: int | None = None,
    api_port: int | None = None,
    **kwargs: Any,
) -> None:
    """Bootstrap the service and run it.

    If ``run`` is not provided it is built from ``config`` (or loaded from
    env vars). ``service_id`` is auto-generated from pod index, a UUID, or
    the service type. ``log_queue`` and ``error_queue`` wire child-process
    logging and error reporting back to the parent. ``health_port`` and
    ``api_port`` expose HTTP endpoints for services that support them.
    Additional ``kwargs`` are forwarded to the service constructor.
    """
    _configure_child_process()

    # Main-process startup sweep: clear stale zero-byte child-stderr logs
    # from prior runs that bypassed cleanup (os._exit, SIGKILL, OS reap).
    # Best-effort; no-op on Linux (the per-process stderr file path is only
    # used on macOS/Windows).
    if multiprocessing.parent_process() is None and (IS_MACOS or IS_WINDOWS):
        sweep_stale_child_stderr_logs()

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    ServiceClass = plugins.get_class(PluginType.SERVICE, service_type)
    service_metadata = plugins.get_service_metadata(service_type)
    resolved_service_id = _resolve_service_id(
        service_type, service_id, service_metadata
    )
    resolved_run = _build_run_if_missing(run, config)

    has_errors = False

    async def _main() -> None:
        nonlocal has_errors
        has_errors = await _run_service(
            ServiceClass,
            service_metadata,
            run=resolved_run,
            service_id=resolved_service_id,
            log_queue=log_queue,
            error_queue=error_queue,
            health_port=health_port,
            api_port=api_port,
            kwargs=kwargs,
        )

    _run_event_loop(_main())

    if has_errors and error_queue is None:
        # Hard-exit so a hung cleanup path (e.g. a cancelled background task
        # blocking on a C-ext call after a failed on_start hook) cannot keep
        # the container alive as a zombie. SystemExit runs atexit handlers
        # that can re-hit the same hang; os._exit skips all of that.
        os._exit(1)
        # Unreachable in production (os._exit terminated the process). The
        # component-integration harness mocks os._exit to a no-op, so propagate
        # the failure as a SystemExit the runner's `except SystemExit` captures
        # as a non-zero exit code. Mirrors cli_runner._single_run.
        sys.exit(1)


def _configure_event_loop_policy_for_platform() -> None:
    """On Windows, switch to ``WindowsSelectorEventLoopPolicy`` before the
    event loop is created.

    pyzmq's async sockets call ``loop.add_reader()`` / ``loop.add_writer()``,
    which the default ``ProactorEventLoop`` on Windows does not implement.
    The selector policy must be set before ``asyncio.run()``/``uvloop.run()``
    constructs the loop.

    uvloop is already auto-disabled on Windows via ``environment.py``, so on
    Windows this only matters for the asyncio path. On non-Windows platforms
    this is a no-op — the default policy is already correct.
    """
    if IS_WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _request_high_resolution_timer_on_windows() -> None:
    """Bump Windows system timer resolution from 15.6ms to 1ms.

    asyncio.sleep on Windows is floored by the OS scheduling timer
    interrupt rate, which defaults to 15.625ms. The aiperf scheduler
    issues credits at sub-15ms intervals for >60 QPS, so without this
    bump credit issuance clumps to the 15.6ms boundary and constant-rate
    / Poisson pacing breaks (CV blows past test thresholds).

    ``winmm.timeBeginPeriod(1)`` requests 1ms timer resolution. On
    Windows 10+ this is scoped per-process — no impact on other apps'
    battery life. We never call ``timeEndPeriod`` because the timer
    bump should hold for the whole aiperf run; Windows restores the
    default automatically when the process exits.

    No-op on every non-Windows platform.
    """
    if not IS_WINDOWS:
        return
    import ctypes

    # winmm is part of Windows and always present, but guard defensively:
    # if it ever fails, aiperf still runs — high-QPS tests may just
    # produce noisier intervals. ``timeBeginPeriod`` also signals failure
    # via a non-zero return code WITHOUT raising, so check the return value
    # too — otherwise a "silent" non-zero leaves users debugging mysterious
    # timing flakes with no breadcrumb.
    try:
        rc = ctypes.WinDLL("winmm").timeBeginPeriod(1)
    except (OSError, AttributeError) as e:
        _logger.warning(
            f"Could not bump Windows timer resolution: {e!r}; high-QPS "
            f"test timing may be coarser than 1ms."
        )
        return
    if rc != 0:
        # MMSYSERR_NOERROR == 0; anything else is a documented failure code.
        _logger.warning(
            f"winmm.timeBeginPeriod(1) returned {rc}; the 1ms timer bump "
            f"did not take effect. High-QPS test timing may be coarser "
            f"than 1ms. See bootstrap.py docstring for context."
        )


def _redirect_stdio_to_devnull() -> None:
    """Redirect stdin/stdout/stderr to NUL/devnull in spawned child processes.

    macOS: avoid Textual UI terminal corruption — children inheriting the
    parent's terminal FDs interfere with Textual's terminal management,
    causing ASCII garbage and freezes on mouse events.

    Windows: when aiperf is launched as a subprocess with stdout/stderr =
    ``subprocess.PIPE`` (e.g. from the integration test runner), Windows marks
    those pipe handles inheritable. ``multiprocessing.spawn`` then propagates
    them into every grandchild service. At shutdown the grandchildren still
    hold those pipe handles, which causes either ``process.communicate()`` to
    hang forever waiting for EOF, or a ``STATUS_ACCESS_VIOLATION`` (0xC0000005)
    during ``DLL_PROCESS_DETACH``. Releasing the inherited pipe FDs to NUL
    early makes shutdown clean. Service log output is already routed through
    the multiprocessing log_queue, so this loses nothing.

    See also: ``src/aiperf/orchestrator/subprocess_runner.py::
    _release_inherited_pipes_on_windows`` — the sibling that calls this
    helper from the sweep-iteration intermediate process. If you extend
    the FD-redirection contract here, audit that call site too.
    """
    # Redirect at the OS level so spawned grandchild processes (e.g.
    # ProcessPoolExecutor workers via 'spawn' context) inherit safe FDs
    # rather than the terminal FDs that Textual manages.
    # Python-level reassignment alone (sys.stdout = ...) is not enough
    # because spawned processes create fresh sys.* from inherited OS FDs.
    #
    # No error handling: if /dev/null can't be opened or dup2 fails, the
    # process is in a broken state and should crash rather than continue
    # with corrupted FDs.
    #
    # Runs inside the event loop as one of the first operations, but
    # os.open on /dev/null hits a kernel fast path (no disk I/O), so
    # the blocking calls are safe here.
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 0)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

    # stderr: redirect to a per-process file rather than NUL. Releases the
    # inherited stderr pipe handle from the parent (same shutdown rationale
    # as fd 1), AND preserves uncaught Python tracebacks for postmortem —
    # otherwise child crashes are invisible because Python's default
    # ``sys.excepthook`` writes to stderr.
    #
    # Filename includes PID + a UUID suffix so a recycled PID (common on
    # Windows) cannot O_TRUNC over a previous process's crash log. An atexit
    # handler removes the file on clean exit if it's still empty — that keeps
    # %TEMP% from accumulating zero-byte ``aiperf_child_*_stderr.log`` files
    # over many runs while still preserving crash evidence (non-empty files
    # are left in place for the user to inspect).
    err_path = (
        f"{tempfile.gettempdir()}{os.sep}"
        f"aiperf_child_{os.getpid()}_{uuid.uuid4().hex[:8]}_stderr.log"
    )
    # 0o600 mode: owner-only read/write. The crash log can contain Python
    # tracebacks with snippets of the user's config (model names, endpoint
    # URLs, request data) and lives in a shared %TEMP%/`/tmp` directory.
    # Restrictive permissions prevent other local users from harvesting it.
    err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(err_fd, 2)
    os.close(err_fd)
    atexit.register(_remove_if_empty, err_path)

    # Recreate Python-level streams from the redirected OS FDs.
    # closefd=False keeps FD ownership at the OS level so that if these
    # stream objects are garbage-collected (e.g. replaced by test frameworks),
    # the underlying FDs 0/1/2 stay open and the /dev/null redirect holds.
    #
    # encoding="utf-8" is critical on Windows: without it, os.fdopen picks
    # the system default (cp1252) which can't encode common Unicode chars
    # (box-drawing arrows, emoji, etc.) used in aiperf's TRACE-level log
    # messages. The first such write triggers UnicodeEncodeError, which
    # Python's logging then re-emits as another UnicodeEncodeError on top,
    # cascading into a flood that wedges the child before it can register.
    # errors="replace" guards against any non-UTF8 binary slipping through.
    sys.stdin = os.fdopen(0, "r", encoding="utf-8", errors="replace", closefd=False)
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = os.fdopen(2, "w", encoding="utf-8", errors="replace", closefd=False)


def _remove_if_empty(path: str) -> None:
    """Delete ``path`` on interpreter exit only if it has zero bytes.

    Used by ``_redirect_stdio_to_devnull`` to clean up the per-process stderr
    file when the process exited cleanly with no uncaught traceback. Files
    with content (real crashes) are preserved for postmortem.

    Args:
        path: Absolute filesystem path to the per-process stderr file. The
            file is unlinked iff ``os.path.getsize(path) == 0``.

    Raises:
        Nothing — errors are suppressed because this runs from ``atexit``
        where any exception would print a misleading traceback to the
        already-shutting-down stderr.
    """
    try:
        if os.path.getsize(path) == 0:
            os.unlink(path)
    except FileNotFoundError:
        # File already gone (concurrent cleanup, race with parent reaping
        # the temp dir, etc.) — benign and expected.
        pass
    except OSError as e:
        # PermissionError, IsADirectoryError, etc. — surface to debug log
        # so the cleanup failure leaves a breadcrumb without breaking exit.
        _logger.debug(lambda exc=e: f"_remove_if_empty({path!r}) failed: {exc!r}")


def sweep_stale_child_stderr_logs(max_age_seconds: int = 86400) -> None:
    """Remove zero-byte ``aiperf_child_*_stderr.log`` files older than the
    cutoff. Sister-cleanup for ``_remove_if_empty``: that ``atexit`` handler
    only fires on clean interpreter exit, so files leaked by ``os._exit``,
    SIGKILL, ``Process.terminate()``, or OS reap of crashed children pile up
    in ``%TEMP%`` / ``/tmp`` across runs. This sweep clears them.

    Non-empty files (real crashes) are preserved for the user to inspect.
    Errors are swallowed per file — this is best-effort housekeeping, not
    a load-bearing path.

    Args:
        max_age_seconds: Files older than this (mtime) are eligible. Default
            24h keeps logs around long enough for someone to investigate a
            morning-after failure without indefinite accumulation.
    """
    pattern = os.path.join(tempfile.gettempdir(), "aiperf_child_*_stderr.log")
    cutoff = time.time() - max_age_seconds
    for path in glob.glob(pattern):
        with contextlib.suppress(OSError):
            st = os.stat(path)
            if st.st_size == 0 and st.st_mtime < cutoff:
                os.unlink(path)


def _start_yappi_profiling() -> None:
    """Start yappi profiling to profile AIPerf's python code."""
    try:
        import yappi

        yappi.set_clock_type("cpu")
        yappi.start()
    except ImportError as e:
        from aiperf.common.exceptions import AIPerfError

        raise AIPerfError(
            "yappi is not installed. Please install yappi to enable profiling. "
            "You can install yappi with `uv add yappi`."
        ) from e


def _stop_yappi_profiling(service_id_: str, run: BenchmarkRun) -> None:
    """Stop yappi profiling and save the profile to a file."""
    import yappi

    yappi.stop()

    # Get profile stats and save to file in the artifact directory
    stats = yappi.get_func_stats()
    yappi_dir = run.cfg.artifacts.dir / "yappi"
    yappi_dir.mkdir(parents=True, exist_ok=True)
    stats.save(
        str(yappi_dir / f"{service_id_}.prof"),
        type="pstat",
    )

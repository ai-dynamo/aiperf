# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import IS_MACOS, IS_WINDOWS
from aiperf.common.enums import LifecycleState
from aiperf.common.environment import Environment
from aiperf.plugin.enums import ServiceType

_logger = AIPerfLogger(__name__)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# Suppress ZMQ RuntimeWarning about dropped messages during shutdown.
# This is expected behavior when async tasks are cancelled while ZMQ messages are in-flight.
warnings.filterwarnings(
    "ignore",
    message=".*Future.*completed while awaiting.*A message has been dropped.*",
    category=RuntimeWarning,
    module="zmq._future",
)


def bootstrap_and_run_service(
    service_type: ServiceType,
    *,
    run: "BenchmarkRun",
    service_id: str | None = None,
    log_queue: "multiprocessing.Queue | None" = None,
    **kwargs,
):
    """Bootstrap the service and run it.

    Constructs an instance of the service from ``run`` and runs its lifecycle.

    Args:
        service_type: The type of the service to run.
        run: BenchmarkRun carrying the v2 BenchmarkConfig + per-run state.
        service_id: Optional unique identifier for this service instance.
        log_queue: Optional multiprocessing queue for child process logging.
        kwargs: Additional keyword arguments to pass to the service constructor.
    """
    is_child_process = multiprocessing.parent_process() is not None

    # Release inherited terminal/pipe FDs in spawned children BEFORE anything
    # else runs in this process. See _redirect_stdio_to_devnull for the
    # per-platform reasoning. Doing it later (e.g. inside the async event
    # loop) is too late on Python 3.13: by that point asyncio/logging have
    # already grabbed C-level references to the inherited fd 1/2, and a
    # later dup2-to-NUL no longer releases the parent's pipe handles fully —
    # the parent's `process.communicate()` never sees EOF and hangs.
    if (IS_MACOS or IS_WINDOWS) and is_child_process:
        _redirect_stdio_to_devnull()

    # Main-process startup sweep: clear stale zero-byte child-stderr logs
    # from prior runs that bypassed ``atexit`` (os._exit, SIGKILL, force-
    # terminate, OS reap). Best-effort cleanup; non-empty files (real
    # crashes) are preserved. Gated to non-child-process startup so each
    # spawned child doesn't repeat the sweep.
    if not is_child_process and (IS_MACOS or IS_WINDOWS):
        sweep_stale_child_stderr_logs()

    # Ignore SIGINT and SIGTERM in child processes. SIGINT is ignored so only
    # the parent handles Ctrl+C. SIGTERM is ignored because graceful shutdown is
    # handled via the message bus (ShutdownCommand); process.terminate() is only
    # called after the message bus path has already timed out, and the manager
    # falls through to SIGKILL after the join timeout anyway. Ignoring SIGTERM
    # prevents SIGSEGV crashes that occur when SIGTERM arrives while C extension
    # code (uvloop, zmq, aiohttp, orjson) is executing.
    if is_child_process:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    ServiceClass = plugins.get_class(PluginType.SERVICE, service_type)
    service_metadata = plugins.get_service_metadata(service_type)
    if not service_id:
        service_id = (
            f"{service_type}_{uuid.uuid4().hex[:8]}"
            if service_metadata.replicable
            else str(service_type)
        )

    async def _run_service():
        # Disable health server in child processes to prevent port conflicts.
        # Multiple child processes on the same host cannot bind to the same port.
        # The main process (SystemController) handles health probes for local mode.
        if is_child_process:
            Environment.SERVICE.HEALTH_ENABLED = False

        if Environment.DEV.ENABLE_YAPPI:
            _start_yappi_profiling()

        if service_metadata.disable_gc:
            # Disable garbage collection in child processes to prevent unpredictable latency spikes.
            # Only required in timing critical services such as Worker and TimingManager.
            import gc

            for _ in range(3):  # Run 3 times to ensure all objects are collected
                gc.collect()
            gc.freeze()
            gc.set_threshold(0)
            gc.disable()

        # Load and apply custom GPU metrics in child process
        if run.cfg.gpu_telemetry.metrics_file:
            from aiperf.gpu_telemetry import constants
            from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

            loader = MetricsConfigLoader()
            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=run.cfg.gpu_telemetry.metrics_file
            )

            constants.GPU_TELEMETRY_METRICS_CONFIG.extend(custom_metrics)
            constants.DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)

        service = ServiceClass(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        from aiperf.common.logging import setup_child_process_logging

        setup_child_process_logging(log_queue, service.service_id, run)

        # Initialize global RandomGenerator for reproducible random number generation
        from aiperf.common import random_generator as rng

        # Always reset and then initialize the global random generator to ensure a clean state
        rng.reset()
        rng.init(run.random_seed if run is not None else None)

        try:
            await service.initialize()
            await service.start()
            await service.stopped_event.wait()
        except Exception as e:
            service.exception(f"Unhandled exception in service: {e}")

        if Environment.DEV.ENABLE_YAPPI:
            _stop_yappi_profiling(service.service_id, run)

        _exit_if_service_failed(service)

    _configure_event_loop_policy_for_platform()
    _request_high_resolution_timer_on_windows()

    with contextlib.suppress(asyncio.CancelledError):
        if not Environment.SERVICE.DISABLE_UVLOOP:
            import uvloop

            uvloop.run(_run_service())
        else:
            asyncio.run(_run_service())


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


def _exit_if_service_failed(service) -> None:
    """Surface accumulated service failures as a non-zero SystemExit.

    The on_stop hook in production calls ``os._exit(1)`` to terminate
    immediately, but the component-integration test harness mocks
    ``os._exit`` to a no-op so the failure must be propagated another
    way. Inspect the service's terminal state and ``_exit_errors`` list
    (populated by ``SERVICE_ERROR`` messages from failing components)
    and raise ``SystemExit(1)`` so the harness — and the production
    ``cli_runner`` ``except SystemExit`` clause — can see the failure.
    """
    exit_errors = getattr(service, "_exit_errors", None)
    state = getattr(service, "state", None)
    if state == LifecycleState.FAILED or bool(exit_errors):
        sys.exit(1)


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
            "You can install yappi with `pip install yappi`."
        ) from e


def _stop_yappi_profiling(service_id_: str, run: "BenchmarkRun") -> None:
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

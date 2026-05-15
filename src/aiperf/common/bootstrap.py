# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import multiprocessing
import os
import platform
import signal
import sys
import uuid
import warnings
from typing import TYPE_CHECKING

from aiperf.common.enums import LifecycleState
from aiperf.common.environment import Environment
from aiperf.plugin.enums import ServiceType

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
    # Ignore SIGINT and SIGTERM in child processes. SIGINT is ignored so only
    # the parent handles Ctrl+C. SIGTERM is ignored because graceful shutdown is
    # handled via the message bus (ShutdownCommand); process.terminate() is only
    # called after the message bus path has already timed out, and the manager
    # falls through to SIGKILL after the join timeout anyway. Ignoring SIGTERM
    # prevents SIGSEGV crashes that occur when SIGTERM arrives while C extension
    # code (uvloop, zmq, aiohttp, orjson) is executing.
    if multiprocessing.parent_process() is not None:
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
        is_child_process = multiprocessing.parent_process() is not None
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

        # NOTE: Prevent child processes from accessing parent's terminal on macOS.
        # This solves the macOS terminal corruption issue with Textual UI where child
        # processes inherit terminal file descriptors and interfere with Textual's
        # terminal management, causing ASCII garbage and freezing when mouse events occur.
        # Only apply this in spawned child processes, NOT in the main process where Textual runs.
        if platform.system() == "Darwin" and is_child_process:
            _redirect_stdio_to_devnull()

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

    with contextlib.suppress(asyncio.CancelledError):
        if not Environment.SERVICE.DISABLE_UVLOOP:
            import uvloop

            uvloop.run(_run_service())
        else:
            asyncio.run(_run_service())


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
    """Redirect stdin/stdout/stderr to /dev/null for macOS child processes.

    Prevents child processes from accessing the parent's terminal, which causes
    Textual UI corruption (ASCII garbage and freezes from inherited terminal FDs).
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
    for fd in (0, 1, 2):
        os.dup2(devnull_fd, fd)
    os.close(devnull_fd)

    # Recreate Python-level streams from the redirected OS FDs.
    # closefd=False keeps FD ownership at the OS level so that if these
    # stream objects are garbage-collected (e.g. replaced by test frameworks),
    # the underlying FDs 0/1/2 stay open and the /dev/null redirect holds.
    sys.stdin = os.fdopen(0, "r", closefd=False)
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)


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

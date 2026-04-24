# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import signal
import sys
import uuid
import warnings
from typing import TYPE_CHECKING, Any

from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.logging import LogQueue
from aiperf.plugin.enums import ServiceType

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

    # Skip HF offline mode in Kubernetes pods: the parent process may not
    # have warmed the cache (e.g. controller pod), so children need network
    # access.  Worker pods prefetch via WorkerGroupManager before spawning
    # subprocesses, but the controller pod does not.
    if not os.environ.get("AIPERF_JOB_ID"):
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
        cfg=config,
        artifact_dir=config.artifacts.dir,
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
    """Run ``coro`` on uvloop or asyncio, suppressing CancelledError on shutdown."""
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

    setup_child_process_logging(log_queue, service.service_id, run.cfg)

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
    rng.init(run.cfg.random_seed)

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

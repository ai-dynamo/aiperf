# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import multiprocessing
import os
import platform
import sys
import uuid
import warnings

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.environment import Environment
from aiperf.plugin.enums import ServiceType

# Suppress ZMQ RuntimeWarning about dropped messages during shutdown.
# This is expected behavior when async tasks are cancelled while ZMQ messages are in-flight.
warnings.filterwarnings(
    "ignore",
    message=".*Future.*completed while awaiting.*A message has been dropped.*",
    category=RuntimeWarning,
    module="zmq._future",
)

# Suppress multiprocessing semaphore leak warnings during shutdown.
# These occur when processes are terminated before cleaning up their semaphores,
# which is expected during Ctrl+C cancellation.
warnings.filterwarnings(
    "ignore",
    message=".*leaked semaphore.*",
    category=UserWarning,
    module="multiprocessing.resource_tracker",
)


def bootstrap_and_run_service(
    service_type: ServiceType,
    service_config: ServiceConfig | None = None,
    user_config: UserConfig | None = None,
    service_id: str | None = None,
    log_queue: "multiprocessing.Queue | None" = None,
    **kwargs,
):
    """Bootstrap the service and run it.

    This function will load the service configuration,
    create an instance of the service, and run it.

    Args:
        service_type: The type of the service to run.
            ServiceType. This should be a string or enum value.
        service_config: The service configuration to use. If not provided, the service
            configuration will be loaded from the environment variables.
        user_config: The user configuration to use. If not provided, the user configuration
            will be loaded from the environment variables.
        log_queue: Optional multiprocessing queue for child process logging. If provided,
            the child process logging will be set up.
        kwargs: Additional keyword arguments to pass to the service constructor.
    """
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

    # Load the service configuration
    if service_config is None:
        from aiperf.common.config.loader import load_service_config

        service_config = load_service_config()

    # Load the user configuration
    if user_config is None:
        from aiperf.common.config.loader import load_user_config

        # TODO: Add support for loading user config from a file/environment variables
        user_config = load_user_config()

    async def _run_service():
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
        if user_config.gpu_telemetry_metrics_file:
            from aiperf.gpu_telemetry import constants
            from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

            loader = MetricsConfigLoader()
            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=user_config.gpu_telemetry_metrics_file
            )

            constants.GPU_TELEMETRY_METRICS_CONFIG.extend(custom_metrics)
            constants.DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)

        service = ServiceClass(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        from aiperf.common.logging import setup_child_process_logging

        setup_child_process_logging(
            log_queue, service.service_id, service_config, user_config
        )

        # NOTE: Prevent child processes from accessing parent's terminal on macOS.
        # This solves the macOS terminal corruption issue with Textual UI where child
        # processes inherit terminal file descriptors and interfere with Textual's
        # terminal management, causing ASCII garbage and freezing when mouse events occur.
        # Only apply this in spawned child processes, NOT in the main process where Textual runs.
        if platform.system() == "Darwin":
            # Only close terminal FDs if we're in a spawned child process
            # The main process name is typically "MainProcess", child processes have other names
            is_child_process = multiprocessing.current_process().name != "MainProcess"

            if is_child_process:
                try:
                    # Close and redirect stdin to prevent reading terminal input (mouse events, etc.)
                    sys.stdin.close()
                    sys.stdin = open(os.devnull)  # noqa: SIM115

                    # Close and redirect stdout/stderr to prevent writing to terminal
                    # All logging goes through the log_queue instead
                    sys.stdout.close()
                    sys.stderr.close()
                    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
                    sys.stderr = open(os.devnull, "w")  # noqa: SIM115
                except Exception:
                    # Silently continue if FD operations fail
                    pass

        # Initialize global RandomGenerator for reproducible random number generation
        from aiperf.common import random_generator as rng

        # Always reset and then initialize the global random generator to ensure a clean state
        rng.reset()
        rng.init(user_config.input.random_seed)

        try:
            await service.initialize()
            await service.start()
            await service.stopped_event.wait()
        except Exception as e:
            service.exception(f"Unhandled exception in service: {e}")

        if Environment.DEV.ENABLE_YAPPI:
            _stop_yappi_profiling(service.service_id, user_config)

    with contextlib.suppress(asyncio.CancelledError):
        if not Environment.SERVICE.DISABLE_UVLOOP:
            import uvloop

            uvloop.run(_run_service())
        else:
            asyncio.run(_run_service())


def _start_yappi_profiling() -> None:
    """Start yappi profiling to profile AIPerf's python code.."""
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


def _stop_yappi_profiling(service_id_: str, user_config: UserConfig) -> None:
    """Stop yappi profiling and save the profile to a file."""
    import yappi

    yappi.stop()

    # Get profile stats and save to file in the artifact directory
    stats = yappi.get_func_stats()
    yappi_dir = user_config.output.artifact_directory / "yappi"
    yappi_dir.mkdir(parents=True, exist_ok=True)
    stats.save(
        str(yappi_dir / f"{service_id_}.prof"),
        type="pstat",
    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import multiprocessing
import random

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.protocols import ServiceProtocol


def bootstrap_and_run_service(
    service_class: type[ServiceProtocol],
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
        service_class: The python class of the service to run. This should be a subclass of
            BaseService. This should be a type and not an instance.
        service_config: The service configuration to use. If not provided, the service
            configuration will be loaded from the environment variables.
        user_config: The user configuration to use. If not provided, the user configuration
            will be loaded from the environment variables.
        log_queue: Optional multiprocessing queue for child process logging. If provided,
            the child process logging will be set up.
        kwargs: Additional keyword arguments to pass to the service constructor.
    """

    # Load the service configuration
    if service_config is None:
        from aiperf.common.config import load_service_config

        service_config = load_service_config()

    # Load the user configuration
    if user_config is None:
        from aiperf.common.config import load_user_config

        # TODO: Add support for loading user config from a file/environment variables
        user_config = load_user_config()

    async def _run_service():
        if service_config.enable_yappi:
            _start_yappi_profiling()

        from aiperf.module_loader import ensure_modules_loaded

        ensure_modules_loaded()

        service = service_class(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        from aiperf.common.logging import setup_child_process_logging

        setup_child_process_logging(
            log_queue, service.service_id, service_config, user_config
        )

        if user_config.input.random_seed is not None:
            random.seed(user_config.input.random_seed)
            # Try and set the numpy random seed
            # https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
            with contextlib.suppress(ImportError):
                import numpy as np

                np.random.seed(user_config.input.random_seed)

        try:
            await service.initialize()
            await service.start()
            await service.stopped_event.wait()
        except Exception as e:
            service.exception(f"Unhandled exception in service: {e}")

        if service_config.enable_yappi:
            _stop_yappi_profiling(service.service_id, user_config)

    with contextlib.suppress(asyncio.CancelledError):
        if service_config.enable_uvloop:
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

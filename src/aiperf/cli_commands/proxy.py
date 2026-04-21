# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command to run a standalone ZMQ proxy in its own process/container.

Used by the Kubernetes sidecar pattern to isolate the event-bus XPUB/XSUB
proxy from the SystemController container, so that large fan-ins of record
processors and workers don't starve the control plane at startup.
"""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="proxy")


@app.default
def proxy(
    *,
    benchmark_run_file: Annotated[
        Path,
        Parameter(
            name="--benchmark-run",
            show_env_var=False,
            negative=False,
            help="Path to the BenchmarkRun JSON file. Used to resolve the proxy's "
            "bind addresses from the resolved communication config.",
        ),
    ],
    kind: Annotated[
        str,
        Parameter(
            show_env_var=False,
            negative=False,
            help="Which proxy to run. Currently only 'event_bus' is supported.",
        ),
    ] = "event_bus",
    health_port: Annotated[
        int | None,
        Parameter(
            show_env_var=False,
            negative=False,
            help="HTTP port for /healthz and /readyz. Falls back to "
            "AIPERF_SERVICE_HEALTH_PORT.",
        ),
    ] = None,
) -> None:
    """Run a single ZMQ proxy in this process until SIGTERM/SIGINT.

    _Advanced use only — this command is invoked by the AIPerf Kubernetes
    sidecar pattern and is not intended for direct human use._
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title=f"Error Running AIPerf Proxy ({kind})"):
        import asyncio
        import signal

        import orjson

        from aiperf.common.environment import Environment
        from aiperf.common.health_server import HealthServer
        from aiperf.config.benchmark import BenchmarkRun
        from aiperf.controller.proxy_manager import ProxyManager

        run = BenchmarkRun.model_validate(orjson.loads(benchmark_run_file.read_bytes()))

        if health_port is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        kind_to_flags = {
            "event_bus": {
                "enable_event_bus": True,
                "enable_dataset_manager": False,
                "enable_raw_inference": False,
            },
        }
        if kind not in kind_to_flags:
            raise ValueError(
                f"Unsupported proxy kind {kind!r}; valid: {sorted(kind_to_flags)}"
            )
        flags = kind_to_flags[kind]

        async def _run() -> None:
            manager = ProxyManager(run=run, **flags)
            health: HealthServer | None = None
            if Environment.SERVICE.HEALTH_ENABLED:
                health = HealthServer(port=Environment.SERVICE.HEALTH_PORT)
                await health.start()

            await manager.initialize_and_start()

            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, stop_event.set)

            try:
                await stop_event.wait()
            finally:
                await manager.stop()
                if health is not None:
                    await health.stop()

        asyncio.run(_run())

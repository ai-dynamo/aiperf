# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command to run a standalone ZMQ proxy in its own process/container.

Used by the Kubernetes sidecar pattern to isolate the event-bus XPUB/XSUB
proxy from the SystemController container, so that large fan-ins of record
processors and workers don't starve the control plane at startup.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Parameter

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

app = App(name="proxy")

_KIND_TO_FLAGS: dict[str, dict[str, bool]] = {
    "event_bus": {
        "enable_event_bus": True,
        "enable_dataset_manager": False,
        "enable_raw_inference": False,
    },
}


def _install_stop_signal_handlers(
    loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event
) -> None:
    """Arrange for SIGTERM/SIGINT (SIGBREAK on Windows) to set ``stop_event``.

    Windows ProactorEventLoop does not implement ``add_signal_handler`` and
    raises NotImplementedError, which would abort ``aiperf proxy`` before the
    proxy serves anything. Fall back to ``signal.signal()`` there.
    """
    import signal

    from aiperf.common.constants import IS_WINDOWS

    if IS_WINDOWS:

        def windows_signal_handler(_sig: int, _frame: object) -> None:
            # signal.signal handlers run outside the event loop thread's
            # normal callback path; marshal onto the loop.
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, windows_signal_handler)
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None:
            signal.signal(sigbreak, windows_signal_handler)
    else:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)


async def _run_proxy(run: "BenchmarkRun", flags: dict[str, bool]) -> None:
    from aiperf.common.environment import Environment
    from aiperf.common.health_server import HealthServer
    from aiperf.controller.proxy_manager import ProxyManager

    manager = ProxyManager(run=run, **flags)
    health: HealthServer | None = None
    if Environment.SERVICE.HEALTH_ENABLED:
        health = HealthServer(port=Environment.SERVICE.HEALTH_PORT)
        await health.start()

    await manager.initialize_and_start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_stop_signal_handlers(loop, stop_event)

    try:
        await stop_event.wait()
    finally:
        await manager.stop()
        if health is not None:
            await health.stop()


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
        import orjson

        from aiperf.common.environment import Environment
        from aiperf.config.resolution.plan import BenchmarkRun

        run = BenchmarkRun.model_validate(orjson.loads(benchmark_run_file.read_bytes()))

        if health_port is not None:
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        if kind not in _KIND_TO_FLAGS:
            raise ValueError(
                f"Unsupported proxy kind {kind!r}; valid: {sorted(_KIND_TO_FLAGS)}"
            )

        asyncio.run(_run_proxy(run, _KIND_TO_FLAGS[kind]))

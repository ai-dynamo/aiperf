# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running individual AIPerf services."""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.plugin.enums import ServiceType

app = App(name="service")


_ServiceTypeArg = Annotated[
    ServiceType,
    Parameter(
        name="--type", show_env_var=False, negative=False, help="Service type to run."
    ),
]
_BenchmarkRunArg = Annotated[
    Path | None,
    Parameter(
        name="--benchmark-run",
        show_env_var=False,
        negative=False,
        help="Path to a BenchmarkRun JSON file. "
        "The service bootstraps with a fully-built BenchmarkRun "
        "including metadata, variation, and trial info.",
    ),
]
_ServiceIdArg = Annotated[
    str | None,
    Parameter(
        show_env_var=False,
        negative=False,
        help="Unique identifier for the service instance. "
        "Useful when running multiple instances of the same service type.",
    ),
]
_HealthHostArg = Annotated[
    str | None,
    Parameter(
        show_env_var=False,
        negative=False,
        help="Host to bind the health server to. "
        "Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable.",
    ),
]
_HealthPortArg = Annotated[
    int | None,
    Parameter(
        show_env_var=False,
        negative=False,
        help="HTTP port for health endpoints (/healthz, /readyz). "
        "Required for Kubernetes liveness and readiness probes. "
        "Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable.",
    ),
]
_ApiPortArg = Annotated[
    int | None,
    Parameter(
        show_env_var=False,
        negative=False,
        help="HTTP port for API endpoints (e.g., /api/dataset, /api/progress). "
        "Only used by services that expose HTTP APIs.",
    ),
]


@app.default
def service(
    *,
    service_type: _ServiceTypeArg,
    benchmark_run_file: _BenchmarkRunArg = None,
    service_id: _ServiceIdArg = None,
    health_host: _HealthHostArg = None,
    health_port: _HealthPortArg = None,
    api_port: _ApiPortArg = None,
) -> None:
    """Run an AIPerf service in a single process.

    _Advanced use only — intended for developers and Kubernetes/distributed
    deployments where services run in separate containers or nodes._

    For standard single-node benchmarking, use the `aiperf profile` command instead.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title=f"Error Running AIPerf Service {service_type}"):
        from aiperf.common.bootstrap import bootstrap_and_run_service

        run = _load_benchmark_run(benchmark_run_file)
        _apply_health_overrides(health_host, health_port)

        bootstrap_and_run_service(
            service_type=service_type,
            run=run,
            config=None,
            service_id=service_id,
            api_port=api_port,
        )


def _load_benchmark_run(benchmark_run_file: Path | None):
    if benchmark_run_file is None:
        return None

    import orjson

    from aiperf.config.benchmark import BenchmarkRun

    return BenchmarkRun.model_validate(orjson.loads(benchmark_run_file.read_bytes()))


def _apply_health_overrides(health_host: str | None, health_port: int | None) -> None:
    from aiperf.common.environment import Environment

    if health_host is not None:
        Environment.SERVICE.HEALTH_ENABLED = True
        Environment.SERVICE.HEALTH_HOST = health_host

    if health_port is not None:
        Environment.SERVICE.HEALTH_ENABLED = True
        Environment.SERVICE.HEALTH_PORT = health_port

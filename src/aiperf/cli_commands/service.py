# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running individual AIPerf services."""

from pathlib import Path
from typing import Annotated

from cyclopts import App

from aiperf.config.cli_parameter import CLIParameter
from aiperf.plugin.enums import ServiceType

app = App(name="service")


@app.default
def service(
    service_type: Annotated[
        ServiceType, CLIParameter(name="--type", help="Service type to run.")
    ],
    benchmark_run_file: Annotated[
        Path,
        CLIParameter(
            name="--benchmark-run",
            help="Path to the pre-built BenchmarkRun JSON file. The service "
            "bootstraps exclusively from this serialized run. Kubernetes "
            "controllers materialize it before starting service containers.",
        ),
    ],
    *,
    api_port: Annotated[
        int | None,
        CLIParameter(
            help="HTTP port for API endpoints (e.g. /api/dataset, /api/progress). "
            "Only used by services that expose HTTP APIs."
        ),
    ] = None,
    service_id: Annotated[
        str | None,
        CLIParameter(
            help="Unique identifier for the service instance. "
            "Useful when running multiple instances of the same service type."
        ),
    ] = None,
    health_host: Annotated[
        str | None,
        CLIParameter(
            help="Host to bind the health server to. "
            "Falls back to AIPERF_SERVICE_HEALTH_HOST environment variable."
        ),
    ] = None,
    health_port: Annotated[
        int | None,
        CLIParameter(
            help="HTTP port for health endpoints (/healthz, /readyz). "
            "Required for Kubernetes liveness and readiness probes. "
            "Falls back to AIPERF_SERVICE_HEALTH_PORT environment variable."
        ),
    ] = None,
) -> None:
    """Run an AIPerf service in a single process.

    _Advanced use only — intended for developers and Kubernetes/distributed
    deployments where services run in separate containers or nodes._

    For standard single-node benchmarking, use the `aiperf profile` command instead.

    Args:
        benchmark_run_file: Controller-rendered BenchmarkRun JSON shared by all
            service containers in the deployment.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title=f"Error Running AIPerf Service {service_type}"):
        import orjson

        from aiperf.common.bootstrap import bootstrap_and_run_service
        from aiperf.common.endpoint_credentials import (
            apply_endpoint_credentials,
            consume_endpoint_credentials,
        )
        from aiperf.common.environment import Environment
        from aiperf.config.resolution.plan import BenchmarkRun
        from aiperf.kubernetes.serialized_run import read_serialized_run_json

        # The launcher resolves and freezes one BenchmarkRun for the whole
        # deployment. Re-resolving flags independently in every service can
        # diverge in seeds, synthesized defaults, and artifact identity.
        run_json = read_serialized_run_json(benchmark_run_file)
        if run_json is None:
            raise ValueError(
                f"Cannot read serialized BenchmarkRun from {benchmark_run_file!s}: "
                "the path is missing, unsafe, not a regular UTF-8 file, or unreadable. "
                "Mount the controller-generated run_config.json and pass its path "
                "with --benchmark-run."
            )
        run = BenchmarkRun.model_validate(orjson.loads(run_json))
        credentials = consume_endpoint_credentials(allow_openai_api_key=True)
        apply_endpoint_credentials(run, credentials, require_resolved=True)

        if health_host is not None:
            # CLI argument takes precedence over environment variable
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_HOST = health_host

        if health_port is not None:
            # CLI argument takes precedence over environment variable
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        extra: dict[str, int] = {}
        if api_port is not None:
            extra["api_port"] = api_port

        bootstrap_and_run_service(
            service_type=service_type,
            run=run,
            service_id=service_id,
            **extra,
        )

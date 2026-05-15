# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for running individual AIPerf services."""

from typing import Annotated

from cyclopts import App

from aiperf.config.cli_parameter import CLIParameter
from aiperf.config.flags import CLIConfig
from aiperf.plugin.enums import ServiceType

app = App(name="service")


@app.default
def service(
    service_type: Annotated[
        ServiceType, CLIParameter(name="--type", help="Service type to run.")
    ],
    *,
    cli_config: CLIConfig,
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
        cli_config: Cyclopts-populated CLIConfig DTO carrying every CLI flag
            (benchmark inputs and service-runtime knobs). Pass ``--config foo.yaml``
            to load defaults from a v2 YAML file; explicit CLI flags overlay on top.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title=f"Error Running AIPerf Service {service_type}"):
        from aiperf.cli_runner import _make_benchmark_run
        from aiperf.common.bootstrap import bootstrap_and_run_service
        from aiperf.common.environment import Environment
        from aiperf.config.flags.resolver import resolve_config
        from aiperf.config.loader import build_benchmark_plan

        # Validate via the AIPerfConfig gate and build a single-variation
        # plan so bootstrap and tokenizer validation can read ``run.cfg``.
        # The service receives the resolved ``run`` (carrying the validated
        # ``BenchmarkConfig``) — service constructors consume that, not the
        # raw ``cli_config`` DTO.
        config = resolve_config(cli_config, cli_config.config_file)
        plan = build_benchmark_plan(config)
        from aiperf.orchestrator.orchestrator import resolve_run_seed

        run = _make_benchmark_run(
            plan.configs[0],
            random_seed=resolve_run_seed(plan, plan.variations[0]),
        )

        if health_host is not None:
            # CLI argument takes precedence over environment variable
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_HOST = health_host

        if health_port is not None:
            # CLI argument takes precedence over environment variable
            Environment.SERVICE.HEALTH_ENABLED = True
            Environment.SERVICE.HEALTH_PORT = health_port

        bootstrap_and_run_service(
            service_type=service_type,
            run=run,
            service_id=service_id,
        )

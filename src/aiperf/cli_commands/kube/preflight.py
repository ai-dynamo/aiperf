# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube preflight command: validate cluster readiness before deploying."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.kube import KubeManageOptions

app = App(name="preflight")


@app.default
async def preflight(
    *,
    manage_options: KubeManageOptions | None = None,
    image: Annotated[
        str | None,
        Parameter(
            name=["-i", "--image"],
            help="Container image to verify accessibility.",
        ),
    ] = None,
    endpoint_url: Annotated[
        str | None,
        Parameter(
            name=["-e", "--endpoint-url"],
            help="LLM endpoint URL to test connectivity.",
        ),
    ] = None,
    workers: Annotated[
        int,
        Parameter(
            name=["-w", "--workers"],
            help="Planned number of worker pods (for resource projection).",
        ),
    ] = 1,
    output: Annotated[
        Literal["text", "json"],
        Parameter(
            name=["-o", "--output"],
            help='Output format: "text" for human-readable, "json" for machine-parseable.',
        ),
    ] = "text",
) -> None:
    """Run pre-flight checks against the target Kubernetes cluster.

    Validates cluster connectivity, API versions, RBAC permissions, resource
    availability, image accessibility, and endpoint reachability before deploying
    a benchmark.

    Examples:
        # Basic cluster check
        aiperf kube preflight

        # Check with image and endpoint
        aiperf kube preflight --image aiperf:latest --endpoint-url http://server:8000

        # JSON output for CI/automation
        aiperf kube preflight -o json

        # Check specific namespace with resource projection
        aiperf kube preflight --namespace my-bench --workers 8
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Running Preflight Checks"):
        await _run_preflight(
            manage_options=manage_options or KubeManageOptions(),
            image=image,
            endpoint_url=endpoint_url,
            workers=workers,
            output=output,
        )


async def _run_preflight(
    *,
    manage_options: KubeManageOptions,
    image: str | None,
    endpoint_url: str | None,
    workers: int,
    output: Literal["text", "json"],
) -> None:
    import logging

    import orjson

    from aiperf.kubernetes.console import console
    from aiperf.kubernetes.preflight import CLIPreflightChecker

    # Suppress text output in JSON mode so only clean JSON goes to stdout
    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)

    try:
        checker = CLIPreflightChecker(
            namespace=manage_options.namespace or "aiperf-benchmarks",
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            image=image,
            endpoint_url=endpoint_url,
            workers=workers,
        )

        results = await checker.run_all_checks()
    finally:
        kube_logger.setLevel(original_level)

    if output == "json":
        json_output = orjson.dumps(
            results.to_dict(), option=orjson.OPT_INDENT_2
        ).decode()
        console.print(json_output, highlight=False)

    if not results.passed:
        raise SystemExit(1)

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
    image_pull_secrets: Annotated[
        list[str] | None,
        Parameter(
            name=["--image-pull-secret", "--image-pull-secrets"],
            negative=(),
            help="Image pull secret name to verify (repeat for multiple secrets).",
        ),
    ] = None,
    secrets: Annotated[
        list[str] | None,
        Parameter(
            name=["--secret", "--secrets"],
            negative=(),
            help="Referenced Kubernetes secret name to verify (repeat for multiple secrets).",
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
    node_selector: Annotated[
        list[str] | None,
        Parameter(
            name="--node-selector",
            negative=(),
            accepts_keys=False,
            n_tokens=-1,
            help="Node selector labels as JSON or repeated key=value values.",
        ),
    ] = None,
    tolerations: Annotated[
        list[str] | None,
        Parameter(
            name="--tolerations",
            negative=(),
            n_tokens=-1,
            help="Pod tolerations as JSON objects or arrays.",
        ),
    ] = None,
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
            image_pull_secrets=image_pull_secrets,
            secrets=secrets,
            endpoint_url=endpoint_url,
            workers=workers,
            node_selector=node_selector,
            tolerations=tolerations,
            output=output,
        )


async def _run_preflight(
    *,
    manage_options: KubeManageOptions,
    image: str | None,
    image_pull_secrets: list[str] | None,
    secrets: list[str] | None,
    endpoint_url: str | None,
    workers: int,
    node_selector: dict[str, str] | list[str] | None,
    tolerations: list[dict[str, object]] | list[str] | None,
    output: Literal["text", "json"],
) -> None:
    from contextlib import nullcontext

    import orjson

    from aiperf.config.kube import _coerce_tolerations, _merge_node_selector_entries
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.cli_helpers import resolve_benchmark_namespace
    from aiperf.kubernetes.preflight import CLIPreflightChecker

    namespace = resolve_benchmark_namespace(
        manage_options.namespace,
        manage_options.kubeconfig,
        manage_options.kube_context,
    )

    # In JSON mode stdout must carry only the JSON document; the checker's own
    # summary (including an ERROR record when checks fail) goes to stderr.
    quiet = (
        kube_console.machine_readable_stdout() if output == "json" else nullcontext()
    )
    with quiet:
        checker = CLIPreflightChecker(
            namespace=namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
            image=image,
            image_pull_secrets=image_pull_secrets,
            secrets=secrets,
            endpoint_url=endpoint_url,
            workers=workers,
            node_selector=_merge_node_selector_entries(node_selector),
            tolerations=_coerce_tolerations(tolerations),
        )

        results = await checker.run_all_checks()

    if output == "json":
        kube_console.emit_raw(
            orjson.dumps(results.to_dict(), option=orjson.OPT_INDENT_2).decode()
        )

    if not results.passed:
        raise SystemExit(1)

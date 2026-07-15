# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``aiperf kube index`` — manual control of the operator's runs index.

Wraps ``GET /admin/index/stats`` and ``POST /admin/index/rebuild`` exposed by
the operator's results-server FastAPI app. Rebuild is the recovery hatch
when the SQLite index has fallen out of sync with the PVC (e.g. after an
external file restore); stats confirms the index is populated post-startup.

The CLI auto-resolves the operator pod via cluster-wide label search
(``app.kubernetes.io/name=aiperf-operator``) and port-forwards to the
results-server container — so the same command works against any chart
install regardless of release name, namespace, or ``resultsServer.port``
override. Use ``--api-url`` only when targeting an externally-exposed
ingress or a pre-pinned port-forward.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import cyclopts
import httpx
import orjson

from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes import console as kube_console

app = cyclopts.App(
    name="index",
    help="Manage the operator's runs/sweep_variations SQLite index.",
)

_MUTATING_ROUTES_TOKEN_ENV = "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN"


def _mutating_route_headers() -> dict[str, str]:
    """Return Authorization headers for protected operator mutating routes."""
    token = os.environ.get(_MUTATING_ROUTES_TOKEN_ENV, "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _raise_mutating_route_error(exc: httpx.HTTPStatusError) -> None:
    status_code = exc.response.status_code
    if status_code not in {401, 403}:
        raise exc
    raise RuntimeError(
        "Operator index rebuild is protected by results-server mutating-route auth. "
        f"Set {_MUTATING_ROUTES_TOKEN_ENV} to the bearer token configured on the "
        "operator, or ask an administrator to set "
        "AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED=1 and "
        f"{_MUTATING_ROUTES_TOKEN_ENV} on the results-server container "
        "(the Helm chart does not template these; see "
        "docs/kubernetes/results-api.md)."
    ) from exc


@asynccontextmanager
async def _operator_api_base(
    api_url: str | None,
    options: KubeManageOptions | None,
    operator_namespace: str | None = None,
):
    """Yield a base URL for the operator's HTTP API.

    Two modes:
    - ``api_url`` set explicitly → use it as-is (caller is responsible for
      reachability; typical use: pre-pinned port-forward, ingress).
    - ``api_url`` None → resolve operator pod via cluster-wide label search
      and port-forward to the results-server container. Mirrors the pattern
      used by ``aiperf kube results list-runs``.

    The benchmark ``--namespace`` does not influence operator-pod lookup;
    pass ``operator_namespace`` (``--operator-namespace``) to pin it, else
    cluster-wide auto-detect + ``DEFAULT_OPERATOR_NAMESPACE`` fallback runs.

    Raises:
        RuntimeError: When auto-resolve cannot find an operator pod (no
            install in the cluster, or wrong context).
    """
    if api_url:
        yield api_url
        return

    from aiperf.kubernetes.client import (
        find_operator_pod,
        k8s_client,
        resolve_operator_namespace,
    )
    from aiperf.kubernetes.port_forward import port_forward_with_status
    from aiperf.kubernetes.results_operator import RESULTS_SERVER_PORT

    opts = options or KubeManageOptions()

    async with k8s_client(kubeconfig=opts.kubeconfig, context=opts.kube_context) as api:
        op_ns = await resolve_operator_namespace(api, explicit=operator_namespace)
        pod_info = await find_operator_pod(api, namespace=op_ns)
        if not pod_info:
            raise RuntimeError(
                f"Operator pod not found in namespace '{op_ns}'. "
                "Is the aiperf-operator deployed? "
                "Pass --api-url to bypass auto-discovery."
            )
        pod_name, _phase = pod_info
        async with port_forward_with_status(
            op_ns,
            pod_name,
            0,
            remote_port=RESULTS_SERVER_PORT,
            verify_api=False,
            kubeconfig=opts.kubeconfig,
            kube_context=opts.kube_context,
        ) as local_port:
            yield f"http://localhost:{local_port}"


@app.command(name="rebuild")
async def rebuild(
    *,
    output: Annotated[
        Literal["text", "json"],
        cyclopts.Parameter(help="Output format."),
    ] = "text",
    api_url: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Operator HTTP API base URL. Default: auto-resolve via "
            "label-selector + port-forward to the results-server container."
        ),
    ] = None,
    operator_namespace: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--operator-namespace",
            help="Namespace where the operator is deployed. Auto-detected "
            "(cluster-wide pod search) when omitted.",
        ),
    ] = None,
    options: KubeManageOptions | None = None,
) -> None:
    """Rebuild the operator's runs index from the PVC."""
    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)

    try:
        async with (
            _operator_api_base(api_url, options, operator_namespace) as base_url,
            httpx.AsyncClient(base_url=base_url, timeout=300.0) as client,
        ):
            resp = await client.post(
                "/admin/index/rebuild", headers=_mutating_route_headers()
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                _raise_mutating_route_error(exc)
            data = orjson.loads(resp.content)
        if output == "json":
            kube_console.console.print(
                orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
            )
        else:
            kube_console.console.print(
                f"Indexed {data['runs_indexed']} runs and "
                f"{data['sweep_variations_indexed']} sweep variations "
                f"in {data['duration_seconds']:.2f}s"
            )
    finally:
        if output == "json":
            kube_logger.setLevel(original_level)


@app.command(name="stats")
async def stats(
    *,
    output: Annotated[
        Literal["text", "json"],
        cyclopts.Parameter(help="Output format."),
    ] = "text",
    api_url: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Operator HTTP API base URL. Default: auto-resolve via "
            "label-selector + port-forward to the results-server container."
        ),
    ] = None,
    operator_namespace: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--operator-namespace",
            help="Namespace where the operator is deployed. Auto-detected "
            "(cluster-wide pod search) when omitted.",
        ),
    ] = None,
    options: KubeManageOptions | None = None,
) -> None:
    """Show runs index statistics."""
    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)
    try:
        async with (
            _operator_api_base(api_url, options, operator_namespace) as base_url,
            httpx.AsyncClient(base_url=base_url) as client,
        ):
            resp = await client.get("/admin/index/stats")
            resp.raise_for_status()
            data = orjson.loads(resp.content)
        if output == "json":
            kube_console.console.print(
                orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
            )
        else:
            kube_console.console.print(
                f"runs={data['runs_count']} "
                f"sweep_variations={data['sweep_variations_count']} "
                f"size={data['db_bytes']}B "
                f"schema_version={data['schema_version']} "
                f"last_bootstrap_unix={data['last_bootstrap_unix']}"
            )
    finally:
        if output == "json":
            kube_logger.setLevel(original_level)

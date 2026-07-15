# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes client — free functions over kubernetes_asyncio.

The canonical interface is ``k8s_client()`` + free functions
(``list_aiperf_jobs``, ``find_jobset``, …) that take an ``ApiClient``
explicitly and call ``CoreV1Api(api)`` / ``CustomObjectsApi(api)``
inline so the reader sees the native kubernetes_asyncio API surface.

Implementation is split by topic across sibling modules; this file
preserves the single import surface:

- selectors (``job_selector``, ``controller_selector``) — :mod:`client_selectors`
- AIPerfJob CR helpers — :mod:`client_jobs`
- JobSet helpers (and ``delete_namespace``) — :mod:`client_jobsets`
- pod helpers and ``cluster_version`` — :mod:`client_pods`
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.client_jobs import (
    cancel_aiperf_job,
    find_aiperf_job,
    find_aiperf_sweep,
    get_raw_aiperfjob,
    get_raw_aiperfjob_status,
    list_aiperf_jobs,
)
from aiperf.kubernetes.client_jobsets import (
    delete_jobset,
    delete_namespace,
    find_jobset,
    list_jobsets,
)
from aiperf.kubernetes.client_pods import (
    cluster_version,
    find_controller_pod,
    find_operator_namespace,
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_pods,
    list_events_for_object,
    list_nodes,
    list_pods_all_namespaces,
    resolve_operator_namespace,
    wait_for_controller_pod_ready,
)
from aiperf.kubernetes.client_selectors import controller_selector, job_selector

__all__ = [
    "asyncio",
    "cancel_aiperf_job",
    "client",
    "cluster_version",
    "controller_selector",
    "delete_jobset",
    "delete_namespace",
    "find_aiperf_job",
    "find_aiperf_sweep",
    "find_aiperfsweep",
    "find_controller_pod",
    "find_jobset",
    "find_operator_namespace",
    "find_operator_pod",
    "find_retrievable_pod",
    "get_pod_summary",
    "get_pods",
    "get_raw_aiperfjob",
    "get_raw_aiperfjob_status",
    "get_raw_aiperfsweep",
    "get_raw_aiperfsweep_status",
    "job_selector",
    "k8s_client",
    "list_aiperf_jobs",
    "list_aiperfsweeps",
    "list_events_for_object",
    "list_jobsets",
    "list_nodes",
    "list_pods_all_namespaces",
    "resolve_operator_namespace",
    "wait_for_controller_pod_ready",
]

# ``client`` (the kubernetes_asyncio module) and ``asyncio`` are re-exported as
# module attributes so tests can patch ``aiperf.kubernetes.client.client.CustomObjectsApi``
# and ``aiperf.kubernetes.client.asyncio.sleep``. Python modules are singletons;
# the patches propagate to the sibling ``client_jobs`` / ``client_jobsets`` /
# ``client_pods`` modules that also import these names.

APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV = "AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE"


def _apply_apiserver_tls_server_name_override() -> None:
    """Apply the chaos-only apiserver TLS hostname override when configured."""
    server_name = os.environ.get(APISERVER_TLS_SERVER_NAME_OVERRIDE_ENV, "").strip()
    if not server_name:
        return
    cfg = client.Configuration.get_default_copy()
    cfg.tls_server_name = server_name
    client.Configuration.set_default(cfg)


@asynccontextmanager
async def k8s_client(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> AsyncIterator[ApiClient]:
    """Load k8s config and yield an ``ApiClient``.

    Tries ``load_incluster_config()`` first (pod-mounted service account), then
    falls back to ``load_kube_config()`` on the given ``kubeconfig``/``context``.
    The ``ApiClient`` is guaranteed to be closed on scope exit.

    Args:
        kubeconfig: Path to a kubeconfig file. ``None`` means use the default
            resolution (``$KUBECONFIG`` or ``~/.kube/config``). Only consulted
            when the in-cluster load fails.
        context: Kubeconfig context name to activate. ``None`` means use the
            current-context from the kubeconfig.

    Raises:
        kubernetes_asyncio.config.ConfigException: If both the in-cluster and
            kubeconfig loaders fail (e.g. no service account mounted AND no
            readable kubeconfig / unknown context).

    Example:
        >>> async with k8s_client() as api:
        ...     jobs = await list_aiperf_jobs(api, namespace="aiperf-bench")
        ...     for job in jobs:
        ...         print(job.name, job.phase)
    """
    suppress_noisy_http_loggers()
    try:
        config.load_incluster_config()
        _apply_apiserver_tls_server_name_override()
    except config.ConfigException:
        await config.load_kube_config(config_file=kubeconfig, context=context)
    api = ApiClient()
    try:
        yield api
    finally:
        await api.close()


async def list_aiperfsweeps(
    api: ApiClient,
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
) -> list[dict[str, Any]]:
    """List AIPerfSweep CRs.

    Args:
        api: The kubernetes_asyncio ApiClient.
        namespace: When set and ``all_namespaces=False``, list only this namespace.
        all_namespaces: When True, list cluster-wide (cluster-scoped permissions
            required).

    Returns:
        List of raw CR dicts; ``items`` array of the apiserver response.
    """
    co = client.CustomObjectsApi(api)
    if all_namespaces:
        resp = await co.list_cluster_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
        )
    else:
        if namespace is None:
            raise ValueError("namespace must be provided when all_namespaces is False")
        resp = await co.list_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
        )
    return list(resp.get("items", []))


async def find_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Fetch a single AIPerfSweep CR. Returns None on 404; raises on other errors."""
    co = client.CustomObjectsApi(api)
    try:
        return await co.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise


async def get_raw_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Alias of :func:`find_aiperfsweep` matching the AIPerfJob naming convention."""
    return await find_aiperfsweep(api, namespace, name)


async def get_raw_aiperfsweep_status(
    api: ApiClient, name: str, namespace: str
) -> dict[str, Any] | None:
    """Fetch ``status`` subresource of a single AIPerfSweep. Returns None on 404."""
    co = client.CustomObjectsApi(api)
    try:
        body = await co.get_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise
    status = body.get("status")
    return status if isinstance(status, dict) else None

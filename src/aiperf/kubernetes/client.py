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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.client_jobs import (
    cancel_aiperf_job,
    find_aiperf_job,
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
    find_operator_pod,
    find_retrievable_pod,
    get_pod_summary,
    get_pods,
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
    "find_controller_pod",
    "find_jobset",
    "find_operator_pod",
    "find_retrievable_pod",
    "get_pod_summary",
    "get_pods",
    "get_raw_aiperfjob_status",
    "job_selector",
    "k8s_client",
    "list_aiperf_jobs",
    "list_jobsets",
    "wait_for_controller_pod_ready",
]

# ``client`` (the kubernetes_asyncio module) and ``asyncio`` are re-exported as
# module attributes so tests can patch ``aiperf.kubernetes.client.client.CustomObjectsApi``
# and ``aiperf.kubernetes.client.asyncio.sleep``. Python modules are singletons;
# the patches propagate to the sibling ``client_jobs`` / ``client_jobsets`` /
# ``client_pods`` modules that also import these names.


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
    except config.ConfigException:
        await config.load_kube_config(config_file=kubeconfig, context=context)
    api = ApiClient()
    try:
        yield api
    finally:
        await api.close()

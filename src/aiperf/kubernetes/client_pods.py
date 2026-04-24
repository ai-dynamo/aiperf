# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-oriented helpers and cluster-version query for the AIPerf k8s client."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client_selectors import controller_selector
from aiperf.kubernetes.constants import JobSetLabels
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import PodSummary

logger = logging.getLogger(__name__)


async def get_pod_summary(
    api: ApiClient,
    jobset_name: str,
    namespace: str,
) -> PodSummary:
    """Pod readiness summary for a JobSet."""
    core = client.CoreV1Api(api)
    try:
        pod_list = await core.list_namespaced_pod(
            namespace,
            label_selector=f"{JobSetLabels.JOBSET_NAME}={jobset_name}",
        )
    except ApiException:
        return PodSummary(ready=0, total=0, restarts=0)

    pods = pod_list.items
    total = len(pods)
    ready = 0
    restarts = 0
    for pod in pods:
        statuses = (pod.status.container_statuses or []) if pod.status else []
        pod_ready = bool(statuses) and all(cs.ready for cs in statuses)
        phase = pod.status.phase if pod.status else None
        if pod_ready and phase == PodPhase.RUNNING:
            ready += 1
        restarts += sum(cs.restart_count or 0 for cs in statuses)
    return PodSummary(ready=ready, total=total, restarts=restarts)


async def find_operator_pod(
    api: ApiClient,
    namespace: str = "aiperf-system",
    label_selector: str = "app.kubernetes.io/name=aiperf-operator",
) -> tuple[str, PodPhase] | None:
    """Find the operator pod; returns (name, phase) or None."""
    core = client.CoreV1Api(api)
    pod_list = await core.list_namespaced_pod(namespace, label_selector=label_selector)
    if not pod_list.items:
        return None
    pod = pod_list.items[0]
    raw_phase = pod.status.phase if pod.status and pod.status.phase else "Unknown"
    return (pod.metadata.name, PodPhase(raw_phase))


async def find_controller_pod(
    api: ApiClient,
    namespace: str,
    job_id: str,
) -> tuple[str, PodPhase] | None:
    """Find the controller pod for a job; returns (name, phase) or None.

    Uses :func:`controller_selector` to filter for the single pod from the
    ``controller`` replicated-job in the JobSet. If the JobSet spec ever
    scales the controller beyond one replica, this returns the first one.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        namespace: Namespace containing the job's pods.
        job_id: AIPerf job ID (``aiperf.nvidia.com/job-id`` label value).

    Returns:
        ``(pod_name, pod_phase)`` for the controller, or ``None`` if no pod
        matches the selector yet.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API failure
            from ``list_namespaced_pod`` (not suppressed — callers decide).
    """
    core = client.CoreV1Api(api)
    pod_list = await core.list_namespaced_pod(
        namespace,
        label_selector=controller_selector(job_id),
    )
    if not pod_list.items:
        return None
    pod = pod_list.items[0]
    raw_phase = pod.status.phase if pod.status and pod.status.phase else "Unknown"
    return (pod.metadata.name, PodPhase(raw_phase))


async def find_retrievable_pod(
    api: ApiClient,
    namespace: str,
    job_id: str,
    *,
    require_running: bool = False,
) -> tuple[str, PodPhase] | None:
    """Find the controller pod only if it is in a retrievable phase."""
    pod_info = await find_controller_pod(api, namespace, job_id)
    if not pod_info:
        return None
    pod_name, pod_phase = pod_info
    if require_running:
        if pod_phase != PodPhase.RUNNING:
            return None
    elif not pod_phase.is_retrievable:
        return None
    return pod_name, pod_phase


async def wait_for_controller_pod_ready(
    api: ApiClient,
    namespace: str,
    job_id: str,
    timeout: int = 300,
) -> str:
    """Poll until the controller pod is Running; returns its name."""
    start = asyncio.get_running_loop().time()
    last_log = 0.0
    while True:
        result = await find_controller_pod(api, namespace, job_id)
        elapsed = asyncio.get_running_loop().time() - start
        if result:
            pod_name, phase = result
            if phase == PodPhase.RUNNING:
                return pod_name
            if elapsed - last_log >= 10:
                logger.info("Controller pod %s: %s (%.0fs)", pod_name, phase, elapsed)
                last_log = elapsed
        elif elapsed - last_log >= 10:
            logger.info("No controller pod found yet (%.0fs)", elapsed)
            last_log = elapsed
        if elapsed > timeout:
            raise TimeoutError(
                f"Controller pod not ready after {timeout}s. "
                f"Check with: kubectl get pods -n {namespace}"
            )
        await asyncio.sleep(2)


async def get_pods(
    api: ApiClient,
    namespace: str,
    label_selector: str,
) -> list[Any]:
    """Return list of ``V1Pod`` matching label selector (typed access).

    Thin wrapper over ``CoreV1Api(api).list_namespaced_pod(...).items`` —
    exposed so callers that need full typed pod access (containers, conditions,
    annotations, etc.) don't re-create a ``CoreV1Api`` instance.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        namespace: Namespace to list pods in.
        label_selector: Comma-separated label selector (see :func:`job_selector`
            / :func:`controller_selector` for canonical AIPerf selectors).

    Returns:
        List of ``kubernetes_asyncio.client.V1Pod`` instances. Empty list if
        no pods match. Return type is ``list[Any]`` because the k8s-asyncio
        ``V1Pod`` class is not a stable import path across versions.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API failure
            (not suppressed).

    Example:
        >>> async with k8s_client() as api:
        ...     pods = await get_pods(api, "aiperf-bench", job_selector("job-abc"))
        ...     print([p.metadata.name for p in pods])
    """
    core = client.CoreV1Api(api)
    return (
        await core.list_namespaced_pod(namespace, label_selector=label_selector)
    ).items


async def cluster_version(api: ApiClient) -> dict[str, Any]:
    """Return Kubernetes cluster version info as a dict.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.

    Returns:
        Dict with keys ``major``, ``minor``, ``gitVersion``, ``gitCommit``,
        ``platform`` — all strings sourced from ``/version`` on the apiserver.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API failure
            (not suppressed — this endpoint is cheap and failure usually means
            the apiserver is unreachable, which callers want to see).
    """
    vinfo = await client.VersionApi(api).get_code()
    return {
        "major": vinfo.major,
        "minor": vinfo.minor,
        "gitVersion": vinfo.git_version,
        "gitCommit": vinfo.git_commit,
        "platform": vinfo.platform,
    }

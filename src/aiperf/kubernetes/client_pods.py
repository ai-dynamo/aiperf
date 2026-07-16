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
from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE, JobSetLabels
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
    namespace: str = DEFAULT_OPERATOR_NAMESPACE,
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


async def find_operator_namespace(
    api: ApiClient,
    label_selector: str = "app.kubernetes.io/name=aiperf-operator",
) -> str | None:
    """Cluster-wide search for an aiperf-operator pod; returns its namespace.

    Returns:
        Namespace of the first matching operator pod, or ``None`` if no pods
        match. Also returns ``None`` (without raising) when the caller lacks
        cluster-wide ``list pods`` RBAC — caller should fall back to a default.

    The caller is expected to log a warning if more than one operator install
    is detected; that's surfaced via the ``logger`` here.
    """
    core = client.CoreV1Api(api)
    try:
        pod_list = await core.list_pod_for_all_namespaces(
            label_selector=label_selector,
        )
    except ApiException as exc:
        # 403 = cluster-wide list-pods forbidden; caller falls back to default.
        if exc.status == 403:
            logger.debug(
                "Cluster-wide pod list forbidden (RBAC); operator namespace "
                "auto-detect unavailable: %s",
                exc.reason,
            )
            return None
        raise
    if not pod_list.items:
        return None
    namespaces = {p.metadata.namespace for p in pod_list.items if p.metadata}
    if not namespaces:
        return None
    chosen = sorted(namespaces)[0]
    if len(namespaces) > 1:
        logger.warning(
            "Multiple aiperf-operator installs detected across namespaces %s; "
            "picking '%s'. Pass --operator-namespace to override.",
            sorted(namespaces),
            chosen,
        )
    return chosen


async def resolve_operator_namespace(
    api: ApiClient,
    *,
    explicit: str | None,
    default: str = DEFAULT_OPERATOR_NAMESPACE,
) -> str:
    """Pick the operator namespace: explicit flag > cluster-wide auto-detect > default.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        explicit: Value of a CLI ``--operator-namespace`` flag, or ``None`` to
            auto-detect.
        default: Fallback when auto-detect finds no match or RBAC blocks the
            cluster-wide pod list. Matches the chart's default install location.

    Returns:
        The resolved namespace string. Never raises for the auto-detect path —
        the worst case is the caller getting back ``default`` and surfacing
        "operator pod not found in namespace 'X'" downstream.
    """
    if explicit is not None:
        return explicit
    detected = await find_operator_namespace(api)
    return detected if detected is not None else default


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
            if phase in (PodPhase.FAILED, PodPhase.SUCCEEDED):
                raise RuntimeError(
                    f"Controller pod {pod_name} reached terminal phase {phase} "
                    f"before Running; check: kubectl logs/describe -n {namespace} "
                    f"{pod_name}"
                )
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


async def list_events_for_object(
    api: ApiClient,
    namespace: str,
    object_name: str,
) -> list[Any]:
    """Return raw ``V1Event`` objects whose ``involvedObject.name`` matches.

    Uses a server-side field selector so the apiserver filters for us — cheap
    even in busy namespaces. ``involvedObject.name`` is not globally unique
    across kinds, so callers that care about disambiguation must filter by
    ``involvedObject.kind`` themselves. The AIPerfJob UI relies on the fact
    that CRs and their pods always have distinct names, so no secondary
    filtering is needed there.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        namespace: Namespace whose event stream is searched.
        object_name: Value matched against ``involvedObject.name``.

    Returns:
        List of ``kubernetes_asyncio.client.V1Event`` instances. Empty list if
        the apiserver returns no matches. Return type is ``list[Any]`` because
        the ``V1Event`` class is not a stable import path across versions.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API failure
            (not suppressed).

    Example:
        >>> async with k8s_client() as api:
        ...     events = await list_events_for_object(api, "ml-lab", "bench-7f2a")
        ...     for ev in events:
        ...         print(ev.last_timestamp, ev.reason, ev.message)
    """
    core = client.CoreV1Api(api)
    resp = await core.list_namespaced_event(
        namespace=namespace,
        field_selector=f"involvedObject.name={object_name}",
    )
    return list(resp.items or [])


async def list_nodes(api: ApiClient) -> list[Any]:
    """Return cluster-wide ``V1Node`` list for the given apiclient.

    Thin wrapper over ``CoreV1Api(api).list_node().items`` exposed so
    the UI's cluster-info endpoint (``_fetch_cluster_gpu_stats``) has a
    single patch point for test fakes alongside :func:`get_pods` /
    :func:`list_events_for_object`. Callers that just need counts or
    GPU allocatable totals iterate the returned list directly.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.

    Returns:
        List of ``kubernetes_asyncio.client.V1Node`` instances. Empty
        list if the cluster is empty. Return type is ``list[Any]``
        because the ``V1Node`` class is not a stable import path.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API
            failure — in particular 403 when the ServiceAccount's
            ClusterRole lacks ``nodes get/list``. Callers decide
            whether to suppress.

    Example:
        >>> async with k8s_client() as api:
        ...     nodes = await list_nodes(api)
        ...     gpus = sum(int(n.status.allocatable.get("nvidia.com/gpu", 0)) for n in nodes)
    """
    return list((await client.CoreV1Api(api).list_node()).items or [])


async def list_pods_all_namespaces(api: ApiClient) -> list[Any]:
    """Return cluster-wide ``V1Pod`` list across every namespace.

    Wraps ``CoreV1Api(api).list_pod_for_all_namespaces().items`` so the UI
    cluster-stats endpoint can compute GPU usage by summing
    ``nvidia.com/gpu`` requests on Running/Pending pods. Requires the
    operator ServiceAccount's ClusterRole to grant ``pods get/list`` —
    already present alongside the ``nodes`` permission used by
    :func:`list_nodes`.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.

    Returns:
        List of ``kubernetes_asyncio.client.V1Pod``. Empty list if the
        cluster has no pods. Return type is ``list[Any]`` for the same
        reason as :func:`list_nodes`.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API
            failure — in particular 403 when the ServiceAccount's
            ClusterRole lacks ``pods list`` cluster-wide.
    """
    return list((await client.CoreV1Api(api).list_pod_for_all_namespaces()).items or [])


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

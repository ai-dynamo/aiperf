# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes client — free functions over kubernetes_asyncio.

The canonical interface is ``k8s_client()`` + free functions
(``list_aiperf_jobs``, ``find_jobset``, …) that take an ``ApiClient``
explicitly and call ``CoreV1Api(api)`` / ``CustomObjectsApi(api)``
inline so the reader sees the native kubernetes_asyncio API surface.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.console import print_info, print_success, print_warning
from aiperf.kubernetes.constants import AIPerfLabels, JobSetLabels
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import (
    AIPerfJobCR,
    AIPerfJobInfo,
    JobSetInfo,
    PodSummary,
)

logger = logging.getLogger(__name__)


# ----- Config + ApiClient ----------------------------------------------------


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


# ----- Label selectors (pure strings) ----------------------------------------


def job_selector(job_id: str) -> str:
    """Build the label selector for all AIPerf resources belonging to a job.

    Combines the repo-wide ``AIPerfLabels.SELECTOR`` (``app.kubernetes.io/part-of=aiperf``)
    with the per-job ``AIPerfLabels.JOB_ID`` into a single comma-separated selector
    string consumable by any ``list_*`` / ``delete_*`` k8s API.

    Args:
        job_id: AIPerf job ID (the value stored on ``metadata.labels[aiperf.nvidia.com/job-id]``).

    Returns:
        A selector string like ``"app.kubernetes.io/part-of=aiperf,aiperf.nvidia.com/job-id=<job_id>"``.

    Raises:
        Never raises — pure string construction.
    """
    return f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}={job_id}"


def controller_selector(job_id: str) -> str:
    """Label selector for the controller pod of a job."""
    return (
        f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}={job_id},"
        f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
    )


# ----- AIPerfJob CR helpers --------------------------------------------------


async def list_aiperf_jobs(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    status_filter: str | None = None,
) -> list[AIPerfJobInfo]:
    """List AIPerfJob CRs, sorted newest-first.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`. Callers own its lifecycle.
        namespace: Namespace to list in. Ignored when ``all_namespaces=True``.
            ``None`` resolves to ``"default"``.
        all_namespaces: If ``True``, lists across the cluster instead of a
            single namespace. Requires cluster-wide list permission on
            ``aiperfjobs.aiperf.nvidia.com``.
        status_filter: If set, keep only jobs whose ``phase`` equals this string
            (e.g. ``"Running"``, ``"Succeeded"``).

    Returns:
        List of :class:`AIPerfJobInfo` sorted by ``created`` descending.
        Empty list if no jobs match or if the CRD is not installed (404 is
        suppressed so fresh clusters look empty, not broken).

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any non-404 API
            failure (403 forbidden, 500, transport error, etc.).

    Example:
        >>> async with k8s_client() as api:
        ...     running = await list_aiperf_jobs(api, all_namespaces=True, status_filter="Running")
        ...     print(f"{len(running)} running jobs")
    """
    custom = client.CustomObjectsApi(api)
    try:
        if all_namespaces:
            result = await custom.list_cluster_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
            )
        else:
            ns = namespace or "default"
            result = await custom.list_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=ns,
            )
    except ApiException as e:
        if e.status == 404:
            return []
        raise

    infos = [
        AIPerfJobCR.model_validate(raw).to_info() for raw in result.get("items", [])
    ]
    if status_filter:
        infos = [i for i in infos if i.phase == status_filter]
    infos.sort(key=lambda x: x.created, reverse=True)
    return infos


async def find_aiperf_job(
    api: ApiClient,
    name: str,
    namespace: str | None = None,
) -> AIPerfJobInfo | None:
    """Find an AIPerfJob by resource name, with fallback to jobId match.

    Resolution order:

    1. If ``namespace`` is given, direct ``get_namespaced_custom_object`` by name.
    2. Otherwise (or on 404 from step 1), cluster-wide list filtered by
       ``metadata.name=<name>``, then match either ``metadata.name`` or
       ``status.jobId`` against the input. This lets callers look up a job by
       either its Kubernetes resource name or its generated ``jobId``.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        name: Either the AIPerfJob resource name (``metadata.name``) or the
            generated ``status.jobId``.
        namespace: Namespace to look in. ``None`` scans all namespaces.

    Returns:
        The matching :class:`AIPerfJobInfo`, or ``None`` if no match found.
        404 is suppressed (treated as "not found").

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any non-404
            failure from either the direct ``get`` or the cluster-wide ``list``.

    Example:
        >>> async with k8s_client() as api:
        ...     job = await find_aiperf_job(api, "my-bench-run", namespace="aiperf-bench")
        ...     if job:
        ...         print(job.phase)
    """
    custom = client.CustomObjectsApi(api)

    # Direct lookup by name — most common path.
    if namespace is not None:
        try:
            raw = await custom.get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
            )
            return AIPerfJobCR.model_validate(raw).to_info()
        except ApiException as e:
            if e.status != 404:
                raise

    # Fallback: scan all namespaces for a status.jobId match.
    try:
        result = await custom.list_cluster_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            field_selector=f"metadata.name={name}" if namespace is None else None,
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise

    for raw in result.get("items", []):
        cr = AIPerfJobCR.model_validate(raw)
        if cr.metadata.name == name or cr.status.job_id == name:
            return cr.to_info()
    return None


async def get_raw_aiperfjob_status(
    api: ApiClient,
    name: str,
    namespace: str,
) -> dict[str, Any]:
    """Return the raw ``status`` dict of an AIPerfJob by name (empty on miss).

    Unlike :func:`find_aiperf_job`, this bypasses the :class:`AIPerfJobCR`
    model and returns the unparsed ``status`` subobject — useful for reading
    controller-written fields that are not yet promoted into the typed model.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        name: AIPerfJob resource name (``metadata.name``).
        namespace: Namespace containing the AIPerfJob.

    Returns:
        The raw ``status`` mapping from the CR (arbitrary keys controller-defined),
        or ``{}`` if the CR is missing, has no status yet, or on any API error.

    Raises:
        Never raises: any :class:`ApiException` is suppressed and returns ``{}``.
        This is intentional — status polling is best-effort.
    """
    custom = client.CustomObjectsApi(api)
    try:
        raw = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
    except ApiException:
        return {}
    return raw.get("status", {}) or {}


async def cancel_aiperf_job(api: ApiClient, name: str, namespace: str) -> None:
    """Cancel an AIPerfJob by setting ``spec.cancel=true`` (merge patch).

    The operator watches this field and triggers graceful teardown (controller
    shutdown, JobSet deletion, phase transition to ``Cancelled``). This call
    returns as soon as the patch is acknowledged; actual cancellation is async.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        name: AIPerfJob resource name (``metadata.name``).
        namespace: Namespace containing the AIPerfJob.

    Returns:
        ``None``. The patch is accepted by the apiserver; the operator handles
        the rest of the cancellation workflow.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any API failure
            (404 if the CR does not exist, 403 forbidden, 409 conflict, etc.).
            Nothing is suppressed — callers decide how to react.

    Example:
        >>> async with k8s_client() as api:
        ...     await cancel_aiperf_job(api, "my-bench-run", namespace="aiperf-bench")
    """
    custom = client.CustomObjectsApi(api)
    await custom.patch_namespaced_custom_object(
        group=AIPERF_JOB_GROUP,
        version=AIPERF_JOB_VERSION,
        plural=AIPERF_JOB_PLURAL,
        namespace=namespace,
        name=name,
        body={"spec": {"cancel": True}},
    )


# ----- JobSet helpers --------------------------------------------------------


async def _list_jobsets_raw(
    api: ApiClient,
    label_selector: str,
    namespace: str | None = None,
    field_selector: str | None = None,
) -> list[dict[str, Any]]:
    """List JobSet raw dicts matching selectors."""
    custom = client.CustomObjectsApi(api)
    kwargs: dict[str, Any] = {"label_selector": label_selector}
    if field_selector:
        kwargs["field_selector"] = field_selector

    if namespace is None:
        result = await custom.list_cluster_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            **kwargs,
        )
    else:
        result = await custom.list_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            **kwargs,
        )
    return result.get("items", []) or []


async def list_jobsets(
    api: ApiClient,
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
    job_id: str | None = None,
    status_filter: str | None = None,
) -> list[JobSetInfo]:
    """List AIPerf-owned JobSets, sorted newest-first.

    Always filters by ``AIPerfLabels.SELECTOR`` (``app.kubernetes.io/part-of=aiperf``)
    so third-party JobSets never appear. ``job_id`` narrows further to a single
    job's JobSet.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        namespace: Namespace to list in. Ignored when ``all_namespaces=True``.
            ``None`` resolves to ``"default"``.
        all_namespaces: If ``True``, lists cluster-wide.
        job_id: If set, AND the selector with ``aiperf.nvidia.com/job-id=<job_id>``.
        status_filter: If set, keep only JobSets whose ``status`` equals this
            string (e.g. ``"Completed"``, ``"Failed"``).

    Returns:
        List of :class:`JobSetInfo` sorted by ``created`` descending. Empty list
        on 404 (JobSet CRD not installed).

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any non-404 API
            failure.
    """
    label_selector = AIPerfLabels.SELECTOR
    if job_id:
        label_selector += f",{AIPerfLabels.JOB_ID}={job_id}"

    ns = None if all_namespaces else (namespace or "default")
    try:
        raws = await _list_jobsets_raw(api, label_selector, ns)
    except ApiException as e:
        if e.status == 404:
            return []
        raise

    infos = [JobSetInfo.from_raw(r) for r in raws]
    if status_filter:
        infos = [i for i in infos if i.status == status_filter]
    infos.sort(key=lambda x: x.created, reverse=True)
    return infos


async def find_jobset(
    api: ApiClient,
    job_id: str,
    namespace: str | None = None,
) -> JobSetInfo | None:
    """Find a JobSet by AIPerf job ID label, falling back to resource name.

    Tries label-selector lookup first (``aiperf.nvidia.com/job-id=<job_id>``);
    if nothing matches, retries with a ``metadata.name=<job_id>`` field
    selector so callers can pass either the labelled job ID or the raw
    JobSet resource name.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        job_id: AIPerf job ID, or a JobSet resource name as a fallback.
        namespace: Namespace to scope the search. ``None`` searches cluster-wide.

    Returns:
        The first matching :class:`JobSetInfo`, or ``None`` if nothing matches
        in either pass. 404 is suppressed.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: On any non-404 API
            failure in either the label-selector or field-selector pass.
    """
    try:
        raws = await _list_jobsets_raw(api, job_selector(job_id), namespace)
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    if raws:
        return JobSetInfo.from_raw(raws[0])

    try:
        raws = await _list_jobsets_raw(
            api,
            AIPerfLabels.SELECTOR,
            namespace,
            field_selector=f"metadata.name={job_id}",
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    return JobSetInfo.from_raw(raws[0]) if raws else None


async def delete_jobset(api: ApiClient, name: str, namespace: str) -> None:
    """Delete a JobSet and its associated ConfigMap/Role/RoleBinding.

    AIPerf provisions four resources per job, all named by suffix off the
    JobSet name. This function deletes all four in order, best-effort:

    1. ``JobSet/<name>``                          (``jobset.x-k8s.io``)
    2. ``ConfigMap/<name>-config``                (``core``)
    3. ``Role/<name>-role``                       (``rbac.authorization.k8s.io``)
    4. ``RoleBinding/<name>-binding``             (``rbac.authorization.k8s.io``)

    Each deletion logs success via :func:`print_success`. ``404 Not Found`` and
    ``409 Conflict`` (namespace terminating) are suppressed per-resource so a
    partially-torn-down job can still be fully cleaned up. Any other failure
    on resources 2-4 is logged via :func:`print_warning` and skipped — only
    an unexpected failure on the JobSet delete itself raises.

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        name: JobSet resource name. The three auxiliary resources are derived
            as ``f"{name}-config"``, ``f"{name}-role"``, ``f"{name}-binding"``.
        namespace: Namespace containing all four resources.

    Returns:
        ``None``. Side effects: up to four ``DELETE`` calls and up to four
        console log lines. Does not wait for finalizers — returns as soon as
        the apiserver accepts the deletion.

    Raises:
        kubernetes_asyncio.client.exceptions.ApiException: Only from the
            JobSet delete itself, and only for non-404 statuses. Failures on
            the ConfigMap/Role/RoleBinding deletes are logged-and-swallowed.

    Example:
        >>> async with k8s_client() as api:
        ...     await delete_jobset(api, "my-bench-run", namespace="aiperf-bench")
    """
    custom = client.CustomObjectsApi(api)
    core = client.CoreV1Api(api)
    rbac = client.RbacAuthorizationV1Api(api)

    try:
        await custom.delete_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=name,
        )
        print_success(f"Deleted JobSet/{name}")
    except ApiException as e:
        if e.status == 404:
            print_warning(f"JobSet/{name} not found")
        else:
            raise

    # Associated resources named "<jobset>-<suffix>"
    targets = [
        (core.delete_namespaced_config_map, f"{name}-config", "ConfigMap"),
        (rbac.delete_namespaced_role, f"{name}-role", "Role"),
        (rbac.delete_namespaced_role_binding, f"{name}-binding", "RoleBinding"),
    ]
    for delete_fn, resource_name, kind in targets:
        try:
            await delete_fn(name=resource_name, namespace=namespace)
            print_success(f"Deleted {kind}/{resource_name}")
        except ApiException as e:
            if e.status in (404, 409):
                # 404 already gone; 409 namespace terminating — both benign.
                continue
            print_warning(f"Failed to delete {kind}/{resource_name}: {e}")


async def delete_namespace(api: ApiClient, name: str) -> None:
    """Delete a Kubernetes namespace (404 treated as already gone)."""
    core = client.CoreV1Api(api)
    try:
        await core.delete_namespace(name=name)
        print_success(f"Deleted Namespace/{name}")
    except ApiException as e:
        if e.status == 404:
            print_info(f"Namespace {name} not found (may already be deleted)")
        else:
            print_warning(f"Failed to delete namespace: {e}")


# ----- Pod helpers -----------------------------------------------------------


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

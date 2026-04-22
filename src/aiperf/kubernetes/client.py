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
from aiperf.kubernetes.constants import JobSetLabels, Labels
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
    """Load k8s config and yield an ApiClient.

    In-cluster first, kubeconfig fallback. The ApiClient is closed on exit.
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
    """Label selector for all AIPerf resources belonging to a job."""
    return f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id}"


def controller_selector(job_id: str) -> str:
    """Label selector for the controller pod of a job."""
    return (
        f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id},"
        f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
    )


# ----- AIPerfJob CR helpers --------------------------------------------------


async def list_aiperf_jobs(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    status_filter: str | None = None,
) -> list[AIPerfJobInfo]:
    """List AIPerfJob CRs, sorted newest-first."""
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
    """Find an AIPerfJob by resource name, with fallback to jobId match."""
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
    """Return the raw ``status`` dict of an AIPerfJob by name (empty on miss)."""
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
    """Cancel an AIPerfJob by setting ``spec.cancel=true`` (merge patch)."""
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
    namespace: str | None = None,
    all_namespaces: bool = False,
    job_id: str | None = None,
    status_filter: str | None = None,
) -> list[JobSetInfo]:
    """List AIPerf-owned JobSets, sorted newest-first."""
    label_selector = Labels.SELECTOR
    if job_id:
        label_selector += f",{Labels.JOB_ID}={job_id}"

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
    """Find a JobSet by AIPerf job ID label, falling back to resource name."""
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
            Labels.SELECTOR,
            namespace,
            field_selector=f"metadata.name={job_id}",
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    return JobSetInfo.from_raw(raws[0]) if raws else None


async def delete_jobset(api: ApiClient, name: str, namespace: str) -> None:
    """Delete a JobSet and its associated ConfigMap/Role/RoleBinding."""
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
    """Find the controller pod for a job; returns (name, phase) or None."""
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
    """Return list of V1Pod matching label selector (typed access)."""
    core = client.CoreV1Api(api)
    return (
        await core.list_namespaced_pod(namespace, label_selector=label_selector)
    ).items


async def cluster_version(api: ApiClient) -> dict[str, Any]:
    """Return Kubernetes cluster version info as a dict."""
    vinfo = await client.VersionApi(api).get_code()
    return {
        "major": vinfo.major,
        "minor": vinfo.minor,
        "gitVersion": vinfo.git_version,
        "gitCommit": vinfo.git_commit,
        "platform": vinfo.platform,
    }

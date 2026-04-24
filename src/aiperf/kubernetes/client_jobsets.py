# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JobSet helpers — list/find/delete free functions plus namespace delete."""

from __future__ import annotations

from typing import Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client_selectors import job_selector
from aiperf.kubernetes.console import print_info, print_success, print_warning
from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.cr_refs import (
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.models import JobSetInfo


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

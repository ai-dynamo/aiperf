# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerfJob CR helpers — list/find/get/cancel free functions."""

from __future__ import annotations

from typing import Any

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
)
from aiperf.kubernetes.models import AIPerfJobCR, AIPerfJobInfo


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


async def get_raw_aiperfjob(
    api: ApiClient,
    namespace: str,
    name: str,
) -> dict[str, Any] | None:
    """Return the full raw AIPerfJob CR dict (spec + status + metadata) or None.

    Unlike :func:`find_aiperf_job` which returns a typed :class:`AIPerfJobInfo`,
    this returns the untouched apiserver response so callers can read the live
    ``spec`` of a running CR that has no on-disk artifacts yet (e.g. the UI
    config endpoint's live-spec fallback for dashboard SLO chips).

    Args:
        api: Open ``ApiClient`` from :func:`k8s_client`.
        namespace: Namespace containing the AIPerfJob.
        name: AIPerfJob resource name (``metadata.name``).

    Returns:
        The raw CR body (``{"apiVersion", "metadata", "spec", "status", ...}``),
        or ``None`` if the CR does not exist (404). Any non-404 API error is
        suppressed and also returns ``None`` — this helper is intended for
        best-effort UI lookups where a missing/erroring CR should fall through
        silently rather than surface as a 5xx.
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
        return None
    return raw or None


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

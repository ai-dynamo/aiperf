# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API router for live Kubernetes job and cluster state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.kubernetes.client import (
    cancel_aiperf_job,
    cluster_version,
    get_pods,
    get_raw_aiperfjob_status,
)
from aiperf.operator.job_union import find_any_job, list_all_jobs

if TYPE_CHECKING:
    from kubernetes_asyncio.client.models import V1Node, V1Pod

logger = logging.getLogger("aiperf.operator.ui")


class JobPodSummary(AIPerfBaseModel):
    """Pod identity + lifecycle summary returned in JobDetailResponse.

    Distinct from ``aiperf.kubernetes.models.PodSummary`` (an aggregate
    ``ready/total/restarts`` snapshot of a JobSet): this model is per-pod and
    includes the pod name / phase.
    """

    name: str = Field(description="Pod name.")
    phase: str = Field(description="Pod phase (Running, Pending, Succeeded, ...).")
    ready: bool = Field(description="True iff at least one container is ready.")
    restarts: int = Field(description="Sum of restart counts across containers.")


def _pod_summary(pod: V1Pod) -> JobPodSummary:
    """Extract pod name, phase, readiness, and restart count for the UI."""
    meta = pod.metadata
    status = pod.status
    container_statuses = (status.container_statuses or []) if status else []
    return JobPodSummary(
        name=(meta.name if meta else "") or "",
        phase=(status.phase if status else None) or "Unknown",
        ready=any(bool(c.ready) for c in container_statuses),
        restarts=sum(int(c.restart_count or 0) for c in container_statuses),
    )


def _node_gpu_count(node: V1Node) -> int:
    """Return the number of nvidia.com/gpu resources allocatable on a node."""
    alloc = (node.status.allocatable or {}) if node.status else {}
    try:
        return int(alloc.get("nvidia.com/gpu", 0))
    except (TypeError, ValueError):
        return 0


async def _fetch_k8s_version(api: ApiClient) -> str:
    """Return the cluster gitVersion, or 'unknown' if the call fails."""
    try:
        version_info = await cluster_version(api)
    except Exception:  # noqa: BLE001 - best-effort; UI tolerates 'unknown'
        return "unknown"
    return version_info.get("gitVersion", "unknown")


async def _fetch_node_gpu_totals(api: ApiClient) -> tuple[int, int]:
    """Return (node_count, total_nvidia_gpus). Returns (0, 0) on failure."""
    try:
        node_list = await client.CoreV1Api(api).list_node()
    except ApiException as e:
        # 403 here is almost always the operator ClusterRole missing
        # `nodes get/list` — log at ERROR so it surfaces in the usual
        # RBAC-misconfig triage instead of masquerading as "0 nodes".
        if (e.status or 0) == 403:
            logger.error(
                "Cluster node listing forbidden (403) — check that the "
                "operator ClusterRole grants `nodes get/list`: %s",
                e,
            )
        else:
            logger.warning("Failed to query nodes (apiserver %s): %s", e.status, e)
        return 0, 0
    except Exception as e:  # noqa: BLE001 - UI tolerates missing cluster-wide query
        logger.warning(f"Failed to query nodes: {e}")
        return 0, 0
    nodes = node_list.items
    return len(nodes), sum(_node_gpu_count(n) for n in nodes)


class ActiveJobListResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs: active AIPerfJob CRs in the cluster."""

    jobs: list[dict[str, Any]] = Field(description="List of AIPerfJob summaries.")


class JobDetailResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs/{namespace}/{name}."""

    job: dict[str, Any] = Field(description="AIPerfJob summary.")
    status: dict[str, Any] = Field(
        description="Raw CR status (phases, conditions, liveMetrics)."
    )
    pods: list[JobPodSummary] = Field(description="Pod summaries for this job.")


class ClusterResponse(AIPerfBaseModel):
    """Response for GET /api/v1/cluster."""

    nodes: int = Field(description="Number of cluster nodes.")
    gpus: int = Field(description="Total allocatable GPUs.")
    kubernetes_version: str = Field(description="Kubernetes server version.")


class CancelResponse(AIPerfBaseModel):
    """Response for POST /api/v1/jobs/{namespace}/{name}/cancel."""

    cancelled: bool = Field(description="Whether cancellation was requested.")


async def _list_jobs_impl(api: ApiClient, results_dir: Path) -> ActiveJobListResponse:
    """Body of GET /api/v1/jobs: union of active CRs + archived PVC directories.

    Returns the unified view from :func:`aiperf.operator.job_union.list_all_jobs`:
    live CRs (``source="live"``), PVC-only historical runs (``source="archived"``),
    and CRs that also have a persisted summary (``source="both"``). Keyed by
    ``(namespace, name)``; overlap entries prefer CR values on live fields and
    backfill from PVC on historical-only fields.

    Raises:
        HTTPException: Any non-404 ``kubernetes_asyncio.client.ApiException``
            status code from the CR half is surfaced verbatim (e.g. 401/403 on
            RBAC denial). The PVC half is tolerant and falls back to an empty
            list on filesystem errors.
    """
    jobs = await list_all_jobs(api, results_dir, all_namespaces=True)
    return ActiveJobListResponse(jobs=[j.model_dump(by_alias=True) for j in jobs])


async def _get_job_impl(
    api: ApiClient,
    results_dir: Path,
    namespace: str,
    name: str,
) -> JobDetailResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}: fetch a CR plus its pod roster.

    Returns three things joined into one response: (1) the AIPerfJob summary
    (same shape as ``list_jobs``), (2) the raw CR ``.status`` subresource
    (phase, conditions, liveMetrics), and (3) the current pod list filtered by
    the ``aiperf.nvidia.com/job-id=<name>`` label selector.

    Archived (PVC-only) jobs have no cluster CR, so the response returns an
    empty ``status`` dict and empty ``pods`` list alongside the archived job
    summary.

    Args:
        api: The kubernetes_asyncio ApiClient.
        results_dir: Base directory on the results PVC.
        namespace: Kubernetes namespace containing the AIPerfJob CR or PVC dir.
        name: Name of the AIPerfJob CR (also the label value matched when
            listing pods, and the PVC subdirectory name).

    Raises:
        HTTPException: 404 if neither a live CR nor a PVC directory exists.
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial).
    """
    job = await find_any_job(api, results_dir, namespace, name)
    if job is None:
        raise HTTPException(404, f"Job {namespace}/{name} not found")

    if job.source == "archived":
        return JobDetailResponse(
            job=job.model_dump(by_alias=True),
            status={},
            pods=[],
        )

    raw_status = await get_raw_aiperfjob_status(api, name, namespace)
    pods_raw = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    return JobDetailResponse(
        job=job.model_dump(by_alias=True),
        status=raw_status or {},
        pods=[_pod_summary(p) for p in pods_raw],
    )


async def _cancel_job_impl(
    api: ApiClient,
    results_dir: Path,
    namespace: str,
    name: str,
) -> CancelResponse:
    """Body of POST /api/v1/jobs/{namespace}/{name}/cancel: set ``spec.cancel=true``.

    This endpoint is *asynchronous*: it patches the AIPerfJob CR's
    ``spec.cancel`` field to ``true`` and returns immediately. The kopf
    operator's reconciler observes the change and drives the benchmark to a
    stopped state (cancelling workers, tearing down pods, finalising results).
    The endpoint does NOT wait for that reconciliation - callers that need to
    observe the terminal phase should poll ``get_job`` until ``status.phase``
    becomes ``Cancelled``/``Failed``/``Succeeded``.

    Archived (PVC-only) jobs cannot be cancelled — their Kubernetes resource no
    longer exists — so the endpoint returns 400 instead of attempting the patch.

    Args:
        api: The kubernetes_asyncio ApiClient.
        results_dir: Base directory on the results PVC (used to detect
            archived-only jobs that have no CR to cancel).
        namespace: Kubernetes namespace containing the AIPerfJob CR.
        name: Name of the AIPerfJob CR to cancel.

    Raises:
        HTTPException: 404 if neither a live CR nor a PVC directory exists.
        HTTPException: 400 if the job is archived-only (no CR on the cluster).
        HTTPException: Other ``kubernetes_asyncio.client.ApiException`` status
            codes propagate (e.g. 401/403 on RBAC denial, 409 on
            concurrent-modification conflicts).
    """
    job = await find_any_job(api, results_dir, namespace, name)
    if job is None:
        raise HTTPException(404, f"Job {namespace}/{name} not found")
    if job.source == "archived":
        raise HTTPException(
            400,
            f"Cannot cancel archived job {namespace}/{name}: "
            "the Kubernetes resource no longer exists.",
        )
    await cancel_aiperf_job(api, name, namespace)
    return CancelResponse(cancelled=True)


async def _cluster_info_impl(api: ApiClient) -> ClusterResponse:
    """Body of GET /api/v1/cluster: best-effort cluster-wide node and GPU totals.

    Calls the core ``/version`` endpoint for the server gitVersion and
    ``list_node`` for node count + ``nvidia.com/gpu`` allocatable totals. Both
    calls are best-effort: failures fall back to ``"unknown"`` / ``(0, 0)``
    rather than surfacing errors, because the UI displays this as supplementary
    context and callers with limited RBAC should not see the page fail.
    """
    k8s_version = await _fetch_k8s_version(api)
    node_count, gpu_count = await _fetch_node_gpu_totals(api)
    return ClusterResponse(
        nodes=node_count,
        gpus=gpu_count,
        kubernetes_version=k8s_version,
    )


def create_jobs_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
) -> APIRouter:
    """Create the jobs/cluster API router.

    All endpoints return 503 if the Kubernetes ApiClient has not been
    initialised (set during FastAPI lifespan startup). See the ``_*_impl``
    helpers above for per-endpoint behaviour and error semantics.

    Args:
        api_holder: Mutable single-element list holding the kubernetes_asyncio
            ApiClient. The client is set during app lifespan startup. If the
            list is empty or contains None, endpoints return 503.
        results_dir: Base directory on the results PVC; passed to the union
            helpers so ``GET /jobs`` and ``GET /jobs/{ns}/{name}`` can surface
            archived (CR-deleted) runs alongside live ones.
    """
    _holder = api_holder if api_holder is not None else [None]
    _results_dir = results_dir if results_dir is not None else Path("/data")
    router = APIRouter(prefix="/api/v1", tags=["jobs"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(
                503,
                "Kubernetes API client not yet initialized by FastAPI lifespan; "
                "retry in a few seconds or check /healthz",
            )
        return api

    @router.get("/jobs", response_model=ActiveJobListResponse)
    async def list_jobs() -> ActiveJobListResponse:
        return await _list_jobs_impl(_require_api(), _results_dir)

    @router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
    async def get_job(namespace: str, name: str) -> JobDetailResponse:
        return await _get_job_impl(_require_api(), _results_dir, namespace, name)

    @router.post("/jobs/{namespace}/{name}/cancel", response_model=CancelResponse)
    async def cancel_job(namespace: str, name: str) -> CancelResponse:
        return await _cancel_job_impl(_require_api(), _results_dir, namespace, name)

    @router.get("/cluster", response_model=ClusterResponse)
    async def cluster_info() -> ClusterResponse:
        return await _cluster_info_impl(_require_api())

    return router

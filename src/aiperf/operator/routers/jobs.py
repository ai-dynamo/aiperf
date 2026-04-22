# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API router for live Kubernetes job and cluster state."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.kubernetes.client import (
    cancel_aiperf_job,
    cluster_version,
    find_aiperf_job,
    get_pods,
    get_raw_aiperfjob_status,
    list_aiperf_jobs,
)

logger = logging.getLogger("aiperf.operator.ui")


class JobListResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs."""

    jobs: list[dict[str, Any]] = Field(description="List of AIPerfJob summaries.")


class JobDetailResponse(AIPerfBaseModel):
    """Response for GET /api/v1/jobs/{namespace}/{name}."""

    job: dict[str, Any] = Field(description="AIPerfJob summary.")
    status: dict[str, Any] = Field(
        description="Raw CR status (phases, conditions, liveMetrics)."
    )
    pods: list[dict[str, Any]] = Field(description="Pod summaries for this job.")


class ClusterResponse(AIPerfBaseModel):
    """Response for GET /api/v1/cluster."""

    nodes: int = Field(description="Number of cluster nodes.")
    gpus: int = Field(description="Total allocatable GPUs.")
    kubernetes_version: str = Field(description="Kubernetes server version.")


class CancelResponse(AIPerfBaseModel):
    """Response for POST /api/v1/jobs/{namespace}/{name}/cancel."""

    cancelled: bool = Field(description="Whether cancellation was requested.")


def create_jobs_router(
    api_holder: list[ApiClient | None] | None = None,
) -> APIRouter:
    """Create the jobs/cluster API router.

    Args:
        api_holder: Mutable single-element list holding the kubernetes_asyncio
            ApiClient. The client is set during app lifespan startup. If the
            list is empty or contains None, endpoints return 503.
    """
    _holder = api_holder if api_holder is not None else [None]
    router = APIRouter(prefix="/api/v1", tags=["jobs"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(503, "Kubernetes API unavailable")
        return api

    @router.get("/jobs", response_model=JobListResponse)
    async def list_jobs() -> JobListResponse:
        """List all AIPerfJob CRs across namespaces."""
        api = _require_api()
        jobs = await list_aiperf_jobs(api, all_namespaces=True)
        return JobListResponse(jobs=[j.model_dump(by_alias=True) for j in jobs])

    @router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
    async def get_job(namespace: str, name: str) -> JobDetailResponse:
        """Get detailed status for a single AIPerfJob."""
        api = _require_api()
        job = await find_aiperf_job(api, name, namespace)
        if not job:
            raise HTTPException(404, f"Job {namespace}/{name} not found")

        raw_status = await get_raw_aiperfjob_status(api, name, namespace)
        pods_raw = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
        pods = [
            {
                "name": (p.metadata.name if p.metadata else "") or "",
                "phase": (p.status.phase if p.status else None) or "Unknown",
                "ready": any(
                    bool(c.ready)
                    for c in ((p.status.container_statuses or []) if p.status else [])
                ),
                "restarts": sum(
                    int(c.restart_count or 0)
                    for c in ((p.status.container_statuses or []) if p.status else [])
                ),
            }
            for p in pods_raw
        ]

        return JobDetailResponse(
            job=job.model_dump(by_alias=True),
            status=raw_status or {},
            pods=pods,
        )

    @router.post("/jobs/{namespace}/{name}/cancel", response_model=CancelResponse)
    async def cancel_job(namespace: str, name: str) -> CancelResponse:
        """Cancel a running AIPerfJob."""
        api = _require_api()
        await cancel_aiperf_job(api, name, namespace)
        return CancelResponse(cancelled=True)

    @router.get("/cluster", response_model=ClusterResponse)
    async def cluster_info() -> ClusterResponse:
        """Get cluster node and GPU information."""
        api = _require_api()
        try:
            version_info = await cluster_version(api)
            k8s_version = version_info.get("gitVersion", "unknown")
        except Exception:
            k8s_version = "unknown"

        try:
            node_list = await client.CoreV1Api(api).list_node()
            nodes = node_list.items
            node_count = len(nodes)
            gpu_count = 0
            for n in nodes:
                alloc = (n.status.allocatable or {}) if n.status else {}
                try:
                    gpu_count += int(alloc.get("nvidia.com/gpu", 0))
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            logger.warning(f"Failed to query nodes: {e}")
            node_count = 0
            gpu_count = 0

        return ClusterResponse(
            nodes=node_count,
            gpus=gpu_count,
            kubernetes_version=k8s_version,
        )

    return router

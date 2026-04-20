# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API router for live Kubernetes job and cluster state."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel

if TYPE_CHECKING:
    from aiperf.kubernetes.client import AIPerfKubeClient

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
    client_holder: list[AIPerfKubeClient | None] | None = None,
) -> APIRouter:
    """Create the jobs/cluster API router.

    Args:
        client_holder: Mutable single-element list holding the kr8s client.
            The client is set during app lifespan startup. If the list is
            empty or contains None, endpoints return 503.
    """
    _holder = client_holder if client_holder is not None else [None]
    router = APIRouter(prefix="/api/v1", tags=["jobs"])

    def _require_client() -> AIPerfKubeClient:
        client = _holder[0] if _holder else None
        if client is None:
            raise HTTPException(503, "Kubernetes API unavailable")
        return client

    @router.get("/jobs", response_model=JobListResponse)
    async def list_jobs() -> JobListResponse:
        """List all AIPerfJob CRs across namespaces."""
        client = _require_client()
        jobs = await client.list_jobs(all_namespaces=True)
        return JobListResponse(jobs=[j.model_dump(by_alias=True) for j in jobs])

    @router.get("/jobs/{namespace}/{name}", response_model=JobDetailResponse)
    async def get_job(namespace: str, name: str) -> JobDetailResponse:
        """Get detailed status for a single AIPerfJob."""
        client = _require_client()
        job = await client.find_job(name, namespace)
        if not job:
            raise HTTPException(404, f"Job {namespace}/{name} not found")

        raw_status = await client.get_raw_status(name, namespace)
        pods_raw = await client.get_pods(namespace, f"aiperf.nvidia.com/job-id={name}")
        pods = [
            {
                "name": p.metadata.get("name", ""),
                "phase": p.status.get("phase", "Unknown"),
                "ready": any(
                    c.get("ready", False) for c in p.status.get("containerStatuses", [])
                ),
                "restarts": sum(
                    c.get("restartCount", 0)
                    for c in p.status.get("containerStatuses", [])
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
        client = _require_client()
        await client.cancel_job(name, namespace)
        return CancelResponse(cancelled=True)

    @router.get("/cluster", response_model=ClusterResponse)
    async def cluster_info() -> ClusterResponse:
        """Get cluster node and GPU information."""
        client = _require_client()
        try:
            version_info = await client.version()
            k8s_version = version_info.get("gitVersion", "unknown")
        except Exception:
            k8s_version = "unknown"

        try:
            # Node is cluster-scoped; do not pass a namespace kwarg.
            nodes = [n async for n in client.api.get("nodes")]
            node_count = len(nodes)
            gpu_count = sum(
                int(
                    n.raw.get("status", {})
                    .get("allocatable", {})
                    .get("nvidia.com/gpu", 0)
                )
                for n in nodes
            )
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

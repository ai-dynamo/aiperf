# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core API router for AIPerf API.

Provides config, run-identity, health, readiness, and shutdown endpoints.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from aiperf.api.api_service import ServiceDep
from aiperf.api.routers.base_router import BaseRouter
from aiperf.common.models.export_models import RunInfo
from aiperf.config.config import BenchmarkConfig

core_router = APIRouter()

# Strong references to in-flight fire-and-forget shutdown tasks.
_SHUTDOWN_TASKS: set[asyncio.Task[None]] = set()


class CoreRouter(BaseRouter):
    """Config, run-identity, health, readiness, and shutdown endpoints."""

    def get_router(self) -> APIRouter:
        return core_router


@core_router.get("/api/config", response_model=BenchmarkConfig, tags=["API"])
async def get_config(svc: ServiceDep) -> dict[str, Any]:
    """Get benchmark configuration."""
    return svc.run.cfg.model_dump(
        mode="json",
        exclude_unset=True,
        exclude_none=True,
        exclude={"endpoint": {"api_key"}},
    )


@core_router.get(
    "/api/run",
    response_model=RunInfo,
    response_model_exclude_none=True,
    tags=["API"],
)
async def get_run(svc: ServiceDep) -> RunInfo:
    """Get run-identity metadata for the currently executing benchmark run.

    Same shape as ``run_info`` in ``profile_export_aiperf.json`` — including
    the matching exclude-None projection, so unset optional fields are omitted
    from the response rather than serialized as ``null``. Includes the redacted
    ``cli_command``, ``benchmark_id``, ``sweep_id``, ``trial``, ``random_seed``,
    and sweep variation coordinates.
    """
    info = RunInfo.from_run(svc.run)
    if info is None:
        raise HTTPException(status_code=503, detail="No active benchmark run.")
    return info


@core_router.get("/healthz", include_in_schema=False)
async def healthz(svc: ServiceDep) -> Response:
    """Kubernetes-style liveness probe."""
    if svc.is_healthy():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="unhealthy")


@core_router.get("/readyz", include_in_schema=False)
async def readyz(svc: ServiceDep) -> Response:
    """Kubernetes-style readiness probe."""
    if svc.is_ready():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="not ready")


@core_router.post("/api/shutdown", tags=["API"])
async def shutdown(svc: ServiceDep) -> dict[str, str]:
    """Trigger graceful shutdown of the API service.

    In Kubernetes mode, the API stays alive after the benchmark completes
    to serve results. This endpoint signals it to shut down, allowing
    the controller pod to exit cleanly.

    Returns 409 if the benchmark is still running.
    """
    results_router = getattr(svc.app.state, "results", None)
    if results_router and not results_router._benchmark_complete:
        raise HTTPException(
            status_code=409,
            detail="Benchmark is still running. Cannot shut down API service.",
        )

    svc.info("Shutdown requested via /api/shutdown")

    async def _delayed_stop() -> None:
        await asyncio.sleep(0.5)
        await svc.stop()

    # asyncio holds only a weak reference to a running task, so a bare
    # create_task here is GC-eligible during its 0.5 s sleep: the endpoint
    # answers "shutting_down" and the pod never stops. Retain until done.
    task = asyncio.create_task(_delayed_stop())
    _SHUTDOWN_TASKS.add(task)
    task.add_done_callback(_SHUTDOWN_TASKS.discard)
    return {"status": "shutting_down"}

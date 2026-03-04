# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core API router for AIPerf API.

Provides config, health, and readiness endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from aiperf.api.api_service import ServiceDep
from aiperf.api.routers.base_router import BaseRouter
from aiperf.common.config import UserConfig

core_router = APIRouter()


class CoreRouter(BaseRouter):
    """Config, health, and readiness endpoints."""

    def get_router(self) -> APIRouter:
        return core_router


@core_router.get("/api/config", response_model=UserConfig, tags=["API"])
async def get_config(svc: ServiceDep) -> dict[str, Any]:
    """Get benchmark configuration."""
    return svc.user_config.model_dump(
        mode="json",
        exclude_unset=True,
        exclude_none=True,
        exclude={"endpoint": {"api_key"}},
    )


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

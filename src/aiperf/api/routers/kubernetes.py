# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes router for AIPerf API.

Provides endpoints for Kubernetes health and readiness probes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from aiperf.api.depends import ServiceDep

kubernetes_router = APIRouter(tags=["Kubernetes"], include_in_schema=False)


@kubernetes_router.get("/healthz")
async def healthz(svc: ServiceDep) -> Response:
    """Kubernetes liveness probe.

    Endpoint contract:
    - ``GET /healthz`` — no query or path parameters.
    - Response: plain-text body (``ok`` or ``unhealthy``).

    Status codes:
    - 200: The service is alive and not deadlocked (``svc.is_healthy()`` is True).
    - 503: The service is in a FAILED state and should be restarted by the kubelet.

    This does not raise ``HTTPException`` — the 503 is returned directly so
    kubelet sees a proper HTTP response rather than an error page.
    """
    if svc.is_healthy():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="unhealthy")


@kubernetes_router.get("/readyz")
async def readyz(svc: ServiceDep) -> Response:
    """Kubernetes readiness probe.

    Endpoint contract:
    - ``GET /readyz`` — no query or path parameters.
    - Response: plain-text body (``ok`` or ``not ready``).

    Status codes:
    - 200: The service is in the ``RUNNING`` state and ready to accept traffic
      (``svc.is_ready()`` is True).
    - 503: The service is still initializing or otherwise not ready. Kubelet
      will withhold traffic via the Service endpoint.

    This does not raise ``HTTPException`` — the 503 is returned directly so
    kubelet sees a proper HTTP response rather than an error page.
    """
    if svc.is_ready():
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="not ready")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reverse proxy from results-server's /dashboard/* to the dashboard sidecar.

The proxy is mounted on the results-server FastAPI app. It forwards
method, path, query, body, and most headers (drops ``host`` and lets
httpx re-set ``content-length``) to ``http://localhost:<PORT>/dashboard/...``
and streams the upstream response back.

When the toggle is off (``AIPERF_DASHBOARD_PROXY_ENABLED`` falsy), the
route returns 503 with a friendly body so the SPA's "Plots ↗" link
fails clearly instead of 404'ing.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Hop-by-hop and otherwise-unsafe headers we don't forward upstream.
_FORWARD_REQUEST_HEADER_DROP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
_FORWARD_RESPONSE_HEADER_DROP = frozenset(
    {"content-encoding", "transfer-encoding", "connection"}
)


def create_dashboard_proxy_router() -> APIRouter:
    """Create the ``/dashboard/{path:path}`` proxy router.

    Reads ``OperatorEnvironment.DASHBOARD`` at request time so a toggle
    flip does not require a reload (env reload is the test concern,
    not prod -- but reading-on-each-request is cheap).
    """
    from aiperf.operator.environment import OperatorEnvironment

    router = APIRouter(tags=["dashboard"])

    @router.api_route(
        "/dashboard/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def proxy(path: str, request: Request) -> Response:
        if not OperatorEnvironment.DASHBOARD.PROXY_ENABLED:
            return Response(
                content=b"Dashboard is disabled on this cluster.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

        port = OperatorEnvironment.DASHBOARD.PORT
        if port <= 0:
            return Response(
                content=b"Dashboard is disabled on this cluster.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

        upstream_url = f"http://localhost:{port}/dashboard/{path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _FORWARD_REQUEST_HEADER_DROP
        }
        body = await request.body()

        try:
            client = httpx.AsyncClient(timeout=30.0)
            try:
                async with client.stream(
                    request.method,
                    upstream_url,
                    headers=forward_headers,
                    content=body,
                ) as upstream:
                    response_headers = {
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in _FORWARD_RESPONSE_HEADER_DROP
                    }
                    content = b""
                    async for chunk in upstream.aiter_raw():
                        content += chunk
                    return Response(
                        content=content,
                        status_code=upstream.status_code,
                        headers=response_headers,
                    )
            finally:
                await client.aclose()
        except httpx.HTTPError as exc:
            logger.warning("dashboard upstream unreachable: %s", exc)
            return Response(
                content=b"Dashboard sidecar is unreachable.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

    return router

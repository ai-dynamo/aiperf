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
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Hop-by-hop and otherwise-unsafe headers we don't forward upstream.
_FORWARD_REQUEST_HEADER_DROP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
_FORWARD_RESPONSE_HEADER_DROP = frozenset({"transfer-encoding", "connection"})


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

        client = httpx.AsyncClient(timeout=30.0)
        try:
            stream_ctx = client.stream(
                request.method,
                upstream_url,
                headers=forward_headers,
                content=body,
            )
            upstream = await stream_ctx.__aenter__()
        except httpx.HTTPError as exc:
            await client.aclose()
            logger.warning("dashboard upstream unreachable: %s", exc)
            return Response(
                content=b"Dashboard sidecar is unreachable.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _FORWARD_RESPONSE_HEADER_DROP
        }

        async def _iter_upstream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await client.aclose()

        return StreamingResponse(
            _iter_upstream(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    return router

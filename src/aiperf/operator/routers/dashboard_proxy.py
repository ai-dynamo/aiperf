# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reverse proxy from results-server's /dashboard/* to the dashboard sidecar.

The proxy is mounted on the results-server FastAPI app. It forwards
method, path, query, body, and most headers (drops ``host`` and lets
aiohttp re-set ``content-length``) to ``http://localhost:<PORT>/dashboard/...``
and streams the upstream response back.

When the toggle is off (``AIPERF_DASHBOARD_PROXY_ENABLED`` falsy), the
route returns 503 with a friendly body so the SPA's "Plots ↗" link
fails clearly instead of 404'ing.
"""

from __future__ import annotations

import logging

import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Hop-by-hop and otherwise-unsafe headers we don't forward upstream.
_FORWARD_REQUEST_HEADER_DROP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
# ``content-length`` must be dropped alongside the hop-by-hop headers: aiohttp
# transparently decodes a compressed upstream body, so the upstream length no
# longer describes the bytes we forward. Leaving it in truncates or hangs the
# response for every gzip-encoded dashboard asset; Starlette re-derives the
# correct framing for the body it actually writes.
_FORWARD_RESPONSE_HEADER_DROP = frozenset(
    {"transfer-encoding", "connection", "content-length", "content-encoding"}
)


def create_dashboard_proxy_router() -> APIRouter:
    """Create the ``/dashboard/{path:path}`` proxy router.

    Reads ``OperatorEnvironment.DASHBOARD`` at request time so a toggle
    flip does not require a reload (env reload is the test concern,
    not prod -- but reading-on-each-request is cheap).
    """
    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.transports.aiohttp_client import create_tcp_connector
    from aiperf.transports.http_defaults import AioHttpDefaults

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

        # auto_decompress=False keeps this a byte-for-byte pass-through:
        # ``content-encoding`` is forwarded to the client below, so the body
        # must stay in its original encoding. aiohttp would otherwise inflate
        # it while the header still advertised gzip.
        session = aiohttp.ClientSession(
            connector=create_tcp_connector(),
            timeout=aiohttp.ClientTimeout(total=30.0),
            trust_env=AioHttpDefaults.TRUST_ENV,
            auto_decompress=False,
        )
        try:
            stream_ctx = session.request(
                request.method,
                upstream_url,
                headers=forward_headers,
                data=body,
            )
            upstream = await stream_ctx.__aenter__()
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            await session.close()
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
                async for chunk in upstream.content.iter_any():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await session.close()

        return StreamingResponse(
            _iter_upstream(),
            status_code=upstream.status,
            headers=response_headers,
        )

    return router

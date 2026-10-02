# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reverse proxy from results-server's /dashboard/* to the dashboard sidecar.

The proxy is mounted on the results-server FastAPI app. It forwards
method, path, query, body, and most headers (drops ``host`` and lets
aiohttp re-set ``content-length``) to ``http://localhost:<PORT>/dashboard/...``
and streams the upstream response back. Request bodies are bounded by
``AIPERF_DASHBOARD_PROXY_MAX_BODY_BYTES`` and rejected with 413 above it.

When the toggle is off (``AIPERF_DASHBOARD_PROXY_ENABLED`` falsy), the
route returns 503 with a friendly body so the SPA's "Plots ↗" link
fails clearly instead of 404'ing.
"""

from __future__ import annotations

import logging
from urllib.parse import unquote

import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Hop-by-hop and otherwise-unsafe headers we don't forward upstream.
# ``accept-encoding`` is deliberately NOT dropped: the client's negotiation is
# passed through so the sidecar can compress large dashboard JSON, and the
# resulting ``content-encoding`` rides back to the client untouched.
_FORWARD_REQUEST_HEADER_DROP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
# ``content-encoding`` is NOT dropped: with ``auto_decompress=False`` below the
# body we relay is still in the upstream's encoding, so stripping the header
# would hand the client compressed bytes labelled as identity. ``content-length``
# is dropped because Starlette re-derives framing for the chunks it actually
# writes, and a stale upstream length truncates or hangs the response.
_FORWARD_RESPONSE_HEADER_DROP = frozenset(
    {"transfer-encoding", "connection", "content-length"}
)


def _decode_to_fixed_point(path: str) -> str:
    """Percent-decode ``path`` until stable.

    Starlette decodes the request path once before handing it to route handlers,
    so a single-encoded ``%2e%2e`` arrives as ``..`` and the dot-segment guard
    catches it. A double-encoded ``%252e%252e`` arrives as ``%2e%2e`` (one decode
    happened), which the guard misses because ``%2e%2e != ".."``; but yarl (used
    by aiohttp below) then decodes ``%2e%2e`` to ``..`` and normalizes the path,
    escaping the ``/dashboard/`` prefix altogether.

    Decoding to a fixed point before the guard closes this bypass.
    """
    while True:
        decoded = unquote(path)
        if decoded == path:
            return path
        path = decoded


def _has_dot_segment(path: str) -> bool:
    """Return True if ``path`` contains a ``.`` or ``..`` segment.

    The caller must first call :func:`_decode_to_fixed_point` so that any
    remaining percent-encoded dots (`%2e`, `%252e`, etc.) are resolved before
    the segment check runs.  See its docstring for the double-encoding bypass
    that makes this necessary.
    """
    return any(segment in (".", "..") for segment in path.replace("\\", "/").split("/"))


async def _read_bounded_body(request: Request, limit: int) -> bytes | None:
    """Buffer the request body, returning ``None`` once ``limit`` bytes are exceeded.

    A ``Content-Length`` precheck alone cannot bound the read: a chunked request
    carries no length header at all, and a declared length is client-supplied
    anyway. So the declared value only short-circuits the obvious case, and the
    bytes actually pulled off the wire are counted as they arrive.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                return None
        except ValueError:
            return None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


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

        path = _decode_to_fixed_point(path)
        if _has_dot_segment(path):
            return Response(
                content=b"Invalid dashboard path.",
                status_code=400,
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
        max_body_bytes = OperatorEnvironment.DASHBOARD.PROXY_MAX_BODY_BYTES
        body = await _read_bounded_body(request, max_body_bytes)
        if body is None:
            return Response(
                content=(
                    f"Dashboard request body exceeds {max_body_bytes} bytes."
                ).encode(),
                status_code=413,
                media_type="text/plain; charset=utf-8",
            )

        # auto_decompress=False keeps this a byte-for-byte pass-through, which
        # is only correct because ``content-encoding`` is forwarded to the
        # client below. aiohttp would otherwise inflate the body while the
        # header still advertised gzip. Starlette's GZipMiddleware on the
        # results-server app skips any response that already carries a
        # ``content-encoding``, so nothing re-compresses these bytes.
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

        response_headers = [
            (key, value)
            for key, value in upstream.raw_headers
            if key.decode("latin-1").lower() not in _FORWARD_RESPONSE_HEADER_DROP
        ]

        async def _iter_upstream():
            try:
                async for chunk in upstream.content.iter_any():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await session.close()

        response = StreamingResponse(
            _iter_upstream(),
            status_code=upstream.status,
        )
        response.raw_headers.extend(response_headers)
        return response

    return router

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebSocket proxy from the operator UI to per-job controller pods.

Browsers viewing ``/v1/job/{ns}/{name}`` cannot reach the controller pod's
``:API_SERVICE/ws`` endpoint directly — controller services are headless
and intra-cluster. The operator already serves the UI and already knows
the routing (it watches the AIPerfJob CR for ``status.jobSetName``), so it
hosts a thin pass-through endpoint:

    WS /api/v1/jobs/{namespace}/{name}/ws

On connect: the proxy reads the CR's ``status.jobSetName``, derives the
controller pod's headless-service DNS, opens an upstream WebSocket via
``aiohttp``, and pumps frames in both directions until either side closes.

The proxy is intentionally transparent — it does not subscribe on behalf
of the client. The browser owns the subscribe protocol the controller
exposes (``{"type": "subscribe", "message_types": [...]}``); see
``src/aiperf/api/static-v2/lib/ws.js`` for the canonical client behavior.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

import aiohttp
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from aiperf.kubernetes.client_jobs import get_raw_aiperfjob_status
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.transports.aiohttp_client import create_tcp_connector

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

__all__ = ["create_jobs_ws_router"]

logger = logging.getLogger(__name__)

# WS close codes (RFC 6455). 4xxx range is application-defined private use.
_CLOSE_NO_API = 4503
_CLOSE_NO_JOBSET = 4404
_CLOSE_UPSTREAM_FAILED = 4502
_CLOSE_NORMAL = 1000

_WS_HEARTBEAT_SEC = K8sEnvironment.PROGRESS_STREAM.WS_HEARTBEAT_SECONDS
_CLOSE_MAX_REASON_BYTES = 123


def _truncate_close_reason(reason: str) -> str:
    """Clamp a WebSocket close reason to the RFC 6455 UTF-8 byte limit."""
    encoded = reason.encode()
    if len(encoded) <= _CLOSE_MAX_REASON_BYTES:
        return reason
    return encoded[:_CLOSE_MAX_REASON_BYTES].decode(errors="ignore")


async def _close_ws(websocket: WebSocket, *, code: int, reason: str) -> None:
    """Close a WebSocket with a protocol-safe reason payload."""
    await websocket.close(code=code, reason=_truncate_close_reason(reason))


async def _pump_upstream_to_client(
    upstream: aiohttp.ClientWebSocketResponse,
    client: WebSocket,
) -> None:
    """Forward frames from the controller WS to the browser. Returns on EOF."""
    async for msg in upstream:
        if msg.type == aiohttp.WSMsgType.TEXT:
            await client.send_text(msg.data)
        elif msg.type == aiohttp.WSMsgType.BINARY:
            await client.send_bytes(msg.data)
        elif msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        ):
            return


async def _pump_client_to_upstream(
    client: WebSocket,
    upstream: aiohttp.ClientWebSocketResponse,
) -> None:
    """Forward frames from the browser to the controller. Returns on disconnect."""
    while True:
        try:
            msg = await client.receive()
        except WebSocketDisconnect:
            return
        msg_type = msg.get("type")
        if msg_type == "websocket.disconnect":
            return
        if msg_type != "websocket.receive":
            continue
        if (text := msg.get("text")) is not None:
            await upstream.send_str(text)
        elif (data := msg.get("bytes")) is not None:
            await upstream.send_bytes(data)


async def _resolve_controller_ws_url(
    api: ApiClient, namespace: str, name: str
) -> str | None:
    """Look up the AIPerfJob CR and return its controller's WS URL, or None.

    Returns None when the CR has no ``status.jobSetName`` yet (operator
    hasn't created the JobSet), in which case the caller should refuse the
    WebSocket with :data:`_CLOSE_NO_JOBSET`.
    """
    status = await get_raw_aiperfjob_status(api, name, namespace)
    jobset_name = (status or {}).get("jobSetName")
    if not jobset_name:
        return None
    host = controller_dns_name(jobset_name, namespace)
    port = K8sEnvironment.PORTS.API_SERVICE
    return f"ws://{host}:{port}/ws"


async def _bridge_frames(
    upstream: aiohttp.ClientWebSocketResponse,
    websocket: WebSocket,
    log_tag: str,
) -> None:
    """Pump frames in both directions until either side closes."""
    up_to_client = asyncio.create_task(_pump_upstream_to_client(upstream, websocket))
    client_to_up = asyncio.create_task(_pump_client_to_upstream(websocket, upstream))
    done, pending = await asyncio.wait(
        {up_to_client, client_to_up},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(
            asyncio.CancelledError,
            aiohttp.ClientError,
            WebSocketDisconnect,
            OSError,
        ):
            await task
    for task in done:
        if (exc := task.exception()) is not None:
            logger.info(f"WS pump for {log_tag} ended with {exc!r}")


async def _proxy_to_controller(websocket: WebSocket, ws_url: str, log_tag: str) -> None:
    """Open the upstream WS to ``ws_url`` and bridge frames to ``websocket``.

    Closes the client WS with :data:`_CLOSE_UPSTREAM_FAILED` if the upstream
    cannot be reached; otherwise closes normally on either-side EOF.
    """
    connector = create_tcp_connector()
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            upstream_cm = session.ws_connect(ws_url, heartbeat=_WS_HEARTBEAT_SEC)
            upstream = await upstream_cm.__aenter__()
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"Upstream WS connect failed for {log_tag}: {e}")
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await _close_ws(
                    websocket,
                    code=_CLOSE_UPSTREAM_FAILED,
                    reason=f"Cannot reach controller: {e}",
                )
            return
        try:
            await _bridge_frames(upstream, websocket, log_tag)
        finally:
            await upstream_cm.__aexit__(None, None, None)


def create_jobs_ws_router(
    api_holder: list[ApiClient | None] | None = None,
) -> APIRouter:
    """Build the per-job WebSocket proxy router.

    Args:
        api_holder: Mutable single-element list holding the kubernetes_asyncio
            ApiClient (set during FastAPI lifespan startup). The endpoint
            closes the WS with code 4503 if the holder is empty / None.
    """
    _holder = api_holder if api_holder is not None else [None]
    router = APIRouter(prefix="/api/v1", tags=["jobs-ws"])

    @router.websocket("/jobs/{namespace}/{name}/ws")
    async def proxy_job_ws(websocket: WebSocket, namespace: str, name: str) -> None:
        await websocket.accept()
        api = _holder[0] if _holder else None
        if api is None:
            await _close_ws(
                websocket,
                code=_CLOSE_NO_API,
                reason="Kubernetes API client not yet initialized",
            )
            return

        ws_url = await _resolve_controller_ws_url(api, namespace, name)
        if ws_url is None:
            await _close_ws(
                websocket,
                code=_CLOSE_NO_JOBSET,
                reason=f"Job {namespace}/{name} has no JobSet yet",
            )
            return

        log_tag = f"{namespace}/{name}"
        logger.debug(f"Proxying WS for {log_tag} -> {ws_url}")
        try:
            await _proxy_to_controller(websocket, ws_url, log_tag)
        finally:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                with contextlib.suppress(RuntimeError, aiohttp.ClientError):
                    await websocket.close(code=_CLOSE_NORMAL)

    return router

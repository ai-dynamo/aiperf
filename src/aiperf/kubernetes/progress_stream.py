# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebSocket progress streaming for the controller API."""

import asyncio
from collections.abc import Awaitable, Callable

import aiohttp
import orjson

from aiperf.kubernetes.console import print_info

# WebSocket reconnection parameters (exponential backoff)
_WS_INITIAL_BACKOFF = 1.0  # seconds
_WS_MAX_BACKOFF = 30.0  # seconds
_WS_HEARTBEAT = 30  # seconds between WebSocket heartbeats


async def _consume_ws_messages(
    ws: aiohttp.ClientWebSocketResponse,
    on_message: Callable[[dict], Awaitable[bool]],
) -> bool:
    """Iterate a WebSocket's frames, returning True if streaming should end.

    Returns:
        True on graceful completion (``on_message`` requested stop, or server
        closed the connection). False on transport error signalling a reconnect.
    """
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = orjson.loads(msg.data)
            should_stop = await on_message(data)
            if should_stop:
                return True
        elif msg.type == aiohttp.WSMsgType.ERROR:
            return False  # Reconnect
        elif msg.type == aiohttp.WSMsgType.CLOSED:
            return True  # Server closed
    return False


async def _stream_once(
    ws_url: str,
    message_types: list[str],
    on_message: Callable[[dict], Awaitable[bool]],
) -> bool:
    """Open one WebSocket, subscribe, and pump messages until done or error.

    Returns:
        True on graceful completion, False if a reconnect is required.
    """
    from aiperf.transports.aiohttp_client import create_tcp_connector

    connector = create_tcp_connector()
    async with (
        aiohttp.ClientSession(connector=connector) as session,
        session.ws_connect(ws_url, heartbeat=_WS_HEARTBEAT) as ws,
    ):
        await ws.send_json({"type": "subscribe", "message_types": message_types})
        await ws.receive_json()  # subscription ack
        return await _consume_ws_messages(ws, on_message)


async def stream_progress_from_api(
    ws_url: str,
    on_message: Callable[[dict], Awaitable[bool]],
    message_types: list[str],
    max_retries: int = 10,
) -> None:
    """Stream progress messages from the controller API WebSocket with auto-reconnection.

    Opens a WebSocket to ``ws_url``, sends a ``{"type": "subscribe",
    "message_types": [...]}`` frame, then invokes ``on_message`` for every
    received text frame. On network error or timeout the connection is
    retried with exponential backoff up to ``max_retries`` attempts.

    The ``on_message`` callback receives a decoded message dict. Its return
    value is load-bearing: returning ``True`` stops streaming (graceful
    completion); returning ``False`` continues receiving.

    Message dicts carry a ``"type"`` field drawn from the controller's
    progress protocol. Common values include:
      - ``"subscribed"`` - initial subscription ack.
      - ``"realtime_metrics"`` - periodic metric snapshots.
      - ``"progress"`` - credit/phase progress updates.
      - ``"benchmark_complete"`` - terminal marker; callers typically stop here.
      - ``"error"`` - controller-reported error payload.

    Args:
        ws_url: WebSocket URL, e.g. ``"ws://localhost:9090/ws"``.
        on_message: Async callback invoked per received message dict.
            Return ``True`` to stop streaming, ``False`` to continue.
        message_types: List of message-type strings to subscribe to.
        max_retries: Maximum reconnection attempts on transport errors.

    Raises:
        ConnectionError: Transport failed after ``max_retries`` attempts.
            The original ``aiohttp.ClientError`` / ``asyncio.TimeoutError``
            is preserved as ``__cause__``.
    """
    retry_count = 0
    backoff = _WS_INITIAL_BACKOFF

    while retry_count < max_retries:
        try:
            if await _stream_once(ws_url, message_types, on_message):
                return
            # _stream_once returned False: server closed without a stop signal
            # (WS ERROR frame or connection dropped cleanly). Treat as a
            # reconnect-eligible transport failure — without this, the while
            # loop never increments retry_count and spins forever, which under
            # test mocks with empty frame lists becomes an unbounded memory
            # leak (the outer asyncio.sleep backoff also never fires).
            retry_count += 1
            if retry_count >= max_retries:
                raise ConnectionError(
                    f"WebSocket stream ended without completion after "
                    f"{max_retries} attempts."
                )
            print_info(
                f"Stream ended without completion, retrying in {backoff:.1f}s... "
                f"({retry_count}/{max_retries})"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _WS_MAX_BACKOFF)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            retry_count += 1
            if retry_count >= max_retries:
                msg = (
                    f"Failed to connect to API after {max_retries} attempts. "
                    "The controller pod may not be running or "
                    "API service may be unavailable."
                )
                raise ConnectionError(msg) from exc

            print_info(
                f"Connection lost, retrying in {backoff:.1f}s... "
                f"({retry_count}/{max_retries})"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _WS_MAX_BACKOFF)

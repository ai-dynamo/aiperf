# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket router component -- owns WebSocket connections and ZMQ message forwarding.

Flow:
- Client connects to WS /ws
- Client sends {"type": "subscribe", "message_types": ["realtime_metrics", "worker_status_summary"]}
- Server responds with {"type": "subscribed", "message_types": ["realtime_metrics", "worker_status_summary"]}
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from typing import Annotated, Any

import orjson
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.messages import Message
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.zmq.zmq_defaults import WILDCARD_TOPIC

WebSocketDep = Annotated["WebSocketRouter", component_dependency("websocket")]

ws_router = APIRouter(tags=["WebSocket"])


class WebSocketRouter(MessageBusClientMixin, BaseRouter):
    """Owns WebSocket connections and forwards ZMQ messages to connected clients."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ws_manager = WebSocketManager()

    def get_router(self) -> APIRouter:
        return ws_router

    @on_init
    async def _subscribe_to_all_message_types(self) -> None:
        await self.subscribe(WILDCARD_TOPIC, self._forward_message)

    async def _forward_message(self, message: Message) -> None:
        sent = await self.ws_manager.broadcast(message)
        self.debug(lambda: f"Forwarded {message.message_type} to {sent} clients")

    @on_stop
    async def _close_all_connections(self) -> None:
        await self.ws_manager.close_all()


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, component: WebSocketDep) -> None:
    """WebSocket endpoint for real-time message streaming."""
    if component.ws_manager.client_count >= component.ws_manager.max_connections:
        await websocket.close(code=1013, reason="Max connections reached")
        return

    await websocket.accept()
    client_host = (
        getattr(websocket.client, "host", "unknown") if websocket.client else "unknown"
    )
    client_id = f"{client_host}:{uuid.uuid4().hex[:8]}"
    component.ws_manager.add(client_id, websocket)

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = orjson.loads(raw_text)
            except orjson.JSONDecodeError as e:
                err = orjson.dumps({"type": "error", "message": f"Invalid JSON: {e}"})
                await websocket.send_text(err.decode())
                continue

            if not isinstance(data, dict):
                err = orjson.dumps(
                    {"type": "error", "message": "Payload must be a JSON object"}
                )
                await websocket.send_text(err.decode())
                continue

            msg_type = data.get("type")

            if msg_type == "subscribe":
                types = data.get("message_types", [])
                if not isinstance(types, list) or not all(
                    isinstance(t, str) for t in types
                ):
                    err = orjson.dumps(
                        {
                            "type": "error",
                            "message": "message_types must be a list of strings",
                        }
                    )
                    await websocket.send_text(err.decode())
                    continue
                component.ws_manager.subscribe(client_id, types)
                resp = orjson.dumps({"type": "subscribed", "message_types": types})
                await websocket.send_text(resp.decode())
                component.info(f"WebSocket: Client {client_id} subscribed to: {types}")

            elif msg_type == "unsubscribe":
                types = data.get("message_types", [])
                if not isinstance(types, list) or not all(
                    isinstance(t, str) for t in types
                ):
                    err = orjson.dumps(
                        {
                            "type": "error",
                            "message": "message_types must be a list of strings",
                        }
                    )
                    await websocket.send_text(err.decode())
                    continue
                component.ws_manager.unsubscribe(client_id, types)
                resp = orjson.dumps({"type": "unsubscribed", "message_types": types})
                await websocket.send_text(resp.decode())
                component.info(
                    f"WebSocket: Client {client_id} unsubscribed from: {types}"
                )

            elif msg_type == "ping":
                await websocket.send_text('{"type":"pong"}')

    except WebSocketDisconnect:
        component.info(f"WebSocket: Client {client_id} disconnected")
    except Exception as e:  # noqa: BLE001 - unexpected WS errors are logged; connection cleanup still runs in finally
        component.exception(f"WebSocket error for {client_id}: {e}")
    finally:
        component.ws_manager.remove(client_id)


class WebSocketManager(AIPerfLoggerMixin):
    """Manages WebSocket client connections and subscriptions.

    Uses copy-on-write snapshots to prevent mutation during iteration.
    """

    MAX_CONNECTIONS = 100

    def __init__(self, max_connections: int = MAX_CONNECTIONS) -> None:
        super().__init__()
        self.max_connections = max_connections
        self._clients: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}
        self._snapshot: tuple[tuple[str, WebSocket], ...] = ()

    def _update_snapshot(self) -> None:
        self._snapshot = tuple(self._clients.items())

    @property
    def client_count(self) -> int:
        return len(self._snapshot)

    def add(self, client_id: str, ws: WebSocket) -> None:
        self._clients[client_id] = ws
        self._subscriptions[client_id] = set()
        self._update_snapshot()
        self.info(f"Client connected: {client_id} (total: {self.client_count})")

    def remove(self, client_id: str) -> set[str]:
        if self._clients.pop(client_id, None) is not None:
            self._update_snapshot()
        subs = self._subscriptions.pop(client_id, set())
        self.info(f"Client disconnected: {client_id} (total: {self.client_count})")
        return subs

    def subscribe(self, client_id: str, message_types: list[str]) -> None:
        if client_id in self._subscriptions:
            self._subscriptions[client_id].update(message_types)

    def unsubscribe(self, client_id: str, message_types: list[str]) -> None:
        if client_id in self._subscriptions:
            self._subscriptions[client_id] -= set(message_types)

    def all_subscriptions(self) -> set[str]:
        if not self._subscriptions:
            return set()
        return set().union(*self._subscriptions.values())

    async def _send_text(
        self, client_id: str, ws: WebSocket, json_str: str
    ) -> str | None:
        """Send text to a WebSocket client. Returns client_id on failure, None on success."""
        try:
            await ws.send_text(json_str)
            return None
        except Exception as e:  # noqa: BLE001 - Starlette/websockets can surface many disconnect error types; any failure means drop this client
            self.warning(f"Send failed for {client_id}: {e}")
            return client_id

    async def broadcast(self, message: Message) -> int:
        """Broadcast to clients subscribed to a message type."""
        snapshot = self._snapshot
        if not snapshot:
            return 0

        targets: list[tuple[str, WebSocket]] = []
        for client_id, ws in snapshot:
            subs = self._subscriptions.get(client_id, set())
            if WILDCARD_TOPIC in subs or message.message_type in subs:
                targets.append((client_id, ws))

        if not targets:
            return 0

        json_str = message.model_dump_json(exclude_none=True)

        results = await asyncio.gather(
            *(self._send_text(cid, ws, json_str) for cid, ws in targets),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, str):
                self.remove(result)

        return sum(1 for r in results if r is None)

    async def close_all(self) -> None:
        """Close all connections gracefully using snapshot."""
        snapshot = self._snapshot
        if not snapshot:
            return
        self._snapshot = ()

        self.info(f"Closing {len(snapshot)} connections")

        async def close_one(ws: WebSocket) -> None:
            if ws.client_state == WebSocketState.CONNECTED:
                with suppress(Exception):
                    await ws.close()

        await asyncio.gather(
            *(close_one(ws) for _, ws in snapshot),
            return_exceptions=True,
        )
        self._clients.clear()
        self._subscriptions.clear()

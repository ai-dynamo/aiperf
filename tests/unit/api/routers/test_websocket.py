# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the WebSocket router and WebSocketManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param
from starlette.testclient import TestClient
from starlette.websockets import WebSocketState

from aiperf.api.api_service import FastAPIService
from aiperf.api.routers.websocket import WebSocketManager, WebSocketRouter
from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.messages import HeartbeatMessage, RealtimeMetricsMessage


def make_mock_websocket(
    closed: bool = False,
    send_side_effect: Exception | None = None,
) -> AsyncMock:
    """Create a mock WebSocket with configurable behavior."""
    ws = AsyncMock()
    ws.closed = closed
    if send_side_effect:
        ws.send_text.side_effect = send_side_effect
        ws.send_str.side_effect = send_side_effect
    return ws


class TestWebSocketManager:
    """Test WebSocketManager functionality."""

    def test_add_and_remove_client(self) -> None:
        """Test adding and removing clients."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        assert manager.client_count == 1

        removed_subs = manager.remove("client-1")
        assert manager.client_count == 0
        assert removed_subs == set()

    def test_remove_returns_subscriptions(self) -> None:
        """Test that remove returns the client's subscriptions."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["topic_a", "topic_b"])

        removed_subs = manager.remove("client-1")
        assert removed_subs == {"topic_a", "topic_b"}

    def test_remove_nonexistent_client(self) -> None:
        """Test removing a client that doesn't exist."""
        manager = WebSocketManager()
        removed_subs = manager.remove("nonexistent")
        assert removed_subs == set()

    def test_subscribe_and_unsubscribe(self) -> None:
        """Test subscription management."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["topic_a", "topic_b"])

        all_subs = manager.all_subscriptions()
        assert "topic_a" in all_subs
        assert "topic_b" in all_subs

        manager.unsubscribe("client-1", ["topic_a"])
        all_subs = manager.all_subscriptions()
        assert "topic_a" not in all_subs
        assert "topic_b" in all_subs

    def test_subscribe_nonexistent_client(self) -> None:
        """Test subscribing for a nonexistent client does nothing."""
        manager = WebSocketManager()
        manager.subscribe("nonexistent", ["topic"])
        assert manager.all_subscriptions() == set()

    def test_unsubscribe_nonexistent_client(self) -> None:
        """Test unsubscribing for a nonexistent client does nothing."""
        manager = WebSocketManager()
        manager.unsubscribe("nonexistent", ["topic"])

    def test_all_subscriptions_empty(self) -> None:
        """Test all_subscriptions with no clients."""
        manager = WebSocketManager()
        assert manager.all_subscriptions() == set()

    def test_all_subscriptions_multiple_clients(self) -> None:
        """Test all_subscriptions aggregates across clients."""
        manager = WebSocketManager()
        ws1, ws2 = MagicMock(), MagicMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", ["topic_a"])
        manager.subscribe("client-2", ["topic_b", "topic_c"])

        all_subs = manager.all_subscriptions()
        assert all_subs == {"topic_a", "topic_b", "topic_c"}

    def test_all_subscriptions_with_overlap(self) -> None:
        """Test all_subscriptions deduplicates overlapping subscriptions."""
        manager = WebSocketManager()
        ws1, ws2 = MagicMock(), MagicMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", ["topic_a", "topic_b"])
        manager.subscribe("client-2", ["topic_b", "topic_c"])

        all_subs = manager.all_subscriptions()
        assert all_subs == {"topic_a", "topic_b", "topic_c"}

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribed_clients(self) -> None:
        """Test broadcasting to subscribed clients."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", [MessageType.REALTIME_METRICS])
        manager.subscribe("client-2", ["other"])

        msg = RealtimeMetricsMessage(service_id="test", metrics=[])
        sent = await manager.broadcast(msg)
        assert sent == 1
        ws1.send_text.assert_called_once()
        ws2.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_wildcard(self) -> None:
        """Test wildcard subscription receives all messages."""
        manager = WebSocketManager()
        ws = AsyncMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["*"])

        msg = HeartbeatMessage(
            service_id="test", service_type="worker", state=LifecycleState.RUNNING
        )
        sent = await manager.broadcast(msg)
        assert sent == 1
        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_empty_snapshot(self) -> None:
        """Test broadcast with no clients returns 0."""
        manager = WebSocketManager()
        msg = HeartbeatMessage(
            service_id="test", service_type="worker", state=LifecycleState.RUNNING
        )
        sent = await manager.broadcast(msg)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self) -> None:
        """Test broadcast when no clients are subscribed to the topic."""
        manager = WebSocketManager()
        ws = AsyncMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["other_topic"])

        msg = HeartbeatMessage(
            service_id="test", service_type="worker", state=LifecycleState.RUNNING
        )
        sent = await manager.broadcast(msg)
        assert sent == 0
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_client(self) -> None:
        """Test that broadcast removes clients that fail to send."""
        manager = WebSocketManager()
        ws = make_mock_websocket(send_side_effect=Exception("Connection lost"))

        manager.add("client-1", ws)
        manager.subscribe("client-1", [MessageType.HEARTBEAT])

        msg = HeartbeatMessage(
            service_id="test", service_type="worker", state=LifecycleState.RUNNING
        )
        sent = await manager.broadcast(msg)
        assert sent == 0
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        """Test closing all WebSocket connections."""
        manager = WebSocketManager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        ws1.client_state = WebSocketState.CONNECTED
        ws2.client_state = WebSocketState.CONNECTED

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)

        await manager.close_all()

        ws1.close.assert_called_once()
        ws2.close.assert_called_once()
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_close_all_empty(self) -> None:
        """Test closing all with no clients does nothing."""
        manager = WebSocketManager()
        await manager.close_all()

    @pytest.mark.asyncio
    async def test_close_all_already_disconnected(self) -> None:
        """Test closing connections that are already disconnected."""
        manager = WebSocketManager()
        ws = AsyncMock()
        ws.client_state = WebSocketState.DISCONNECTED

        manager.add("client-1", ws)
        await manager.close_all()

        ws.close.assert_not_called()
        assert manager.client_count == 0


class TestWebSocketManagerClientCount:
    """Test client_count tracking through add/remove operations."""

    def test_client_count_tracks_additions(self) -> None:
        """Test that client_count increases with each add."""
        manager = WebSocketManager()
        for i in range(3):
            manager.add(f"client-{i}", MagicMock())
        assert manager.client_count == 3

    def test_client_count_tracks_removals(self) -> None:
        """Test that client_count decreases with each remove."""
        manager = WebSocketManager()
        for i in range(3):
            manager.add(f"client-{i}", MagicMock())

        manager.remove("client-1")
        assert manager.client_count == 2


class TestWebSocketAdditiveSubscriptions:
    """Test that multiple subscribe calls are additive."""

    def test_subscribe_is_additive(self) -> None:
        """Test that calling subscribe twice merges topics."""
        manager = WebSocketManager()
        manager.add("c1", MagicMock())

        manager.subscribe("c1", ["topic_a"])
        manager.subscribe("c1", ["topic_b"])

        assert manager.all_subscriptions() == {"topic_a", "topic_b"}

    def test_subscribe_deduplicates_topics(self) -> None:
        """Test that subscribing to the same topic twice doesn't duplicate."""
        manager = WebSocketManager()
        manager.add("c1", MagicMock())

        manager.subscribe("c1", ["topic_a", "topic_a"])
        assert manager.all_subscriptions() == {"topic_a"}


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    def test_websocket_subscribe(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket subscribe message."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "subscribe", "message_types": ["realtime_metrics"]}
            )
            response = websocket.receive_json()
            assert response["type"] == "subscribed"
            assert "realtime_metrics" in response["message_types"]

    def test_websocket_subscribe_multiple_types(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket subscribe to multiple message types."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "subscribe",
                    "message_types": ["realtime_metrics", "worker_status"],
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "subscribed"
            assert set(response["message_types"]) == {
                "realtime_metrics",
                "worker_status",
            }

    def test_websocket_unsubscribe(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket unsubscribe message."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "subscribe", "message_types": ["realtime_metrics"]}
            )
            websocket.receive_json()

            websocket.send_json(
                {"type": "unsubscribe", "message_types": ["realtime_metrics"]}
            )
            response = websocket.receive_json()
            assert response["type"] == "unsubscribed"
            assert "realtime_metrics" in response["message_types"]

    def test_websocket_ping_pong(self, api_test_client: TestClient) -> None:
        """Test WebSocket ping/pong."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_invalid_json(self, api_test_client: TestClient) -> None:
        """Test WebSocket handles invalid JSON gracefully."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_text("not valid json")
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["message"]

    @pytest.mark.parametrize(
        "invalid_text",
        [
            param("not valid json", id="plain-text"),
            param("{incomplete", id="incomplete-json"),
            param("{'single': 'quotes'}", id="single-quotes"),
        ],
    )  # fmt: skip
    def test_websocket_various_invalid_json(
        self, api_test_client: TestClient, invalid_text: str
    ) -> None:
        """Test WebSocket handles various forms of invalid JSON."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_text(invalid_text)
            response = websocket.receive_json()
            assert response["type"] == "error"


class TestWebSocketPayloadValidation:
    """Test WebSocket payload validation."""

    @pytest.mark.parametrize(
        "payload",
        [
            param("[]", id="json-array"),
            param('"a string"', id="json-string"),
            param("42", id="json-number"),
            param("true", id="json-boolean"),
            param("null", id="json-null"),
        ],
    )  # fmt: skip
    def test_non_object_payload_returns_error(
        self, api_test_client: TestClient, payload: str
    ) -> None:
        """Test that non-object JSON payloads return an error without disconnecting."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_text(payload)
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "JSON object" in response["message"]
            websocket.send_json({"type": "ping"})
            pong = websocket.receive_json()
            assert pong["type"] == "pong"

    @pytest.mark.parametrize(
        "msg_type",
        [param("subscribe", id="subscribe"), param("unsubscribe", id="unsubscribe")],
    )  # fmt: skip
    def test_string_message_types_returns_error(
        self, api_test_client: TestClient, msg_type: str
    ) -> None:
        """Test that a string message_types value returns an error."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": msg_type, "message_types": "realtime_metrics"})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "list of strings" in response["message"]

    def test_non_string_items_in_message_types_returns_error(
        self, api_test_client: TestClient
    ) -> None:
        """Test that non-string items in message_types returns an error."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "subscribe", "message_types": [1, 2]})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "list of strings" in response["message"]


class TestWebSocketUnknownMessageType:
    """Test WebSocket behavior with unknown message types."""

    def test_unknown_message_type_ignored(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that unknown message types are silently ignored (no error response)."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "unknown_action", "data": "test"})
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"


class TestWebSocketRouterLifecycle:
    """Test WebSocketRouter lifecycle hooks."""

    @pytest.fixture
    def ws_router(self, mock_zmq, router_config) -> WebSocketRouter:
        return WebSocketRouter(run=router_config)

    @pytest.mark.asyncio
    async def test_subscribe_to_all_message_types(
        self, ws_router: WebSocketRouter
    ) -> None:
        """Test @on_init subscribes to wildcard topic."""
        ws_router.subscribe = AsyncMock()
        await ws_router._subscribe_to_all_message_types()
        ws_router.subscribe.assert_called_once_with("*", ws_router._forward_message)

    @pytest.mark.asyncio
    async def test_forward_message_broadcasts(self, ws_router: WebSocketRouter) -> None:
        """Test _forward_message broadcasts to ws_manager."""
        ws_router.ws_manager.broadcast = AsyncMock(return_value=2)
        ws_router.debug = MagicMock()

        msg = HeartbeatMessage(
            service_id="test", service_type="worker", state=LifecycleState.RUNNING
        )
        await ws_router._forward_message(msg)

        ws_router.ws_manager.broadcast.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_close_all_connections(self, ws_router: WebSocketRouter) -> None:
        """Test @on_stop closes all WebSocket connections."""
        ws_router.ws_manager.close_all = AsyncMock()
        await ws_router._close_all_connections()
        ws_router.ws_manager.close_all.assert_called_once()


class TestWebSocketEndpointExceptionHandler:
    """Test the generic exception handler in the WebSocket endpoint."""

    def test_generic_exception_logged(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that a non-disconnect exception is logged via router.exception."""
        ws_router = mock_fastapi_service._routers["websocket"]
        ws_router.exception = MagicMock()

        with (
            patch(
                "aiperf.api.routers.websocket.WebSocket.receive_text",
                side_effect=RuntimeError("boom"),
            ),
            api_test_client.websocket_connect("/ws"),
        ):
            pass

        ws_router.exception.assert_called_once()
        assert "boom" in ws_router.exception.call_args[0][0]

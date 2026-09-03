# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Responses WebSocket transport."""

import asyncio
import time
from unittest.mock import AsyncMock

import aiohttp
import orjson
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.models import RequestInfo, SSEMessage
from aiperf.plugin.enums import TransportType
from aiperf.transports.websocket_transport import (
    WebSocketTransport,
    _has_ws_scheme,
    _sanitize_stream_id,
)
from tests.unit.transports.conftest import create_model_endpoint_info


def _text(event: dict) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, orjson.dumps(event).decode(), None)


def _binary(event: dict) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, orjson.dumps(event), None)


def _control(msg_type: aiohttp.WSMsgType) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(msg_type, None, None)


class FakeWS:
    """Minimal stand-in for aiohttp.ClientWebSocketResponse."""

    def __init__(self, messages: list[aiohttp.WSMessage]) -> None:
        self._messages = list(messages)
        self.closed = False
        self.sent: list[str] = []
        self._exc: Exception | None = None

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self) -> aiohttp.WSMessage:
        if not self._messages:
            self.closed = True
            return _control(aiohttp.WSMsgType.CLOSED)
        return self._messages.pop(0)

    async def close(self) -> None:
        self.closed = True

    def exception(self) -> Exception | None:
        return self._exc


def _request_info(
    *,
    x_correlation_id: str = "conv-1",
    is_final_turn: bool = True,
    turn_index: int = 0,
    previous_response_id: str | None = None,
) -> RequestInfo:
    model_endpoint = create_model_endpoint_info(
        base_url="ws://localhost:8000", custom_endpoint="/v1/responses"
    )
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        turn_index=turn_index,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-1",
        x_correlation_id=x_correlation_id,
        conversation_id="conv-1",
        is_final_turn=is_final_turn,
        previous_response_id=previous_response_id,
    )


class TestSanitizeStreamId:
    def test_none_and_empty_return_none(self) -> None:
        assert _sanitize_stream_id(None) is None
        assert _sanitize_stream_id("") is None

    def test_invalid_chars_replaced(self) -> None:
        assert _sanitize_stream_id("a b/c:d") == "a_b_c_d"

    def test_allowed_chars_preserved(self) -> None:
        assert _sanitize_stream_id("Conv_1.2-3") == "Conv_1.2-3"

    def test_truncated_to_256(self) -> None:
        assert len(_sanitize_stream_id("x" * 500)) == 256

    def test_all_invalid_collapses_but_nonempty(self) -> None:
        # Non-alphanumeric chars are replaced with '_', never dropped.
        assert _sanitize_stream_id("///") == "___"


class TestHasWsScheme:
    def test_ws_and_wss(self) -> None:
        assert _has_ws_scheme("ws://h") is True
        assert _has_ws_scheme("WSS://h") is True

    def test_http_is_not_ws(self) -> None:
        assert _has_ws_scheme("http://h") is False


class TestMetadata:
    def test_metadata(self) -> None:
        meta = WebSocketTransport.metadata()
        assert meta.transport_type == TransportType.WEBSOCKET
        assert meta.url_schemes == ["ws", "wss"]


class TestGetUrl:
    def _transport(self) -> WebSocketTransport:
        return WebSocketTransport(
            model_endpoint=create_model_endpoint_info(base_url="ws://localhost:8000")
        )

    def test_ws_scheme_joins_endpoint_path(self) -> None:
        transport = self._transport()
        info = _request_info()
        assert transport.get_url(info) == "ws://localhost:8000/v1/responses"

    def test_wss_scheme_preserved(self) -> None:
        transport = WebSocketTransport(
            model_endpoint=create_model_endpoint_info(
                base_url="wss://example.com", custom_endpoint="/v1/responses"
            )
        )
        info = RequestInfo(
            model_endpoint=create_model_endpoint_info(
                base_url="wss://example.com", custom_endpoint="/v1/responses"
            ),
            turns=[],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="r",
            x_correlation_id="c",
            conversation_id="c",
        )
        assert transport.get_url(info) == "wss://example.com/v1/responses"

    def test_no_duplicate_path_when_already_present(self) -> None:
        me = create_model_endpoint_info(
            base_url="ws://localhost:8000/v1/responses",
            custom_endpoint="/v1/responses",
        )
        transport = WebSocketTransport(model_endpoint=me)
        info = RequestInfo(
            model_endpoint=me,
            turns=[],
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="r",
            x_correlation_id="c",
            conversation_id="c",
        )
        assert transport.get_url(info) == "ws://localhost:8000/v1/responses"


class TestBuildEnvelope:
    def _transport(self) -> WebSocketTransport:
        return WebSocketTransport(
            model_endpoint=create_model_endpoint_info(base_url="ws://localhost:8000")
        )

    def test_strips_http_only_keys_and_sets_type(self) -> None:
        transport = self._transport()
        payload = {
            "model": "m",
            "input": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "background": True,
        }
        envelope = transport._build_envelope(payload, _request_info())
        assert envelope["type"] == "response.create"
        assert "stream" not in envelope
        assert "stream_options" not in envelope
        assert "background" not in envelope
        assert envelope["model"] == "m"

    def test_accepts_bytes_payload(self) -> None:
        transport = self._transport()
        payload = orjson.dumps({"model": "m", "input": []})
        envelope = transport._build_envelope(payload, _request_info())
        assert envelope["type"] == "response.create"
        assert envelope["model"] == "m"

    def test_stream_id_derived_from_correlation(self) -> None:
        transport = self._transport()
        envelope = transport._build_envelope(
            {"model": "m"}, _request_info(x_correlation_id="conv/1")
        )
        assert envelope["stream_id"] == "conv_1"


class TestErrorFromEvent:
    def test_top_level_error(self) -> None:
        err = WebSocketTransport._error_from_event(
            {"type": "error", "error": {"code": 400, "type": "bad", "message": "boom"}}
        )
        assert err.code == 400
        assert err.type == "bad"
        assert err.message == "boom"

    def test_nested_response_error(self) -> None:
        err = WebSocketTransport._error_from_event(
            {
                "type": "response.failed",
                "response": {"error": {"type": "server_error", "message": "nope"}},
            }
        )
        assert err.type == "server_error"
        assert err.message == "nope"

    def test_fallback(self) -> None:
        err = WebSocketTransport._error_from_event({"type": "response.incomplete"})
        assert err.type == "response.incomplete"
        assert "did not complete" in err.message


@pytest.mark.asyncio
class TestSendRequest:
    async def _transport(self) -> WebSocketTransport:
        transport = WebSocketTransport(
            model_endpoint=create_model_endpoint_info(base_url="ws://localhost:8000")
        )
        await transport.initialize()
        return transport

    async def test_not_initialized_raises(self) -> None:
        transport = WebSocketTransport(
            model_endpoint=create_model_endpoint_info(base_url="ws://localhost:8000")
        )
        with pytest.raises(NotInitializedError):
            await transport.send_request(_request_info(), {"model": "m"})

    async def test_success_streams_events(self) -> None:
        transport = await self._transport()
        fake = FakeWS(
            [
                _text({"type": "response.created", "response": {"id": "resp_1"}}),
                _text({"type": "response.output_text.delta", "delta": "Hi"}),
                _text(
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_1", "status": "completed"},
                    }
                ),
            ]
        )
        transport._open = AsyncMock(return_value=fake)

        record = await transport.send_request(_request_info(), {"model": "m"})

        assert record.status == 200
        assert record.error is None
        assert len(record.responses) == 3
        assert all(isinstance(r, SSEMessage) for r in record.responses)
        # The envelope was sent as a JSON text frame.
        assert fake.sent
        sent = orjson.loads(fake.sent[0])
        assert sent["type"] == "response.create"
        await transport.stop()

    async def test_first_token_callback_fires(self) -> None:
        transport = await self._transport()
        fake = FakeWS(
            [
                _text({"type": "response.output_text.delta", "delta": "Hi"}),
                _text({"type": "response.completed", "response": {"id": "r"}}),
            ]
        )
        transport._open = AsyncMock(return_value=fake)
        calls: list[int] = []

        async def cb(ttft_ns: int, sse: SSEMessage) -> bool:
            calls.append(ttft_ns)
            return True

        await transport.send_request(
            _request_info(), {"model": "m"}, first_token_callback=cb
        )
        assert len(calls) == 1
        await transport.stop()

    async def test_failure_event_sets_error(self) -> None:
        transport = await self._transport()
        fake = FakeWS(
            [
                _text(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": "r",
                            "error": {"type": "server_error", "message": "kaboom"},
                        },
                    }
                ),
            ]
        )
        transport._open = AsyncMock(return_value=fake)

        record = await transport.send_request(_request_info(), {"model": "m"})
        assert record.status != 200
        assert record.error is not None
        assert record.error.message == "kaboom"
        await transport.stop()

    async def test_binary_frame_decoded(self) -> None:
        transport = await self._transport()
        fake = FakeWS(
            [
                _binary({"type": "response.completed", "response": {"id": "r"}}),
            ]
        )
        transport._open = AsyncMock(return_value=fake)
        record = await transport.send_request(_request_info(), {"model": "m"})
        assert record.status == 200
        await transport.stop()

    async def test_premature_close_sets_error(self) -> None:
        transport = await self._transport()
        fake = FakeWS([_control(aiohttp.WSMsgType.CLOSE)])
        transport._open = AsyncMock(return_value=fake)
        record = await transport.send_request(_request_info(), {"model": "m"})
        assert record.error is not None
        assert record.error.type == "ConnectionClosed"
        await transport.stop()

    async def test_cancellation_records_499(self) -> None:
        transport = await self._transport()

        class CancelWS(FakeWS):
            async def receive(self) -> aiohttp.WSMessage:
                raise asyncio.CancelledError

        fake = CancelWS([])
        transport._open = AsyncMock(return_value=fake)
        with pytest.raises(asyncio.CancelledError):
            await transport.send_request(_request_info(), {"model": "m"})
        await transport.stop()


@pytest.mark.asyncio
class TestPoolAndAffinity:
    async def _transport(self) -> WebSocketTransport:
        transport = WebSocketTransport(
            model_endpoint=create_model_endpoint_info(base_url="ws://localhost:8000")
        )
        await transport.initialize()
        return transport

    def _completed(self) -> FakeWS:
        return FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])

    async def test_sequential_turns_reuse_socket(self) -> None:
        transport = await self._transport()
        opened: list[FakeWS] = []

        async def fake_open(
            request_info: RequestInfo, headers: dict[str, str]
        ) -> FakeWS:
            fake = FakeWS(
                [_text({"type": "response.completed", "response": {"id": "r"}})]
            )
            opened.append(fake)
            return fake

        transport._open = fake_open

        # First (non-final) turn opens a socket and keeps it leased.
        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        # Second turn on the same conversation reuses the leased socket.
        first_socket = opened[0]
        first_socket._messages = [
            _text({"type": "response.completed", "response": {"id": "r"}})
        ]
        await transport.send_request(
            _request_info(is_final_turn=True, turn_index=1), {"model": "m"}
        )
        assert len(opened) == 1
        await transport.stop()

    async def test_final_turn_drops_lease(self) -> None:
        transport = await self._transport()
        transport._open = AsyncMock(side_effect=lambda ri, h: self._completed())
        await transport.send_request(_request_info(is_final_turn=True), {"model": "m"})
        assert transport._leases == {}
        await transport.stop()

    async def test_no_correlation_id_gets_fresh_socket_per_request(self) -> None:
        transport = await self._transport()
        opened: list[FakeWS] = []

        async def fake_open(
            request_info: RequestInfo, headers: dict[str, str]
        ) -> FakeWS:
            fake = self._completed()
            opened.append(fake)
            return fake

        transport._open = fake_open
        # Without a conversation identity, each request opens and closes its own
        # socket; nothing is leased.
        await transport.send_request(
            _request_info(x_correlation_id="", is_final_turn=False), {"model": "m"}
        )
        await transport.send_request(
            _request_info(x_correlation_id="", is_final_turn=False), {"model": "m"}
        )
        assert len(opened) == 2
        assert transport._leases == {}
        assert all(f.closed for f in opened)
        await transport.stop()

    async def test_stop_closes_all_sockets(self) -> None:
        transport = await self._transport()
        fake = self._completed()
        transport._open = AsyncMock(return_value=fake)
        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        await transport.stop()
        assert fake.closed is True
        assert transport._all == set()


class _HangingWS(FakeWS):
    """A socket whose peer never sends another frame (silent peer)."""

    async def receive(self) -> aiohttp.WSMessage:
        await asyncio.Event().wait()  # never set
        raise AssertionError("unreachable")  # pragma: no cover


@pytest.mark.asyncio
class TestReceiveTimeout:
    def _ws_endpoint(self, timeout: float):
        me = create_model_endpoint_info(
            base_url="ws://localhost:8000", custom_endpoint="/v1/responses"
        )
        me.endpoint.timeout = timeout
        return me

    async def test_receive_raises_when_deadline_passed(self) -> None:
        # A deadline already in the past short-circuits without awaiting.
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await WebSocketTransport._receive(
                _HangingWS([]), time.perf_counter_ns() - 1
            )

    async def test_receive_without_deadline_returns_frame(self) -> None:
        fake = FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])
        msg = await WebSocketTransport._receive(fake, None)
        assert msg.type is aiohttp.WSMsgType.TEXT

    async def test_silent_peer_times_out_turn(self) -> None:
        transport = WebSocketTransport(model_endpoint=self._ws_endpoint(0.05))
        await transport.initialize()
        transport._open = AsyncMock(return_value=_HangingWS([]))

        record = await transport.send_request(_request_info(), {"model": "m"})

        assert record.error is not None
        assert record.error.type == "TimeoutError"
        assert record.error.code == 408
        # The doomed socket is dropped, not left leased.
        assert transport._leases == {}
        await transport.stop()


@pytest.mark.asyncio
class TestReconnectChaining:
    def _endpoint(self, extra: list[tuple[str, object]] | None = None):
        me = create_model_endpoint_info(
            base_url="ws://localhost:8000", custom_endpoint="/v1/responses"
        )
        if extra is not None:
            me.endpoint.extra = extra
        return me

    def _open_recorder(self, opened: list[FakeWS]):
        async def fake_open(
            request_info: RequestInfo, headers: dict[str, str]
        ) -> FakeWS:
            fake = FakeWS(
                [_text({"type": "response.completed", "response": {"id": "r"}})]
            )
            opened.append(fake)
            return fake

        return fake_open

    async def test_reconnect_without_store_fails_chained_turn(self) -> None:
        transport = WebSocketTransport(model_endpoint=self._endpoint())
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        # Turn 1 leases a socket for the conversation.
        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        # The peer drops the idle socket between turns.
        opened[0].closed = True
        # Turn 2 chains onto the prior response but must reconnect; the
        # connection-local cache is gone, so the chained turn cannot succeed.
        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert record.error is not None
        assert record.error.type == "ChainingContextLost"
        # A fresh socket was opened but the doomed turn was never sent on it.
        assert len(opened) == 2
        assert opened[1].sent == []
        await transport.stop()

    async def test_reconnect_with_store_proceeds(self) -> None:
        transport = WebSocketTransport(
            model_endpoint=self._endpoint(extra=[("store", True)])
        )
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        opened[0].closed = True
        # With server-side store the id persists, so a reconnect resolves it.
        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert record.error is None
        assert record.status == 200
        assert len(opened) == 2
        assert opened[1].sent  # the turn was actually sent on the fresh socket
        await transport.stop()

    async def test_reconnect_with_per_turn_store_proceeds(self) -> None:
        # Endpoint-level extra has no store, but the per-turn payload does; that
        # still persists the response server-side, so a reconnect resolves the id.
        transport = WebSocketTransport(model_endpoint=self._endpoint())
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        opened[0].closed = True
        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1", "store": True},
        )

        assert record.error is None
        assert record.status == 200
        assert len(opened) == 2
        assert opened[1].sent
        await transport.stop()


@pytest.mark.asyncio
class TestForkCrossConnectionChaining:
    """A FORK child gets a fresh x_correlation_id, so on WebSockets it opens its
    own socket. Chaining onto an inherited previous_response_id there is not
    portable without store, so the worker drops the id and the child replays full
    history instead (see worker/session_manager). These tests only assert the
    transport still chains normally when a previous_response_id IS present -- the
    replay decision is exercised in the worker/session-manager suites."""

    def _endpoint(self, extra: list[tuple[str, object]] | None = None):
        me = create_model_endpoint_info(
            base_url="ws://localhost:8000", custom_endpoint="/v1/responses"
        )
        if extra is not None:
            me.endpoint.extra = extra
        return me

    def _open_recorder(self, opened: list[FakeWS]):
        async def fake_open(
            request_info: RequestInfo, headers: dict[str, str]
        ) -> FakeWS:
            fake = FakeWS(
                [_text({"type": "response.completed", "response": {"id": "r"}})]
            )
            opened.append(fake)
            return fake

        return fake_open

    async def test_fresh_socket_with_previous_id_sends_chained_turn(self) -> None:
        # When a previous_response_id reaches the transport (e.g. store:true FORK,
        # or any chained turn), it is sent as-is on a fresh socket -- the transport
        # does not fail or rewrite it. The replay decision happens upstream.
        transport = WebSocketTransport(
            model_endpoint=self._endpoint(extra=[("store", True)])
        )
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        record = await transport.send_request(
            _request_info(
                x_correlation_id="conv-child",
                previous_response_id="resp_parent",
            ),
            {"model": "m", "previous_response_id": "resp_parent"},
        )

        assert record.status == 200
        assert opened[0].sent  # the chained turn was sent on the fresh socket
        sent = orjson.loads(opened[0].sent[0])
        assert sent["previous_response_id"] == "resp_parent"
        await transport.stop()

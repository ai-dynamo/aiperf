# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Responses WebSocket transport."""

import asyncio
import time
from unittest.mock import AsyncMock

import aiohttp
import orjson
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
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
    cancel_after_ns: int | None = None,
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
        cancel_after_ns=cancel_after_ns,
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


class TestHasWsScheme:
    def test_ws_and_wss(self) -> None:
        assert _has_ws_scheme("ws://h") is True
        assert _has_ws_scheme("WSS://h") is True


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

    def test_v1_base_url_collapses_overlap(self) -> None:
        """A ``ws(s)://host/v1`` base URL must not become ``/v1/v1/responses``."""
        me = create_model_endpoint_info(
            base_url="ws://localhost:8000/v1", custom_endpoint="/v1/responses"
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
        # A failure event carrying no error payload still yields a usable error.
        err = WebSocketTransport._error_from_event({"type": "response.failed"})
        assert err.type == "response.failed"
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
        assert fake.sent
        sent = orjson.loads(fake.sent[0])
        assert sent["type"] == "response.create"
        await transport.stop()

    async def test_payload_bytes_updated_to_sent_envelope(self) -> None:
        transport = await self._transport()
        fake = FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])
        transport._open = AsyncMock(return_value=fake)
        info = _request_info()

        await transport.send_request(
            info, {"model": "m", "stream": True, "background": True}
        )

        # The raw-record exporter replays payload_bytes verbatim, so it must be
        # the exact wire frame, not the pre-envelope HTTP-style body.
        assert info.payload_bytes == fake.sent[0].encode()
        payload = orjson.loads(info.payload_bytes)
        assert payload["type"] == "response.create"
        assert payload["stream_id"]
        assert "stream" not in payload
        assert "background" not in payload
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

    async def test_incomplete_event_ends_turn_successfully(self) -> None:
        # response.incomplete is the contract's normal terminal status for a
        # truncated output (e.g. max_output_tokens); like the HTTP SSE path it is
        # a success, not an error, and leaves the socket reusable.
        transport = await self._transport()
        fake = FakeWS(
            [
                _text({"type": "response.output_text.delta", "delta": "Hi"}),
                _text(
                    {
                        "type": "response.incomplete",
                        "response": {
                            "id": "r",
                            "status": "incomplete",
                            "incomplete_details": {"reason": "max_output_tokens"},
                        },
                    }
                ),
            ]
        )
        transport._open = AsyncMock(return_value=fake)

        record = await transport.send_request(_request_info(), {"model": "m"})

        assert record.status == 200
        assert record.error is None
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

    async def test_cancellation_propagates_and_closes_socket(self) -> None:
        transport = await self._transport()

        class CancelWS(FakeWS):
            async def receive(self) -> aiohttp.WSMessage:
                raise asyncio.CancelledError

        fake = CancelWS([])
        transport._open = AsyncMock(return_value=fake)
        # send_request re-raises CancelledError rather than returning the record,
        # so the only observable contract is propagation plus socket cleanup: the
        # leased socket is closed and its lease dropped in the finally block.
        with pytest.raises(asyncio.CancelledError):
            await transport.send_request(_request_info(), {"model": "m"})
        assert fake.closed is True
        assert transport._leases == {}
        await transport.stop()

    async def test_handshake_timeout_records_408(self) -> None:
        transport = await self._transport()
        transport._open = AsyncMock(side_effect=TimeoutError)
        record = await transport.send_request(_request_info(), {"model": "m"})
        assert record.error is not None
        assert record.error.code == 408
        assert record.error.type == "TimeoutError"
        assert "handshake exceeded" in record.error.message
        await transport.stop()

    async def test_connect_failure_records_error(self) -> None:
        transport = await self._transport()
        transport._open = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        record = await transport.send_request(_request_info(), {"model": "m"})
        assert record.error is not None
        assert record.status != 200
        await transport.stop()

    async def test_cancel_after_ns_records_499(self) -> None:
        transport = await self._transport()
        fake = FakeWS([])
        transport._open = AsyncMock(return_value=fake)

        async def _hang(*_a: object, **_k: object) -> bool:
            await asyncio.Event().wait()
            return False

        transport._read_until_terminal = _hang
        record = await transport.send_request(
            _request_info(cancel_after_ns=1_000_000), {"model": "m"}
        )
        assert record.error is not None
        assert record.error.code == 499
        assert record.error.type == "RequestCancellationError"
        assert record.cancellation_perf_ns is not None
        await transport.stop()

    async def test_cancel_after_ns_completes_before_deadline(self) -> None:
        transport = await self._transport()
        fake = FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])
        transport._open = AsyncMock(return_value=fake)
        record = await transport.send_request(
            _request_info(cancel_after_ns=60 * NANOS_PER_SECOND), {"model": "m"}
        )
        assert record.status == 200
        assert record.error is None
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

        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
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

        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
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

    async def test_reconnect_discards_stale_socket_from_all(self) -> None:
        # A peer-closed leased socket is replaced on reconnect; the old one must
        # not linger in _all, or every reconnect leaks a socket until shutdown.
        transport = WebSocketTransport(
            model_endpoint=self._endpoint(extra=[("store", True)])
        )
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        await transport.send_request(_request_info(is_final_turn=False), {"model": "m"})
        opened[0].closed = True
        await transport.send_request(
            _request_info(
                is_final_turn=False, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert opened[0] not in transport._all
        assert opened[1] in transport._all
        assert len(transport._all) == 1
        await transport.stop()

    async def test_reconnect_with_per_turn_store_proceeds(self) -> None:
        # Endpoint-level extra has no store, but turn 1's per-turn payload does;
        # that persists resp_1 server-side, so a reconnect on turn 2 resolves it.
        transport = WebSocketTransport(model_endpoint=self._endpoint())
        await transport.initialize()
        opened: list[FakeWS] = []
        transport._open = self._open_recorder(opened)

        await transport.send_request(
            _request_info(is_final_turn=False), {"model": "m", "store": True}
        )
        opened[0].closed = True
        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert record.error is None
        assert record.status == 200
        assert len(opened) == 2
        assert opened[1].sent
        await transport.stop()

    def _seq_opener(self, sockets: list[FakeWS], opened: list[FakeWS]):
        """Hand out prepared sockets in order, recording each open.

        Unlike ``_open_recorder`` (which fabricates a fresh clean socket per open),
        this lets a turn end on a socket that is already dirty/closed, so the test
        can exercise the reconnect the *release* triggers rather than the peer.
        """
        queue = list(sockets)

        async def fake_open(
            request_info: RequestInfo, headers: dict[str, str]
        ) -> FakeWS:
            ws = queue.pop(0)
            opened.append(ws)
            return ws

        return fake_open

    async def test_dirty_release_forces_guard_on_next_chained_turn(self) -> None:
        # Turn 1 ends dirty (premature close, no terminal event), so _release drops
        # the lease. The next chained turn must still be forced onto a fresh socket
        # and tripped by the guard -- keying off "was a stale lease present" would
        # miss this, because the dirty release left nothing stale to find.
        transport = WebSocketTransport(model_endpoint=self._endpoint())
        await transport.initialize()
        opened: list[FakeWS] = []
        turn1 = FakeWS([])  # empty -> immediate CLOSED -> dirty premature close
        turn2 = FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])
        transport._open = self._seq_opener([turn1, turn2], opened)

        first = await transport.send_request(
            _request_info(is_final_turn=False), {"model": "m"}
        )
        assert first.error is not None  # turn 1 was dirty
        assert "conv-1" not in transport._leases  # dirty release dropped the lease

        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert record.error is not None
        assert record.error.type == "ChainingContextLost"
        assert len(opened) == 2
        assert opened[1].sent == []
        await transport.stop()

    async def test_socket_closed_at_release_forces_guard_on_next_chained_turn(
        self,
    ) -> None:
        # Turn 1 succeeds but the peer closes the socket as it delivers the terminal
        # event, so _release sees ws.closed and drops the lease (keep requires an
        # open socket). The next chained turn reconnects and must hit the guard.
        class _ClosesOnTerminalWS(FakeWS):
            async def receive(self) -> aiohttp.WSMessage:
                msg = await super().receive()
                if not self._messages:  # terminal event just handed out
                    self.closed = True
                return msg

        transport = WebSocketTransport(model_endpoint=self._endpoint())
        await transport.initialize()
        opened: list[FakeWS] = []
        turn1 = _ClosesOnTerminalWS(
            [_text({"type": "response.completed", "response": {"id": "r"}})]
        )
        turn2 = FakeWS([_text({"type": "response.completed", "response": {"id": "r"}})])
        transport._open = self._seq_opener([turn1, turn2], opened)

        first = await transport.send_request(
            _request_info(is_final_turn=False), {"model": "m"}
        )
        assert first.error is None
        assert "conv-1" not in transport._leases  # closed socket dropped at release

        record = await transport.send_request(
            _request_info(
                is_final_turn=True, turn_index=1, previous_response_id="resp_1"
            ),
            {"model": "m", "previous_response_id": "resp_1"},
        )

        assert record.error is not None
        assert record.error.type == "ChainingContextLost"
        assert len(opened) == 2
        assert opened[1].sent == []
        await transport.stop()

    async def test_reconnect_prior_turn_unstored_fails_despite_current_store(
        self,
    ) -> None:
        # resp_1 came from an unstored turn 1; turn 2 setting store:true cannot
        # retroactively persist it, so a reconnect still loses the chain. The
        # guard must key off the prior turn's store, not the current turn's.
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

        assert record.error is not None
        assert record.error.type == "ChainingContextLost"
        assert len(opened) == 2
        assert opened[1].sent == []
        await transport.stop()


@pytest.mark.asyncio
class TestForkCrossConnectionChaining:
    """The transport forwards a previous_response_id as-is; the FORK replay
    decision is made upstream in the worker/session-manager suites."""

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
        assert opened[0].sent
        sent = orjson.loads(opened[0].sent[0])
        assert sent["previous_response_id"] == "resp_parent"
        await transport.stop()

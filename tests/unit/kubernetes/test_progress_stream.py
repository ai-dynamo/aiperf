# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.progress_stream.

Complements ``tests.unit.kubernetes.test_port_forward`` (which re-exports
``stream_progress_from_api`` via ``port_forward``) by exercising the
contract of ``progress_stream`` directly:

- Subscription handshake happens before ``on_message`` is invoked.
- ``asyncio.TimeoutError`` is treated like ``aiohttp.ClientError`` for retry.
- ``ConnectionError`` after ``max_retries`` preserves the original exception
  as ``__cause__``.
- Exceptions raised inside ``on_message`` propagate after the session /
  websocket context managers run their ``__aexit__`` cleanup.
- Backoff is clamped to ``_WS_MAX_BACKOFF`` during long retry chains.
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, NonCallableMock, patch

import aiohttp
import pytest

from aiperf.kubernetes import progress_stream
from aiperf.kubernetes.progress_stream import (
    _WS_MAX_BACKOFF,
    stream_progress_from_api,
)

# ============================================================
# Helpers
# ============================================================


async def _async_iter(items: list) -> AsyncIterator:
    """Build an async iterator over the given items."""
    for item in items:
        yield item


def _text_frame(payload: bytes) -> MagicMock:
    """Build a mock aiohttp TEXT WS frame carrying ``payload`` bytes."""
    frame = MagicMock()
    frame.type = aiohttp.WSMsgType.TEXT
    frame.data = payload
    return frame


def _make_mock_ws(
    frames: list,
    *,
    ack: dict | None = None,
    ack_side_effect: Exception | None = None,
) -> AsyncMock:
    """Build a mock websocket that yields ``frames`` and supports async-with."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    if ack_side_effect is not None:
        ws.receive_json = AsyncMock(side_effect=ack_side_effect)
    else:
        ws.receive_json = AsyncMock(return_value=ack or {"type": "subscribed"})
    ws.close = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)
    ws.__aiter__ = lambda self: _async_iter(frames)
    return ws


def _make_mock_session(ws_or_sider) -> AsyncMock:
    """Build a mock aiohttp.ClientSession wrapping the given ws (or side_effect).

    ``MagicMock`` / ``AsyncMock`` are themselves ``callable``, so a bare
    ``callable()`` check would wrongly treat a mocked websocket as a factory.
    Route Exceptions and genuine factory functions to ``side_effect``; treat
    everything else (including Mock instances) as the ``return_value``.
    """
    session = AsyncMock()
    if isinstance(ws_or_sider, Exception) or (
        callable(ws_or_sider)
        and not isinstance(ws_or_sider, (MagicMock, AsyncMock, NonCallableMock))
    ):
        session.ws_connect = MagicMock(side_effect=ws_or_sider)
    else:
        session.ws_connect = MagicMock(return_value=ws_or_sider)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _patch_session(session_or_factory):
    """Patch ``aiohttp.ClientSession`` at the progress_stream module path."""
    # A plain function acts as a factory (side_effect); an AsyncMock/MagicMock
    # is a pre-built session to be returned directly. ``isinstance(x, Mock)``
    # misses AsyncMock unless we check NonCallableMock.
    if callable(session_or_factory) and not isinstance(
        session_or_factory, (MagicMock, AsyncMock, NonCallableMock)
    ):
        return patch(
            "aiperf.kubernetes.progress_stream.aiohttp.ClientSession",
            side_effect=session_or_factory,
        )
    return patch(
        "aiperf.kubernetes.progress_stream.aiohttp.ClientSession",
        return_value=session_or_factory,
    )


def _patch_connector():
    """Stub ``create_tcp_connector`` so no real socket is opened."""
    return patch(
        "aiperf.transports.aiohttp_client.create_tcp_connector",
        return_value=MagicMock(),
    )


# ============================================================
# Subscription handshake
# ============================================================


class TestSubscriptionHandshake:
    """Verify the connect -> subscribe -> ack ordering before dispatch."""

    async def test_subscribe_frame_sent_before_ack_consumed(self) -> None:
        """send_json subscribe must be awaited before receive_json ack."""
        call_order: list[str] = []

        async def track_send(_payload: dict) -> None:
            call_order.append("send")

        async def track_receive() -> dict:
            call_order.append("receive")
            return {"type": "subscribed"}

        ws = _make_mock_ws([])
        ws.send_json = AsyncMock(side_effect=track_send)
        ws.receive_json = AsyncMock(side_effect=track_receive)

        session = _make_mock_session(ws)

        async def on_message(_data: dict) -> bool:
            return True

        with _patch_session(session), _patch_connector():
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress", "realtime_metrics"],
            )

        ws.send_json.assert_awaited_once_with(
            {
                "type": "subscribe",
                "message_types": ["progress", "realtime_metrics"],
            }
        )
        assert call_order == ["send", "receive"]

    async def test_ack_consumed_before_on_message_dispatch(self) -> None:
        """on_message is not invoked with the subscription ack frame."""
        received: list[dict] = []

        async def on_message(data: dict) -> bool:
            received.append(data)
            return True

        ws = _make_mock_ws(
            [_text_frame(b'{"type":"progress","value":7}')],
            ack={"type": "subscribed"},
        )
        session = _make_mock_session(ws)

        with _patch_session(session), _patch_connector():
            await stream_progress_from_api(
                "ws://localhost:9090/ws", on_message, ["progress"]
            )

        # The ack ({"type": "subscribed"}) was consumed by receive_json and
        # must NOT appear in on_message's call history.
        assert received == [{"type": "progress", "value": 7}]


# ============================================================
# Transport errors / retry
# ============================================================


class TestTransportErrorRetry:
    """Verify retry behaviour on transport errors."""

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(aiohttp.ClientError("refused"), id="aiohttp-client-error"),
            pytest.param(asyncio.TimeoutError(), id="asyncio-timeout-error"),
            pytest.param(
                aiohttp.ClientConnectorError(
                    MagicMock(ssl=None), OSError("conn refused")
                ),
                id="aiohttp-client-connector-error",
            ),
        ],
    )  # fmt: skip
    async def test_transport_error_triggers_retry(
        self, exc: Exception
    ) -> None:
        """Subclasses of ClientError and TimeoutError are caught and retried."""
        # First attempt raises; second attempt succeeds with a stop message.
        ws_ok = _make_mock_ws([_text_frame(b'{"done":true}')])

        call_count = 0

        def session_factory(**_kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_session(exc)
            return _make_mock_session(ws_ok)

        async def on_message(_data: dict) -> bool:
            return True

        with (
            _patch_session(session_factory),
            _patch_connector(),
            patch.object(progress_stream, "print_info"),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
                max_retries=3,
            )

        assert call_count == 2

    async def test_connection_error_preserves_original_cause(self) -> None:
        """ConnectionError.__cause__ is the last transport exception raised."""
        original = aiohttp.ClientError("boom")
        session = _make_mock_session(original)

        with (
            _patch_session(session),
            _patch_connector(),
            patch.object(progress_stream, "print_info"),
            pytest.raises(ConnectionError) as excinfo,
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws", AsyncMock(), ["progress"], max_retries=2
            )

        assert excinfo.value.__cause__ is original
        assert "Failed to connect to API after 2 attempts" in str(excinfo.value)

    async def test_timeout_error_preserved_as_cause(self) -> None:
        """asyncio.TimeoutError is also preserved as __cause__."""
        original = asyncio.TimeoutError()
        session = _make_mock_session(original)

        with (
            _patch_session(session),
            _patch_connector(),
            patch.object(progress_stream, "print_info"),
            pytest.raises(ConnectionError) as excinfo,
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws", AsyncMock(), ["progress"], max_retries=1
            )

        assert excinfo.value.__cause__ is original

    async def test_non_transport_exception_is_not_caught(self) -> None:
        """Unrelated exceptions bubble immediately without triggering retry."""

        class WeirdError(Exception):
            """Not a transport error — must propagate."""

        session = _make_mock_session(WeirdError("kaboom"))

        with (
            _patch_session(session),
            _patch_connector(),
            patch.object(progress_stream, "print_info") as mock_info,
            pytest.raises(WeirdError, match="kaboom"),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws", AsyncMock(), ["progress"], max_retries=5
            )

        # No retry log emitted — error escaped on the first attempt.
        mock_info.assert_not_called()


# ============================================================
# Backoff progression
# ============================================================


class TestBackoffProgression:
    """Verify exponential backoff schedule and cap."""

    async def test_backoff_doubles_then_caps_at_max(self) -> None:
        """Successive retries double the delay up to _WS_MAX_BACKOFF."""
        # Fail enough times to blow past the cap (1, 2, 4, 8, 16, 30, 30, ...).
        attempts_needed = 8
        session = _make_mock_session(aiohttp.ClientError("down"))

        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        with (
            _patch_session(session),
            _patch_connector(),
            patch.object(progress_stream, "print_info"),
            patch.object(progress_stream.asyncio, "sleep", fake_sleep),
            pytest.raises(ConnectionError),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                AsyncMock(),
                ["progress"],
                max_retries=attempts_needed,
            )

        # asyncio.sleep is called after every retry except the one that
        # raises ConnectionError. sleeps[i] is the delay *before* attempt i+2.
        assert sleeps[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
        # Cap reached; all subsequent sleeps equal _WS_MAX_BACKOFF.
        assert all(d == _WS_MAX_BACKOFF for d in sleeps[5:])
        # And never exceed the cap.
        assert max(sleeps) == _WS_MAX_BACKOFF


# ============================================================
# Callback failure propagation
# ============================================================


class TestCallbackFailure:
    """Verify behaviour when on_message raises."""

    async def test_on_message_exception_propagates_and_closes_ws(self) -> None:
        """Exceptions from on_message propagate and the WS context exits cleanly."""

        class CallbackExplosion(RuntimeError):
            """Sentinel exception for the test."""

        async def on_message(_data: dict) -> bool:
            raise CallbackExplosion("dispatch failed")

        ws = _make_mock_ws([_text_frame(b'{"n":1}')])
        session = _make_mock_session(ws)

        with (
            _patch_session(session),
            _patch_connector(),
            pytest.raises(CallbackExplosion, match="dispatch failed"),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws", on_message, ["progress"]
            )

        # Both async context managers must unwind, even on exception.
        ws.__aexit__.assert_awaited()
        session.__aexit__.assert_awaited()

    async def test_on_message_exception_does_not_trigger_retry(self) -> None:
        """A raising callback is NOT treated as a transport error."""

        async def on_message(_data: dict) -> bool:
            raise RuntimeError("not transport")

        ws = _make_mock_ws([_text_frame(b'{"n":1}')])

        call_count = 0

        def session_factory(**_kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            return _make_mock_session(ws)

        with (
            _patch_session(session_factory),
            _patch_connector(),
            pytest.raises(RuntimeError, match="not transport"),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
                max_retries=5,
            )

        # Only one attempt: non-transport exception must not trigger reconnect.
        assert call_count == 1

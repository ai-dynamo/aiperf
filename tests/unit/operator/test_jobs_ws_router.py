# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator's per-job WebSocket proxy router.

The proxy lives at ``WS /api/v1/jobs/{ns}/{name}/ws``. It looks up the CR's
``status.jobSetName`` to derive the controller pod's headless-service DNS,
opens an upstream WS via ``aiohttp``, and bridges frames bidirectionally.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.operator.routers.jobs_ws import create_jobs_ws_router

_NS = "aiperf-benchmarks"
_NAME = "test-bench"
_JOBSET = f"aiperf-{_NAME}"


def _make_app(api: Any | None) -> FastAPI:
    """Build a FastAPI app exposing the WS proxy with ``api`` in the holder."""
    app = FastAPI()
    holder: list[Any] = [api]
    app.include_router(create_jobs_ws_router(holder))
    return app


def _ws_url() -> str:
    return f"/api/v1/jobs/{_NS}/{_NAME}/ws"


def _patch_status(jobset_name: str | None) -> Any:
    """Patch ``get_raw_aiperfjob_status`` used inside the proxy module."""
    status = {"jobSetName": jobset_name} if jobset_name else {}
    return patch(
        "aiperf.operator.routers.jobs_ws.get_raw_aiperfjob_status",
        new=AsyncMock(return_value=status),
    )


class _FakeUpstreamWS:
    """Minimal stand-in for ``aiohttp.ClientWebSocketResponse``.

    Yields the queued frames in order via ``async for`` (mirroring aiohttp's
    own iteration). When the queue is empty and ``block_when_empty=True``,
    iteration awaits forever until externally cancelled — useful for tests
    that need the bridge to keep running long enough for the client side to
    forward a frame upstream before the wait collapses.
    """

    def __init__(self, frames: list[Any], *, block_when_empty: bool = False) -> None:
        self._frames = list(frames)
        self._block_when_empty = block_when_empty
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._frames:
            return self._frames.pop(0)
        if not self._block_when_empty:
            raise StopAsyncIteration
        # Block until the surrounding bridge cancels this coroutine. The
        # client→upstream pump is the one that should drive completion in
        # the tests that opt into this mode.
        await asyncio.Event().wait()
        raise StopAsyncIteration  # unreachable; satisfies type checker

    async def send_str(self, text: str) -> None:
        self.sent_text.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self) -> None:
        self.closed = True


def _ws_msg(msg_type: aiohttp.WSMsgType, data: Any = "") -> Any:
    """Build a duck-typed WSMessage-shaped object."""
    msg = MagicMock()
    msg.type = msg_type
    msg.data = data
    return msg


def _patch_upstream(upstream: _FakeUpstreamWS) -> Any:
    """Patch ``aiohttp.ClientSession`` so ``session.ws_connect`` returns ``upstream``.

    The proxy uses ``async with aiohttp.ClientSession(...) as session`` and then
    ``session.ws_connect(url, ...)`` as another async context manager — both
    legs need __aenter__/__aexit__ wired.
    """
    cm_ws = MagicMock()
    cm_ws.__aenter__ = AsyncMock(return_value=upstream)
    cm_ws.__aexit__ = AsyncMock(return_value=None)

    fake_session = MagicMock()
    fake_session.ws_connect = MagicMock(return_value=cm_ws)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)

    return patch(
        "aiperf.operator.routers.jobs_ws.aiohttp.ClientSession",
        return_value=fake_session,
    )


def _patch_upstream_connect_error(exc: Exception) -> Any:
    """Patch ``aiohttp.ClientSession`` so ``ws_connect.__aenter__`` raises ``exc``."""
    cm_ws = MagicMock()
    cm_ws.__aenter__ = AsyncMock(side_effect=exc)
    cm_ws.__aexit__ = AsyncMock(return_value=None)

    fake_session = MagicMock()
    fake_session.ws_connect = MagicMock(return_value=cm_ws)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)

    return patch(
        "aiperf.operator.routers.jobs_ws.aiohttp.ClientSession",
        return_value=fake_session,
    )


class TestRefusalCloses:
    def test_no_api_client_closes_with_4503(self) -> None:
        app = _make_app(api=None)
        with (
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
            pytest.raises(Exception) as exc_info,
        ):
            ws.receive_text()
        assert (
            "4503" in str(exc_info.value)
            or getattr(exc_info.value, "code", None) == 4503
        )

    def test_no_jobset_closes_with_4404(self) -> None:
        app = _make_app(api=MagicMock())
        with (
            _patch_status(jobset_name=None),
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
            pytest.raises(Exception) as exc_info,
        ):
            ws.receive_text()
        assert (
            "4404" in str(exc_info.value)
            or getattr(exc_info.value, "code", None) == 4404
        )

    def test_upstream_connect_failure_closes_with_4502(self) -> None:
        app = _make_app(api=MagicMock())
        err = aiohttp.ClientConnectorError(
            connection_key=MagicMock(),
            os_error=OSError("nodename nor servname provided, or not known"),
        )
        with (
            _patch_status(jobset_name=_JOBSET),
            _patch_upstream_connect_error(err),
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
            pytest.raises(Exception) as exc_info,
        ):
            ws.receive_text()
        assert (
            "4502" in str(exc_info.value)
            or getattr(exc_info.value, "code", None) == 4502
        )


class TestProxyForwarding:
    def test_upstream_text_frames_arrive_at_client(self) -> None:
        upstream = _FakeUpstreamWS(
            [
                _ws_msg(aiohttp.WSMsgType.TEXT, '{"type":"subscribed"}'),
                _ws_msg(
                    aiohttp.WSMsgType.TEXT,
                    '{"type":"realtime_metrics","metrics":[{"tag":"request_throughput","avg":42.0}]}',
                ),
                _ws_msg(aiohttp.WSMsgType.CLOSE),
            ]
        )
        app = _make_app(api=MagicMock())
        with (
            _patch_status(jobset_name=_JOBSET),
            _patch_upstream(upstream),
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
        ):
            first = ws.receive_text()
            second = ws.receive_text()
            assert '"subscribed"' in first
            assert '"realtime_metrics"' in second
            assert '"request_throughput"' in second

    def test_client_text_frame_forwarded_upstream(self) -> None:
        # Upstream blocks on __anext__ so the bridge keeps running until the
        # client disconnects, giving the client→upstream pump time to forward
        # the subscribe frame before tear-down.
        upstream = _FakeUpstreamWS([], block_when_empty=True)
        app = _make_app(api=MagicMock())
        with (
            _patch_status(jobset_name=_JOBSET),
            _patch_upstream(upstream),
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
        ):
            ws.send_text('{"type":"subscribe","message_types":["realtime_metrics"]}')
            # Wait briefly for the proxy to forward; TestClient runs the
            # ASGI app in a background thread so this resolves in <100ms.
            deadline = time.monotonic() + 2.0
            while not upstream.sent_text and time.monotonic() < deadline:
                time.sleep(0.01)
            ws.close()
        assert any("subscribe" in t for t in upstream.sent_text), (
            f"upstream did not see the subscribe frame; sent_text={upstream.sent_text}"
        )

    def test_upstream_close_propagates_to_client(self) -> None:
        from starlette.websockets import WebSocketDisconnect

        upstream = _FakeUpstreamWS([_ws_msg(aiohttp.WSMsgType.CLOSE)])
        app = _make_app(api=MagicMock())
        with (
            _patch_status(jobset_name=_JOBSET),
            _patch_upstream(upstream),
            TestClient(app) as client,
            client.websocket_connect(_ws_url()) as ws,
            pytest.raises(WebSocketDisconnect),
        ):
            ws.receive_text()

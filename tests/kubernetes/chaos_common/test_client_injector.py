# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`ClientInjector`.

Mocks :py:class:`aiohttp.ClientSession` so the contract (mid-stream cancel
closes the session, oversized-payload POST size + status capture, prefix
matching, precondition errors on missing url) is exercised without touching
a real HTTP server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from tests.kubernetes.chaos_common.base import (
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.client import ClientInjector


class _FakeStreamContent:
    """Stand-in for ``resp.content``: yields a configurable chunk sequence."""

    def __init__(self, chunks: list[bytes], delay_per_chunk: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay_per_chunk

    def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        async def _gen() -> AsyncIterator[bytes]:
            import asyncio

            for chunk in self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield chunk

        return _gen()

    async def read(self, n: int = -1) -> bytes:
        joined = b"".join(self._chunks)
        if n < 0:
            return joined
        return joined[:n]


def _make_mock_response(
    *,
    chunks: list[bytes] | None = None,
    delay_per_chunk: float = 0.0,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> MagicMock:
    """Build a MagicMock that mimics :py:class:`aiohttp.ClientResponse`."""
    resp = MagicMock()
    resp.status = status
    resp.headers = headers if headers is not None else {}
    if chunks is not None:
        resp.content = _FakeStreamContent(chunks, delay_per_chunk=delay_per_chunk)
    else:
        resp.content = _FakeStreamContent([body])
    resp.release = MagicMock()
    # The async context-manager surface used by the overflow path.
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


def _make_mock_session(
    response: MagicMock,
    *,
    capture: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics :py:class:`aiohttp.ClientSession`.

    If ``capture`` is provided, the POST url and json body are recorded so
    tests can assert on the wire payload.
    """
    session = MagicMock()

    def _post(url: str, *, json: Any = None, **_: Any) -> MagicMock:
        if capture is not None:
            capture["url"] = url
            capture["json"] = json
        return response

    session.post = MagicMock(side_effect=_post)
    session.close = AsyncMock(return_value=None)
    # Async context-manager surface for the overflow path's `async with session:`.
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.mark.asyncio
async def test_cancel_request_closes_session_after_deadline() -> None:
    # Slow stream: 50 chunks, ~10ms each = ~500ms total; deadline 50ms forces
    # the timeout-induced cancel path to fire.
    response = _make_mock_response(
        chunks=[b"data: chunk\n\n"] * 50,
        delay_per_chunk=0.01,
    )
    session = _make_mock_session(response)
    spec = FaultSpec(
        fault_id="client.cancel_request",
        target={"url": "http://example.invalid/v1/chat/completions"},
        params={
            "payload": {"model": "x", "messages": []},
            "cancel_after_seconds": 0.05,
        },
    )

    with patch("aiohttp.ClientSession", return_value=session):
        handle = await ClientInjector().inject(spec)

    session.close.assert_awaited_once()
    response.release.assert_called_once()
    assert handle.metadata["url"] == "http://example.invalid/v1/chat/completions"


@pytest.mark.asyncio
async def test_cancel_request_records_bytes_received() -> None:
    response = _make_mock_response(
        chunks=[b"abc", b"defg", b"hi"],
        delay_per_chunk=0.01,
    )
    session = _make_mock_session(response)
    spec = FaultSpec(
        fault_id="client.cancel_request",
        target={"url": "http://example.invalid/v1/chat/completions"},
        params={
            "payload": {"model": "x"},
            # Long enough to fully drain the small fixture stream before the
            # cancel deadline; we only care that bytes_received is populated.
            "cancel_after_seconds": 5.0,
        },
    )

    with patch("aiohttp.ClientSession", return_value=session):
        handle = await ClientInjector().inject(spec)

    assert handle.metadata["bytes_received"] >= 0
    # If the stream finished naturally, all three chunks were read.
    assert handle.metadata["bytes_received"] == len(b"abcdefghi")


@pytest.mark.asyncio
async def test_cancel_request_connection_refused_raises_mechanism_error() -> None:
    session = MagicMock()

    class _FailingPostCM:
        async def __aenter__(self) -> Any:
            raise aiohttp.ClientConnectionError("connection refused")

        async def __aexit__(self, *_: Any) -> None:
            return None

    session.post = MagicMock(return_value=_FailingPostCM())
    session.close = AsyncMock(return_value=None)

    spec = FaultSpec(
        fault_id="client.cancel_request",
        target={"url": "http://example.invalid/v1/chat/completions"},
        params={
            "payload": {"model": "x"},
            "cancel_after_seconds": 0.05,
        },
    )

    with (
        patch("aiohttp.ClientSession", return_value=session),
        pytest.raises(FaultMechanismError, match="could not establish connection"),
    ):
        await ClientInjector().inject(spec)

    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_overflow_tokens_posts_payload_at_size() -> None:
    capture: dict[str, Any] = {}
    response = _make_mock_response(
        status=413,
        headers={"Content-Type": "application/json"},
        body=b'{"error":"payload too large"}',
    )
    session = _make_mock_session(response, capture=capture)
    target_size = 64 * 1024
    spec = FaultSpec(
        fault_id="client.overflow_tokens",
        target={"url": "http://example.invalid/v1/completions"},
        params={
            "payload_template": {"model": "x", "max_tokens": 1},
            "payload_size_bytes": target_size,
        },
    )

    with patch("aiohttp.ClientSession", return_value=session):
        await ClientInjector().inject(spec)

    assert capture["url"] == "http://example.invalid/v1/completions"
    sent = capture["json"]
    assert sent["model"] == "x"
    # The oversized field landed in the default "prompt" key at exactly the
    # requested length (the test asserts approximate size by asserting equal
    # length on the synthesized field).
    assert len(sent["prompt"]) == target_size


@pytest.mark.asyncio
async def test_overflow_tokens_records_response_status_in_metadata() -> None:
    response = _make_mock_response(
        status=413,
        headers={"Content-Type": "application/json", "X-Limit-MB": "45"},
        body=b'{"error":"payload too large"}',
    )
    session = _make_mock_session(response)
    spec = FaultSpec(
        fault_id="client.overflow_tokens",
        target={"url": "http://example.invalid/v1/completions"},
        params={
            "payload_template": {"model": "x"},
            "payload_size_bytes": 1024,
            "prompt_field": "messages",
        },
    )

    with patch("aiohttp.ClientSession", return_value=session):
        handle = await ClientInjector().inject(spec)

    assert handle.metadata["status"] == 413
    assert handle.metadata["headers"]["X-Limit-MB"] == "45"
    assert "payload too large" in handle.metadata["body_preview"]
    assert handle.metadata["prompt_field"] == "messages"
    assert handle.metadata["payload_size_bytes"] == 1024


@pytest.mark.asyncio
async def test_missing_url_raises_precondition() -> None:
    spec = FaultSpec(
        fault_id="client.cancel_request",
        target={},
        params={
            "payload": {"model": "x"},
            "cancel_after_seconds": 0.05,
        },
    )

    with pytest.raises(FaultPreconditionError, match="url"):
        await ClientInjector().inject(spec)


def test_handles_prefix_match_client() -> None:
    assert ClientInjector.handles("client")
    assert ClientInjector.handles("client.cancel_request")
    assert ClientInjector.handles("client.overflow_tokens")
    assert not ClientInjector.handles("network")
    assert not ClientInjector.handles("pod")
    # Guard against accidental prefix-extension matches.
    assert not ClientInjector.handles("clientfoo")

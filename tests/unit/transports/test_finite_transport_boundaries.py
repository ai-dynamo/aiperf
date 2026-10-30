# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP invocation and body EOF are separate from the last SSE event."""

import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from pytest import param

from aiperf.transports.aiohttp_client import AioHttpClient
from tests.unit.transports.conftest import create_mock_response, setup_mock_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        param("text/event-stream", id="sse"),
        param("application/octet-stream", id="binary"),
        param("application/json", id="text"),
    ],
)  # fmt: skip
@pytest.mark.parametrize(
    ("at_eof", "callback_fails", "error_type"),
    [
        param(True, False, None, id="complete"),
        param(False, False, "ResponseBodyIncompleteError", id="incomplete"),
        param(True, True, "RuntimeError", id="callback-failure"),
    ],
)  # fmt: skip
async def test_response_body_eof_preserves_callback_order_and_errors(
    aiohttp_client: AioHttpClient,
    content_type: str,
    at_eof: bool,
    callback_fails: bool,
    error_type: str | None,
) -> None:
    response = create_mock_response(content_type=content_type)
    response.content.at_eof = Mock(return_value=at_eof)
    response.read = AsyncMock(return_value=b"{}")
    callback = Mock(
        side_effect=RuntimeError("EOF observer failed") if callback_fails else None
    )

    async def chunks() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'

    async def decode_text() -> str:
        callback.assert_called_once()
        return "{}"

    response.content.iter_any = Mock(side_effect=chunks)
    response.text.side_effect = decode_text

    with patch("aiohttp.ClientSession") as session:
        setup_mock_session(session, response, ["request"])
        record = await aiohttp_client.post_request(
            "http://test.com/api", b"{}", {}, transport_eof_callback=callback
        )

    if error_type is None:
        assert record.error is None
    else:
        assert record.error is not None
        assert record.error.type == error_type
        if not at_eof:
            assert record.error.code == 500
            assert "did not reach body EOF" in record.error.message

    if at_eof:
        callback.assert_called_once_with(record.response_body_eof_perf_ns)
        assert record.response_body_eof_perf_ns is not None
        assert record.end_perf_ns >= record.response_body_eof_perf_ns
    else:
        callback.assert_not_called()
        assert record.response_body_eof_perf_ns is None

    if content_type == "application/json":
        response.read.assert_awaited_once()
        assert response.text.await_count == (error_type is None)
    elif content_type == "application/octet-stream":
        response.read.assert_awaited_once()


@pytest.mark.asyncio
async def test_sse_eof_is_captured_after_done_and_exported_separately() -> None:
    done_sent = asyncio.Event()
    release_eof = asyncio.Event()
    server_received_at: list[int] = []

    async def stream(request: web.Request) -> web.StreamResponse:
        server_received_at.append(time.perf_counter_ns())
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        done_sent.set()
        await release_eof.wait()
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = AioHttpClient()
    starts: list[int] = []
    eofs: list[int] = []
    try:
        request = asyncio.create_task(
            client.post_request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                b"{}",
                {"Content-Type": "application/json"},
                transport_start_callback=starts.append,
                transport_eof_callback=eofs.append,
            )
        )
        await asyncio.wait_for(done_sent.wait(), timeout=2)
        assert not eofs
        asyncio.get_running_loop().call_later(0.06, release_eof.set)
        record = await asyncio.wait_for(request, timeout=2)
        assert record.error is None
        assert len(starts) == len(eofs) == 1
        assert starts[0] <= server_received_at[0]
        assert record.response_body_eof_perf_ns == eofs[0]
        assert record.end_perf_ns >= eofs[0]
        assert (
            eofs[0] - max(message.perf_ns for message in record.responses) >= 30_000_000
        )
    finally:
        release_eof.set()
        await client.close()
        await runner.cleanup()

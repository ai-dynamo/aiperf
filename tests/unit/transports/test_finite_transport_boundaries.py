# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP invocation and body EOF are separate from the last SSE event."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import ClientTimeout, web
from pytest import param

from aiperf.common.models import RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient
from tests.unit.transports.conftest import create_mock_response, setup_mock_session


@asynccontextmanager
async def _serve_http(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/"
    finally:
        await runner.cleanup()


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
@pytest.mark.parametrize("observe", [param(False, id="ordinary"), param(True, id="finite")])  # fmt: skip
async def test_sse_eof_is_captured_after_done_and_exported_separately(
    observe: bool,
) -> None:
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

    client = AioHttpClient()
    starts: list[int] = []
    eofs: list[int] = []
    try:
        async with _serve_http(stream) as url:
            request = asyncio.create_task(
                client.post_request(
                    url,
                    b"{}",
                    {"Content-Type": "application/json"},
                    transport_start_callback=starts.append if observe else None,
                    transport_eof_callback=eofs.append if observe else None,
                )
            )
            try:
                await asyncio.wait_for(done_sent.wait(), timeout=2)
                assert not request.done()
                assert not eofs
                asyncio.get_running_loop().call_later(0.06, release_eof.set)
                record = await asyncio.wait_for(request, timeout=2)
            finally:
                release_eof.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
        assert record.error is None
        if observe:
            assert len(starts) == len(eofs) == 1
            assert starts[0] <= server_received_at[0]
            assert record.response_body_eof_perf_ns == eofs[0]
        else:
            assert not starts and not eofs
        assert record.response_body_eof_perf_ns is not None
        assert record.end_perf_ns >= record.response_body_eof_perf_ns
        assert (
            record.response_body_eof_perf_ns
            - max(message.perf_ns for message in record.responses)
            >= 30_000_000
        )
    finally:
        release_eof.set()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [param(200, id="success"), param(400, id="client-error"), param(500, id="server-error")])  # fmt: skip
@pytest.mark.parametrize("observe", [param(False, id="ordinary"), param(True, id="finite")])  # fmt: skip
@pytest.mark.parametrize("content_type", [param("application/json", id="text"), param("application/octet-stream", id="binary")])  # fmt: skip
async def test_fully_read_http_body_records_eof_independently_of_success(
    aiohttp_client: AioHttpClient, status: int, observe: bool, content_type: str
) -> None:
    body_started = asyncio.Event()
    release_eof = asyncio.Event()
    callback = Mock()

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=status, headers={"Content-Type": content_type}
        )
        await response.prepare(request)
        await response.write(b'{"message":')
        body_started.set()
        await release_eof.wait()
        await response.write(b'"complete"}')
        await response.write_eof()
        return response

    async with _serve_http(stream) as url:
        task = asyncio.create_task(
            aiohttp_client.post_request(
                url, b"{}", {}, transport_eof_callback=callback if observe else None
            )
        )
        try:
            await asyncio.wait_for(body_started.wait(), timeout=2)
            assert not task.done()
            callback.assert_not_called()
            released_at = time.perf_counter_ns()
            release_eof.set()
            record = await asyncio.wait_for(task, timeout=2)
        finally:
            release_eof.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert record.status == status
    assert released_at <= record.response_body_eof_perf_ns <= record.end_perf_ns
    if status == 200:
        assert record.error is None
        if observe:
            callback.assert_called_once_with(record.response_body_eof_perf_ns)
    else:
        assert record.error.code == status
        assert record.error.message == '{"message":"complete"}'
        callback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [param("timeout", id="timeout"), param("cancel", id="cancel"), param("truncated", id="connection-lost")])  # fmt: skip
@pytest.mark.parametrize("status", [param(200, id="success-status"), param(500, id="error-status")])  # fmt: skip
async def test_http_body_interrupted_before_eof_has_no_completion_observation(
    aiohttp_client: AioHttpClient, outcome: str, status: int
) -> None:
    release_body = asyncio.Event()
    callback = Mock()
    if outcome == "timeout":
        aiohttp_client.timeout = ClientTimeout(total=0.05)

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=status,
            headers={"Content-Type": "application/json", "Content-Length": "1024"},
        )
        await response.prepare(request)
        await response.write(b'{"partial":')
        if outcome == "truncated":
            request.transport.close()
        else:
            await release_body.wait()
        return response

    async with _serve_http(stream) as url:
        try:
            record = await asyncio.wait_for(
                aiohttp_client.post_request(
                    url,
                    b"{}",
                    {},
                    cancel_after_ns=50_000_000 if outcome == "cancel" else None,
                    transport_eof_callback=callback,
                ),
                timeout=2,
            )
        finally:
            release_body.set()

    assert record.error is not None
    assert record.response_body_eof_perf_ns is None
    callback.assert_not_called()
    if outcome == "cancel":
        assert record.error.type == "RequestCancellationError"


@pytest.mark.asyncio
async def test_cancellation_after_eof_retains_the_transport_observation(
    aiohttp_client: AioHttpClient,
) -> None:
    observed: list[int] = []

    async def finish_body_then_block(
        *args: object,
        record: RequestRecord,
        on_request_sent: asyncio.Event,
        **kwargs: object,
    ) -> RequestRecord:
        record.response_body_eof_perf_ns = time.perf_counter_ns()
        observed.append(record.response_body_eof_perf_ns)
        on_request_sent.set()
        await asyncio.Event().wait()
        return record

    with patch.object(aiohttp_client, "_request", side_effect=finish_body_then_block):
        record = await aiohttp_client.post_request(
            "http://test.com/", b"{}", {}, cancel_after_ns=1_000_000
        )

    assert record.error.type == "RequestCancellationError"
    assert record.response_body_eof_perf_ns == observed[0]

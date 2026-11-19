# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP invocation and body EOF are separate from the last SSE event."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from aiohttp import ClientTimeout, web
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    AioHttpTraceData,
    ParsedResponseRecord,
    RecordContext,
    RequestRecord,
)
from aiperf.common.utils import compute_time_ns
from aiperf.post_processors.raw_record_writer_processor import RawRecordWriterProcessor
from aiperf.records.record_processor_service import RecordProcessor
from aiperf.transports.aiohttp_client import AioHttpClient, _record_response_body_eof
from aiperf.transports.aws.eventstream import EVENTSTREAM_CONTENT_TYPE
from tests.unit.transports.conftest import create_mock_response, setup_mock_session


@asynccontextmanager
async def _serve_http(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    """Serve one POST handler on an ephemeral localhost port and close it on exit."""
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


@pytest.mark.parametrize("observe", [param(False, id="ordinary"), param(True, id="finite")])  # fmt: skip
def test_record_response_body_eof_existing_timestamp_is_preserved(
    observe: bool,
) -> None:
    """Repeated EOF observation retains the original time, including callback delivery."""
    response = create_mock_response()
    record = RequestRecord(response_body_eof_perf_ns=123)
    callback = Mock()

    _record_response_body_eof(response, record, callback if observe else None)

    assert record.response_body_eof_perf_ns == 123
    if observe:
        callback.assert_called_once_with(123)
    else:
        callback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body", "error_type", "error_message"),
    [
        param("text/event-stream", b"event: error\ndata: failure", "SSEResponseError", "Unknown error", id="sse-named-error"),
        param("text/event-stream", b'data: {"error": {"message": "model failed"}}', "SSEResponseError", "model failed", id="sse-data-error"),
        param(EVENTSTREAM_CONTENT_TYPE, b"\x00", "AwsEventStreamError", "1 trailing byte(s)", id="aws-incomplete-frame"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("observe", [param(False, id="ordinary"), param(True, id="finite")])  # fmt: skip
@pytest.mark.parametrize("first_token", [param(False, id="fast-path"), param(True, id="first-token-path")])  # fmt: skip
async def test_stream_error_after_http_eof_retains_timestamp_and_exports(
    aiohttp_client: AioHttpClient,
    content_type: str,
    body: bytes,
    error_type: str,
    error_message: str,
    observe: bool,
    first_token: bool,
) -> None:
    """Fully read SSE/AWS failures export EOF and errors without releasing replay dependencies."""
    callback = Mock()
    token_callback = AsyncMock(return_value=True)
    eof_sent: list[int] = []

    async def stream(request: web.Request) -> web.StreamResponse:
        """Finish HTTP framing normally while leaving an application error in the body."""
        response = web.StreamResponse(headers={"Content-Type": content_type})
        await response.prepare(request)
        await response.write(body)
        eof_sent.append(time.perf_counter_ns())
        await response.write_eof()
        return response

    async with _serve_http(stream) as url:
        record = await asyncio.wait_for(
            aiohttp_client.post_request(
                url,
                b"{}",
                {},
                first_token_callback=token_callback if first_token else None,
                transport_eof_callback=callback if observe else None,
            ),
            timeout=2,
        )

    assert record.status == 200
    assert record.error is not None
    assert record.error.type == error_type
    assert error_message in record.error.message
    assert record.response_body_eof_perf_ns is not None
    assert eof_sent[0] <= record.response_body_eof_perf_ns <= record.end_perf_ns
    callback.assert_not_called()
    token_callback.assert_not_awaited()

    record.request_info = RecordContext(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="stream-error",
        turn_index=0,
        x_request_id="request-0",
        x_correlation_id="credit-0",
    )
    metadata = RecordProcessor._create_metric_record_metadata(
        Mock(service_id="processor-0"), record, "worker-0"
    )
    raw = RawRecordWriterProcessor._build_export_record(
        Mock(spec=RawRecordWriterProcessor),
        ParsedResponseRecord(request=record, responses=[]),
        metadata,
    )
    exported = orjson.loads(raw.model_dump_json(exclude_none=True))
    assert exported["response_body_eof_perf_ns"] == record.response_body_eof_perf_ns
    assert exported["metadata"]["response_body_eof_ns"] == compute_time_ns(
        record.controller_timestamp_ns,
        record.start_perf_ns,
        record.response_body_eof_perf_ns,
    )
    assert exported["error"]["type"] == error_type


@pytest.mark.asyncio
@pytest.mark.parametrize("first_token", [param(False, id="fast-path"), param(True, id="first-token-path")])  # fmt: skip
async def test_stream_error_before_http_eof_does_not_drain_or_record_completion(
    aiohttp_client: AioHttpClient, first_token: bool
) -> None:
    """An early SSE error leaves EOF absent and stops before reading the remaining body."""
    response = create_mock_response(content_type="text/event-stream")
    response.content.at_eof.return_value = False
    response.read = AsyncMock()
    callback = Mock()
    token_callback = AsyncMock(return_value=True)
    remaining_body_read = Mock()

    async def chunks() -> AsyncIterator[bytes]:
        """Expose an error before a remaining chunk that must never be requested."""
        yield b'event: error\ndata: {"error": {"message": "model failed"}}\n\n'
        remaining_body_read()
        yield b"data: [DONE]\n\n"

    response.content.iter_any = Mock(side_effect=chunks)
    with patch("aiohttp.ClientSession") as session:
        setup_mock_session(session, response, ["request"])
        record = await aiohttp_client.post_request(
            "http://test.com/",
            b"{}",
            {},
            first_token_callback=token_callback if first_token else None,
            transport_eof_callback=callback,
        )

    assert record.error is not None
    assert record.error.type == "SSEResponseError"
    assert "model failed" in record.error.message
    assert record.response_body_eof_perf_ns is None
    callback.assert_not_called()
    token_callback.assert_not_awaited()
    remaining_body_read.assert_not_called()
    response.read.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("at_eof", [param(False, id="unfinished-body"), param(True, id="fully-read-body")])  # fmt: skip
async def test_stream_cancellation_records_only_observed_eof(
    aiohttp_client: AioHttpClient, at_eof: bool
) -> None:
    """Cancellation during a token callback retains observed EOF without releasing replay."""
    response = create_mock_response(content_type="text/event-stream")
    response.content.at_eof.return_value = at_eof
    record = RequestRecord(trace_data=AioHttpTraceData())
    callback = Mock()

    async def chunks() -> AsyncIterator[bytes]:
        """Deliver a valid message before cancellation interrupts stream processing."""
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'

    response.content.iter_any = Mock(side_effect=chunks)
    with patch("aiohttp.ClientSession") as session:
        setup_mock_session(session, response, ["request"])
        with pytest.raises(asyncio.CancelledError):
            await aiohttp_client._request(
                "POST",
                "http://test.com/",
                {},
                record=record,
                first_token_callback=AsyncMock(side_effect=asyncio.CancelledError),
                transport_eof_callback=callback,
            )

    assert record.error.type == "RequestCancellationError"
    assert (record.response_body_eof_perf_ns is not None) == at_eof
    if at_eof:
        assert (
            record.start_perf_ns
            <= record.response_body_eof_perf_ns
            <= record.end_perf_ns
        )
    callback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "first_token"),
    [
        param("text/event-stream", False, id="sse-fast-path"),
        param("text/event-stream", True, id="sse-first-token-path"),
        param("application/octet-stream", False, id="binary"),
        param("application/json", False, id="text"),
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
    first_token: bool,
) -> None:
    """Successful bodies notify once; incomplete bodies and callback failures retain their errors."""
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
            "http://test.com/api",
            b"{}",
            {},
            first_token_callback=AsyncMock(return_value=True) if first_token else None,
            transport_eof_callback=callback,
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
@pytest.mark.parametrize("content_type", [param("application/json", id="text"), param("text/event-stream", id="sse")])  # fmt: skip
async def test_http_body_interrupted_before_eof_has_no_completion_observation(
    aiohttp_client: AioHttpClient, outcome: str, status: int, content_type: str
) -> None:
    """Timeout, cancellation, and connection loss leave unfinished text/SSE bodies unstamped."""
    release_body = asyncio.Event()
    callback = Mock()
    if outcome == "timeout":
        aiohttp_client.timeout = ClientTimeout(total=0.05)

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=status,
            headers={"Content-Type": content_type, "Content-Length": "1024"},
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

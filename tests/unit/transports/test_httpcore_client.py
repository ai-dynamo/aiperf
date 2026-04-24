# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HttpCoreClient SSE handling, first_token_callback, and error inspection."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ConnectionReuseStrategy, SSEEventType, SSEFieldType
from aiperf.common.models import ErrorDetails, SSEField, SSEMessage
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.plugin.enums import EndpointType
from aiperf.transports.httpcore_client import HttpCoreClient
from aiperf.transports.httpcore_transport import HttpCoreTransport
from aiperf.transports.sse_utils import AsyncSSEStreamReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> HttpCoreClient:
    """Return a HttpCoreClient with a mocked pool (no real TCP connections)."""
    with patch("httpcore.AsyncConnectionPool"):
        client = HttpCoreClient(timeout=30.0)
    return client


def make_sse_response(
    status: int = 200,
    content_type: str = "text/event-stream",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> MagicMock:
    """Build a minimal mock httpcore response for SSE streams."""
    if headers is None:
        headers = [
            (b"content-type", content_type.encode()),
        ]
    response = MagicMock()
    response.status = status
    response.headers = headers
    response.extensions = {"http_version": b"2"}
    return response


def make_non_sse_response(
    status: int = 200,
    body: bytes = b'{"ok": true}',
    content_type: str = "application/json",
) -> MagicMock:
    """Build a minimal mock httpcore response for plain JSON responses."""
    headers = [(b"content-type", content_type.encode())]
    response = MagicMock()
    response.status = status
    response.headers = headers
    response.extensions = {"http_version": b"2"}

    async def _iter_stream():
        yield body

    response.aiter_stream = _iter_stream
    return response


def build_stream_context(response: MagicMock, sse_chunks: list[bytes]):
    """Return an async context manager that yields *response* and sets aiter_stream."""

    async def _iter_stream():
        for chunk in sse_chunks:
            yield chunk

    response.aiter_stream = _iter_stream

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield response

    return _ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def httpcore_client() -> HttpCoreClient:
    """Provide a fresh HttpCoreClient with a mocked pool."""
    return make_client()


@pytest.fixture
def sse_headers() -> dict[str, str]:
    """Standard SSE request headers."""
    return {"Accept": "text/event-stream", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Basic SSE parsing (no callback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHttpCoreClientSSEBasic:
    """Verify basic SSE parsing works without a callback."""

    async def test_sse_messages_collected_without_callback(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """SSE messages are collected into record.responses when no callback supplied."""
        sse_chunks = [b"data: hello\n\n", b"data: world\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b'{"prompt": "hi"}',
                sse_headers,
            )

        assert record.error is None
        assert len(record.responses) == 2
        assert all(isinstance(m, SSEMessage) for m in record.responses)

    async def test_non_sse_response_returns_text_response(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """Non-SSE responses are stored as a single TextResponse."""
        from aiperf.common.models import TextResponse

        body = b'{"result": "ok"}'
        response = make_non_sse_response(body=body)

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.error is None
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert record.responses[0].text == '{"result": "ok"}'


# ---------------------------------------------------------------------------
# first_token_callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFirstTokenCallback:
    """Verify first_token_callback fires with correct arguments."""

    async def test_callback_receives_ttft_ns_and_message(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """Callback is called with (ttft_ns: int, message: SSEMessage)."""
        received: list[tuple[int, SSEMessage]] = []

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            received.append((ttft_ns, message))
            return True  # acquired on first call

        sse_chunks = [b"data: token1\n\n", b"data: token2\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=callback,
            )

        # Callback fired exactly once (returned True on first call)
        assert len(received) == 1
        ttft_ns, message = received[0]
        assert isinstance(ttft_ns, int)
        assert ttft_ns >= 0
        assert isinstance(message, SSEMessage)
        # Both messages still collected into record
        assert len(record.responses) == 2
        assert record.error is None

    async def test_callback_ttft_ns_is_non_negative(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """ttft_ns passed to callback must be >= 0 (message.perf_ns - start_perf_ns)."""
        ttft_values: list[int] = []

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            ttft_values.append(ttft_ns)
            return True

        sse_chunks = [b"data: first\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=callback,
            )

        assert len(ttft_values) == 1
        assert ttft_values[0] >= 0

    async def test_callback_not_called_when_none(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """When callback is None, all messages are still collected (fast path)."""
        sse_chunks = [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=None,
            )

        assert record.error is None
        assert len(record.responses) == 3

    async def test_callback_stops_being_called_after_returning_true(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """Callback is not invoked again once it returns True."""
        call_count = 0

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal call_count
            call_count += 1
            return True  # acquired immediately

        sse_chunks = [b"data: t1\n\n", b"data: t2\n\n", b"data: t3\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=callback,
            )

        assert call_count == 1
        assert len(record.responses) == 3

    async def test_callback_continues_until_returns_true(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """Callback keeps being called while it returns False."""
        return_values = [False, False, True]
        call_count = 0

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal call_count
            result = return_values[call_count]
            call_count += 1
            return result

        sse_chunks = [
            b"data: t1\n\n",
            b"data: t2\n\n",
            b"data: t3\n\n",
            b"data: t4\n\n",
        ]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=callback,
            )

        assert call_count == 3
        assert len(record.responses) == 4


# ---------------------------------------------------------------------------
# SSE error event inspection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSSEErrorInspection:
    """Verify SSE error events are caught via inspect_message_for_error."""

    async def test_sse_error_event_caught_without_callback(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """SSE error events without callback are caught and stored in record.error."""
        # Build raw SSE bytes that parse to an error event with a comment
        sse_chunks = [b"event: error\n: rate limit exceeded\ndata: {}\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        assert record.error is not None
        assert record.error.code == 502
        assert record.error.type == "SSEResponseError"
        assert "rate limit exceeded" in record.error.message

    async def test_sse_error_event_caught_with_callback(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """SSE error events with callback are caught and stored in record.error."""
        callback_calls: list[int] = []

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            callback_calls.append(ttft_ns)
            return True

        sse_chunks = [b"event: error\n: backend error\ndata: {}\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
                first_token_callback=callback,
            )

        assert record.error is not None
        assert record.error.type == "SSEResponseError"
        assert "backend error" in record.error.message

    @pytest.mark.parametrize(
        "comment_value,expected_text",
        [
            ("Rate limit exceeded", "Rate limit exceeded"),
            (None, "Unknown error in SSE response"),
        ],
    )  # fmt: skip
    async def test_sse_error_message_content(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
        comment_value: str | None,
        expected_text: str,
    ) -> None:
        """Error message is the comment if present, otherwise a generic fallback."""
        packets = [SSEField(name=SSEFieldType.EVENT, value=SSEEventType.ERROR)]
        if comment_value:
            packets.append(SSEField(name=SSEFieldType.COMMENT, value=comment_value))
        packets.append(SSEField(name=SSEFieldType.DATA, value="{}"))
        error_message = SSEMessage(perf_ns=100_000_000, packets=packets)

        response = make_sse_response()

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        async def _mock_iter():
            yield b""  # never actually used; reader is patched

        response.aiter_stream = _mock_iter

        with (
            patch.object(httpcore_client.pool, "stream", _ctx),
            patch(
                "aiperf.transports.httpcore_client.AsyncSSEStreamReader"
            ) as mock_reader_class,
        ):

            async def _aiter():
                yield error_message
                AsyncSSEStreamReader.inspect_message_for_error(error_message)

            mock_reader = MagicMock()
            mock_reader.__aiter__ = MagicMock(return_value=_aiter())
            mock_reader_class.return_value = mock_reader

            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        assert record.error is not None
        assert record.error.code == 502
        assert record.error.type == "SSEResponseError"
        assert expected_text in record.error.message

    async def test_successful_sse_then_error_message(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """A good SSE message followed by an error event: responses has one item, error is set."""
        sse_chunks = [
            b"data: good token\n\n",
            b"event: error\n: quota exceeded\ndata: {}\n\n",
        ]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        # First message was appended before the error was raised
        assert len(record.responses) == 1
        assert record.error is not None
        assert "quota exceeded" in record.error.message


# ---------------------------------------------------------------------------
# HTTP error responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHttpCoreClientHTTPErrors:
    """Verify non-200 HTTP responses are handled correctly."""

    @pytest.mark.parametrize(
        "status_code",
        [400, 401, 404, 500, 503],
    )  # fmt: skip
    async def test_http_error_stored_in_record(
        self,
        httpcore_client: HttpCoreClient,
        status_code: int,
    ) -> None:
        """Non-200 status codes are recorded as ErrorDetails."""
        error_body = b"something went wrong"
        response = MagicMock()
        response.status = status_code
        response.headers = []
        response.extensions = {"http_version": b"1.1"}

        async def _iter_stream():
            yield error_body

        response.aiter_stream = _iter_stream

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/api",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.error is not None
        assert record.error.code == status_code
        assert record.status == status_code


# ---------------------------------------------------------------------------
# Binary response handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBinaryResponseHandling:
    """Verify binary content types return BinaryResponse instead of TextResponse."""

    @pytest.mark.parametrize(
        "content_type",
        [
            "video/mp4",
            "image/png",
            "audio/wav",
            "application/octet-stream",
        ],
    )  # fmt: skip
    async def test_binary_content_types_return_binary_response(
        self,
        httpcore_client: HttpCoreClient,
        content_type: str,
    ) -> None:
        """Binary content types yield a BinaryResponse with the raw bytes."""
        from aiperf.common.models import BinaryResponse

        raw_data = b"\x00\x01\x02\x03\xff"
        response = make_non_sse_response(body=raw_data, content_type=content_type)

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/generate",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.error is None
        assert len(record.responses) == 1
        resp = record.responses[0]
        assert isinstance(resp, BinaryResponse)
        assert resp.raw_bytes == raw_data
        assert resp.content_type == content_type

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json",
            "text/plain",
            "text/html",
            "application/xml",
        ],
    )  # fmt: skip
    async def test_text_content_types_return_text_response(
        self,
        httpcore_client: HttpCoreClient,
        content_type: str,
    ) -> None:
        """Non-binary content types still yield a TextResponse."""
        from aiperf.common.models import TextResponse

        body = b'{"ok": true}'
        response = make_non_sse_response(body=body, content_type=content_type)

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.error is None
        assert len(record.responses) == 1
        resp = record.responses[0]
        assert isinstance(resp, TextResponse)
        assert resp.text == '{"ok": true}'
        assert resp.content_type == content_type

    async def test_binary_response_preserves_exact_bytes(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """BinaryResponse raw_bytes is an exact copy of the response body."""
        from aiperf.common.models import BinaryResponse

        # Simulate arbitrary binary payload (e.g. video frame header)
        raw_data = bytes(range(256))
        response = make_non_sse_response(body=raw_data, content_type="video/mp4")

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/video",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert isinstance(record.responses[0], BinaryResponse)
        assert record.responses[0].raw_bytes == raw_data
        assert len(record.responses[0].raw_bytes) == 256


# ---------------------------------------------------------------------------
# Trace data population
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHttpCoreTraceData:
    """Verify BaseTraceData is populated on both SSE and non-SSE paths."""

    async def test_trace_data_not_none_non_sse(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """record.trace_data is set after a non-SSE request."""
        response = make_non_sse_response(body=b'{"ok": true}')

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.trace_data is not None

    async def test_trace_type_is_httpcore_non_sse(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """trace_data.trace_type is 'httpcore' for non-SSE responses."""
        response = make_non_sse_response(body=b'{"ok": true}')

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.trace_data.trace_type == "httpcore"

    async def test_trace_data_timing_populated_non_sse(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """response_receive_start/end_perf_ns are populated for non-SSE responses."""
        response = make_non_sse_response(body=b'{"result": "data"}')

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.trace_data.response_receive_start_perf_ns is not None
        assert record.trace_data.response_receive_end_perf_ns is not None

    async def test_trace_data_bytes_and_chunks_non_sse(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """response_bytes_total > 0 and response_chunks_count > 0 for non-SSE."""
        body = b'{"result": "data"}'
        response = make_non_sse_response(body=body)

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
            )

        assert record.trace_data.response_bytes_total == len(body)
        assert record.trace_data.response_chunks_count == 1

    async def test_trace_data_not_none_sse(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """record.trace_data is set after an SSE request."""
        sse_chunks = [b"data: hello\n\n", b"data: world\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        assert record.trace_data is not None

    async def test_trace_type_is_httpcore_sse(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """trace_data.trace_type is 'httpcore' for SSE responses."""
        sse_chunks = [b"data: hello\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        assert record.trace_data.trace_type == "httpcore"

    async def test_trace_data_timing_populated_sse(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """response_receive_start/end_perf_ns are populated for SSE responses."""
        sse_chunks = [b"data: hello\n\n", b"data: world\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        assert record.trace_data.response_receive_start_perf_ns is not None
        assert record.trace_data.response_receive_end_perf_ns is not None

    async def test_trace_data_bytes_and_chunks_sse(
        self,
        httpcore_client: HttpCoreClient,
        sse_headers: dict[str, str],
    ) -> None:
        """response_bytes_total > 0 and response_chunks_count > 0 for SSE responses."""
        sse_chunks = [b"data: hello\n\n", b"data: world\n\n"]
        response = make_sse_response()
        ctx = build_stream_context(response, sse_chunks)

        with patch.object(httpcore_client.pool, "stream", ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/stream",
                b"{}",
                sse_headers,
            )

        expected_bytes = sum(len(c) for c in sse_chunks)
        assert record.trace_data.response_bytes_total == expected_bytes
        assert record.trace_data.response_chunks_count == len(sse_chunks)


# ---------------------------------------------------------------------------
# cancel_after_ns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancelAfterNs:
    """Verify cancel_after_ns causes cancellation and correct record fields."""

    async def test_cancel_after_ns_returns_499_error(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """cancel_after_ns causes the request to be cancelled with error code 499."""
        import asyncio
        from contextlib import asynccontextmanager

        response = MagicMock()
        response.status = 200
        response.headers = [(b"content-type", b"application/json")]
        response.extensions = {"http_version": b"2"}

        # Use an asyncio.Event that never resolves — not affected by looptime speedup
        never_resolves = asyncio.Event()

        async def _blocking_stream():
            await never_resolves.wait()
            yield b'{"ok": true}'

        response.aiter_stream = _blocking_stream

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
                cancel_after_ns=1,  # 1ns — fires immediately
            )

        assert record.error is not None
        assert record.error.code == 499
        assert record.error.type == "RequestCancellationError"

    async def test_cancel_after_ns_sets_cancellation_perf_ns(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """cancel_after_ns sets cancellation_perf_ns on the returned record."""
        import asyncio
        from contextlib import asynccontextmanager

        response = MagicMock()
        response.status = 200
        response.headers = [(b"content-type", b"application/json")]
        response.extensions = {"http_version": b"2"}

        never_resolves = asyncio.Event()

        async def _blocking_stream():
            await never_resolves.wait()
            yield b'{"ok": true}'

        response.aiter_stream = _blocking_stream

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
                cancel_after_ns=1,  # 1ns — fires immediately
            )

        assert record.cancellation_perf_ns is not None
        assert record.cancellation_perf_ns > 0
        assert record.end_perf_ns is not None
        assert record.cancellation_perf_ns == record.end_perf_ns

    async def test_cancel_after_ns_none_completes_normally(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """Without cancel_after_ns, requests complete normally."""
        from aiperf.common.models import TextResponse

        body = b'{"result": "ok"}'
        response = make_non_sse_response(body=body)

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
                cancel_after_ns=None,
            )

        assert record.error is None
        assert record.cancellation_perf_ns is None
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)

    async def test_cancel_after_ns_error_message_contains_type(
        self,
        httpcore_client: HttpCoreClient,
    ) -> None:
        """Cancellation error message references RequestCancellationError."""
        import asyncio
        from contextlib import asynccontextmanager

        response = MagicMock()
        response.status = 200
        response.headers = [(b"content-type", b"application/json")]
        response.extensions = {"http_version": b"2"}

        never_resolves = asyncio.Event()

        async def _blocking_stream():
            await never_resolves.wait()
            yield b'{"ok": true}'

        response.aiter_stream = _blocking_stream

        @asynccontextmanager
        async def _ctx(*args, **kwargs):
            yield response

        with patch.object(httpcore_client.pool, "stream", _ctx):
            record = await httpcore_client.post_request(
                "http://test.example.com/v1/complete",
                b"{}",
                {"Content-Type": "application/json"},
                cancel_after_ns=1,
            )

        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert "cancelled" in record.error.message.lower()


# ---------------------------------------------------------------------------
# create_ephemeral
# ---------------------------------------------------------------------------


class TestCreateEphemeral:
    """Verify HttpCoreClient.create_ephemeral() builds a minimal single-connection pool."""

    def test_create_ephemeral_returns_httpcore_client(self) -> None:
        """create_ephemeral() returns an HttpCoreClient instance."""
        with patch("httpcore.AsyncConnectionPool"):
            client = HttpCoreClient.create_ephemeral(timeout=10.0)

        assert isinstance(client, HttpCoreClient)

    def test_create_ephemeral_uses_single_connection(self) -> None:
        """create_ephemeral() creates pool with max_connections=1."""
        with patch("httpcore.AsyncConnectionPool") as mock_pool_cls:
            HttpCoreClient.create_ephemeral(timeout=10.0)

        kwargs = mock_pool_cls.call_args[1]
        assert kwargs["max_connections"] == 1

    def test_create_ephemeral_disables_keepalive(self) -> None:
        """create_ephemeral() sets max_keepalive_connections=0 and keepalive_expiry=0."""
        with patch("httpcore.AsyncConnectionPool") as mock_pool_cls:
            HttpCoreClient.create_ephemeral(timeout=5.0)

        kwargs = mock_pool_cls.call_args[1]
        assert kwargs["max_keepalive_connections"] == 0
        assert kwargs["keepalive_expiry"] == 0

    def test_create_ephemeral_no_retries(self) -> None:
        """create_ephemeral() disables retries."""
        with patch("httpcore.AsyncConnectionPool") as mock_pool_cls:
            HttpCoreClient.create_ephemeral()

        kwargs = mock_pool_cls.call_args[1]
        assert kwargs["retries"] == 0

    def test_create_ephemeral_sets_timeout(self) -> None:
        """create_ephemeral() stores the given timeout on the instance."""
        with patch("httpcore.AsyncConnectionPool"):
            client = HttpCoreClient.create_ephemeral(timeout=42.0)

        assert client.timeout_seconds == 42.0

    def test_create_ephemeral_default_timeout(self) -> None:
        """create_ephemeral() defaults to 300s when timeout is None."""
        with patch("httpcore.AsyncConnectionPool"):
            client = HttpCoreClient.create_ephemeral()

        assert client.timeout_seconds == 300.0


# ---------------------------------------------------------------------------
# Connection reuse strategies in HttpCoreTransport
# ---------------------------------------------------------------------------


def _make_transport_run(
    reuse_strategy: ConnectionReuseStrategy = ConnectionReuseStrategy.POOLED,
) -> BenchmarkRun:
    """Build a minimal BenchmarkRun with the given connection reuse strategy."""
    cfg = BenchmarkConfig(
        models=["test-model"],
        endpoint={
            "type": EndpointType.CHAT,
            "urls": ["http://localhost:8000"],
            "path": "/v1/chat/completions",
            "streaming": False,
            "connection_reuse": reuse_strategy,
        },
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 1, "concurrency": 1}},
    )
    return BenchmarkRun(benchmark_id="test", cfg=cfg, artifact_dir=Path("/tmp/test"))


def _make_transport(run: BenchmarkRun) -> HttpCoreTransport:
    """Instantiate HttpCoreTransport with a mocked httpcore_client."""
    with patch("httpcore.AsyncConnectionPool"):
        transport = HttpCoreTransport(run=run)
        transport.httpcore_client = HttpCoreClient(timeout=30.0)
    return transport


def _make_request_info(run: BenchmarkRun) -> MagicMock:
    """Build a minimal RequestInfo mock."""
    ri = MagicMock()
    ri.cancel_after_ns = None
    ri.endpoint_headers = {}
    ri.config = run.cfg
    return ri


@pytest.mark.asyncio
class TestHttpCoreTransportConnectionReuse:
    """Verify connection reuse strategies are applied correctly in HttpCoreTransport."""

    async def test_pooled_strategy_uses_shared_client(self) -> None:
        """POOLED strategy calls post_request on the shared httpcore_client."""
        run = _make_transport_run(ConnectionReuseStrategy.POOLED)
        transport = _make_transport(run)

        record_mock = MagicMock()
        record_mock.request_headers = {}
        transport.httpcore_client.post_request = AsyncMock(return_value=record_mock)

        ri = _make_request_info(run)
        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
        ):
            await transport.send_request(ri, {"messages": []})

        transport.httpcore_client.post_request.assert_awaited_once()

    async def test_never_strategy_creates_and_closes_ephemeral_client(self) -> None:
        """NEVER strategy creates an ephemeral client and closes it after the request."""
        run = _make_transport_run(ConnectionReuseStrategy.NEVER)
        transport = _make_transport(run)

        record_mock = MagicMock()
        record_mock.request_headers = {}

        ephemeral_mock = MagicMock(spec=HttpCoreClient)
        ephemeral_mock.post_request = AsyncMock(return_value=record_mock)
        ephemeral_mock.close = AsyncMock()

        ri = _make_request_info(run)
        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
            patch(
                "aiperf.transports.httpcore_transport.HttpCoreClient.create_ephemeral",
                return_value=ephemeral_mock,
            ),
        ):
            await transport.send_request(ri, {"messages": []})

        ephemeral_mock.post_request.assert_awaited_once()
        ephemeral_mock.close.assert_awaited_once()

    async def test_never_strategy_closes_ephemeral_on_error(self) -> None:
        """NEVER strategy closes ephemeral client even if post_request raises."""
        run = _make_transport_run(ConnectionReuseStrategy.NEVER)
        transport = _make_transport(run)

        ephemeral_mock = MagicMock(spec=HttpCoreClient)
        ephemeral_mock.post_request = AsyncMock(side_effect=RuntimeError("boom"))
        ephemeral_mock.close = AsyncMock()

        ri = _make_request_info(run)
        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
            patch(
                "aiperf.transports.httpcore_transport.HttpCoreClient.create_ephemeral",
                return_value=ephemeral_mock,
            ),
        ):
            record = await transport.send_request(ri, {"messages": []})

        ephemeral_mock.close.assert_awaited_once()
        assert record.error is not None

    async def test_sticky_user_sessions_uses_per_session_client(self) -> None:
        """STICKY_USER_SESSIONS uses a dedicated client per session via lease manager."""
        run = _make_transport_run(ConnectionReuseStrategy.STICKY_USER_SESSIONS)
        transport = _make_transport(run)

        from aiperf.transports.httpcore_transport import HttpCoreLeaseManager

        transport.lease_manager = HttpCoreLeaseManager(timeout=30.0)

        record_mock = MagicMock()
        record_mock.request_headers = {}
        record_mock.cancellation_perf_ns = None
        record_mock.error = None

        ri = _make_request_info(run)
        ri.x_correlation_id = "session-1"
        ri.is_final_turn = False

        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
            patch.object(
                transport.lease_manager,
                "get_client",
                return_value=MagicMock(
                    post_request=AsyncMock(return_value=record_mock)
                ),
            ) as mock_get_client,
        ):
            await transport.send_request(ri, {"messages": []})

        mock_get_client.assert_called_once_with("session-1")

    async def test_sticky_user_sessions_releases_on_final_turn(self) -> None:
        """Lease is released when is_final_turn is True."""
        run = _make_transport_run(ConnectionReuseStrategy.STICKY_USER_SESSIONS)
        transport = _make_transport(run)

        from aiperf.transports.httpcore_transport import HttpCoreLeaseManager

        transport.lease_manager = HttpCoreLeaseManager(timeout=30.0)

        record_mock = MagicMock()
        record_mock.request_headers = {}
        record_mock.cancellation_perf_ns = None
        record_mock.error = None

        ri = _make_request_info(run)
        ri.x_correlation_id = "session-2"
        ri.is_final_turn = True

        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
            patch.object(
                transport.lease_manager,
                "get_client",
                return_value=MagicMock(
                    post_request=AsyncMock(return_value=record_mock)
                ),
            ),
            patch.object(
                transport.lease_manager, "release", new_callable=AsyncMock
            ) as mock_release,
        ):
            await transport.send_request(ri, {"messages": []})

        mock_release.assert_awaited_once_with("session-2")

    async def test_sticky_user_sessions_releases_on_error(self) -> None:
        """Lease is released when the request returns an error."""
        run = _make_transport_run(ConnectionReuseStrategy.STICKY_USER_SESSIONS)
        transport = _make_transport(run)

        from aiperf.transports.httpcore_transport import HttpCoreLeaseManager

        transport.lease_manager = HttpCoreLeaseManager(timeout=30.0)

        record_mock = MagicMock()
        record_mock.request_headers = {}
        record_mock.cancellation_perf_ns = None
        record_mock.error = ErrorDetails(type="TestError", message="fail")

        ri = _make_request_info(run)
        ri.x_correlation_id = "session-3"
        ri.is_final_turn = False

        with (
            patch.object(
                transport,
                "build_url",
                return_value="http://localhost:8000/v1/chat/completions",
            ),
            patch.object(transport, "build_headers", return_value={}),
            patch.object(
                transport.lease_manager,
                "get_client",
                return_value=MagicMock(
                    post_request=AsyncMock(return_value=record_mock)
                ),
            ),
            patch.object(
                transport.lease_manager, "release", new_callable=AsyncMock
            ) as mock_release,
        ):
            await transport.send_request(ri, {"messages": []})

        mock_release.assert_awaited_once_with("session-3")

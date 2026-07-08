# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for aiohttp client components."""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from aiperf.common.enums import SSEEventType, SSEFieldType
from aiperf.common.environment import Environment
from aiperf.common.models import SSEField, SSEMessage
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.http_defaults import AioHttpDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader
from tests.unit.transports.conftest import (
    MockStreamReader,
    assert_error_request_record,
    assert_successful_request_record,
    create_aiohttp_exception,
    create_mock_error_response,
    create_mock_response,
    setup_mock_session,
)


class TestAioHttpClient:
    """Test suite for AioHttpClient class."""

    def test_init_creates_connector_and_timeout(self) -> None:
        """Test that initialization creates TCP connector and timeout configurations."""
        with patch(
            "aiperf.transports.aiohttp_client.create_tcp_connector"
        ) as mock_create:
            mock_connector = Mock()
            mock_create.return_value = mock_connector

            client = AioHttpClient(timeout=600.0)

            assert client.tcp_connector == mock_connector
            assert isinstance(client.timeout, aiohttp.ClientTimeout)
            assert client.timeout.total == 600.0
            mock_create.assert_called_once()

    async def test_cleanup_closes_connector(
        self, aiohttp_client: AioHttpClient
    ) -> None:
        """Test that cleanup closes the TCP connector."""
        mock_connector = Mock()
        mock_connector.close = AsyncMock()
        aiohttp_client.tcp_connector = mock_connector

        await aiohttp_client.close()

        mock_connector.close.assert_called_once()
        assert aiohttp_client.tcp_connector is None

    async def test_cleanup_handles_none_connector(
        self, aiohttp_client: AioHttpClient
    ) -> None:
        """Test that cleanup handles None connector gracefully."""
        aiohttp_client.tcp_connector = None

        await aiohttp_client.close()

        assert aiohttp_client.tcp_connector is None

    @pytest.mark.asyncio
    async def test_successful_json_request(
        self, aiohttp_client: AioHttpClient, mock_aiohttp_response: Mock
    ) -> None:
        """Test successful JSON request handling."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_aiohttp_response, ["request"])

            record = await aiohttp_client.post_request(
                "http://test.com/api",
                '{"test": "data"}',
                {"Content-Type": "application/json"},
            )

            assert_successful_request_record(record)

    @pytest.mark.asyncio
    async def test_sse_stream_request(
        self, aiohttp_client: AioHttpClient, mock_sse_response: Mock
    ) -> None:
        """Test SSE stream request handling."""
        mock_messages = [
            SSEMessage(perf_ns=123456789),
            SSEMessage(perf_ns=123456790),
        ]

        with (
            patch("aiohttp.ClientSession") as mock_session_class,
            patch(
                "aiperf.transports.aiohttp_client.AsyncSSEStreamReader"
            ) as mock_reader_class,
        ):
            mock_sse_response.content = MockStreamReader([b"data: test\n\n"])
            setup_mock_session(mock_session_class, mock_sse_response, ["request"])

            async def mock_aiter():
                for msg in mock_messages:
                    yield msg

            mock_reader = Mock()
            mock_reader.__aiter__ = Mock(return_value=mock_aiter())
            mock_reader_class.return_value = mock_reader

            record = await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
            )

            assert_successful_request_record(
                record, expected_response_count=2, expected_response_type=SSEMessage
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "comment_value,expected_error_text",
        [
            ("Rate limit exceeded", "Rate limit exceeded"),
            (None, "Unknown error in SSE response"),
        ],
    )
    async def test_sse_stream_error_event_handling(
        self,
        aiohttp_client: AioHttpClient,
        mock_sse_response: Mock,
        comment_value: str | None,
        expected_error_text: str,
    ) -> None:
        """Test that SSE error events are properly caught and handled in the client."""

        packets = [
            SSEField(name=SSEFieldType.EVENT, value=SSEEventType.ERROR),
        ]
        if comment_value:
            packets.append(SSEField(name=SSEFieldType.COMMENT, value=comment_value))
        packets.append(SSEField(name=SSEFieldType.DATA, value="{}"))

        mock_error_message = SSEMessage(perf_ns=123456789, packets=packets)

        with (
            patch("aiohttp.ClientSession") as mock_session_class,
            patch(
                "aiperf.transports.aiohttp_client.AsyncSSEStreamReader"
            ) as mock_reader_class,
        ):
            chunks = [b"event: error\n"]
            if comment_value:
                chunks.append(f": {comment_value}\n".encode())
            chunks.append(b"data: {}\n\n")
            mock_sse_response.content = MockStreamReader(chunks)

            setup_mock_session(mock_session_class, mock_sse_response, ["request"])

            async def mock_aiter():
                yield mock_error_message

                AsyncSSEStreamReader.inspect_message_for_error(mock_error_message)

            mock_reader = Mock()
            mock_reader.__aiter__ = Mock(return_value=mock_aiter())
            mock_reader_class.return_value = mock_reader

            record = await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
            )

            assert record.error is not None
            assert record.error.code == 502
            assert record.error.type == "SSEResponseError"
            assert expected_error_text in record.error.message
            assert len(record.responses) == 1
            assert isinstance(record.responses[0], SSEMessage)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,reason,error_text",
        [
            (400, "Bad Request", "Invalid request format"),
            (401, "Unauthorized", "Authentication failed"),
            (404, "Not Found", "Resource not found"),
            (500, "Internal Server Error", "Server error occurred"),
            (503, "Service Unavailable", "Service temporarily unavailable"),
        ],
    )
    async def test_http_error_handling(
        self,
        aiohttp_client: AioHttpClient,
        status_code: int,
        reason: str,
        error_text: str,
    ) -> None:
        """Test HTTP error response handling."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_error_response(status_code, reason, error_text)
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_error_request_record(
                record,
                expected_error_code=status_code,
                expected_error_type=reason,
                expected_error_message=error_text,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_class,exception_message",
        [
            (aiohttp.ClientConnectionError, "Request timeout"),
            (ConnectionError, "Network connection failed"),
            (ValueError, "Invalid value provided"),
        ],
    )
    async def test_exception_handling(
        self,
        aiohttp_client: AioHttpClient,
        exception_class: type[Exception],
        exception_message: str,
    ) -> None:
        """Test various exception handling scenarios."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = exception_class(exception_message)

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_error_request_record(
                record,
                expected_error_type=exception_class.__name__,
                expected_error_message=exception_message,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_class,message,expected_type",
        [
            (aiohttp.ClientConnectorError, "Connection failed", "ClientConnectorError"),
            (aiohttp.ClientResponseError, "Internal Server Error", "ClientResponseError"),
        ],
    )  # fmt: skip
    async def test_aiohttp_specific_exceptions(
        self,
        aiohttp_client: AioHttpClient,
        exception_class: type[Exception],
        message: str,
        expected_type: str,
    ) -> None:
        """Test handling of aiohttp-specific exceptions."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            exception = create_aiohttp_exception(exception_class, message)
            mock_session_class.side_effect = exception

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_error_request_record(record, expected_error_type=expected_type)

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_session_post(
        self, aiohttp_client: AioHttpClient, mock_aiohttp_response: Mock
    ) -> None:
        """Test that additional kwargs are passed to session.post."""
        extra_kwargs = {"ssl": False, "proxy": "http://proxy.example.com"}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = setup_mock_session(
                mock_session_class, mock_aiohttp_response, ["request"]
            )

            record = await aiohttp_client.post_request(
                "http://test.com", "{}", {}, **extra_kwargs
            )

            assert_successful_request_record(record)
            mock_session.request.assert_called_once()
            call_kwargs = mock_session.request.call_args[1]
            assert "ssl" in call_kwargs
            assert "proxy" in call_kwargs

    @pytest.mark.asyncio
    async def test_session_configuration(
        self, aiohttp_client: AioHttpClient, mock_aiohttp_response: Mock
    ) -> None:
        """Test that ClientSession is configured correctly."""
        headers = {"Authorization": "Bearer token", "Custom-Header": "value"}

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_aiohttp_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", "{}", headers)

            assert_successful_request_record(record)
            mock_session_class.assert_called_once()
            call_kwargs = mock_session_class.call_args[1]
            assert call_kwargs["connector"] == aiohttp_client.tcp_connector
            assert call_kwargs["timeout"] == aiohttp_client.timeout
            assert call_kwargs["headers"] == headers
            assert call_kwargs["connector_owner"] is False
            assert "Authorization" in call_kwargs["skip_auto_headers"]
            assert "Custom-Header" in call_kwargs["skip_auto_headers"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trust_env_value", [True, False])
    async def test_trust_env_passed_to_session(
        self,
        aiohttp_client: AioHttpClient,
        mock_aiohttp_response: Mock,
        trust_env_value: bool,
        monkeypatch,
    ) -> None:
        """Test that TRUST_ENV setting is passed to ClientSession."""
        monkeypatch.setattr(Environment.HTTP, "TRUST_ENV", trust_env_value)
        monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", trust_env_value)

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_aiohttp_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_successful_request_record(record)
            mock_session_class.assert_called_once()
            call_kwargs = mock_session_class.call_args[1]
            assert call_kwargs["trust_env"] == trust_env_value

    @pytest.mark.asyncio
    async def test_end_to_end_json_request(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """Test end-to-end JSON request flow."""
        test_response = {"message": "success", "data": [1, 2, 3]}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response(text_content=json.dumps(test_response))
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request(
                "http://test.com/api",
                json.dumps({"query": "test"}),
                {"Content-Type": "application/json"},
            )

            assert_successful_request_record(record)

    @pytest.mark.asyncio
    async def test_end_to_end_sse_request(
        self, aiohttp_client: AioHttpClient, mock_sse_response: Mock
    ) -> None:
        """Test end-to-end SSE request flow."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_sse_response.content = MockStreamReader(
                [
                    b"data: Hello\nevent: message\n\n",
                    b"data: World\n\n",
                ]
            )
            setup_mock_session(mock_session_class, mock_sse_response, ["request"])

            with patch("time.perf_counter_ns", side_effect=range(123456789, 123456799)):
                record = await aiohttp_client.post_request(
                    "http://test.com/stream",
                    json.dumps({"stream": True}),
                    {"Accept": "text/event-stream"},
                )

            assert_successful_request_record(
                record, expected_response_count=2, expected_response_type=SSEMessage
            )

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self,
        aiohttp_client: AioHttpClient,
    ) -> None:
        """Test handling of concurrent requests."""
        num_requests = 5

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response()
            setup_mock_session(mock_session_class, mock_response, ["request"])

            tasks = []
            for i in range(num_requests):
                task = aiohttp_client.post_request(
                    f"http://test.com/api/{i}",
                    f'{{"request": {i}}}',
                    {"Content-Type": "application/json"},
                )
                tasks.append(task)

            records = await asyncio.gather(*tasks, return_exceptions=True)

            assert len(records) == num_requests
            for record in records:
                assert_successful_request_record(record)

    @pytest.mark.asyncio
    async def test_empty_response_body(self, aiohttp_client: AioHttpClient) -> None:
        """Test handling of empty response body."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response(text_content="")
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_successful_request_record(record)

    @pytest.mark.asyncio
    async def test_very_large_payload(self, aiohttp_client: AioHttpClient) -> None:
        """Test handling of very large payloads."""
        large_payload = b"x" * (1024 * 1024)  # 1MB payload

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response(text_content='{"received": "ok"}')
            mock_session = setup_mock_session(
                mock_session_class, mock_response, ["request"]
            )

            record = await aiohttp_client.post_request(
                "http://test.com", large_payload, {}
            )

            assert_successful_request_record(record)
            mock_session.request.assert_called_once()
            call_args = mock_session.request.call_args
            assert call_args[1]["data"] == large_payload


# --- FirstToken Callback Test Helpers ---


def setup_sse_stream_mock(
    mock_sse_response: Mock,
    mock_messages: list[SSEMessage],
):
    """Setup mocks for SSE streaming with given messages.

    Returns a context manager that patches ClientSession and AsyncSSEStreamReader.

    Args:
        mock_sse_response: Mock response with SSE content type
        mock_messages: List of SSEMessage objects to yield from the stream

    Returns:
        Context manager for the patches
    """

    @contextmanager
    def _setup():
        with (
            patch("aiohttp.ClientSession") as mock_session_class,
            patch(
                "aiperf.transports.aiohttp_client.AsyncSSEStreamReader"
            ) as mock_reader_class,
        ):
            mock_sse_response.content = MockStreamReader([b"data: test\n\n"])
            setup_mock_session(mock_session_class, mock_sse_response, ["request"])

            async def mock_aiter():
                for msg in mock_messages:
                    yield msg

            mock_reader = Mock()
            mock_reader.__aiter__ = Mock(return_value=mock_aiter())
            mock_reader_class.return_value = mock_reader

            yield

    return _setup()


@pytest.mark.asyncio
class TestFirstTokenCallback:
    """Test suite for FirstToken callback behavior in AioHttpClient."""

    async def test_callback_receives_ttft_ns_and_sse_message(
        self, aiohttp_client: AioHttpClient, mock_sse_response: Mock
    ) -> None:
        """Test that callback receives ttft_ns (int) and SSEMessage."""
        received_calls: list[tuple[int, SSEMessage]] = []

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            received_calls.append((ttft_ns, message))
            return True  # Stop after first message

        mock_messages = [
            SSEMessage(perf_ns=100_000_000),
            SSEMessage(perf_ns=200_000_000),
        ]

        with setup_sse_stream_mock(mock_sse_response, mock_messages):
            await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
                first_token_callback=callback,
            )

        # Callback should be called with ttft_ns (int) and SSEMessage
        assert len(received_calls) == 1
        ttft_ns, message = received_calls[0]
        assert isinstance(ttft_ns, int)
        assert isinstance(message, SSEMessage)

    @pytest.mark.parametrize(
        "return_pattern,expected_call_count,description",
        [
            # fmt: off
            pytest.param(
                [True],
                1,
                "returns True immediately - stops after first",
                id="true_stops",
            ),
            pytest.param(
                [False, False, True],
                3,
                "returns False twice then True - stops on third",
                id="false_continues",
            ),
            # fmt: on
        ],
    )
    async def test_callback_return_value_controls_continuation(
        self,
        aiohttp_client: AioHttpClient,
        mock_sse_response: Mock,
        return_pattern: list[bool],
        expected_call_count: int,
        description: str,
    ) -> None:
        """Test that callback return value controls whether to continue calling."""
        call_count = 0

        async def callback(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal call_count
            call_count += 1
            # Return value from pattern, or False if pattern exhausted
            idx = call_count - 1
            return return_pattern[idx] if idx < len(return_pattern) else False

        # More messages than pattern to verify stopping behavior
        mock_messages = [SSEMessage(perf_ns=i * 100_000_000) for i in range(1, 5)]

        with setup_sse_stream_mock(mock_sse_response, mock_messages):
            await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
                first_token_callback=callback,
            )

        assert call_count == expected_call_count, f"Failed for: {description}"

    async def test_no_callback_fast_path(
        self, aiohttp_client: AioHttpClient, mock_sse_response: Mock
    ) -> None:
        """Test that no callback works correctly (fast path)."""
        mock_messages = [
            SSEMessage(perf_ns=100_000_000),
            SSEMessage(perf_ns=200_000_000),
        ]

        with setup_sse_stream_mock(mock_sse_response, mock_messages):
            record = await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
                # No callback - using fast path
            )

        # All messages should be collected
        assert len(record.responses) == 2
        assert record.error is None


class _FakeClock:
    """Deterministic stand-in for the module ``time`` used by the client.

    Both clocks advance together (via :meth:`advance`) so the wall-vs-perf
    offset stays constant, mirroring real hardware. Injected into
    ``aiperf.transports.aiohttp_client.time`` so a test can force a measurable
    gap between record construction and the post-session-build re-pairing.
    """

    def __init__(self, perf_ns: int, wall_ns: int) -> None:
        self._perf = perf_ns
        self._wall = wall_ns

    def perf_counter_ns(self) -> int:
        return self._perf

    def time_ns(self) -> int:
        return self._wall

    def advance(self, ns: int) -> None:
        self._perf += ns
        self._wall += ns


class TestRequestStartClockPairing:
    """Regression guard for PR #867 (commit a63eeba03).

    ``_execute_request`` re-samples ``record.start_perf_ns`` after the session
    is built; it must re-sample ``record.timestamp_ns`` at the same instant so
    the (wall, perf) pair fed to ``compute_time_ns`` stays consistent. Dropping
    the ``timestamp_ns`` companion lets the two clocks drift by the whole
    session-build duration.
    """

    async def test_timestamp_ns_is_repaired_with_start_perf_ns_after_build(
        self, aiohttp_client: AioHttpClient
    ) -> None:
        perf0 = 1_000_000_000
        wall0 = 1_700_000_000_000_000_000
        build_ns = 250_000_000  # session setup consumes 250 ms of both clocks
        fake = _FakeClock(perf_ns=perf0, wall_ns=wall0)

        mock_response = create_mock_response()

        resp_ctx = AsyncMock()
        resp_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        resp_ctx.__aexit__ = AsyncMock(return_value=None)

        session_obj = AsyncMock()
        session_obj.request = Mock(return_value=resp_ctx)

        def _enter_session(*_args, **_kwargs):
            # Session build moves start_perf_ns forward; timestamp_ns must follow.
            fake.advance(build_ns)
            return session_obj

        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(side_effect=_enter_session)
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("aiperf.transports.aiohttp_client.time", fake),
            patch("aiohttp.ClientSession", return_value=session_ctx),
        ):
            record = await aiohttp_client.post_request(
                "http://test.com/v1/chat/completions",
                b'{"prompt": "hi"}',
                {"Content-Type": "application/json"},
            )

        assert record.error is None
        # start_perf_ns re-sampled after the build advance.
        assert record.start_perf_ns == perf0 + build_ns
        # The #867 fix: timestamp_ns re-sampled at the SAME instant. Without it,
        # timestamp_ns would keep its construction-time value and this fails.
        assert record.timestamp_ns == wall0 + build_ns
        # Pairing preserved: the wall-vs-perf offset is exactly the original.
        assert record.timestamp_ns - record.start_perf_ns == wall0 - perf0

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for aiohttp client components."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from aiperf.clients.http.aiohttp_client import (
    AioHttpClientMixin,
)
from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.config import UserConfig
from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import (
    SSEMessage,
)
from tests.clients.http.conftest import (
    assert_error_request_record,
    assert_successful_request_record,
    create_aiohttp_exception,
    create_mock_error_response,
    create_mock_response,
    setup_mock_session,
    setup_sse_content_mock,
)

################################################################################
# Test AioHttpClientMixin
################################################################################


class TestAioHttpClientMixin:
    """Test suite for AioHttpClientMixin class."""

    def test_init_creates_connector_and_timeout(self, user_config: UserConfig) -> None:
        """Test that initialization creates TCP connector and timeout configurations."""
        with patch(
            "aiperf.clients.http.aiohttp_client.create_tcp_connector"
        ) as mock_create:
            mock_connector = Mock()
            mock_create.return_value = mock_connector

            client = AioHttpClientMixin(
                model_endpoint=ModelEndpointInfo.from_user_config(user_config)
            )

            assert client.model_endpoint == ModelEndpointInfo.from_user_config(
                user_config
            )
            assert client.tcp_connector == mock_connector
            assert isinstance(client.timeout, aiohttp.ClientTimeout)
            assert client.timeout.total == 600.0
            assert client.timeout.connect == 600.0
            mock_create.assert_called_once()

    @pytest.mark.parametrize(
        "timeout_ms,expected_seconds",
        [
            (1000, 1.0),
            (5000, 5.0),
            (30000, 30.0),
            (300000, 300.0),
        ],
    )
    def test_timeout_conversion(self, timeout_ms: int, expected_seconds: float) -> None:
        """Test that timeout milliseconds are correctly converted to seconds."""

        with patch("aiperf.clients.http.aiohttp_client.create_tcp_connector"):
            client = AioHttpClientMixin(
                model_endpoint=ModelEndpointInfo(
                    endpoint=EndpointInfo(
                        type=EndpointType.CHAT,
                        base_url="http://test.com",
                        timeout=timeout_ms / 1000,
                    ),
                    models=ModelListInfo(
                        models=[ModelInfo(name="gpt-4")],
                        model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
                    ),
                )
            )

            assert client.timeout.total == expected_seconds
            assert client.timeout.connect == expected_seconds
            assert client.timeout.sock_connect == expected_seconds
            assert client.timeout.sock_read == expected_seconds
            assert client.timeout.ceil_threshold == expected_seconds

    async def test_cleanup_closes_connector(
        self, aiohttp_client: AioHttpClientMixin
    ) -> None:
        """Test that cleanup properly closes the TCP connector."""
        mock_connector = Mock()
        mock_connector.close = AsyncMock()
        aiohttp_client.tcp_connector = mock_connector

        await aiohttp_client.close()

        mock_connector.close.assert_called_once()
        assert aiohttp_client.tcp_connector is None

    async def test_cleanup_handles_none_connector(
        self, aiohttp_client: AioHttpClientMixin
    ) -> None:
        """Test that cleanup handles None connector gracefully."""
        aiohttp_client.tcp_connector = None

        # Should not raise an exception
        await aiohttp_client.close()

        assert aiohttp_client.tcp_connector is None

    @pytest.mark.asyncio
    async def test_successful_json_request(
        self, aiohttp_client: AioHttpClientMixin, mock_aiohttp_response: Mock
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
        self, aiohttp_client: AioHttpClientMixin, mock_sse_response: Mock
    ) -> None:
        """Test SSE stream request handling."""
        mock_messages = [
            SSEMessage(perf_ns=123456789),
            SSEMessage(perf_ns=123456790),
        ]

        with (
            patch("aiohttp.ClientSession") as mock_session_class,
            patch(
                "aiperf.clients.http.aiohttp_client.AioHttpSSEStreamReader"
            ) as mock_reader_class,
        ):
            setup_mock_session(mock_session_class, mock_sse_response, ["request"])

            mock_reader = Mock()
            mock_reader.read_complete_stream = AsyncMock(return_value=mock_messages)
            mock_reader_class.return_value = mock_reader

            record = await aiohttp_client.post_request(
                "http://test.com/stream",
                '{"stream": true}',
                {"Accept": "text/event-stream"},
            )

            assert_successful_request_record(
                record, expected_response_count=2, expected_response_type=SSEMessage
            )
            mock_reader.read_complete_stream.assert_called_once()

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
        aiohttp_client: AioHttpClientMixin,
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
        aiohttp_client: AioHttpClientMixin,
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
            (
                aiohttp.ClientResponseError,
                "Internal Server Error",
                "ClientResponseError",
            ),
        ],
    )
    async def test_aiohttp_specific_exceptions(
        self,
        aiohttp_client: AioHttpClientMixin,
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
        self, aiohttp_client: AioHttpClientMixin, mock_aiohttp_response: Mock
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
        self, aiohttp_client: AioHttpClientMixin, mock_aiohttp_response: Mock
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
    async def test_end_to_end_json_request(
        self,
        aiohttp_client: AioHttpClientMixin,
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
        self, aiohttp_client: AioHttpClientMixin, mock_sse_response: Mock
    ) -> None:
        """Test end-to-end SSE request flow."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            sse_chunks = [
                (b"d", b"ata: Hello\nevent: message\n\n"),
                (b"d", b"ata: World\n\n"),
            ]
            setup_sse_content_mock(mock_sse_response, sse_chunks)
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
        aiohttp_client: AioHttpClientMixin,
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
    async def test_empty_response_body(
        self, aiohttp_client: AioHttpClientMixin
    ) -> None:
        """Test handling of empty response body."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = create_mock_response(text_content="")
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.post_request("http://test.com", "{}", {})

            assert_successful_request_record(record)

    @pytest.mark.asyncio
    async def test_very_large_payload(self, aiohttp_client: AioHttpClientMixin) -> None:
        """Test handling of very large payloads."""
        large_payload = "x" * (1024 * 1024)  # 1MB payload

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

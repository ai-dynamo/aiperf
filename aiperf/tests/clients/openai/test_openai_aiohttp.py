#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.clients.client_interfaces import InferenceClientFactory
from aiperf.clients.model_endpoint_info import EndpointInfo, ModelEndpointInfo
from aiperf.clients.openai.openai_aiohttp import OpenAIClientAioHttp
from aiperf.common.enums import EndpointType
from aiperf.common.record_models import (
    RequestRecord,
    SSEMessage,
    TextResponse,
)


@pytest.mark.asyncio
class TestOpenAIClientAioHttp:
    """Test cases for OpenAIClientAioHttp."""

    async def test_client_initialization(self, sample_model_endpoint_info):
        """Test client can be initialized with model endpoint info."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        assert client.model_endpoint == sample_model_endpoint_info
        assert hasattr(client, "logger")
        assert hasattr(client, "tcp_connector")
        assert hasattr(client, "timeout")

    async def test_client_factory_registration(self, sample_model_endpoint_info):
        """Test that client is registered with factory for all OpenAI endpoint types."""
        openai_endpoints = [
            EndpointType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_COMPLETIONS,
            EndpointType.OPENAI_EMBEDDINGS,
            EndpointType.OPENAI_RESPONSES,
        ]

        for endpoint_type in openai_endpoints:
            client = InferenceClientFactory.create_instance(
                endpoint_type, model_endpoint=sample_model_endpoint_info
            )
            assert isinstance(client, OpenAIClientAioHttp)

    async def test_get_headers_basic(self, sample_model_endpoint_info):
        """Test basic header generation."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        headers = client.get_headers(sample_model_endpoint_info)

        expected_headers = {
            "User-Agent": "aiperf/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-api-key",
            "Custom-Header": "test-value",
        }

        assert headers == expected_headers

    async def test_get_headers_streaming(self, sample_model_endpoint_info):
        """Test header generation for streaming requests."""
        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        headers = client.get_headers(sample_model_endpoint_info)

        assert headers["Accept"] == "text/event-stream"

    async def test_get_headers_no_api_key(self, sample_model_endpoint_info):
        """Test header generation without API key."""
        # Remove API key
        sample_model_endpoint_info.endpoint.api_key = None

        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        headers = client.get_headers(sample_model_endpoint_info)

        assert "Authorization" not in headers

    async def test_get_headers_no_custom_headers(self, sample_model_endpoint_info):
        """Test header generation without custom headers."""
        # Remove custom headers
        sample_model_endpoint_info.endpoint.headers = None

        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        headers = client.get_headers(sample_model_endpoint_info)

        assert "Custom-Header" not in headers
        assert "User-Agent" in headers
        assert "Content-Type" in headers

    async def test_get_url_with_http_prefix(self, sample_model_endpoint_info):
        """Test URL generation with http prefix."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        url = client.get_url(sample_model_endpoint_info)

        assert url == "https://api.openai.com/v1/chat/completions"

    async def test_get_url_without_http_prefix(self, sample_model_endpoint_info):
        """Test URL generation without http prefix."""
        # Remove http prefix
        sample_model_endpoint_info.endpoint.base_url = "api.openai.com"

        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        url = client.get_url(sample_model_endpoint_info)

        assert url == "http://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_send_request_successful(
        self, sample_model_endpoint_info, openai_chat_response
    ):
        """Test successful request sending."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(
                request=None,
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns() + 1000000,
                status=200,
                responses=[
                    TextResponse(
                        perf_ns=time.perf_counter_ns(),
                        content_type="application/json",
                        text=json.dumps(openai_chat_response),
                    )
                ],
            )
            mock_post.return_value = mock_record

            record = await client.send_request(sample_model_endpoint_info, payload)

            assert record is mock_record
            assert record.request == payload
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_with_exception(self, sample_model_endpoint_info):
        """Test request sending with exception."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Mock the post_request method to raise an exception
        with patch.object(client, "post_request") as mock_post:
            mock_post.side_effect = Exception("Connection error")

            record = await client.send_request(sample_model_endpoint_info, payload)

            assert record.request == payload
            assert record.error is not None
            assert record.error.type == "Exception"
            assert record.error.message == "Connection error"
            assert record.start_perf_ns > 0
            assert record.end_perf_ns is not None
            assert record.end_perf_ns > record.start_perf_ns

    @pytest.mark.asyncio
    async def test_send_request_debug_logging(self, sample_model_endpoint_info, caplog):
        """Test that debug logging is performed."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Enable debug logging
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(status=200)
            mock_post.return_value = mock_record

            await client.send_request(sample_model_endpoint_info, payload)

            # Check that debug log was created
            assert "Sending OpenAI request" in caplog.text

    @pytest.mark.asyncio
    async def test_send_request_error_logging(self, sample_model_endpoint_info, caplog):
        """Test that error logging is performed."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Mock the post_request method to raise an exception
        with patch.object(client, "post_request") as mock_post:
            mock_post.side_effect = Exception("Connection error")

            await client.send_request(sample_model_endpoint_info, payload)

            # Check that error log was created
            assert "Error in OpenAI request" in caplog.text
            assert "Exception" in caplog.text
            assert "Connection error" in caplog.text

    @pytest.mark.asyncio
    async def test_close_method(self, sample_model_endpoint_info):
        """Test close method (inherited from mixin)."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        # Mock the tcp_connector
        mock_close = AsyncMock()
        client.tcp_connector = AsyncMock()
        client.tcp_connector.close = mock_close

        await client.close()

        mock_close.assert_called_once()
        assert client.tcp_connector is None

    async def test_timeout_configuration(self, sample_model_endpoint_info):
        """Test that timeout is configured correctly."""
        # Set custom timeout
        sample_model_endpoint_info.endpoint.timeout = 60.0

        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        # Check that timeout is set correctly
        assert client.timeout.total == 60.0
        assert client.timeout.connect == 60.0
        assert client.timeout.sock_connect == 60.0
        assert client.timeout.sock_read == 60.0
        assert client.timeout.ceil_threshold == 60.0

    @pytest.mark.asyncio
    async def test_send_request_different_payload_types(
        self, sample_model_endpoint_info
    ):
        """Test sending requests with different payload types."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        payloads = [
            # Chat completion payload
            {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"},
            # Completion payload
            {"prompt": "Hello", "model": "gpt-4"},
            # Embeddings payload
            {"input": "Hello", "model": "text-embedding-ada-002"},
            # Responses payload
            {"input": "Hello", "model": "gpt-4", "max_output_tokens": 100},
        ]

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(status=200)
            mock_post.return_value = mock_record

            for payload in payloads:
                record = await client.send_request(sample_model_endpoint_info, payload)
                assert record.request == payload

    @pytest.mark.asyncio
    async def test_send_request_with_streaming_endpoint(
        self, sample_model_endpoint_info
    ):
        """Test sending request with streaming endpoint."""
        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": True,
        }

        # Mock the post_request method with SSE responses
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(
                status=200,
                responses=[SSEMessage(perf_ns=time.perf_counter_ns(), packets=[])],
            )
            mock_post.return_value = mock_record

            record = await client.send_request(sample_model_endpoint_info, payload)

            assert record.request == payload
            assert len(record.responses) == 1
            assert isinstance(record.responses[0], SSEMessage)

    @pytest.mark.asyncio
    async def test_send_request_preserves_payload_structure(
        self, sample_model_endpoint_info
    ):
        """Test that request preserves complex payload structure."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        # Complex payload with nested structures
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                    ],
                }
            ],
            "model": "gpt-4-vision-preview",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["Human:", "AI:"],
        }

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(status=200)
            mock_post.return_value = mock_record

            record = await client.send_request(sample_model_endpoint_info, payload)

            # Verify that the payload is preserved exactly
            assert record.request == payload
            assert record.request["messages"][0]["content"][0]["type"] == "text"
            assert record.request["messages"][0]["content"][1]["type"] == "image_url"
            assert record.request["stop"] == ["Human:", "AI:"]

    @pytest.mark.asyncio
    async def test_send_request_error_details_format(self, sample_model_endpoint_info):
        """Test that error details are formatted correctly."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Test different exception types
        exceptions = [
            Exception("Generic error"),
            ValueError("Invalid value"),
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
        ]

        for exception in exceptions:
            with patch.object(client, "post_request") as mock_post:
                mock_post.side_effect = exception

                record = await client.send_request(sample_model_endpoint_info, payload)

                assert record.error is not None
                assert record.error.type == exception.__class__.__name__
                assert record.error.message == str(exception)
                assert record.error.code is None

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, sample_model_endpoint_info):
        """Test that client can handle concurrent requests."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        # Create multiple payloads
        payloads = [
            {"messages": [{"role": "user", "content": f"Hello {i}"}], "model": "gpt-4"}
            for i in range(5)
        ]

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:

            def create_mock_record(payload):
                return RequestRecord(
                    request=payload,
                    status=200,
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns() + 1000000,
                )

            mock_post.side_effect = [create_mock_record(p) for p in payloads]

            # Send concurrent requests
            import asyncio

            tasks = [
                client.send_request(sample_model_endpoint_info, payload)
                for payload in payloads
            ]
            records = await asyncio.gather(*tasks)

            # Verify all requests completed
            assert len(records) == 5
            for i, record in enumerate(records):
                assert record.request == payloads[i]
                assert record.status == 200

    async def test_inheritance_from_aiohttp_mixin(self, sample_model_endpoint_info):
        """Test that client properly inherits from AioHttpClientMixin."""
        from aiperf.clients.http.aiohttp_client import AioHttpClientMixin

        client = OpenAIClientAioHttp(sample_model_endpoint_info)

        assert isinstance(client, AioHttpClientMixin)
        assert hasattr(client, "post_request")
        assert hasattr(client, "close")
        assert hasattr(client, "tcp_connector")
        assert hasattr(client, "timeout")

    @pytest.mark.asyncio
    async def test_headers_passed_to_post_request(self, sample_model_endpoint_info):
        """Test that headers are properly passed to post_request."""
        client = OpenAIClientAioHttp(sample_model_endpoint_info)
        payload = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}

        # Mock the post_request method
        with patch.object(client, "post_request") as mock_post:
            mock_record = RequestRecord(status=200)
            mock_post.return_value = mock_record

            await client.send_request(sample_model_endpoint_info, payload)

            # Verify post_request was called with correct arguments
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check URL
            assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

            # Check payload (should be JSON string)
            assert call_args[0][1] == json.dumps(payload)

            # Check headers
            headers = call_args[0][2]
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            EndpointType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_COMPLETIONS,
            EndpointType.OPENAI_EMBEDDINGS,
            EndpointType.OPENAI_RESPONSES,
        ],
    )
    async def test_client_works_with_all_endpoint_types(
        self, sample_model_list_info, endpoint_type
    ):
        """Test that client works with all supported endpoint types."""
        endpoint_info = EndpointInfo(
            type=endpoint_type, base_url="https://api.openai.com", api_key="test-key"
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        client = OpenAIClientAioHttp(model_endpoint_info)

        # Verify URL is generated correctly for each endpoint type
        url = client.get_url(model_endpoint_info)
        expected_paths = {
            EndpointType.OPENAI_CHAT_COMPLETIONS: "/v1/chat/completions",
            EndpointType.OPENAI_COMPLETIONS: "/v1/completions",
            EndpointType.OPENAI_EMBEDDINGS: "/v1/embeddings",
            EndpointType.OPENAI_RESPONSES: "/v1/responses",
        }

        assert url.endswith(expected_paths[endpoint_type])

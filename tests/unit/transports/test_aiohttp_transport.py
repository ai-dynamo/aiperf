# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import TransportType
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info


class TestAioHttpTransport:
    """Comprehensive tests for AioHttpTransport."""

    @pytest.fixture
    def transport(self, model_endpoint_non_streaming):
        """Create an AioHttpTransport instance."""
        return AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

    @pytest.fixture
    def transport_with_tcp_kwargs(self, model_endpoint_non_streaming):
        """Create an AioHttpTransport with custom TCP settings."""
        tcp_kwargs = {"limit": 200, "limit_per_host": 50}
        return AioHttpTransport(
            model_endpoint=model_endpoint_non_streaming, tcp_kwargs=tcp_kwargs
        )

    @pytest.fixture
    async def initialized_transport(self, transport):
        """Initialize transport and yield for testing."""
        await transport.initialize()
        yield transport
        await transport.stop()

    def _create_request_info(
        self,
        model_endpoint,
        endpoint_headers=None,
        endpoint_params=None,
        x_request_id=None,
        x_correlation_id=None,
    ):
        """Helper to create RequestInfo with defaults."""
        return RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers=endpoint_headers or {},
            endpoint_params=endpoint_params or {},
            x_request_id=x_request_id,
            x_correlation_id=x_correlation_id,
        )

    def _extract_call_args(self, mock_call_args):
        """Extract URL, JSON, and headers from mock call_args."""
        return {
            "url": mock_call_args[0][0],
            "json_str": mock_call_args[0][1],
            "headers": mock_call_args[0][2],
        }

    async def _setup_initialized_transport_with_mock(self, transport):
        """Initialize transport and setup mock post_request."""
        await transport.initialize()
        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)
        return mock_record

    @pytest.mark.asyncio
    async def test_init_with_default_tcp_kwargs(self, transport):
        """Test initialization with default TCP kwargs."""
        assert transport.tcp_kwargs is None
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_init_with_custom_tcp_kwargs(self, transport_with_tcp_kwargs):
        """Test initialization with custom TCP kwargs."""
        assert transport_with_tcp_kwargs.tcp_kwargs is not None
        assert transport_with_tcp_kwargs.tcp_kwargs["limit"] == 200
        assert transport_with_tcp_kwargs.tcp_kwargs["limit_per_host"] == 50

    @pytest.mark.asyncio
    async def test_init_hook_creates_aiohttp_client(self, transport):
        """Test that lifecycle initialize creates AioHttpClient."""
        await transport.initialize()
        assert transport.aiohttp_client is not None

    @pytest.mark.asyncio
    async def test_stop_hook_closes_aiohttp_client(self, transport):
        """Test that lifecycle stop closes AioHttpClient."""
        await transport.initialize()
        assert transport.aiohttp_client is not None

        await transport.stop()
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_stop_hook_handles_none_client(self, transport):
        """Test that stop hook handles None client."""
        await transport.stop()
        assert transport.aiohttp_client is None

    def test_metadata(self, transport):
        """Test metadata returns correct transport info."""
        metadata = transport.metadata()
        assert metadata.transport_type == TransportType.HTTP
        assert "http" in metadata.url_schemes
        assert "https" in metadata.url_schemes

    @pytest.mark.parametrize(
        "streaming,expected_accept",
        [(False, "application/json"), (True, "text/event-stream")],
        ids=["non-streaming", "streaming"],
    )
    def test_get_transport_headers(self, transport, streaming, expected_accept):
        """Test transport headers for different streaming modes."""
        model_endpoint = create_model_endpoint_info(streaming=streaming)
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])
        headers = transport.get_transport_headers(request_info)

        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == expected_accept

    @pytest.mark.parametrize(
        "base_url,custom_endpoint,expected_url",
        [
            (
                "http://localhost:8000",
                "/v1/chat/completions",
                "http://localhost:8000/v1/chat/completions",
            ),
            ("localhost:8000", "/v1/chat", "http://localhost:8000/v1/chat"),
            ("https://api.example.com", "/v1/chat", "https://api.example.com/v1/chat"),
        ],
        ids=["http-prefix", "no-scheme", "https-prefix"],
    )
    def test_get_url(
        self, model_endpoint_non_streaming, base_url, custom_endpoint, expected_url
    ):
        """Test get_url with various base URLs and endpoints."""
        model_endpoint_non_streaming.endpoint.base_url = base_url
        model_endpoint_non_streaming.endpoint.custom_endpoint = custom_endpoint

        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming, turns=[]
        )
        url = transport.get_url(request_info)
        assert url == expected_url

    @pytest.mark.asyncio
    async def test_send_request_success(self, transport, model_endpoint_non_streaming):
        """Test successful HTTP request."""
        await transport.initialize()

        # Mock the aiohttp_client
        mock_record = RequestRecord(responses=[], error=None)
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = RequestInfo(
            model_endpoint=model_endpoint_non_streaming,
            turns=[],
            endpoint_headers={"Authorization": "Bearer token"},
            endpoint_params={},
        )
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.error is None
        transport.aiohttp_client.post_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_builds_correct_url(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that send_request builds URL correctly with params."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = self._create_request_info(
            model_endpoint_non_streaming,
            endpoint_params={"api-version": "2024-10-01"},
        )
        payload = {"test": "data"}

        await transport.send_request(request_info, payload)

        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        assert "api-version=2024-10-01" in args["url"]

    @pytest.mark.asyncio
    async def test_send_request_builds_correct_headers(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that send_request builds headers correctly."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = self._create_request_info(
            model_endpoint_non_streaming,
            endpoint_headers={"Authorization": "Bearer token123"},
            x_request_id="req-456",
        )
        payload = {"test": "data"}

        await transport.send_request(request_info, payload)

        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        headers = args["headers"]

        assert headers["Authorization"] == "Bearer token123"
        assert headers["User-Agent"] == "aiperf/1.0"
        assert headers["X-Request-ID"] == "req-456"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_request_serializes_payload_with_orjson(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that payload is serialized using orjson."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"messages": [{"role": "user", "content": "Test"}], "model": "gpt-4"}

        await transport.send_request(request_info, payload)

        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        json_str = args["json_str"]

        assert isinstance(json_str, str)
        assert "messages" in json_str
        assert "gpt-4" in json_str

    @pytest.mark.asyncio
    async def test_send_request_handles_exception(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that exceptions are caught and recorded."""
        await transport.initialize()
        transport.aiohttp_client.post_request = AsyncMock(
            side_effect=ValueError("Test error")
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"test": "data"}

        record = await transport.send_request(request_info, payload)

        assert record.error is not None
        assert record.error.type == "ValueError"
        assert "Test error" in record.error.message
        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None

    @pytest.mark.asyncio
    async def test_send_request_timing_on_error(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that timing is recorded even on errors."""
        await transport.initialize()
        transport.aiohttp_client.post_request = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {"test": "data"}

        record = await transport.send_request(request_info, payload)

        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None
        assert record.end_perf_ns >= record.start_perf_ns
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_send_request_streaming_headers(self, model_endpoint_streaming):
        """Test correct headers for streaming requests."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_streaming)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = RequestInfo(
            model_endpoint=model_endpoint_streaming,
            turns=[],
            endpoint_headers={},
            endpoint_params={},
        )
        payload = {"stream": True}

        await transport.send_request(request_info, payload)

        call_args = transport.aiohttp_client.post_request.call_args
        headers = call_args[0][2]
        assert headers["Accept"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_send_request_empty_payload(
        self, transport, model_endpoint_non_streaming
    ):
        """Test send_request with empty payload."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {}

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        assert args["json_str"] == "{}"

    @pytest.mark.asyncio
    async def test_send_request_complex_payload(
        self, transport, model_endpoint_non_streaming
    ):
        """Test send_request with complex nested payload."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = self._create_request_info(model_endpoint_non_streaming)
        payload = {
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Response"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                },
            ],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 500,
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        json_str = args["json_str"]
        assert "messages" in json_str
        assert "image_url" in json_str
        assert "0.7" in json_str


class TestAioHttpTransportLifecycle:
    """Test lifecycle management of AioHttpTransport."""

    @pytest.mark.asyncio
    async def test_init_creates_client(self, model_endpoint_non_streaming):
        """Test that init creates aiohttp client."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        assert transport.aiohttp_client is None

        await transport.initialize()
        assert transport.aiohttp_client is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_client(self, model_endpoint_non_streaming):
        """Test that stop closes aiohttp client."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        client = transport.aiohttp_client
        assert client is not None

        await transport.stop()
        assert transport.aiohttp_client is None

    @pytest.mark.asyncio
    async def test_multiple_init_calls(self, model_endpoint_non_streaming):
        """Test that multiple init calls are handled correctly."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

        await transport.initialize()
        _ = transport.aiohttp_client

        await transport.initialize()
        client2 = transport.aiohttp_client

        assert client2 is not None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_without_init(self, model_endpoint_non_streaming):
        """Test that stop works if init was never called."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.stop()
        assert transport.aiohttp_client is None


class TestAioHttpTransportIntegration:
    """Integration tests for AioHttpTransport with full request flow."""

    @pytest.mark.asyncio
    async def test_full_request_flow_non_streaming(self):
        """Test complete request flow for non-streaming."""
        model_endpoint = create_model_endpoint_info(
            base_url="https://api.example.com",
            api_key="test-key",
            headers=[("Custom-Header", "value")],
        )

        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={
                "Authorization": "Bearer test-key",
                "Custom-Header": "value",
            },
            endpoint_params={"api-version": "2024-10-01"},
            x_request_id="req-123",
            x_correlation_id="corr-456",
        )

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
        }

        await transport.send_request(request_info, payload)

        assert transport.aiohttp_client.post_request.called
        args = {
            "url": transport.aiohttp_client.post_request.call_args[0][0],
            "json_str": transport.aiohttp_client.post_request.call_args[0][1],
            "headers": transport.aiohttp_client.post_request.call_args[0][2],
        }

        assert "https://api.example.com/v1/chat/completions" in args["url"]
        assert "api-version=2024-10-01" in args["url"]
        assert "Hello" in args["json_str"]
        assert args["headers"]["Authorization"] == "Bearer test-key"
        assert args["headers"]["Custom-Header"] == "value"
        assert args["headers"]["X-Request-ID"] == "req-123"
        assert args["headers"]["X-Correlation-ID"] == "corr-456"
        assert args["headers"]["Accept"] == "application/json"

        await transport.stop()

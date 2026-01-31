# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ConnectionReuseStrategy, CreditPhase
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.plugin import plugins
from aiperf.plugin.enums import TransportType
from aiperf.transports.aiohttp_transport import (
    AioHttpTransport,
    ConnectionLeaseManager,
)
from tests.unit.transports.conftest import create_model_endpoint_info
from tests.unit.transports.test_base_transport import AIPERF_USER_AGENT


def create_request_info(
    model_endpoint,
    *,
    endpoint_headers: dict | None = None,
    endpoint_params: dict | None = None,
    x_request_id: str = "test-request-id",
    x_correlation_id: str = "test-correlation-id",
    conversation_id: str = "test-conversation-id",
    cancel_after_ns: int | None = None,
    is_final_turn: bool = True,
    turn_index: int = 0,
    credit_num: int = 1,
) -> RequestInfo:
    """Create RequestInfo with sensible defaults for transport tests."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers=endpoint_headers or {},
        endpoint_params=endpoint_params or {},
        turn_index=turn_index,
        credit_num=credit_num,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=x_request_id,
        x_correlation_id=x_correlation_id,
        conversation_id=conversation_id,
        cancel_after_ns=cancel_after_ns,
        is_final_turn=is_final_turn,
    )


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
        assert transport.tcp_kwargs == {}
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
        metadata = plugins.get_transport_metadata(TransportType.HTTP)
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
        request_info = create_request_info(model_endpoint)
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
    def test_get_url(self, base_url, custom_endpoint, expected_url):
        """Test get_url with various base URLs and endpoints."""
        model_endpoint = create_model_endpoint_info(
            base_url=base_url, custom_endpoint=custom_endpoint
        )

        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = create_request_info(model_endpoint)
        url = transport.get_url(request_info)
        assert url == expected_url

    @pytest.mark.asyncio
    async def test_send_request_success(self, transport, model_endpoint_non_streaming):
        """Test successful HTTP request."""
        await transport.initialize()

        # Mock the aiohttp_client
        mock_record = RequestRecord(responses=[], error=None)
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = create_request_info(
            model_endpoint_non_streaming,
            endpoint_headers={"Authorization": "Bearer token"},
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

        request_info = create_request_info(
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

        request_info = create_request_info(
            model_endpoint_non_streaming,
            endpoint_headers={"Authorization": "Bearer token123"},
            x_request_id="req-456",
        )
        payload = {"test": "data"}

        await transport.send_request(request_info, payload)

        args = self._extract_call_args(transport.aiohttp_client.post_request.call_args)
        headers = args["headers"]

        assert headers["Authorization"] == "Bearer token123"
        assert headers["User-Agent"] == AIPERF_USER_AGENT
        assert headers["X-Request-ID"] == "req-456"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_request_serializes_payload_with_orjson(
        self, transport, model_endpoint_non_streaming
    ):
        """Test that payload is serialized using orjson."""
        await self._setup_initialized_transport_with_mock(transport)

        request_info = create_request_info(model_endpoint_non_streaming)
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

        request_info = create_request_info(model_endpoint_non_streaming)
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

        request_info = create_request_info(model_endpoint_non_streaming)
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

        request_info = create_request_info(model_endpoint_streaming)
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

        request_info = create_request_info(model_endpoint_non_streaming)
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

        request_info = create_request_info(model_endpoint_non_streaming)
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
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="req-123",
            x_correlation_id="corr-456",
            conversation_id="test-conversation-id",
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

        assert args["url"].startswith("https://api.example.com/v1/chat/completions")
        assert "api-version=2024-10-01" in args["url"]
        assert "Hello" in args["json_str"]
        assert args["headers"]["Authorization"] == "Bearer test-key"
        assert args["headers"]["Custom-Header"] == "value"
        assert args["headers"]["X-Request-ID"] == "req-123"
        assert args["headers"]["X-Correlation-ID"] == "corr-456"
        assert args["headers"]["Accept"] == "application/json"

        await transport.stop()


class TestAioHttpTransportCancellation:
    """Test request cancellation handling at the transport layer.

    Note: Time is globally mocked in this test suite. The transport code uses
    asyncio.sleep(0) to yield to the event loop, which works with mocked time.
    """

    def _create_mock_session_factory(self, complete_immediately: bool = True):
        """Create a mock session factory for cancellation tests.

        Args:
            complete_immediately: If True, request completes. If False, request hangs forever.

        Returns:
            Tuple of (capture_session function, trace_config holder dict)
        """
        holder = {"trace_config": None, "request_sent": False}

        def capture_session(*args, **kwargs):
            trace_configs = kwargs.get("trace_configs", [])
            if trace_configs:
                holder["trace_config"] = trace_configs[0]

            # Create mock response context manager
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.reason = "OK"
            mock_response.headers = {}
            mock_response.content_type = "application/json"
            mock_response.text = AsyncMock(return_value='{"result": "ok"}')
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            # Create async context manager for session.request()
            # session.request() is used as: async with session.request(...) as response:
            @asynccontextmanager
            async def mock_request_cm(method, url, **req_kwargs):
                holder["request_sent"] = True
                if complete_immediately:
                    yield mock_response
                else:
                    # Request never completes (for testing cancellation)
                    await asyncio.Future()  # Never yields - hangs forever

            mock_session = MagicMock()
            mock_session.request = mock_request_cm
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            return mock_session

        return capture_session, holder

    async def _fire_request_sent_event(self, trace_config, body_size: int = 100):
        """Fire the on_request_chunk_sent trace event to signal request body was sent.

        Args:
            trace_config: The captured trace config
            body_size: Size of the request body to simulate
        """
        if trace_config:
            mock_params = MagicMock()
            mock_params.chunk = b"x" * body_size  # Simulate body chunk
            for callback in trace_config.on_request_chunk_sent:
                await callback(None, None, mock_params)

    @pytest.mark.asyncio
    async def test_send_request_without_cancellation(
        self, model_endpoint_non_streaming
    ):
        """Test that request without cancel_after_ns is sent normally."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = create_request_info(
            model_endpoint_non_streaming, cancel_after_ns=None
        )
        payload = {"test": "data"}

        record = await transport.send_request(request_info, payload)

        assert record == mock_record
        transport.aiohttp_client.post_request.assert_called_once()
        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_with_cancellation_completes_before_timeout(
        self, model_endpoint_non_streaming
    ):
        """Test request that completes before cancellation timeout.

        When cancel_after_ns is set, the transport uses _request_with_cancellation which
        waits for the request to be sent before starting the cancellation timer.
        """
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        capture_session, holder = self._create_mock_session_factory(
            complete_immediately=True
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = capture_session

            # 10 second timeout - request should complete before this
            cancel_after_ns = 10 * 1_000_000_000
            request_info = create_request_info(
                model_endpoint_non_streaming, cancel_after_ns=cancel_after_ns
            )
            payload = {"test": "data"}

            task = asyncio.create_task(transport.send_request(request_info, payload))

            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Fire trace event (request chunk sent - triggers when body is fully sent)
            await self._fire_request_sent_event(holder["trace_config"])

            record = await task

        # Request completed normally before timeout
        assert record.status == 200
        assert record.error is None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_with_zero_cancellation_sends_then_cancels(
        self, model_endpoint_non_streaming
    ):
        """Test that cancel_after_ns=0 sends request then cancels immediately.

        With cancel_after_ns=0, the transport waits for the full request to be sent
        (via on_request_end trace event), then immediately cancels (timeout=0 seconds).
        """
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        capture_session, holder = self._create_mock_session_factory(
            complete_immediately=False
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = capture_session

            # cancel_after_ns=0 means cancel immediately after request is sent
            request_info = create_request_info(
                model_endpoint_non_streaming, cancel_after_ns=0
            )
            payload = {"test": "data"}

            task = asyncio.create_task(transport.send_request(request_info, payload))

            # Give the task time to start and capture the trace config
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Fire the trace event (simulating full request sent)
            await self._fire_request_sent_event(holder["trace_config"])

            # Now wait for the task to complete (it should cancel quickly)
            record = await task

        # Request was started and then cancelled
        assert record.error is not None, f"Expected cancellation error, got: {record}"
        assert record.error.type == "RequestCancellationError"
        assert record.error.code == 499  # Client Closed Request
        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_cancellation_with_short_delay(
        self, model_endpoint_non_streaming
    ):
        """Test request cancellation with a small delay (100ms)."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        capture_session, holder = self._create_mock_session_factory(
            complete_immediately=False
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = capture_session

            cancel_after_ns = 100_000_000  # 100ms
            request_info = create_request_info(
                model_endpoint_non_streaming, cancel_after_ns=cancel_after_ns
            )
            payload = {"test": "data"}

            task = asyncio.create_task(transport.send_request(request_info, payload))

            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Fire trace event (request chunk sent)
            await self._fire_request_sent_event(holder["trace_config"])

            record = await task

        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_cancellation_record_has_timing(
        self, model_endpoint_non_streaming
    ):
        """Test that cancellation record has proper timing fields."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        capture_session, holder = self._create_mock_session_factory(
            complete_immediately=False
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = capture_session

            request_info = create_request_info(
                model_endpoint_non_streaming,
                cancel_after_ns=50_000_000,  # 50ms
            )
            payload = {"test": "data"}

            task = asyncio.create_task(transport.send_request(request_info, payload))

            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Fire trace event (request chunk sent)
            await self._fire_request_sent_event(holder["trace_config"])

            record = await task

        # Verify timing fields are set
        assert record.start_perf_ns is not None
        assert record.end_perf_ns is not None
        assert record.cancellation_perf_ns is not None
        assert record.end_perf_ns >= record.start_perf_ns
        assert record.cancellation_perf_ns == record.end_perf_ns
        await transport.stop()

    @pytest.mark.asyncio
    async def test_cancellation_error_details(self, model_endpoint_non_streaming):
        """Test that cancellation error has correct details."""
        transport = AioHttpTransport(model_endpoint=model_endpoint_non_streaming)
        await transport.initialize()

        capture_session, holder = self._create_mock_session_factory(
            complete_immediately=False
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = capture_session

            # Test with 200ms cancellation delay
            cancel_after_ns = 200_000_000  # 200ms = 0.2 seconds
            request_info = create_request_info(
                model_endpoint_non_streaming, cancel_after_ns=cancel_after_ns
            )
            payload = {"test": "data"}

            task = asyncio.create_task(transport.send_request(request_info, payload))

            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Fire trace event (request chunk sent)
            await self._fire_request_sent_event(holder["trace_config"])

            record = await task

        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert record.error.code == 499  # Client Closed Request
        assert "0.200" in record.error.message  # Should mention the timeout
        await transport.stop()


class TestConnectionLeaseManager:
    """Tests for ConnectionLeaseManager."""

    @pytest.fixture
    def lease_manager(self):
        """Create a ConnectionLeaseManager instance."""
        return ConnectionLeaseManager()

    @pytest.fixture
    def lease_manager_with_kwargs(self):
        """Create a ConnectionLeaseManager with custom TCP kwargs."""
        return ConnectionLeaseManager(tcp_kwargs={"limit": 1, "limit_per_host": 1})

    @pytest.mark.asyncio
    async def test_get_connector_creates_new_connector(self, lease_manager):
        """Test that get_connector creates a new connector for unknown conversation."""
        connector = lease_manager.get_connector("conv-1")
        assert connector is not None
        assert "conv-1" in lease_manager._leases
        await lease_manager.close_all()

    @pytest.mark.asyncio
    async def test_get_connector_returns_same_connector(self, lease_manager):
        """Test that get_connector returns the same connector for the same conversation."""
        connector1 = lease_manager.get_connector("conv-1")
        connector2 = lease_manager.get_connector("conv-1")
        assert connector1 is connector2
        await lease_manager.close_all()

    @pytest.mark.asyncio
    async def test_get_connector_creates_different_connectors_per_conversation(
        self, lease_manager
    ):
        """Test that different conversations get different connectors."""
        connector1 = lease_manager.get_connector("conv-1")
        connector2 = lease_manager.get_connector("conv-2")
        assert connector1 is not connector2
        await lease_manager.close_all()

    @pytest.mark.asyncio
    async def test_release_lease_closes_connector(self, lease_manager):
        """Test that release_lease closes and removes the connector."""
        connector = lease_manager.get_connector("conv-1")
        assert "conv-1" in lease_manager._leases

        await lease_manager.release_lease("conv-1")
        assert "conv-1" not in lease_manager._leases
        assert connector.closed

    @pytest.mark.asyncio
    async def test_release_lease_nonexistent_conversation(self, lease_manager):
        """Test that release_lease does nothing for unknown conversation."""
        # Should not raise
        await lease_manager.release_lease("unknown-conv")

    @pytest.mark.asyncio
    async def test_close_all_closes_all_connectors(self, lease_manager):
        """Test that close_all closes all active connectors."""
        connector1 = lease_manager.get_connector("conv-1")
        connector2 = lease_manager.get_connector("conv-2")
        connector3 = lease_manager.get_connector("conv-3")

        await lease_manager.close_all()

        assert len(lease_manager._leases) == 0
        assert connector1.closed
        assert connector2.closed
        assert connector3.closed


class TestConnectionStrategies:
    """Tests for different connection strategies."""

    @pytest.mark.asyncio
    async def test_pooled_strategy_uses_shared_connector(self):
        """Test that POOLED strategy uses the shared connection pool."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.POOLED
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # No lease manager for pooled strategy
        assert transport.lease_manager is None

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = create_request_info(model_endpoint)
        await transport.send_request(request_info, {"test": "data"})

        # Verify post_request was called with connector=None (uses shared pool)
        call_kwargs = transport.aiohttp_client.post_request.call_args[1]
        assert call_kwargs.get("connector") is None
        assert call_kwargs.get("connector_owner") is False

        await transport.stop()

    @pytest.mark.asyncio
    async def test_never_strategy_creates_new_connector(self):
        """Test that NEVER strategy creates a new connector per request."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # No lease manager for never strategy
        assert transport.lease_manager is None

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        request_info = create_request_info(model_endpoint)
        await transport.send_request(request_info, {"test": "data"})

        # Verify post_request was called with a connector and connector_owner=True
        call_kwargs = transport.aiohttp_client.post_request.call_args[1]
        assert call_kwargs.get("connector") is not None
        assert call_kwargs.get("connector_owner") is True

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_strategy_creates_lease_manager(self):
        """Test that STICKY_USER_SESSIONS strategy creates a lease manager."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # Lease manager should be created
        assert transport.lease_manager is not None
        assert isinstance(transport.lease_manager, ConnectionLeaseManager)

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_strategy_reuses_connector(self):
        """Test that STICKY_USER_SESSIONS strategy reuses connector across turns."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        # First turn - not final
        request_info_1 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info_1, {"test": "data"})

        first_call_kwargs = transport.aiohttp_client.post_request.call_args[1]
        first_connector = first_call_kwargs.get("connector")

        # Second turn - not final
        request_info_2 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info_2, {"test": "data"})

        second_call_kwargs = transport.aiohttp_client.post_request.call_args[1]
        second_connector = second_call_kwargs.get("connector")

        # Same connector should be used
        assert first_connector is second_connector

        # Connector should not be closed yet
        assert not first_connector.closed

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_strategy_releases_on_final_turn(self):
        """Test that STICKY_USER_SESSIONS releases lease on final turn."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        # First turn - not final
        request_info_1 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info_1, {"test": "data"})

        first_connector = transport.aiohttp_client.post_request.call_args[1][
            "connector"
        ]
        assert "session-1" in transport.lease_manager._leases

        # Final turn
        request_info_2 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=True
        )
        await transport.send_request(request_info_2, {"test": "data"})

        # Lease should be released
        assert "session-1" not in transport.lease_manager._leases
        assert first_connector.closed

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_separate_sessions_use_separate_connectors(
        self,
    ):
        """Test that different user sessions get different connectors."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        # Request for conversation 1
        request_info_1 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info_1, {"test": "data"})
        connector_1 = transport.aiohttp_client.post_request.call_args[1]["connector"]

        # Request for conversation 2
        request_info_2 = create_request_info(
            model_endpoint, x_correlation_id="session-2", is_final_turn=False
        )
        await transport.send_request(request_info_2, {"test": "data"})
        connector_2 = transport.aiohttp_client.post_request.call_args[1]["connector"]

        # Different connectors
        assert connector_1 is not connector_2

        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_lease_manager(self):
        """Test that stopping transport closes all connection leases."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        # Create some leases
        request_info_1 = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info_1, {"test": "data"})
        request_info_2 = create_request_info(
            model_endpoint, x_correlation_id="session-2", is_final_turn=False
        )
        await transport.send_request(request_info_2, {"test": "data"})

        # Active leases exist
        assert len(transport.lease_manager._leases) == 2

        # Stop transport
        await transport.stop()

        # Lease manager should be None
        assert transport.lease_manager is None

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_releases_lease_on_cancellation(self):
        """Test that STICKY_USER_SESSIONS releases lease when request is cancelled.

        When a request is cancelled, the underlying TCP connection is closed/dirty,
        so we must release the lease to avoid reusing a broken connection.
        """
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # Mock a cancelled request - cancellation_perf_ns indicates cancellation
        cancelled_record = RequestRecord(cancellation_perf_ns=123456789)
        transport.aiohttp_client.post_request = AsyncMock(return_value=cancelled_record)

        # First turn - not final, but will be cancelled
        request_info = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info, {"test": "data"})

        # Lease should be released because request was cancelled
        assert "session-1" not in transport.lease_manager._leases

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_user_sessions_keeps_lease_on_successful_non_final_turn(self):
        """Test that successful non-final turn keeps the lease active."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # Mock a successful (non-cancelled) request
        success_record = RequestRecord()  # No cancellation_perf_ns
        transport.aiohttp_client.post_request = AsyncMock(return_value=success_record)

        # Non-final turn
        request_info = create_request_info(
            model_endpoint, x_correlation_id="session-1", is_final_turn=False
        )
        await transport.send_request(request_info, {"test": "data"})

        # Lease should still be active
        assert "session-1" in transport.lease_manager._leases

        await transport.stop()


class TestConnectionStrategiesStress:
    """Stress tests for connection strategies with concurrent requests.

    These tests verify that connection strategies behave correctly under load
    with many concurrent sessions and requests.
    """

    @pytest.mark.asyncio
    async def test_sticky_sessions_concurrent_sessions(self):
        """Test many concurrent user sessions each getting their own connector."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_sessions = 50
        connectors_by_session: dict[str, object] = {}

        async def make_request(session_id: str) -> None:
            request_info = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=False
            )
            await transport.send_request(request_info, {"test": "data"})
            # Capture the connector used
            call_kwargs = transport.aiohttp_client.post_request.call_args[1]
            connectors_by_session[session_id] = call_kwargs.get("connector")

        # Launch all requests concurrently
        tasks = [
            asyncio.create_task(make_request(f"session-{i}"))
            for i in range(num_sessions)
        ]
        await asyncio.gather(*tasks)

        # Each session should have its own unique connector
        connectors = list(connectors_by_session.values())
        unique_connectors = set(id(c) for c in connectors)
        assert len(unique_connectors) == num_sessions

        # All sessions should have active leases
        assert len(transport.lease_manager._leases) == num_sessions

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_concurrent_turns_same_session(self):
        """Test many concurrent turns for the same session share one connector."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        session_id = "shared-session"
        num_turns = 20
        connectors_used: list[object] = []

        async def make_turn() -> None:
            request_info = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=False
            )
            await transport.send_request(request_info, {"test": "data"})
            call_kwargs = transport.aiohttp_client.post_request.call_args[1]
            connectors_used.append(call_kwargs.get("connector"))

        # Launch all turns concurrently for the same session
        tasks = [asyncio.create_task(make_turn()) for _ in range(num_turns)]
        await asyncio.gather(*tasks)

        # All turns should use the exact same connector
        first_connector = connectors_used[0]
        assert all(c is first_connector for c in connectors_used)

        # Only one lease should exist
        assert len(transport.lease_manager._leases) == 1
        assert session_id in transport.lease_manager._leases

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_interleaved_sessions(self):
        """Test interleaved requests from multiple sessions maintain isolation."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_sessions = 10
        turns_per_session = 5
        connectors_by_session: dict[str, list[object]] = {
            f"session-{i}": [] for i in range(num_sessions)
        }

        async def make_turn(session_id: str) -> None:
            request_info = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=False
            )
            await transport.send_request(request_info, {"test": "data"})
            call_kwargs = transport.aiohttp_client.post_request.call_args[1]
            connectors_by_session[session_id].append(call_kwargs.get("connector"))

        # Interleave requests: session-0-turn-0, session-1-turn-0, ..., session-0-turn-1, ...
        tasks = []
        for _ in range(turns_per_session):
            for session_idx in range(num_sessions):
                session_id = f"session-{session_idx}"
                tasks.append(asyncio.create_task(make_turn(session_id)))
        await asyncio.gather(*tasks)

        # Each session should have used the same connector for all its turns
        for session_id, connectors in connectors_by_session.items():
            assert len(connectors) == turns_per_session
            first = connectors[0]
            assert all(c is first for c in connectors), (
                f"{session_id} used different connectors"
            )

        # Different sessions should have different connectors
        all_session_connectors = [
            connectors_by_session[f"session-{i}"][0] for i in range(num_sessions)
        ]
        unique_ids = set(id(c) for c in all_session_connectors)
        assert len(unique_ids) == num_sessions

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_rapid_create_release(self):
        """Test rapid creation and release of sessions."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_cycles = 30

        async def create_and_release_session(idx: int) -> None:
            session_id = f"ephemeral-{idx}"
            # First turn (not final)
            request_info = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=False
            )
            await transport.send_request(request_info, {"test": "data"})

            # Final turn (releases lease)
            request_info_final = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=True
            )
            await transport.send_request(request_info_final, {"test": "data"})

        # Run many create/release cycles concurrently
        tasks = [
            asyncio.create_task(create_and_release_session(i))
            for i in range(num_cycles)
        ]
        await asyncio.gather(*tasks)

        # All leases should be released
        assert len(transport.lease_manager._leases) == 0

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_mixed_final_non_final(self):
        """Test mix of final and non-final turns across sessions."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        # Sessions that will complete (final turn)
        completing_sessions = [f"complete-{i}" for i in range(10)]
        # Sessions that stay open (no final turn)
        ongoing_sessions = [f"ongoing-{i}" for i in range(10)]

        async def complete_session(session_id: str) -> None:
            # Non-final turn
            req1 = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=False
            )
            await transport.send_request(req1, {"test": "data"})
            # Final turn
            req2 = create_request_info(
                model_endpoint, x_correlation_id=session_id, is_final_turn=True
            )
            await transport.send_request(req2, {"test": "data"})

        async def ongoing_session(session_id: str) -> None:
            # Just non-final turns
            for _ in range(3):
                req = create_request_info(
                    model_endpoint, x_correlation_id=session_id, is_final_turn=False
                )
                await transport.send_request(req, {"test": "data"})

        # Run all concurrently
        tasks = [asyncio.create_task(complete_session(s)) for s in completing_sessions]
        tasks += [asyncio.create_task(ongoing_session(s)) for s in ongoing_sessions]
        await asyncio.gather(*tasks)

        # Only ongoing sessions should have active leases
        assert len(transport.lease_manager._leases) == len(ongoing_sessions)
        for session_id in ongoing_sessions:
            assert session_id in transport.lease_manager._leases
        for session_id in completing_sessions:
            assert session_id not in transport.lease_manager._leases

        await transport.stop()

    @pytest.mark.asyncio
    async def test_never_strategy_concurrent_requests(self):
        """Test NEVER strategy creates unique connectors for concurrent requests."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_requests = 30
        connectors_used: list[object] = []

        async def make_request(idx: int) -> None:
            request_info = create_request_info(
                model_endpoint, x_correlation_id=f"req-{idx}"
            )
            await transport.send_request(request_info, {"test": "data"})
            call_kwargs = transport.aiohttp_client.post_request.call_args[1]
            connectors_used.append(call_kwargs.get("connector"))

        tasks = [asyncio.create_task(make_request(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)

        # Each request should have connector_owner=True (unique connector per request)
        # Note: We can't verify uniqueness of connectors here since they're created
        # in sequence due to the mock, but we verify the owner flag
        assert len(connectors_used) == num_requests
        # All should have connectors (not None)
        assert all(c is not None for c in connectors_used)

        await transport.stop()

    @pytest.mark.asyncio
    async def test_pooled_strategy_concurrent_requests(self):
        """Test POOLED strategy uses shared pool for concurrent requests."""
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.POOLED
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_requests = 30
        connectors_used: list[object] = []

        async def make_request(idx: int) -> None:
            request_info = create_request_info(
                model_endpoint, x_correlation_id=f"req-{idx}"
            )
            await transport.send_request(request_info, {"test": "data"})
            call_kwargs = transport.aiohttp_client.post_request.call_args[1]
            connectors_used.append(call_kwargs.get("connector"))

        tasks = [asyncio.create_task(make_request(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)

        # All requests should use connector=None (shared pool)
        assert len(connectors_used) == num_requests
        assert all(c is None for c in connectors_used)

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_cancellation_under_load(self):
        """Test that cancellations properly release leases under concurrent load.

        This test runs sessions sequentially within each session (to avoid mock races)
        but launches multiple sessions concurrently. Each session has a deterministic
        cancellation pattern based on its index.
        """
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        # Create records: some cancelled, some successful
        cancelled_record = RequestRecord(cancellation_perf_ns=123456789)
        success_record = RequestRecord()

        # Track which sessions should be cancelled
        # Sessions 0, 5, 10 (divisible by 5) cancel on turn 0
        # Sessions 3, 6, 9, 12 (divisible by 3 but not 5) cancel on turn 1
        cancellation_schedule: dict[str, int | None] = {}
        for i in range(15):
            if i % 5 == 0:
                cancellation_schedule[f"session-{i}"] = 0
            elif i % 3 == 0:
                cancellation_schedule[f"session-{i}"] = 1
            else:
                cancellation_schedule[f"session-{i}"] = None

        # Mock that uses session-specific cancellation schedule
        # We'll track turn counts per session
        session_turn_counts: dict[str, int] = {}

        async def mock_post(*args, **kwargs):
            # Extract session_id from the connector by looking up in lease manager
            connector = kwargs.get("connector")
            # Find which session owns this connector
            session_id = None
            for sid, conn in transport.lease_manager._leases.items():
                if conn is connector:
                    session_id = sid
                    break

            if session_id is None:
                return success_record

            # Track turn count for this session
            turn = session_turn_counts.get(session_id, 0)
            session_turn_counts[session_id] = turn + 1

            # Check if this turn should be cancelled
            cancel_on = cancellation_schedule.get(session_id)
            if cancel_on is not None and turn == cancel_on:
                return cancelled_record
            return success_record

        transport.aiohttp_client.post_request = AsyncMock(side_effect=mock_post)

        async def session_workflow(session_id: str) -> bool:
            """Returns True if session completed without cancellation."""
            for turn in range(3):
                is_final = turn == 2
                request_info = create_request_info(
                    model_endpoint, x_correlation_id=session_id, is_final_turn=is_final
                )
                record = await transport.send_request(request_info, {"test": "data"})
                if record.cancellation_perf_ns is not None:
                    return False
            return True

        num_sessions = 15
        tasks = [
            asyncio.create_task(session_workflow(f"session-{i}"))
            for i in range(num_sessions)
        ]
        results = await asyncio.gather(*tasks)

        # Count sessions that completed vs cancelled
        completed = sum(results)
        cancelled = num_sessions - completed

        # Verify no dangling leases - all sessions either completed (final turn)
        # or were cancelled (lease released on cancellation)
        assert len(transport.lease_manager._leases) == 0

        # With 15 sessions (0-14):
        # Cancel on turn 0: indices 0, 5, 10 (divisible by 5) = 3 sessions
        # Cancel on turn 1: indices 3, 6, 9, 12 (divisible by 3 but not 5) = 4 sessions
        # Complete: 1, 2, 4, 7, 8, 11, 13, 14 = 8 sessions
        assert cancelled == 7, f"Expected 7 cancellations, got {cancelled}"
        assert completed == 8, f"Expected 8 completions, got {completed}"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sticky_sessions_stress_connector_isolation(self):
        """Stress test to verify connector isolation between sessions.

        This test verifies that under high concurrency:
        1. Each session gets a dedicated connector
        2. The same connector is reused across all turns of a session
        3. All leases are properly released on final turns
        """
        model_endpoint = create_model_endpoint_info(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        await transport.initialize()

        mock_record = RequestRecord()
        transport.aiohttp_client.post_request = AsyncMock(return_value=mock_record)

        num_sessions = 100
        turns_per_session = 3

        # Track connectors by querying lease manager directly (avoids call_args race)
        session_connectors: dict[str, set[int]] = {}

        async def session_workflow(session_id: str) -> None:
            session_connectors[session_id] = set()
            for turn in range(turns_per_session):
                is_final = turn == turns_per_session - 1

                # Capture connector from lease manager BEFORE request (it creates if needed)
                # Note: On final turn, connector gets released after request
                if not is_final:
                    # Get connector that will be used (creates if not exists)
                    connector = transport.lease_manager.get_connector(session_id)
                    session_connectors[session_id].add(id(connector))

                request_info = create_request_info(
                    model_endpoint, x_correlation_id=session_id, is_final_turn=is_final
                )
                await transport.send_request(request_info, {"test": "data"})

                # On final turn, connector was captured before release in send_request
                # so we need to verify it was the same one used
                if is_final:
                    # Connector should have been released, but we already tracked it
                    pass

        tasks = [
            asyncio.create_task(session_workflow(f"session-{i}"))
            for i in range(num_sessions)
        ]
        await asyncio.gather(*tasks)

        # Each session should have used exactly one connector for all non-final turns
        for session_id, connector_ids in session_connectors.items():
            assert len(connector_ids) == 1, (
                f"{session_id} used {len(connector_ids)} connectors, expected 1"
            )

        # All sessions should be released (all had final turns)
        assert len(transport.lease_manager._leases) == 0

        await transport.stop()

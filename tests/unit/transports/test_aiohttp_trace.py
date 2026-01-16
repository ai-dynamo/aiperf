# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for aiohttp trace functionality."""

from unittest.mock import Mock

import aiohttp
import pytest

from aiperf.common.models import AioHttpTraceData
from aiperf.transports.aiohttp_trace import create_aiohttp_trace_config


def create_mock_response(
    status: int = 200,
    reason: str = "OK",
    headers: dict | None = None,
    local_addr: tuple[str, int] = ("192.168.1.100", 54321),
    remote_addr: tuple[str, int] = ("10.0.0.1", 8080),
) -> Mock:
    """Create a mock response with connection/transport for socket info capture.

    Args:
        status: HTTP status code
        reason: HTTP status reason phrase
        headers: Response headers dict
        local_addr: Local (IP, port) tuple
        remote_addr: Remote (IP, port) tuple

    Returns:
        Mock response object with properly configured connection.transport
    """
    mock_response = Mock()
    mock_response.status = status
    mock_response.reason = reason
    mock_response.headers = headers or {}

    mock_transport = Mock()
    mock_transport.get_extra_info.side_effect = lambda key: {
        "sockname": local_addr,
        "peername": remote_addr,
    }.get(key)
    mock_response.connection.transport = mock_transport

    return mock_response


class TestCreateAioHttpTraceConfig:
    """Tests for create_aiohttp_trace_config function."""

    def test_creates_trace_config(self):
        """Function creates aiohttp TraceConfig."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        assert isinstance(trace_config, aiohttp.TraceConfig)
        assert trace_data.reference_time_ns is not None
        assert trace_data.reference_perf_ns is not None

    def test_initializes_reference_timestamps(self):
        """Trace config initialization sets reference timestamps."""
        trace_data = AioHttpTraceData()
        create_aiohttp_trace_config(trace_data)

        assert trace_data.reference_time_ns is not None
        assert trace_data.reference_perf_ns is not None
        # time_ns should be much larger than perf_counter_ns
        assert trace_data.reference_time_ns > trace_data.reference_perf_ns

    @pytest.mark.asyncio
    async def test_connection_pool_events(self):
        """Connection pool events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        # Create mock parameters
        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Simulate connection pool events
        await trace_config.on_connection_queued_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_pool_wait_start_perf_ns is not None

        await trace_config.on_connection_queued_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_pool_wait_end_perf_ns is not None
        assert (
            trace_data.connection_pool_wait_end_perf_ns
            >= trace_data.connection_pool_wait_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_dns_resolution_events(self):
        """DNS resolution events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # DNS resolution
        await trace_config.on_dns_resolvehost_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_lookup_start_perf_ns is not None

        await trace_config.on_dns_resolvehost_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_lookup_end_perf_ns is not None
        assert trace_data.dns_lookup_end_perf_ns >= trace_data.dns_lookup_start_perf_ns

    @pytest.mark.asyncio
    async def test_dns_cache_events(self):
        """DNS cache hit/miss events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Cache hit
        await trace_config.on_dns_cache_hit[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.dns_cache_hit_perf_ns is not None

        # Cache miss (separate trace data to avoid conflict)
        trace_data2 = AioHttpTraceData()
        trace_config2 = create_aiohttp_trace_config(trace_data2)
        await trace_config2.on_dns_cache_miss[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data2.dns_cache_miss_perf_ns is not None

    @pytest.mark.asyncio
    async def test_tcp_connection_events(self):
        """TCP connection creation events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_trace_ctx.url = "http://example.com"  # HTTP URL
        mock_params = Mock()

        # TCP connection
        await trace_config.on_connection_create_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.tcp_connect_start_perf_ns is not None

        await trace_config.on_connection_create_end[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.tcp_connect_end_perf_ns is not None
        assert (
            trace_data.tcp_connect_end_perf_ns >= trace_data.tcp_connect_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_connection_reuse_event(self):
        """Connection reuse event is tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        await trace_config.on_connection_reuseconn[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.connection_reused_perf_ns is not None

    @pytest.mark.asyncio
    async def test_request_events(self):
        """Request phase events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Request start
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.request_send_start_perf_ns is not None

        # Request headers sent
        mock_params.headers = {"content-type": "application/json"}
        await trace_config.on_request_headers_sent[0](
            mock_session, mock_trace_ctx, mock_params
        )
        assert trace_data.request_headers_sent_perf_ns is not None
        assert trace_data.request_headers == {"content-type": "application/json"}

        # Request end (now captures response metadata)
        mock_params.response = create_mock_response(
            status=200, headers={"content-type": "application/json"}
        )
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)
        # on_request_end sets response_headers_received_perf_ns, not request_send_end_perf_ns
        assert trace_data.response_headers_received_perf_ns is not None
        assert (
            trace_data.response_headers_received_perf_ns
            >= trace_data.request_send_start_perf_ns
        )
        # Verify response metadata was captured
        assert trace_data.response_status_code == 200
        assert trace_data.response_headers == {"content-type": "application/json"}
        # Verify socket info was captured
        assert trace_data.local_ip == "192.168.1.100"
        assert trace_data.local_port == 54321
        assert trace_data.remote_ip == "10.0.0.1"
        assert trace_data.remote_port == 8080

    @pytest.mark.asyncio
    async def test_request_chunk_sent(self):
        """Request chunk sent events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()

        # Send multiple chunks
        for chunk_data in [b"chunk1", b"chunk2", b"chunk3"]:
            mock_params = Mock()
            mock_params.chunk = chunk_data
            await trace_config.on_request_chunk_sent[0](
                mock_session, mock_trace_ctx, mock_params
            )

        # Verify all chunks are tracked (tuples of (timestamp, size))
        assert len(trace_data.request_chunks) == 3
        assert [size for _, size in trace_data.request_chunks] == [
            6,
            6,
            6,
        ]  # len("chunkN")

        # Timestamps should be in increasing order
        assert trace_data.request_chunks[0][0] <= trace_data.request_chunks[1][0]
        assert trace_data.request_chunks[1][0] <= trace_data.request_chunks[2][0]

    @pytest.mark.asyncio
    async def test_response_chunk_received(self):
        """Response chunk received events are tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()

        # First chunk (note: response metadata is captured in on_request_end, not here)
        mock_params = Mock()
        mock_params.chunk = b"first chunk"

        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify first chunk tracked (tuples of (timestamp, size))
        assert len(trace_data.response_chunks) == 1
        assert trace_data.response_chunks[0][1] == 11  # len("first chunk")
        # Verify start/end timestamps are set
        assert trace_data.response_receive_start_perf_ns is not None
        assert trace_data.response_receive_end_perf_ns is not None

        # Second chunk
        mock_params.chunk = b"second chunk"
        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify second chunk tracked
        assert len(trace_data.response_chunks) == 2
        assert trace_data.response_chunks[1][1] == 12  # len("second chunk")
        # Verify end timestamp updated
        assert (
            trace_data.response_receive_end_perf_ns
            >= trace_data.response_receive_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_request_exception_event(self):
        """Request exception event is tracked."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()
        mock_params.exception = Exception("Test error")

        await trace_config.on_request_exception[0](
            mock_session, mock_trace_ctx, mock_params
        )

        assert trace_data.error_timestamp_perf_ns is not None

    @pytest.mark.asyncio
    async def test_all_callbacks_registered(self):
        """All trace callbacks are registered on TraceConfig."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        # Connection pool
        assert len(trace_config.on_connection_queued_start) == 1
        assert len(trace_config.on_connection_queued_end) == 1

        # Connection creation
        assert len(trace_config.on_connection_create_start) == 1
        assert len(trace_config.on_connection_create_end) == 1

        # Connection reuse
        assert len(trace_config.on_connection_reuseconn) == 1

        # DNS
        assert len(trace_config.on_dns_resolvehost_start) == 1
        assert len(trace_config.on_dns_resolvehost_end) == 1
        assert len(trace_config.on_dns_cache_hit) == 1
        assert len(trace_config.on_dns_cache_miss) == 1

        # Request
        assert len(trace_config.on_request_start) == 1
        assert len(trace_config.on_request_chunk_sent) == 1
        assert len(trace_config.on_request_end) == 1
        assert len(trace_config.on_request_headers_sent) == 1

        # Response
        assert len(trace_config.on_response_chunk_received) == 1

        # Exception
        assert len(trace_config.on_request_exception) == 1

    @pytest.mark.asyncio
    async def test_socket_info_capture(self):
        """Socket info (local/remote IP and port) is captured in on_request_end."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Set up mock response with custom socket addresses
        mock_params.response = create_mock_response(
            status=200,
            headers={},
            local_addr=("10.20.30.40", 12345),
            remote_addr=("203.0.113.50", 443),
        )
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # Verify socket info was captured
        assert trace_data.local_ip == "10.20.30.40"
        assert trace_data.local_port == 12345
        assert trace_data.remote_ip == "203.0.113.50"
        assert trace_data.remote_port == 443

    @pytest.mark.asyncio
    async def test_socket_info_with_none_connection(self):
        """Handles None connection gracefully."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Set up mock response with None connection
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.connection = None
        mock_params.response = mock_response
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # Socket info should remain None
        assert trace_data.local_ip is None
        assert trace_data.local_port is None
        assert trace_data.remote_ip is None
        assert trace_data.remote_port is None


# Integration and edge case tests
class TestAioHttpTraceIntegration:
    """Integration tests for aiohttp trace functionality."""

    @pytest.mark.asyncio
    async def test_complete_request_lifecycle(self):
        """Complete request lifecycle with all connection and request events."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Simulate complete lifecycle
        # 1. Connection pool
        await trace_config.on_connection_queued_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_connection_queued_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 2. DNS resolution
        await trace_config.on_dns_resolvehost_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_dns_resolvehost_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 3. TCP connection (includes TLS for HTTPS)
        await trace_config.on_connection_create_start[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_connection_create_end[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # 4. Request
        mock_request_params = Mock()
        mock_request_params.url = "https://example.com"
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_request_params
        )
        mock_params.headers = {"user-agent": "test"}
        await trace_config.on_request_headers_sent[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Set up mock response for on_request_end
        mock_params.response = create_mock_response(
            status=200, headers={"content-type": "application/json"}
        )
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # 6. Response chunks
        mock_params.chunk = b"response data"
        await trace_config.on_response_chunk_received[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify complete trace data
        assert trace_data.connection_pool_wait_start_perf_ns is not None
        assert trace_data.dns_lookup_start_perf_ns is not None
        assert trace_data.tcp_connect_start_perf_ns is not None
        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.response_status_code == 200

        # Verify export works with all computed fields
        export = trace_data.to_export()
        assert export.blocked_ns is not None
        assert export.dns_lookup_ns is not None
        assert export.connecting_ns is not None

    @pytest.mark.asyncio
    async def test_connection_reuse_scenario(self):
        """Request using reused connection skips DNS/TCP."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Connection reuse scenario
        await trace_config.on_connection_reuseconn[0](
            mock_session, mock_trace_ctx, mock_params
        )
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify connection reuse recorded
        assert trace_data.connection_reused_perf_ns is not None

        # DNS and TCP should not be set for reused connection
        assert trace_data.dns_lookup_start_perf_ns is None
        assert trace_data.tcp_connect_start_perf_ns is None

    @pytest.mark.asyncio
    async def test_error_during_request(self):
        """Error during request captures error timestamp."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Start request
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Error occurs
        mock_params.exception = aiohttp.ClientError("Connection failed")
        await trace_config.on_request_exception[0](
            mock_session, mock_trace_ctx, mock_params
        )

        assert trace_data.request_send_start_perf_ns is not None
        assert trace_data.error_timestamp_perf_ns is not None
        assert (
            trace_data.error_timestamp_perf_ns >= trace_data.request_send_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_empty_response_body(self):
        """Response with no body still captures metadata."""
        trace_data = AioHttpTraceData()
        trace_config = create_aiohttp_trace_config(trace_data)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Request with response metadata but no body chunks
        mock_request_params = Mock()
        mock_request_params.url = "http://example.com"
        await trace_config.on_request_start[0](
            mock_session, mock_trace_ctx, mock_request_params
        )

        # Set up mock response for on_request_end (204 No Content)
        mock_params.response = create_mock_response(status=204, headers={})
        await trace_config.on_request_end[0](mock_session, mock_trace_ctx, mock_params)

        # No response chunks received (empty body)
        assert len(trace_data.response_chunks) == 0
        # But response status is captured in on_request_end
        assert trace_data.response_status_code == 204

    @pytest.mark.asyncio
    async def test_multiple_trace_configs_independent(self):
        """Multiple trace configs operate independently."""
        trace_data1 = AioHttpTraceData()
        trace_data2 = AioHttpTraceData()

        trace_config1 = create_aiohttp_trace_config(trace_data1)
        trace_config2 = create_aiohttp_trace_config(trace_data2)

        mock_session = Mock(spec=aiohttp.ClientSession)
        mock_trace_ctx = Mock()
        mock_params = Mock()

        # Trigger events on first config
        await trace_config1.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Verify only first trace data is affected
        assert trace_data1.request_send_start_perf_ns is not None
        assert trace_data2.request_send_start_perf_ns is None

        # Trigger events on second config
        await trace_config2.on_request_start[0](
            mock_session, mock_trace_ctx, mock_params
        )

        # Both should now have data
        assert trace_data1.request_send_start_perf_ns is not None
        assert trace_data2.request_send_start_perf_ns is not None

        # But values should be different (different timestamps)
        assert (
            trace_data1.request_send_start_perf_ns
            != trace_data2.request_send_start_perf_ns
        )

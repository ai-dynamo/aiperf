# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GrpcTransport."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    CreditPhase,
    ModelSelectionStrategy,
)
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.models import TextResponse
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin.enums import EndpointType
from aiperf.transports.grpc.grpc_client import GrpcUnaryResult
from aiperf.transports.grpc.grpc_transport import (
    GrpcChannelLeaseManager,
    GrpcTransport,
    _metadata_to_dict,
)
from aiperf.transports.grpc.kserve_v2_serializers import KServeV2GrpcSerializer
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.trace_data import GrpcTraceData

# Method paths for tests (match plugins.yaml kserve_v2_infer metadata)
_V2_UNARY_METHOD = "/inference.GRPCInferenceService/ModelInfer"
_V2_STREAM_METHOD = "/inference.GRPCInferenceService/ModelStreamInfer"

_SIMPLE_PAYLOAD = {
    "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
}


def create_grpc_model_endpoint(
    base_url: str = "grpc://localhost:8001",
    base_urls: list[str] | None = None,
    model_name: str = "test-model",
    streaming: bool = False,
    connection_reuse_strategy: ConnectionReuseStrategy = ConnectionReuseStrategy.POOLED,
) -> ModelEndpointInfo:
    """Create ModelEndpointInfo for gRPC transport tests."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.KSERVE_V2_INFER,
            base_urls=base_urls or [base_url],
            streaming=streaming,
            connection_reuse_strategy=connection_reuse_strategy,
        ),
    )


def create_request_info(
    model_endpoint: ModelEndpointInfo,
    *,
    x_request_id: str = "test-request-id",
    x_correlation_id: str = "test-correlation-id",
    cancel_after_ns: int | None = None,
    is_final_turn: bool = True,
) -> RequestInfo:
    """Create RequestInfo for transport tests."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=x_request_id,
        x_correlation_id=x_correlation_id,
        conversation_id="test-conversation",
        cancel_after_ns=cancel_after_ns,
        is_final_turn=is_final_turn,
    )


def make_infer_response(text: str = "Generated text") -> pb2.ModelInferResponse:
    """Create a ModelInferResponse protobuf with BYTES output."""
    response = pb2.ModelInferResponse()
    response.model_name = "test-model"
    output = response.outputs.add()
    output.name = "text_output"
    output.datatype = "BYTES"
    output.shape.append(1)
    output.contents.bytes_contents.append(text.encode("utf-8"))
    return response


def make_stream_response(text: str = "token") -> pb2.ModelStreamInferResponse:
    """Create a ModelStreamInferResponse wrapping an infer response."""
    stream_resp = pb2.ModelStreamInferResponse()
    stream_resp.infer_response.CopyFrom(make_infer_response(text))
    return stream_resp


def make_stream_error_response(error: str) -> pb2.ModelStreamInferResponse:
    """Create a ModelStreamInferResponse with an error."""
    stream_resp = pb2.ModelStreamInferResponse()
    stream_resp.error_message = error
    return stream_resp


def make_unary_result(
    text: str = "Generated text",
    trailing_metadata: tuple[tuple[str, str], ...] = (),
) -> GrpcUnaryResult:
    """Create a GrpcUnaryResult wrapping a ModelInferResponse."""
    return GrpcUnaryResult(
        data=make_infer_response(text).SerializeToString(),
        trailing_metadata=trailing_metadata,
    )


class MockStreamCall:
    """Mock GrpcStreamCall for testing streaming transport."""

    def __init__(
        self,
        chunks: list[bytes],
        trailing: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._chunks = chunks
        self._trailing = trailing

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._trailing

    async def initial_metadata(self) -> tuple[()]:
        return ()

    def cancel(self) -> bool:
        return True


def make_mock_stream_call(
    *responses: pb2.ModelStreamInferResponse,
    trailing_metadata: tuple[tuple[str, str], ...] = (),
) -> MockStreamCall:
    """Create a MockStreamCall from protobuf responses."""
    chunks = [r.SerializeToString() for r in responses]
    return MockStreamCall(chunks, trailing=trailing_metadata)


def _init_transport_serializer(transport: GrpcTransport) -> None:
    """Wire V2 serializer and method paths onto a transport for testing."""
    transport._serializer = KServeV2GrpcSerializer()
    transport._unary_method = _V2_UNARY_METHOD
    transport._stream_method = _V2_STREAM_METHOD


def _make_mock_client(
    *,
    unary_result: GrpcUnaryResult | None = None,
    stream_call: MockStreamCall | None = None,
    wait_for_ready: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock gRPC client with common defaults."""
    mock_client = MagicMock()
    if unary_result is not None:
        mock_client.unary = AsyncMock(return_value=unary_result)
    if stream_call is not None:
        mock_client.server_stream = MagicMock(return_value=stream_call)
    mock_client.wait_for_ready = wait_for_ready or AsyncMock()
    mock_client.close = AsyncMock()
    return mock_client


def _set_pool(
    transport: GrpcTransport,
    mock_client: MagicMock,
    target: str = "localhost:8001",
) -> None:
    """Set the channel pool on a transport for testing."""
    transport._channel_pool = {target: mock_client}


class TestGrpcTransportGetUrl:
    """Tests for URL handling."""

    def test_get_url_strips_grpc_scheme(self) -> None:
        """get_url should return host:port without scheme."""
        endpoint = create_grpc_model_endpoint(base_url="grpc://triton:8001")
        transport = GrpcTransport(model_endpoint=endpoint)
        request_info = create_request_info(endpoint)

        url = transport.get_url(request_info)
        assert url == "triton:8001"

    def test_get_url_strips_grpcs_scheme(self) -> None:
        """get_url should return host:port for grpcs scheme."""
        endpoint = create_grpc_model_endpoint(base_url="grpcs://secure-triton:8001")
        transport = GrpcTransport(model_endpoint=endpoint)
        request_info = create_request_info(endpoint)

        url = transport.get_url(request_info)
        assert url == "secure-triton:8001"

    def test_get_url_multi_url(self) -> None:
        """get_url should use url_index for multi-URL load balancing."""
        endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="m")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.KSERVE_V2_INFER,
                base_urls=["grpc://host1:8001", "grpc://host2:8001"],
                streaming=False,
            ),
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        ri0 = create_request_info(endpoint)
        ri0.url_index = 0
        assert transport.get_url(ri0) == "host1:8001"

        ri1 = create_request_info(endpoint)
        ri1.url_index = 1
        assert transport.get_url(ri1) == "host2:8001"


class TestGrpcTransportHeaders:
    """Tests for header/metadata handling."""

    def test_get_transport_headers_returns_empty(self) -> None:
        """gRPC transport returns empty transport headers."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        request_info = create_request_info(endpoint)

        assert transport.get_transport_headers(request_info) == {}


class TestGrpcTransportSendRequest:
    """Tests for send_request method."""

    @pytest.fixture
    def transport_and_endpoint(self):
        """Create transport with mocked gRPC client and V2 serializer."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        return transport, endpoint

    @pytest.mark.asyncio
    async def test_send_request_unary_success(self, transport_and_endpoint) -> None:
        """Successful unary ModelInfer request."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello"],
                }
            ]
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert record.error is None
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert record.start_perf_ns > 0
        assert record.end_perf_ns is not None
        assert record.end_perf_ns >= record.start_perf_ns
        mock_client.unary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_request_streaming_success(self) -> None:
        """Successful streaming ModelStreamInfer request."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("Hello"),
            make_stream_response(" world"),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["prompt"],
                }
            ]
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        assert record.error is None
        assert len(record.responses) == 2
        assert all(isinstance(r, TextResponse) for r in record.responses)

    @pytest.mark.asyncio
    async def test_send_request_streaming_with_first_token_callback(self) -> None:
        """Streaming request fires first_token_callback on first response."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("token1"),
            make_stream_response("token2"),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        callback_calls = []

        async def callback(ttft_ns: int, message) -> bool:
            callback_calls.append((ttft_ns, message))
            return True  # First token is meaningful

        request_info = create_request_info(endpoint)

        record = await transport.send_request(
            request_info, _SIMPLE_PAYLOAD, first_token_callback=callback
        )

        assert len(callback_calls) == 1
        assert callback_calls[0][0] > 0  # ttft_ns > 0
        assert record.status == 200

    @pytest.mark.asyncio
    async def test_send_request_streaming_error_in_stream(self) -> None:
        """Streaming request with error in stream sets error on record."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("token1"),
            make_stream_error_response("Model crashed"),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert "Model crashed" in record.error.message
        # First token before error should still be collected
        assert len(record.responses) == 1

    @pytest.mark.asyncio
    async def test_send_request_grpc_error(self, transport_and_endpoint) -> None:
        """gRPC RPC error maps to correct HTTP status."""
        transport, endpoint = transport_and_endpoint

        # Create a proper AioRpcError
        rpc_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Server not available",
        )

        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=rpc_error)
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert record.status == 503  # UNAVAILABLE -> 503
        assert "gRPC:UNAVAILABLE" in record.error.type

    @pytest.mark.asyncio
    async def test_send_request_with_cancel_after_ns(
        self, transport_and_endpoint
    ) -> None:
        """Request with cancel_after_ns that completes before timeout."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=5_000_000_000)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.status == 200
        assert record.error is None
        assert record.cancellation_perf_ns is None

    @pytest.mark.asyncio
    async def test_send_request_cancelled_by_timeout(
        self, transport_and_endpoint
    ) -> None:
        """Request cancelled by cancel_after_ns timeout."""
        transport, endpoint = transport_and_endpoint

        async def slow_unary(*args, **kwargs):
            # Use a Future that never resolves to simulate a slow request.
            # asyncio.sleep is auto-mocked to be instant in tests.
            await asyncio.get_event_loop().create_future()
            return make_unary_result()

        mock_client = _make_mock_client()
        mock_client.unary = slow_unary
        _set_pool(transport, mock_client)

        # Cancel after 1ns (effectively immediate)
        request_info = create_request_info(endpoint, cancel_after_ns=1)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert record.error.code == 499
        assert record.cancellation_perf_ns is not None

    @pytest.mark.asyncio
    async def test_headers_passed_as_grpc_metadata(
        self, transport_and_endpoint
    ) -> None:
        """Headers from build_headers are passed as gRPC metadata."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {"Authorization": "Bearer token123"}

        await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        call_kwargs = mock_client.unary.call_args[1]
        metadata = call_kwargs.get("metadata")
        assert metadata is not None
        metadata_dict = dict(metadata)
        assert "authorization" in metadata_dict
        assert metadata_dict["authorization"] == "Bearer token123"

    @pytest.mark.asyncio
    async def test_trace_data_populated(self, transport_and_endpoint) -> None:
        """Trace data should be populated with gRPC timing info."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert isinstance(record.trace_data, GrpcTraceData)
        assert record.trace_data.trace_type == "grpc"
        assert record.trace_data.grpc_status_code == 0  # OK
        assert record.trace_data.request_send_start_perf_ns is not None
        assert record.trace_data.response_receive_start_perf_ns is not None
        assert record.trace_data.response_receive_end_perf_ns is not None
        assert len(record.trace_data.request_chunks) > 0
        assert len(record.trace_data.response_chunks) > 0


class TestGrpcTransportTraceFields:
    """Tests for trace data field population (parity with aiohttp)."""

    @pytest.fixture
    def transport_and_endpoint(self):
        """Create transport with mocked gRPC client and V2 serializer."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        return transport, endpoint

    @pytest.mark.asyncio
    async def test_unary_request_headers_recorded(self, transport_and_endpoint) -> None:
        """Request headers (gRPC metadata) should be recorded in trace data."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {"X-Custom": "value"}

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.trace_data.request_headers is not None
        assert "x-custom" in record.trace_data.request_headers
        assert record.trace_data.request_headers["x-custom"] == "value"

    @pytest.mark.asyncio
    async def test_unary_request_headers_sent_timestamp(
        self, transport_and_endpoint
    ) -> None:
        """request_headers_sent_perf_ns should be set for unary requests."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.request_headers_sent_perf_ns is not None
        assert (
            record.trace_data.request_headers_sent_perf_ns
            == record.trace_data.request_send_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_unary_response_status_fields(self, transport_and_endpoint) -> None:
        """response_status_code and response_reason should be set on success."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_status_code == 200
        assert record.trace_data.response_reason == "OK"

    @pytest.mark.asyncio
    async def test_unary_response_headers_received_timestamp(
        self, transport_and_endpoint
    ) -> None:
        """response_headers_received_perf_ns should be set for unary requests."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_headers_received_perf_ns is not None
        assert (
            record.trace_data.response_headers_received_perf_ns
            == record.trace_data.response_receive_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_unary_trailing_metadata_as_response_headers(
        self, transport_and_endpoint
    ) -> None:
        """Trailing metadata should be captured as response_headers."""
        transport, endpoint = transport_and_endpoint
        result = make_unary_result(
            trailing_metadata=(("x-server-id", "gpu-0"), ("x-model-version", "2")),
        )
        mock_client = _make_mock_client(unary_result=result)
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_headers is not None
        assert record.trace_data.response_headers["x-server-id"] == "gpu-0"
        assert record.trace_data.response_headers["x-model-version"] == "2"

    @pytest.mark.asyncio
    async def test_grpc_error_populates_response_status_fields(
        self, transport_and_endpoint
    ) -> None:
        """gRPC error should set response_status_code and response_reason in trace."""
        transport, endpoint = transport_and_endpoint

        rpc_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Server not available",
        )
        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=rpc_error)
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_status_code == 503
        assert record.trace_data.response_reason == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_streaming_response_headers_received_on_first_chunk(self) -> None:
        """Streaming: response_headers_received_perf_ns set on first chunk."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("a"),
            make_stream_response("b"),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_headers_received_perf_ns is not None
        assert (
            record.trace_data.response_headers_received_perf_ns
            == record.trace_data.response_receive_start_perf_ns
        )

    @pytest.mark.asyncio
    async def test_streaming_trailing_metadata_captured(self) -> None:
        """Streaming: trailing metadata captured as response_headers on success."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("token"),
            trailing_metadata=(("x-request-id", "abc123"),),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_headers is not None
        assert record.trace_data.response_headers["x-request-id"] == "abc123"

    @pytest.mark.asyncio
    async def test_streaming_status_fields_on_success(self) -> None:
        """Streaming: response_status_code and response_reason set on success."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(make_stream_response("ok"))
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.trace_data.response_status_code == 200
        assert record.trace_data.response_reason == "OK"

    @pytest.mark.asyncio
    async def test_default_headers_always_recorded(
        self, transport_and_endpoint
    ) -> None:
        """Default headers (user-agent, x-request-id, etc.) are always recorded."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {}

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        # Default headers are always present (user-agent, x-request-id, x-correlation-id)
        assert record.trace_data.request_headers is not None
        assert "x-request-id" in record.trace_data.request_headers


class TestGrpcTransportCancellation:
    """Tests for two-stage cancellation (channel-ready + cancel timer)."""

    @pytest.fixture
    def transport_and_endpoint(self):
        """Create non-streaming transport with serializer."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        return transport, endpoint

    @pytest.mark.asyncio
    async def test_request_send_timeout_when_channel_not_ready(
        self, transport_and_endpoint
    ) -> None:
        """Channel not ready within timeout should produce RequestSendTimeout."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(
            wait_for_ready=AsyncMock(side_effect=asyncio.TimeoutError),
        )
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=5_000_000_000)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert record.error.type == "RequestSendTimeout"
        assert record.error.code == 0
        assert (
            record.cancellation_perf_ns is None
        )  # Not a cancellation, it's a send timeout

    @pytest.mark.asyncio
    async def test_request_send_timeout_channel_shutdown(
        self, transport_and_endpoint
    ) -> None:
        """Channel in SHUTDOWN state should produce RequestSendTimeout."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(
            wait_for_ready=AsyncMock(
                side_effect=ConnectionError("gRPC channel is shutdown")
            ),
        )
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=5_000_000_000)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert record.error.type == "RequestSendTimeout"
        assert "shutdown" in record.error.message

    @pytest.mark.asyncio
    async def test_two_stage_cancellation_unary(self, transport_and_endpoint) -> None:
        """Unary: channel ready succeeds, then request times out -> RequestCancellationError."""
        transport, endpoint = transport_and_endpoint

        async def slow_unary(*args, **kwargs):
            await asyncio.get_event_loop().create_future()
            return make_unary_result()

        mock_client = _make_mock_client()
        mock_client.unary = slow_unary
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=1)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        # wait_for_ready was called (stage 1)
        mock_client.wait_for_ready.assert_awaited_once()
        # Then timed out waiting for response (stage 2)
        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert record.error.code == 499
        assert record.cancellation_perf_ns is not None

    @pytest.mark.asyncio
    async def test_two_stage_cancellation_streaming(self) -> None:
        """Streaming: channel ready succeeds, then stream times out -> RequestCancellationError."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        class SlowStreamCall(MockStreamCall):
            """Stream that yields one chunk then hangs forever."""

            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield make_stream_response("first").SerializeToString()
                await asyncio.get_event_loop().create_future()

        slow_call = SlowStreamCall(chunks=[], trailing=())
        mock_client = _make_mock_client(stream_call=slow_call)
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=1)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        mock_client.wait_for_ready.assert_awaited_once()
        assert record.error is not None
        assert record.error.type == "RequestCancellationError"
        assert record.cancellation_perf_ns is not None

    @pytest.mark.asyncio
    async def test_streaming_send_timeout(self) -> None:
        """Streaming: channel not ready should produce RequestSendTimeout."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(make_stream_response("x"))
        mock_client = _make_mock_client(
            stream_call=stream_call,
            wait_for_ready=AsyncMock(side_effect=asyncio.TimeoutError),
        )
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=5_000_000_000)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.error is not None
        assert record.error.type == "RequestSendTimeout"

    @pytest.mark.asyncio
    async def test_no_cancel_skips_wait_for_ready(self, transport_and_endpoint) -> None:
        """Without cancel_after_ns, wait_for_ready should not be called."""
        transport, endpoint = transport_and_endpoint
        mock_client = _make_mock_client(unary_result=make_unary_result())
        _set_pool(transport, mock_client)

        request_info = create_request_info(endpoint, cancel_after_ns=None)

        record = await transport.send_request(request_info, _SIMPLE_PAYLOAD)

        assert record.status == 200
        mock_client.wait_for_ready.assert_not_awaited()


class TestGrpcTransportInit:
    """Tests for transport initialization."""

    @pytest.mark.asyncio
    async def test_pooled_creates_channels_for_all_base_urls(self) -> None:
        """POOLED strategy should pre-create channels for all base_urls."""
        endpoint = create_grpc_model_endpoint(
            base_urls=["grpc://host1:8001", "grpc://host2:8001"],
            connection_reuse_strategy=ConnectionReuseStrategy.POOLED,
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        with patch(
            "aiperf.transports.grpc.grpc_transport.GenericGrpcClient"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            await transport._init_grpc_client()

        assert "host1:8001" in transport._channel_pool
        assert "host2:8001" in transport._channel_pool
        assert len(transport._channel_pool) == 2

    @pytest.mark.asyncio
    async def test_never_creates_no_channels(self) -> None:
        """NEVER strategy should not pre-create any channels."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        await transport._init_grpc_client()

        assert len(transport._channel_pool) == 0
        assert transport._lease_manager is None

    @pytest.mark.asyncio
    async def test_sticky_creates_lease_manager(self) -> None:
        """STICKY_USER_SESSIONS strategy should create a lease manager."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        await transport._init_grpc_client()

        assert transport._lease_manager is not None
        assert isinstance(transport._lease_manager, GrpcChannelLeaseManager)
        assert len(transport._channel_pool) == 0

    @pytest.mark.asyncio
    async def test_secure_detected_from_grpcs_scheme(self) -> None:
        """Transport should detect secure=True from grpcs:// scheme."""
        endpoint = create_grpc_model_endpoint(
            base_url="grpcs://secure-host:443",
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER,
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        await transport._init_grpc_client()

        assert transport._secure is True

    @pytest.mark.asyncio
    async def test_mixed_schemes_raises_error(self) -> None:
        """Mixed grpc:// and grpcs:// schemes should raise ValueError."""
        endpoint = create_grpc_model_endpoint(
            base_urls=["grpc://host1:8001", "grpcs://host2:8001"],
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        with pytest.raises(ValueError, match="mixed"):
            await transport._init_grpc_client()


class TestGrpcTransportShutdown:
    """Tests for transport shutdown."""

    @pytest.mark.asyncio
    async def test_close_closes_pool_and_lease_manager(self) -> None:
        """Shutdown should close all pool channels and lease manager."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)

        mock_client1 = _make_mock_client()
        mock_client2 = _make_mock_client()
        transport._channel_pool = {
            "host1:8001": mock_client1,
            "host2:8001": mock_client2,
        }
        mock_lease_mgr = AsyncMock()
        transport._lease_manager = mock_lease_mgr

        await transport._close_grpc_client()

        mock_lease_mgr.close_all.assert_awaited_once()
        mock_client1.close.assert_awaited_once()
        mock_client2.close.assert_awaited_once()
        assert len(transport._channel_pool) == 0
        assert transport._lease_manager is None


class TestGrpcChannelLeaseManager:
    """Tests for GrpcChannelLeaseManager."""

    def test_get_or_create_creates_new(self) -> None:
        """get_or_create should call factory for a new session."""
        manager = GrpcChannelLeaseManager()
        mock_client = _make_mock_client()
        factory = MagicMock(return_value=mock_client)

        result = manager.get_or_create("session-1", factory)

        assert result is mock_client
        factory.assert_called_once()

    def test_get_or_create_returns_existing(self) -> None:
        """get_or_create should return existing client for same session."""
        manager = GrpcChannelLeaseManager()
        mock_client = _make_mock_client()
        factory = MagicMock(return_value=mock_client)

        result1 = manager.get_or_create("session-1", factory)
        result2 = manager.get_or_create("session-1", factory)

        assert result1 is result2
        factory.assert_called_once()  # Only called once

    def test_get_or_create_different_sessions_get_different_clients(self) -> None:
        """Different sessions should get different clients."""
        manager = GrpcChannelLeaseManager()
        client1 = _make_mock_client()
        client2 = _make_mock_client()
        calls = iter([client1, client2])
        factory = MagicMock(side_effect=lambda: next(calls))

        result1 = manager.get_or_create("session-1", factory)
        result2 = manager.get_or_create("session-2", factory)

        assert result1 is client1
        assert result2 is client2
        assert factory.call_count == 2

    @pytest.mark.asyncio
    async def test_release_lease_closes_client(self) -> None:
        """release_lease should close the client and remove it."""
        manager = GrpcChannelLeaseManager()
        mock_client = _make_mock_client()
        manager.get_or_create("session-1", lambda: mock_client)

        await manager.release_lease("session-1")

        mock_client.close.assert_awaited_once()
        # Creating again should use the factory
        mock_client2 = _make_mock_client()
        result = manager.get_or_create("session-1", lambda: mock_client2)
        assert result is mock_client2

    @pytest.mark.asyncio
    async def test_release_lease_noop_for_unknown_session(self) -> None:
        """release_lease should be a no-op for unknown sessions."""
        manager = GrpcChannelLeaseManager()
        await manager.release_lease("unknown")  # Should not raise

    @pytest.mark.asyncio
    async def test_close_all_closes_everything(self) -> None:
        """close_all should close all active leases."""
        manager = GrpcChannelLeaseManager()
        client1 = _make_mock_client()
        client2 = _make_mock_client()
        manager.get_or_create("s1", lambda: client1)
        manager.get_or_create("s2", lambda: client2)

        await manager.close_all()

        client1.close.assert_awaited_once()
        client2.close.assert_awaited_once()
        assert len(manager._leases) == 0


class TestGrpcConnectionStrategies:
    """Tests for connection reuse strategy behavior."""

    @pytest.mark.asyncio
    async def test_pooled_uses_correct_target_per_url_index(self) -> None:
        """POOLED: should use the correct channel for each url_index."""
        endpoint = create_grpc_model_endpoint(
            base_urls=["grpc://host1:8001", "grpc://host2:8001"],
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        client1 = _make_mock_client(unary_result=make_unary_result("resp1"))
        client2 = _make_mock_client(unary_result=make_unary_result("resp2"))
        transport._channel_pool = {
            "host1:8001": client1,
            "host2:8001": client2,
        }

        ri0 = create_request_info(endpoint)
        ri0.url_index = 0
        await transport.send_request(ri0, _SIMPLE_PAYLOAD)
        client1.unary.assert_awaited_once()
        client2.unary.assert_not_awaited()

        ri1 = create_request_info(endpoint)
        ri1.url_index = 1
        await transport.send_request(ri1, _SIMPLE_PAYLOAD)
        client2.unary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pooled_lazily_creates_missing_target(self) -> None:
        """POOLED: should lazily create a channel for a target not in the pool."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False

        # Empty pool - should create lazily
        with patch.object(transport, "_create_client") as mock_create:
            mock_client = _make_mock_client(unary_result=make_unary_result())
            mock_create.return_value = mock_client

            ri = create_request_info(endpoint)
            record = await transport.send_request(ri, _SIMPLE_PAYLOAD)

            mock_create.assert_called_once_with("localhost:8001")
            assert record.status == 200
            assert "localhost:8001" in transport._channel_pool

    @pytest.mark.asyncio
    async def test_never_creates_new_channel_per_request(self) -> None:
        """NEVER: should create a new channel for each request and close after."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False

        mock_client = _make_mock_client(unary_result=make_unary_result())
        with patch.object(transport, "_create_client", return_value=mock_client):
            record = await transport.send_request(
                create_request_info(endpoint), _SIMPLE_PAYLOAD
            )

        assert record.status == 200
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_never_closes_channel_on_error(self) -> None:
        """NEVER: should close the per-request channel even on error."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False

        rpc_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Server not available",
        )
        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=rpc_error)
        with patch.object(transport, "_create_client", return_value=mock_client):
            record = await transport.send_request(
                create_request_info(endpoint), _SIMPLE_PAYLOAD
            )

        assert record.error is not None
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sticky_creates_lease_and_reuses(self) -> None:
        """STICKY: should reuse the same channel across turns of a conversation."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False
        transport._lease_manager = GrpcChannelLeaseManager()

        mock_client = _make_mock_client(unary_result=make_unary_result())
        with patch.object(transport, "_create_client", return_value=mock_client):
            # Turn 1 (not final)
            ri1 = create_request_info(
                endpoint,
                x_correlation_id="conv-1",
                is_final_turn=False,
            )
            await transport.send_request(ri1, _SIMPLE_PAYLOAD)
            mock_client.close.assert_not_awaited()

            # Turn 2 (not final) - should reuse same client
            ri2 = create_request_info(
                endpoint,
                x_correlation_id="conv-1",
                is_final_turn=False,
            )
            await transport.send_request(ri2, _SIMPLE_PAYLOAD)
            mock_client.close.assert_not_awaited()
            assert mock_client.unary.await_count == 2

    @pytest.mark.asyncio
    async def test_sticky_releases_on_final_turn(self) -> None:
        """STICKY: should release the lease on the final turn."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False
        transport._lease_manager = GrpcChannelLeaseManager()

        mock_client = _make_mock_client(unary_result=make_unary_result())
        with patch.object(transport, "_create_client", return_value=mock_client):
            ri = create_request_info(
                endpoint,
                x_correlation_id="conv-1",
                is_final_turn=True,
            )
            await transport.send_request(ri, _SIMPLE_PAYLOAD)

        mock_client.close.assert_awaited_once()
        assert "conv-1" not in transport._lease_manager._leases

    @pytest.mark.asyncio
    async def test_sticky_releases_on_error(self) -> None:
        """STICKY: should release the lease when request errors."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False
        transport._lease_manager = GrpcChannelLeaseManager()

        rpc_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Server not available",
        )
        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=rpc_error)
        with patch.object(transport, "_create_client", return_value=mock_client):
            ri = create_request_info(
                endpoint,
                x_correlation_id="conv-1",
                is_final_turn=False,
            )
            await transport.send_request(ri, _SIMPLE_PAYLOAD)

        mock_client.close.assert_awaited_once()
        assert "conv-1" not in transport._lease_manager._leases

    @pytest.mark.asyncio
    async def test_sticky_releases_on_cancelled_error(self) -> None:
        """STICKY: should release the lease when request is externally cancelled."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False
        transport._lease_manager = GrpcChannelLeaseManager()

        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=asyncio.CancelledError)
        with patch.object(transport, "_create_client", return_value=mock_client):
            ri = create_request_info(
                endpoint,
                x_correlation_id="conv-1",
                is_final_turn=False,
            )
            with pytest.raises(asyncio.CancelledError):
                await transport.send_request(ri, _SIMPLE_PAYLOAD)

        mock_client.close.assert_awaited_once()
        assert "conv-1" not in transport._lease_manager._leases

    @pytest.mark.asyncio
    async def test_never_creates_new_channel_per_streaming_request(self) -> None:
        """NEVER + streaming: should create and close a channel per request."""
        endpoint = create_grpc_model_endpoint(
            streaming=True,
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._secure = False

        stream_call = make_mock_stream_call(make_stream_response("token"))
        mock_client = _make_mock_client(stream_call=stream_call)
        with patch.object(transport, "_create_client", return_value=mock_client):
            record = await transport.send_request(
                create_request_info(endpoint), _SIMPLE_PAYLOAD
            )

        assert record.status == 200
        mock_client.close.assert_awaited_once()


class TestMetadataToDict:
    """Tests for _metadata_to_dict helper function."""

    def test_none_returns_none(self) -> None:
        """None metadata should return None."""
        assert _metadata_to_dict(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Empty metadata list should return None."""
        assert _metadata_to_dict([]) is None

    def test_string_values(self) -> None:
        """String metadata values should be preserved."""
        metadata = [("key1", "val1"), ("key2", "val2")]
        result = _metadata_to_dict(metadata)
        assert result == {"key1": "val1", "key2": "val2"}

    def test_binary_values_decoded(self) -> None:
        """Binary metadata values should be decoded to UTF-8."""
        metadata = [("content-type-bin", b"application/grpc")]
        result = _metadata_to_dict(metadata)
        assert result == {"content-type-bin": "application/grpc"}

    def test_binary_values_with_invalid_utf8(self) -> None:
        """Binary metadata with invalid UTF-8 should use replacement character."""
        metadata = [("data-bin", b"\xff\xfe")]
        result = _metadata_to_dict(metadata)
        assert result is not None
        assert "data-bin" in result

    def test_mixed_str_and_bytes(self) -> None:
        """Metadata with mixed str and bytes values should work."""
        metadata = [("text-key", "text-val"), ("binary-bin", b"binary-val")]
        result = _metadata_to_dict(metadata)
        assert result == {"text-key": "text-val", "binary-bin": "binary-val"}


class TestParseTarget:
    """Tests for GrpcTransport._parse_target static method."""

    def test_grpc_scheme(self) -> None:
        """Should strip grpc:// scheme."""
        assert GrpcTransport._parse_target("grpc://host:8001") == "host:8001"

    def test_grpcs_scheme(self) -> None:
        """Should strip grpcs:// scheme."""
        assert GrpcTransport._parse_target("grpcs://host:443") == "host:443"

    def test_empty_url_raises_value_error(self) -> None:
        """Empty URL should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            GrpcTransport._parse_target("")

    def test_scheme_only_raises_value_error(self) -> None:
        """URL with scheme only and no host should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            GrpcTransport._parse_target("grpc://")


class TestBuildGrpcMetadata:
    """Tests for _build_grpc_metadata method."""

    def test_empty_headers_returns_none(self) -> None:
        """No headers should return None metadata."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        transport.base_headers = {}
        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {}
        request_info.x_request_id = ""
        request_info.x_correlation_id = ""

        result = transport._build_grpc_metadata(request_info)
        assert result is None

    def test_header_keys_lowercased(self) -> None:
        """Header keys should be lowercased for gRPC metadata."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {"Authorization": "Bearer tok", "X-Custom": "v"}

        result = transport._build_grpc_metadata(request_info)
        assert result is not None
        keys = [k for k, _ in result]
        assert all(k == k.lower() for k in keys)
        assert "authorization" in keys


class TestSendRequestErrorPaths:
    """Tests for error paths in send_request not covered elsewhere."""

    @pytest.mark.asyncio
    async def test_serializer_none_raises_not_initialized(self) -> None:
        """send_request should raise NotInitializedError if serializer is None."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)

        with pytest.raises(NotInitializedError, match="not initialized"):
            await transport.send_request(create_request_info(endpoint), _SIMPLE_PAYLOAD)

    @pytest.mark.asyncio
    async def test_sticky_no_lease_manager_raises_not_initialized(self) -> None:
        """send_request should raise NotInitializedError if lease_manager is None for sticky."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        with pytest.raises(NotInitializedError, match="LeaseManager"):
            await transport.send_request(create_request_info(endpoint), _SIMPLE_PAYLOAD)

    @pytest.mark.asyncio
    async def test_generic_exception_populates_error_details(self) -> None:
        """Generic exceptions should be caught and wrapped in ErrorDetails."""
        endpoint = create_grpc_model_endpoint()
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=RuntimeError("Something broke"))
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.error is not None
        assert "RuntimeError" in record.error.type
        assert "Something broke" in record.error.message
        assert record.end_perf_ns is not None
        assert record.trace_data.error_timestamp_perf_ns is not None

    @pytest.mark.asyncio
    async def test_generic_exception_releases_sticky_lease(self) -> None:
        """Generic exceptions should release sticky lease."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        )
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)
        transport._lease_manager = GrpcChannelLeaseManager()

        mock_client = _make_mock_client()
        mock_client.unary = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(transport, "_create_client", return_value=mock_client):
            ri = create_request_info(
                endpoint, x_correlation_id="c1", is_final_turn=False
            )
            await transport.send_request(ri, _SIMPLE_PAYLOAD)

        assert "c1" not in transport._lease_manager._leases


class TestStreamingFirstTokenCallback:
    """Tests for first_token_callback edge cases in streaming."""

    @pytest.mark.asyncio
    async def test_callback_returning_false_fires_again(self) -> None:
        """first_token_callback returning False should fire again on next chunk."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        stream_call = make_mock_stream_call(
            make_stream_response("empty"),
            make_stream_response("real_token"),
            make_stream_response("more"),
        )
        mock_client = _make_mock_client(stream_call=stream_call)
        _set_pool(transport, mock_client)

        callback_calls: list[int] = []

        async def callback(ttft_ns: int, message) -> bool:
            callback_calls.append(ttft_ns)
            return len(callback_calls) >= 2

        ri = create_request_info(endpoint)
        record = await transport.send_request(
            ri, _SIMPLE_PAYLOAD, first_token_callback=callback
        )

        assert len(callback_calls) == 2
        assert record.status == 200


class TestStreamingGrpcError:
    """Tests for gRPC-level errors during streaming."""

    @pytest.mark.asyncio
    async def test_aio_rpc_error_during_stream_caught(self) -> None:
        """AioRpcError from the stream should be caught by outer handler."""
        endpoint = create_grpc_model_endpoint(streaming=True)
        transport = GrpcTransport(model_endpoint=endpoint)
        _init_transport_serializer(transport)

        rpc_error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Connection lost",
        )

        class ErrorStreamCall:
            async def __aiter__(self):
                raise rpc_error
                yield  # noqa: RUF028 - make this an async generator

            async def trailing_metadata(self):
                return ()

            def cancel(self) -> bool:
                return True

        mock_client = _make_mock_client()
        mock_client.server_stream = MagicMock(return_value=ErrorStreamCall())
        _set_pool(transport, mock_client)

        record = await transport.send_request(
            create_request_info(endpoint), _SIMPLE_PAYLOAD
        )

        assert record.error is not None
        assert record.status == 503
        assert "UNAVAILABLE" in record.error.type
        assert record.trace_data.grpc_status_code is not None

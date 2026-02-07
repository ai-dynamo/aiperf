# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GrpcTransport."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    CreditPhase,
    ModelSelectionStrategy,
)
from aiperf.common.models import TextResponse
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin.enums import EndpointType
from aiperf.transports.grpc.grpc_transport import GrpcTransport
from aiperf.transports.grpc.kserve_v2_serializers import KServeV2GrpcSerializer
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.trace_data import GrpcTraceData

# Method paths for tests (match plugins.yaml kserve_v2_infer metadata)
_V2_UNARY_METHOD = "/inference.GRPCInferenceService/ModelInfer"
_V2_STREAM_METHOD = "/inference.GRPCInferenceService/ModelStreamInfer"


def create_grpc_model_endpoint(
    base_url: str = "grpc://localhost:8001",
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
            base_urls=[base_url],
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
        is_final_turn=True,
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


def _init_transport_serializer(transport: GrpcTransport) -> None:
    """Wire V2 serializer and method paths onto a transport for testing."""
    transport._serializer = KServeV2GrpcSerializer()
    transport._unary_method = _V2_UNARY_METHOD
    transport._stream_method = _V2_STREAM_METHOD


class TestGrpcTransportMetadata:
    """Tests for GrpcTransport class metadata."""

    def test_metadata_transport_type(self) -> None:
        """Transport metadata should report grpc type."""
        meta = GrpcTransport.metadata()
        assert meta.transport_type == "grpc"

    def test_metadata_url_schemes(self) -> None:
        """Transport metadata should support grpc and grpcs schemes."""
        meta = GrpcTransport.metadata()
        assert "grpc" in meta.url_schemes
        assert "grpcs" in meta.url_schemes


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
        mock_client = MagicMock()
        mock_client.unary = AsyncMock(
            return_value=make_infer_response().SerializeToString()
        )
        transport._grpc_client = mock_client

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

        responses = [
            make_stream_response("Hello"),
            make_stream_response(" world"),
        ]

        async def mock_stream(*args, **kwargs):
            for resp in responses:
                yield resp.SerializeToString()

        mock_client = MagicMock()
        mock_client.server_stream = mock_stream
        transport._grpc_client = mock_client

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

        responses = [
            make_stream_response("token1"),
            make_stream_response("token2"),
        ]

        async def mock_stream(*args, **kwargs):
            for resp in responses:
                yield resp.SerializeToString()

        mock_client = MagicMock()
        mock_client.server_stream = mock_stream
        transport._grpc_client = mock_client

        callback_calls = []

        async def callback(ttft_ns: int, message) -> bool:
            callback_calls.append((ttft_ns, message))
            return True  # First token is meaningful

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(
            request_info, payload, first_token_callback=callback
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

        responses = [
            make_stream_response("token1"),
            make_stream_error_response("Model crashed"),
        ]

        async def mock_stream(*args, **kwargs):
            for resp in responses:
                yield resp.SerializeToString()

        mock_client = MagicMock()
        mock_client.server_stream = mock_stream
        transport._grpc_client = mock_client

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(request_info, payload)

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

        mock_client = MagicMock()
        mock_client.unary = AsyncMock(side_effect=rpc_error)
        transport._grpc_client = mock_client

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(request_info, payload)

        assert record.error is not None
        assert record.status == 503  # UNAVAILABLE -> 503
        assert "gRPC:UNAVAILABLE" in record.error.type

    @pytest.mark.asyncio
    async def test_send_request_with_cancel_after_ns(
        self, transport_and_endpoint
    ) -> None:
        """Request with cancel_after_ns that completes before timeout."""
        transport, endpoint = transport_and_endpoint
        mock_client = MagicMock()
        mock_client.unary = AsyncMock(
            return_value=make_infer_response().SerializeToString()
        )
        transport._grpc_client = mock_client

        request_info = create_request_info(endpoint, cancel_after_ns=5_000_000_000)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(request_info, payload)

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
            return make_infer_response().SerializeToString()

        mock_client = MagicMock()
        mock_client.unary = slow_unary
        transport._grpc_client = mock_client

        # Cancel after 1ns (effectively immediate)
        request_info = create_request_info(endpoint, cancel_after_ns=1)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(request_info, payload)

        assert record.error is not None
        assert record.error.code == 499
        assert record.cancellation_perf_ns is not None

    @pytest.mark.asyncio
    async def test_headers_passed_as_grpc_metadata(
        self, transport_and_endpoint
    ) -> None:
        """Headers from build_headers are passed as gRPC metadata."""
        transport, endpoint = transport_and_endpoint
        mock_client = MagicMock()
        mock_client.unary = AsyncMock(
            return_value=make_infer_response().SerializeToString()
        )
        transport._grpc_client = mock_client

        request_info = create_request_info(endpoint)
        request_info.endpoint_headers = {"Authorization": "Bearer token123"}
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        await transport.send_request(request_info, payload)

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
        mock_client = MagicMock()
        mock_client.unary = AsyncMock(
            return_value=make_infer_response().SerializeToString()
        )
        transport._grpc_client = mock_client

        request_info = create_request_info(endpoint)
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}]
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record.trace_data, GrpcTraceData)
        assert record.trace_data.trace_type == "grpc"
        assert record.trace_data.grpc_status_code == 0  # OK
        assert record.trace_data.request_send_start_perf_ns is not None
        assert record.trace_data.response_receive_start_perf_ns is not None
        assert record.trace_data.response_receive_end_perf_ns is not None
        assert len(record.trace_data.request_chunks) > 0
        assert len(record.trace_data.response_chunks) > 0


class TestGrpcTransportInit:
    """Tests for transport initialization."""

    @pytest.mark.asyncio
    async def test_connection_reuse_never_warns(self) -> None:
        """NEVER connection reuse strategy should log a warning."""
        endpoint = create_grpc_model_endpoint(
            connection_reuse_strategy=ConnectionReuseStrategy.NEVER
        )
        transport = GrpcTransport(model_endpoint=endpoint)

        with patch.object(transport, "warning") as mock_warn:
            with patch("aiperf.transports.grpc.grpc_transport.GenericGrpcClient"):
                await transport._init_grpc_client()

            mock_warn.assert_called_once()
            assert "not applicable" in mock_warn.call_args[0][0]

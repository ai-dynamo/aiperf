# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for KServeV2GrpcSerializer."""

from __future__ import annotations

from aiperf.transports.grpc.grpc_transport import GrpcSerializerProtocol
from aiperf.transports.grpc.kserve_v2_serializers import KServeV2GrpcSerializer
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk


def _make_infer_response(text: str = "Generated text") -> pb2.ModelInferResponse:
    """Create a ModelInferResponse protobuf with BYTES output."""
    response = pb2.ModelInferResponse()
    response.model_name = "test-model"
    output = response.outputs.add()
    output.name = "text_output"
    output.datatype = "BYTES"
    output.shape.append(1)
    output.contents.bytes_contents.append(text.encode("utf-8"))
    return response


class TestKServeV2GrpcSerializerProtocol:
    """Tests for protocol conformance."""

    def test_implements_protocol(self) -> None:
        """KServeV2GrpcSerializer should satisfy GrpcSerializerProtocol."""
        assert isinstance(KServeV2GrpcSerializer(), GrpcSerializerProtocol)


class TestSerializeRequest:
    """Tests for KServeV2GrpcSerializer.serialize_request."""

    def test_roundtrip_bytes(self) -> None:
        """Serialized bytes should parse back to a valid ModelInferRequest."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello world"],
                }
            ]
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(
            payload, model_name="my-model", request_id="r1"
        )
        assert isinstance(data, bytes)
        assert len(data) > 0

        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)
        assert parsed.model_name == "my-model"
        assert parsed.id == "r1"
        assert len(parsed.inputs) == 1
        assert parsed.inputs[0].name == "text_input"

    def test_with_parameters(self) -> None:
        """Request-level parameters should survive serialization."""
        payload = {
            "inputs": [{"name": "t", "shape": [1], "datatype": "BYTES", "data": ["p"]}],
            "parameters": {"stream": True, "max_tokens": 100},
        }
        serializer = KServeV2GrpcSerializer()
        data = serializer.serialize_request(payload, model_name="m")
        parsed = pb2.ModelInferRequest()
        parsed.ParseFromString(data)
        assert parsed.parameters["stream"].bool_param is True
        assert parsed.parameters["max_tokens"].int64_param == 100


class TestDeserializeResponse:
    """Tests for KServeV2GrpcSerializer.deserialize_response."""

    def test_response_dict_and_size(self) -> None:
        """Should return correct dict and wire size."""
        proto = _make_infer_response("Hello")
        data = proto.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        result_dict, size = serializer.deserialize_response(data)

        assert size == len(data)
        assert result_dict["model_name"] == "test-model"
        assert len(result_dict["outputs"]) == 1
        assert result_dict["outputs"][0]["name"] == "text_output"
        assert result_dict["outputs"][0]["data"] == ["Hello"]


class TestDeserializeStreamResponse:
    """Tests for KServeV2GrpcSerializer.deserialize_stream_response."""

    def test_success_chunk(self) -> None:
        """Normal streaming chunk should produce StreamChunk with response_dict."""
        stream_resp = pb2.ModelStreamInferResponse()
        stream_resp.infer_response.CopyFrom(_make_infer_response("token"))
        data = stream_resp.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        chunk = serializer.deserialize_stream_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message is None
        assert chunk.response_dict is not None
        assert chunk.response_dict["outputs"][0]["data"] == ["token"]
        assert chunk.response_size == len(data)

    def test_error_chunk(self) -> None:
        """Error streaming chunk should produce StreamChunk with error_message."""
        stream_resp = pb2.ModelStreamInferResponse()
        stream_resp.error_message = "Model crashed"
        data = stream_resp.SerializeToString()

        serializer = KServeV2GrpcSerializer()
        chunk = serializer.deserialize_stream_response(data)

        assert isinstance(chunk, StreamChunk)
        assert chunk.error_message == "Model crashed"
        assert chunk.response_dict is None
        assert chunk.response_size == len(data)

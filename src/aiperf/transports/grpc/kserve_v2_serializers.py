# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KServe V2 gRPC serializer.

This module is the ONLY place (besides payload_converter.py which it wraps)
that imports V2 protobuf types. All proto knowledge is isolated here so the
transport and client layers remain protocol-agnostic.

Discovered via plugins.yaml endpoint metadata (``grpc.serializer``).
"""

from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.payload_converter import (
    dict_to_model_infer_request,
    model_infer_response_to_dict,
)
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk


class KServeV2GrpcSerializer:
    """KServe V2 gRPC serializer for the generic gRPC transport.

    Converts between endpoint dict payloads and V2 protobuf wire bytes.
    Implements the GrpcSerializerProtocol expected by GrpcTransport.
    """

    @staticmethod
    def serialize_request(
        payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes:
        """Convert a dict payload to serialized ModelInferRequest bytes.

        Args:
            payload: V2 JSON-format dict from the endpoint's format_payload().
            model_name: Model name for the gRPC request.
            request_id: Optional request ID.

        Returns:
            Serialized protobuf bytes ready to send on the wire.
        """
        proto = dict_to_model_infer_request(
            payload, model_name=model_name, request_id=request_id
        )
        return proto.SerializeToString()

    @staticmethod
    def deserialize_response(data: bytes) -> tuple[dict[str, Any], int]:
        """Deserialize ModelInferResponse bytes to a dict and wire size.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            Tuple of (V2 JSON-format response dict, wire size in bytes).
        """
        response = pb2.ModelInferResponse()
        response.ParseFromString(data)
        return model_infer_response_to_dict(response), len(data)

    @staticmethod
    def deserialize_stream_response(data: bytes) -> StreamChunk:
        """Deserialize ModelStreamInferResponse bytes to a StreamChunk.

        Args:
            data: Raw bytes from the gRPC wire.

        Returns:
            Protocol-agnostic StreamChunk with either error or response data.
        """
        stream_resp = pb2.ModelStreamInferResponse()
        stream_resp.ParseFromString(data)

        if stream_resp.error_message:
            return StreamChunk(
                error_message=stream_resp.error_message,
                response_dict=None,
                response_size=len(data),
            )

        resp_dict = model_infer_response_to_dict(stream_resp.infer_response)
        return StreamChunk(
            error_message=None,
            response_dict=resp_dict,
            response_size=len(data),
        )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KServe V2 gRPC serializer and dict/protobuf converters.

This module is the ONLY place that imports V2 protobuf types. All proto
knowledge is isolated here so the transport and client layers remain
protocol-agnostic.

Discovered via plugins.yaml endpoint metadata (``grpc.serializer``).
"""

from __future__ import annotations

import struct
from typing import Any

from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.stream_chunk import StreamChunk

# ---------------------------------------------------------------------------
# Dict -> Protobuf conversion helpers
# ---------------------------------------------------------------------------


def _set_tensor_contents(
    contents: pb2.InferTensorContents, datatype: str, data: list[Any]
) -> None:
    """Populate InferTensorContents based on the V2 datatype string.

    Args:
        contents: Protobuf InferTensorContents to populate.
        datatype: V2 datatype string (BYTES, INT32, INT64, FP32, FP64, BOOL, etc.).
        data: List of values to set.
    """
    datatype = datatype.upper()
    if datatype == "BYTES":
        for item in data:
            contents.bytes_contents.append(
                item.encode("utf-8") if isinstance(item, str) else item
            )
    elif datatype in ("INT8", "INT16", "INT32"):
        contents.int_contents.extend(int(v) for v in data)
    elif datatype == "INT64":
        contents.int64_contents.extend(int(v) for v in data)
    elif datatype in ("UINT8", "UINT16", "UINT32"):
        contents.uint_contents.extend(int(v) for v in data)
    elif datatype == "UINT64":
        contents.uint64_contents.extend(int(v) for v in data)
    elif datatype in ("FP16", "FP32"):
        contents.fp32_contents.extend(float(v) for v in data)
    elif datatype == "FP64":
        contents.fp64_contents.extend(float(v) for v in data)
    elif datatype == "BOOL":
        contents.bool_contents.extend(bool(v) for v in data)
    else:
        # Fallback: treat as bytes
        for item in data:
            contents.bytes_contents.append(
                item.encode("utf-8") if isinstance(item, str) else bytes(item)
            )


def _make_infer_parameter(value: Any) -> pb2.InferParameter:
    """Create an InferParameter from a Python value, auto-detecting the type.

    Args:
        value: Python value (str, int, float, or bool).

    Returns:
        InferParameter protobuf message.
    """
    param = pb2.InferParameter()
    if isinstance(value, bool):
        param.bool_param = value
    elif isinstance(value, int):
        param.int64_param = value
    elif isinstance(value, float):
        param.double_param = value
    else:
        param.string_param = str(value)
    return param


def dict_to_model_infer_request(
    payload: dict[str, Any],
    model_name: str,
    model_version: str = "",
    request_id: str = "",
) -> pb2.ModelInferRequest:
    """Convert an endpoint dict payload to a ModelInferRequest protobuf.

    The dict format matches the KServe V2 JSON inference protocol:
    ``{"inputs": [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]}``

    Args:
        payload: V2 JSON-format dict from the endpoint's format_payload().
        model_name: Model name for the gRPC request.
        model_version: Model version (empty string for server default).
        request_id: Optional request ID.

    Returns:
        ModelInferRequest protobuf ready to send via gRPC.
    """
    request = pb2.ModelInferRequest()
    request.model_name = model_name
    if model_version:
        request.model_version = model_version
    if request_id:
        request.id = request_id

    # Convert parameters
    if "parameters" in payload:
        for key, value in payload["parameters"].items():
            request.parameters[key].CopyFrom(_make_infer_parameter(value))

    # Convert input tensors
    for inp in payload.get("inputs", []):
        tensor = pb2.ModelInferRequest.InferInputTensor()
        tensor.name = inp["name"]
        tensor.datatype = inp["datatype"]
        tensor.shape.extend(int(s) for s in inp["shape"])

        # Set tensor contents from data
        if "data" in inp:
            _set_tensor_contents(tensor.contents, inp["datatype"], inp["data"])

        # Input-level parameters
        if "parameters" in inp:
            for key, value in inp["parameters"].items():
                tensor.parameters[key].CopyFrom(_make_infer_parameter(value))

        request.inputs.append(tensor)

    return request


# ---------------------------------------------------------------------------
# Protobuf -> Dict conversion helpers
# ---------------------------------------------------------------------------


def _extract_tensor_data(contents: pb2.InferTensorContents, datatype: str) -> list[Any]:
    """Extract data from InferTensorContents based on datatype.

    Args:
        contents: Protobuf InferTensorContents.
        datatype: V2 datatype string.

    Returns:
        List of Python values.
    """
    datatype = datatype.upper()
    if datatype == "BYTES":
        return [b.decode("utf-8", errors="replace") for b in contents.bytes_contents]
    elif datatype in ("INT8", "INT16", "INT32"):
        return list(contents.int_contents)
    elif datatype == "INT64":
        return list(contents.int64_contents)
    elif datatype in ("UINT8", "UINT16", "UINT32"):
        return list(contents.uint_contents)
    elif datatype == "UINT64":
        return list(contents.uint64_contents)
    elif datatype in ("FP16", "FP32"):
        return list(contents.fp32_contents)
    elif datatype == "FP64":
        return list(contents.fp64_contents)
    elif datatype == "BOOL":
        return list(contents.bool_contents)
    else:
        return [b.decode("utf-8", errors="replace") for b in contents.bytes_contents]


def _extract_raw_tensor_data(
    raw_bytes: bytes, datatype: str, shape: list[int]
) -> list[Any]:
    """Extract data from raw_output_contents bytes based on datatype.

    Triton may populate ``raw_output_contents`` instead of
    ``InferTensorContents``. For BYTES, the format is repeated
    ``[4-byte LE uint32 length][UTF-8 data]`` segments. For numeric
    types the bytes are the flattened binary representation.

    Args:
        raw_bytes: Raw binary tensor data.
        datatype: V2 datatype string.
        shape: Tensor shape for computing element count.

    Returns:
        List of Python values.
    """
    datatype = datatype.upper()
    if datatype == "BYTES":
        results: list[Any] = []
        offset = 0
        while offset < len(raw_bytes):
            if offset + 4 > len(raw_bytes):
                break
            (length,) = struct.unpack_from("<I", raw_bytes, offset)
            offset += 4
            results.append(
                raw_bytes[offset : offset + length].decode("utf-8", errors="replace")
            )
            offset += length
        return results

    # Numeric types: compute element count from shape
    num_elements = 1
    for dim in shape:
        if dim > 0:
            num_elements *= dim

    fmt_map: dict[str, str] = {
        "BOOL": "?",
        "INT8": "b",
        "INT16": "<h",
        "INT32": "<i",
        "INT64": "<q",
        "UINT8": "B",
        "UINT16": "<H",
        "UINT32": "<I",
        "UINT64": "<Q",
        "FP16": "<e",
        "FP32": "<f",
        "FP64": "<d",
    }
    fmt = fmt_map.get(datatype)
    if fmt is None:
        # Unknown type, try BYTES decoding
        return [raw_bytes.decode("utf-8", errors="replace")]

    elem_size = struct.calcsize(fmt)
    return [
        struct.unpack_from(fmt, raw_bytes, i * elem_size)[0]
        for i in range(num_elements)
    ]


def model_infer_response_to_dict(response: pb2.ModelInferResponse) -> dict[str, Any]:
    """Convert a ModelInferResponse protobuf to a V2 JSON-compatible dict.

    Returns a dict matching the V2 JSON inference protocol response format
    so the endpoint's parse_response() sees identical structure as HTTP.

    Handles both ``InferTensorContents`` and ``raw_output_contents``
    representations — Triton commonly uses the latter for streaming.

    Args:
        response: ModelInferResponse protobuf from gRPC.

    Returns:
        V2 JSON-format dict: ``{"outputs": [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]}``.
    """
    raw_contents = list(response.raw_output_contents)
    outputs: list[dict[str, Any]] = []
    for i, output in enumerate(response.outputs):
        data = _extract_tensor_data(output.contents, output.datatype)
        if not data and i < len(raw_contents):
            data = _extract_raw_tensor_data(
                raw_contents[i], output.datatype, list(output.shape)
            )
        output_dict: dict[str, Any] = {
            "name": output.name,
            "datatype": output.datatype,
            "shape": list(output.shape),
            "data": data,
        }
        outputs.append(output_dict)

    result: dict[str, Any] = {"outputs": outputs}

    if response.model_name:
        result["model_name"] = response.model_name
    if response.model_version:
        result["model_version"] = response.model_version
    if response.id:
        result["id"] = response.id

    return result


# ---------------------------------------------------------------------------
# Serializer class (plugin entry point)
# ---------------------------------------------------------------------------


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

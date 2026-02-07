# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Converters between endpoint dict payloads and gRPC protobuf messages."""

from __future__ import annotations

from typing import Any

from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2


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


def model_infer_response_to_dict(response: pb2.ModelInferResponse) -> dict[str, Any]:
    """Convert a ModelInferResponse protobuf to a V2 JSON-compatible dict.

    Returns a dict matching the V2 JSON inference protocol response format
    so the endpoint's parse_response() sees identical structure as HTTP.

    Args:
        response: ModelInferResponse protobuf from gRPC.

    Returns:
        V2 JSON-format dict: ``{"outputs": [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]}``.
    """
    outputs: list[dict[str, Any]] = []
    for output in response.outputs:
        output_dict: dict[str, Any] = {
            "name": output.name,
            "datatype": output.datatype,
            "shape": list(output.shape),
            "data": _extract_tensor_data(output.contents, output.datatype),
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

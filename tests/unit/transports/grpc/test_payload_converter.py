# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gRPC payload converter (dict <-> protobuf)."""

import struct

import pytest

from aiperf.transports.grpc.kserve_v2_serializers import (
    _extract_raw_tensor_data,
    dict_to_model_infer_request,
    model_infer_response_to_dict,
)
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2


class TestDictToModelInferRequest:
    """Tests for converting endpoint dict -> ModelInferRequest protobuf."""

    def test_basic_bytes_input(self) -> None:
        """Convert a basic BYTES tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello, world!"],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="test-model")

        assert request.model_name == "test-model"
        assert len(request.inputs) == 1
        assert request.inputs[0].name == "text_input"
        assert request.inputs[0].datatype == "BYTES"
        assert list(request.inputs[0].shape) == [1]
        assert request.inputs[0].contents.bytes_contents == [b"Hello, world!"]

    def test_int32_input(self) -> None:
        """Convert an INT32 tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [128],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.inputs[0].contents.int_contents == [128]

    def test_fp32_input(self) -> None:
        """Convert an FP32 tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "temperature",
                    "shape": [1],
                    "datatype": "FP32",
                    "data": [0.7],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert len(request.inputs[0].contents.fp32_contents) == 1
        assert abs(request.inputs[0].contents.fp32_contents[0] - 0.7) < 0.01

    def test_bool_input(self) -> None:
        """Convert a BOOL tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "stream",
                    "shape": [1],
                    "datatype": "BOOL",
                    "data": [True],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.inputs[0].contents.bool_contents == [True]

    def test_int64_input(self) -> None:
        """Convert an INT64 tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "seed",
                    "shape": [1],
                    "datatype": "INT64",
                    "data": [42],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.inputs[0].contents.int64_contents == [42]

    def test_fp64_input(self) -> None:
        """Convert an FP64 tensor input."""
        payload = {
            "inputs": [
                {
                    "name": "threshold",
                    "shape": [1],
                    "datatype": "FP64",
                    "data": [0.95],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert len(request.inputs[0].contents.fp64_contents) == 1
        assert abs(request.inputs[0].contents.fp64_contents[0] - 0.95) < 0.001

    def test_multiple_inputs(self) -> None:
        """Convert multiple input tensors."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["prompt"],
                },
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [256],
                },
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert len(request.inputs) == 2
        assert request.inputs[0].name == "text_input"
        assert request.inputs[1].name == "max_tokens"

    def test_model_version_and_request_id(self) -> None:
        """Model version and request ID are set correctly."""
        payload = {"inputs": []}
        request = dict_to_model_infer_request(
            payload,
            model_name="m",
            model_version="1",
            request_id="req-123",
        )

        assert request.model_name == "m"
        assert request.model_version == "1"
        assert request.id == "req-123"

    def test_parameters_string(self) -> None:
        """Convert string parameters."""
        payload = {
            "inputs": [],
            "parameters": {"stop": "</s>"},
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.parameters["stop"].string_param == "</s>"

    def test_parameters_int(self) -> None:
        """Convert int parameters."""
        payload = {
            "inputs": [],
            "parameters": {"top_k": 50},
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.parameters["top_k"].int64_param == 50

    def test_parameters_bool(self) -> None:
        """Convert bool parameters."""
        payload = {
            "inputs": [],
            "parameters": {"stream": True},
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.parameters["stream"].bool_param is True

    def test_parameters_float(self) -> None:
        """Convert float parameters (stored as double_param)."""
        payload = {
            "inputs": [],
            "parameters": {"temperature": 0.7},
        }
        request = dict_to_model_infer_request(payload, model_name="m")

        assert abs(request.parameters["temperature"].double_param - 0.7) < 0.001

    def test_empty_inputs(self) -> None:
        """Empty inputs list produces valid request."""
        payload = {"inputs": []}
        request = dict_to_model_infer_request(payload, model_name="m")

        assert request.model_name == "m"
        assert len(request.inputs) == 0


class TestModelInferResponseToDict:
    """Tests for converting ModelInferResponse protobuf -> dict."""

    def test_basic_bytes_output(self) -> None:
        """Convert a BYTES output tensor response."""
        response = pb2.ModelInferResponse()
        response.model_name = "test-model"
        response.id = "req-1"
        output = response.outputs.add()
        output.name = "text_output"
        output.datatype = "BYTES"
        output.shape.append(1)
        output.contents.bytes_contents.append(b"Generated text")

        result = model_infer_response_to_dict(response)

        assert result["model_name"] == "test-model"
        assert result["id"] == "req-1"
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["name"] == "text_output"
        assert result["outputs"][0]["datatype"] == "BYTES"
        assert result["outputs"][0]["data"] == ["Generated text"]

    def test_int32_output(self) -> None:
        """Convert an INT32 output tensor."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "token_count"
        output.datatype = "INT32"
        output.shape.append(1)
        output.contents.int_contents.append(42)

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == [42]

    def test_multiple_outputs(self) -> None:
        """Convert response with multiple output tensors."""
        response = pb2.ModelInferResponse()
        out1 = response.outputs.add()
        out1.name = "text_output"
        out1.datatype = "BYTES"
        out1.shape.append(1)
        out1.contents.bytes_contents.append(b"text")

        out2 = response.outputs.add()
        out2.name = "scores"
        out2.datatype = "FP32"
        out2.shape.extend([1, 3])
        out2.contents.fp32_contents.extend([0.1, 0.5, 0.9])

        result = model_infer_response_to_dict(response)

        assert len(result["outputs"]) == 2
        assert result["outputs"][0]["name"] == "text_output"
        assert result["outputs"][1]["name"] == "scores"
        assert len(result["outputs"][1]["data"]) == 3

    def test_round_trip_bytes(self) -> None:
        """Round-trip test: dict -> protobuf -> dict preserves BYTES data."""
        original = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Hello, world!"],
                }
            ]
        }

        request = dict_to_model_infer_request(original, model_name="m")

        # Simulate server response with same tensor as output
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "text_output"
        output.datatype = "BYTES"
        output.shape.append(1)
        # Copy the input data to the output
        output.contents.bytes_contents.extend(request.inputs[0].contents.bytes_contents)

        result = model_infer_response_to_dict(response)
        assert result["outputs"][0]["data"] == ["Hello, world!"]

    def test_empty_response(self) -> None:
        """Empty response produces valid dict."""
        response = pb2.ModelInferResponse()
        result = model_infer_response_to_dict(response)

        assert result["outputs"] == []

    def test_raw_output_contents_bytes(self) -> None:
        """Extract data from raw_output_contents for BYTES tensor."""
        response = pb2.ModelInferResponse()
        response.model_name = "m"
        output = response.outputs.add()
        output.name = "text_output"
        output.datatype = "BYTES"
        output.shape.append(1)
        # Empty InferTensorContents — data in raw_output_contents
        text = b"Hello world"
        response.raw_output_contents.append(struct.pack("<I", len(text)) + text)

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == ["Hello world"]

    def test_raw_output_contents_int32(self) -> None:
        """Extract data from raw_output_contents for INT32 tensor."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "count"
        output.datatype = "INT32"
        output.shape.append(3)
        response.raw_output_contents.append(struct.pack("<iii", 10, 20, 30))

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == [10, 20, 30]

    def test_raw_output_contents_fp32(self) -> None:
        """Extract data from raw_output_contents for FP32 tensor."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "scores"
        output.datatype = "FP32"
        output.shape.extend([1, 2])
        response.raw_output_contents.append(struct.pack("<ff", 0.5, 1.5))

        result = model_infer_response_to_dict(response)

        assert len(result["outputs"][0]["data"]) == 2
        assert abs(result["outputs"][0]["data"][0] - 0.5) < 0.001
        assert abs(result["outputs"][0]["data"][1] - 1.5) < 0.001

    def test_raw_output_contents_multiple_outputs(self) -> None:
        """raw_output_contents indexed by output position."""
        response = pb2.ModelInferResponse()
        # Output 0: BYTES
        out0 = response.outputs.add()
        out0.name = "text"
        out0.datatype = "BYTES"
        out0.shape.append(1)
        text = b"token"
        response.raw_output_contents.append(struct.pack("<I", len(text)) + text)

        # Output 1: INT32
        out1 = response.outputs.add()
        out1.name = "count"
        out1.datatype = "INT32"
        out1.shape.append(1)
        response.raw_output_contents.append(struct.pack("<i", 42))

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == ["token"]
        assert result["outputs"][1]["data"] == [42]

    def test_contents_preferred_over_raw(self) -> None:
        """InferTensorContents takes precedence over raw_output_contents."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "text"
        output.datatype = "BYTES"
        output.shape.append(1)
        output.contents.bytes_contents.append(b"from_contents")
        # Also set raw (should be ignored)
        text = b"from_raw"
        response.raw_output_contents.append(struct.pack("<I", len(text)) + text)

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == ["from_contents"]


class TestExtractRawTensorData:
    """Tests for _extract_raw_tensor_data()."""

    def test_bytes_single_element(self) -> None:
        """Parse single length-prefixed BYTES element."""
        text = b"Hello"
        raw = struct.pack("<I", len(text)) + text
        result = _extract_raw_tensor_data(raw, "BYTES", [1])

        assert result == ["Hello"]

    def test_bytes_multiple_elements(self) -> None:
        """Parse multiple length-prefixed BYTES elements."""
        raw = b""
        for s in [b"Hello", b"world"]:
            raw += struct.pack("<I", len(s)) + s
        result = _extract_raw_tensor_data(raw, "BYTES", [2])

        assert result == ["Hello", "world"]

    def test_bytes_empty(self) -> None:
        """Empty raw bytes for BYTES produces empty list."""
        result = _extract_raw_tensor_data(b"", "BYTES", [0])

        assert result == []

    @pytest.mark.parametrize(
        "datatype,fmt,values",
        [
            ("INT32", "<i", [1, 2, 3]),
            ("INT64", "<q", [100, 200]),
            ("FP32", "<f", [0.5, 1.5]),
            ("FP64", "<d", [0.25]),
            ("BOOL", "?", [True, False]),
            ("UINT32", "<I", [10, 20]),
        ],
    )
    def test_numeric_types(self, datatype: str, fmt: str, values: list) -> None:
        """Numeric types correctly unpacked from raw bytes."""
        raw = struct.pack(
            f"{len(values)}{fmt[-1]}" if len(fmt) > 1 else f"{len(values)}{fmt}",
            *values,
        )
        result = _extract_raw_tensor_data(raw, datatype, [len(values)])

        for expected, actual in zip(values, result, strict=True):
            if isinstance(expected, float):
                assert abs(expected - actual) < 0.001
            else:
                assert expected == actual

    @pytest.mark.parametrize(
        "datatype,fmt,values",
        [
            ("UINT64", "<Q", [100, 200]),
            ("UINT8", "B", [1, 2, 255]),
            ("UINT16", "<H", [1000, 2000]),
            ("INT8", "b", [-1, 0, 127]),
            ("INT16", "<h", [-100, 100]),
            ("FP16", "<e", [0.5, 1.0]),
            ("FP64", "<d", [0.123456789]),
        ],
    )
    def test_additional_numeric_types(
        self, datatype: str, fmt: str, values: list
    ) -> None:
        """Additional numeric types correctly unpacked from raw bytes."""
        raw = struct.pack(
            f"{len(values)}{fmt[-1]}" if len(fmt) > 1 else f"{len(values)}{fmt}",
            *values,
        )
        result = _extract_raw_tensor_data(raw, datatype, [len(values)])

        for expected, actual in zip(values, result, strict=True):
            if isinstance(expected, float):
                assert abs(expected - actual) < 0.01
            else:
                assert expected == actual

    def test_multi_dimensional_shape(self) -> None:
        """Multi-dimensional shape should compute correct element count."""
        # Shape [2, 3] = 6 elements of INT32
        values = [1, 2, 3, 4, 5, 6]
        raw = struct.pack("<6i", *values)
        result = _extract_raw_tensor_data(raw, "INT32", [2, 3])

        assert result == values

    def test_bytes_truncated_length_prefix(self) -> None:
        """Truncated BYTES data (incomplete length prefix) should stop gracefully."""
        # Only 2 bytes, not enough for 4-byte length prefix
        raw = b"\x01\x02"
        result = _extract_raw_tensor_data(raw, "BYTES", [1])

        assert result == []

    def test_unknown_datatype_fallback(self) -> None:
        """Unknown datatype should fall back to UTF-8 decoding of raw bytes."""
        raw = b"hello"
        result = _extract_raw_tensor_data(raw, "BFLOAT16", [1])

        assert result == ["hello"]


class TestDictToModelInferRequestAdditional:
    """Additional tests for dict_to_model_infer_request."""

    @pytest.mark.parametrize(
        "datatype,proto_field,values",
        [
            ("UINT64", "uint64_contents", [42, 100]),
            ("UINT8", "uint_contents", [1, 255]),
            ("UINT16", "uint_contents", [1000]),
            ("UINT32", "uint_contents", [50000]),
            ("INT8", "int_contents", [-1, 127]),
            ("INT16", "int_contents", [-100, 100]),
            ("FP16", "fp32_contents", [0.5, 1.0]),
        ],
    )
    def test_additional_input_datatypes(
        self, datatype: str, proto_field: str, values: list
    ) -> None:
        """Additional datatypes should be set on the correct protobuf field."""
        payload = {
            "inputs": [
                {
                    "name": "tensor",
                    "shape": [len(values)],
                    "datatype": datatype,
                    "data": values,
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")
        contents = request.inputs[0].contents
        actual = list(getattr(contents, proto_field))

        assert len(actual) == len(values)
        for expected, got in zip(values, actual, strict=True):
            if isinstance(expected, float):
                assert abs(expected - got) < 0.01
            else:
                assert expected == got

    def test_unknown_datatype_stored_as_bytes(self) -> None:
        """Unknown datatype should fall back to bytes_contents."""
        payload = {
            "inputs": [
                {
                    "name": "tensor",
                    "shape": [1],
                    "datatype": "CUSTOM_TYPE",
                    "data": ["hello"],
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")
        assert request.inputs[0].contents.bytes_contents == [b"hello"]

    def test_input_level_parameters(self) -> None:
        """Input-level parameters should be set on the tensor."""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["prompt"],
                    "parameters": {"binary_data_size": 1024},
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")
        assert request.inputs[0].parameters["binary_data_size"].int64_param == 1024

    def test_input_without_data_key(self) -> None:
        """Input tensor without 'data' key should produce empty contents."""
        payload = {
            "inputs": [
                {
                    "name": "tensor",
                    "shape": [0],
                    "datatype": "INT32",
                }
            ]
        }
        request = dict_to_model_infer_request(payload, model_name="m")
        assert len(request.inputs[0].contents.int_contents) == 0


class TestModelInferResponseToDictAdditional:
    """Additional tests for model_infer_response_to_dict."""

    def test_model_version_included(self) -> None:
        """model_version should appear in result dict when set."""
        response = pb2.ModelInferResponse()
        response.model_name = "m"
        response.model_version = "3"
        output = response.outputs.add()
        output.name = "out"
        output.datatype = "INT32"
        output.shape.append(1)
        output.contents.int_contents.append(42)

        result = model_infer_response_to_dict(response)

        assert result["model_version"] == "3"

    def test_model_version_absent_when_empty(self) -> None:
        """model_version should not appear in result dict when empty."""
        response = pb2.ModelInferResponse()
        response.model_name = "m"
        output = response.outputs.add()
        output.name = "out"
        output.datatype = "INT32"
        output.shape.append(1)
        output.contents.int_contents.append(42)

        result = model_infer_response_to_dict(response)

        assert "model_version" not in result

    @pytest.mark.parametrize(
        "datatype,proto_field,values",
        [
            ("UINT64", "uint64_contents", [42]),
            ("INT64", "int64_contents", [100]),
            ("FP64", "fp64_contents", [0.5]),
            ("BOOL", "bool_contents", [True, False]),
        ],
    )
    def test_extract_additional_output_datatypes(
        self, datatype: str, proto_field: str, values: list
    ) -> None:
        """Additional output datatypes should be extracted correctly."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "tensor"
        output.datatype = datatype
        output.shape.append(len(values))
        getattr(output.contents, proto_field).extend(values)

        result = model_infer_response_to_dict(response)

        for expected, actual in zip(values, result["outputs"][0]["data"], strict=True):
            if isinstance(expected, float):
                assert abs(expected - actual) < 0.001
            else:
                assert expected == actual

    def test_unknown_datatype_falls_back_to_bytes(self) -> None:
        """Unknown datatype should fall back to bytes_contents decoding."""
        response = pb2.ModelInferResponse()
        output = response.outputs.add()
        output.name = "tensor"
        output.datatype = "CUSTOM"
        output.shape.append(1)
        output.contents.bytes_contents.append(b"raw_data")

        result = model_infer_response_to_dict(response)

        assert result["outputs"][0]["data"] == ["raw_data"]

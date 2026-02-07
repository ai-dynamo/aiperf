# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gRPC payload converter (dict <-> protobuf)."""

from aiperf.transports.grpc.payload_converter import (
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

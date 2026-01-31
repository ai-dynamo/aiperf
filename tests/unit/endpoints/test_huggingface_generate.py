# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import ParsedResponse
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse, Turn
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import create_request_info


class TestHuggingFaceGenerateEndpoint:
    """Unit tests for HuggingFaceGenerateEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        endpoint_info = EndpointInfo(
            type=EndpointType.HUGGINGFACE_GENERATE,
            base_url="http://localhost:8081",
            custom_endpoint=None,
        )
        model_list = ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.RANDOM,
        )
        return ModelEndpointInfo(models=model_list, endpoint=endpoint_info)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        ep = HuggingFaceGenerateEndpoint(model_endpoint)
        ep.debug = Mock()
        ep.make_text_response_data = Mock(return_value={"text": "parsed"})
        return ep

    def test_metadata_values(self):
        meta = plugins.get_endpoint_metadata(EndpointType.HUGGINGFACE_GENERATE)
        assert isinstance(meta, EndpointMetadata)
        assert meta.endpoint_path == "/generate"
        assert meta.streaming_path == "/generate_stream"
        assert meta.supports_streaming
        assert meta.produces_tokens
        assert meta.tokenizes_input
        assert meta.metrics_title == "LLM Metrics"

    def test_format_payload_basic(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["Hello world"]}])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["inputs"] == "Hello world"
        assert payload["parameters"] == {}

    def test_format_payload_with_max_tokens_and_extra(self, endpoint, model_endpoint):
        turn = Turn(texts=[{"contents": ["hi"]}], max_tokens=25)
        model_endpoint.endpoint.extra = {"temperature": 0.9}
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["parameters"]["max_new_tokens"] == 25
        assert payload["parameters"]["temperature"] == 0.9

    def test_format_payload_multiple_turns_raises(self, endpoint, model_endpoint):
        turn1 = Turn(texts=[{"contents": ["a"]}])
        turn2 = Turn(texts=[{"contents": ["b"]}])
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2]
        )
        with pytest.raises(ValueError):
            endpoint.format_payload(request_info)

    def test_parse_response_streaming_calls_streaming(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        endpoint.model_endpoint.endpoint.streaming = True
        endpoint._parse_streaming = Mock(return_value="stream_result")
        result = endpoint.parse_response(response)
        assert result == "stream_result"
        endpoint._parse_streaming.assert_called_once_with(response)

    def test_parse_response_non_streaming_calls_non_streaming(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        endpoint.model_endpoint.endpoint.streaming = False
        endpoint._parse_non_streaming = Mock(return_value="non_stream_result")
        result = endpoint.parse_response(response)
        assert result == "non_stream_result"
        endpoint._parse_non_streaming.assert_called_once_with(response)

    def test_parse_non_streaming_with_list(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = [{"generated_text": "ok"}]
        response.perf_ns = 123
        result = endpoint._parse_non_streaming(response)
        endpoint.make_text_response_data.assert_called_once_with("ok")
        assert isinstance(result, ParsedResponse)
        assert result.data.text == "parsed"

    def test_parse_non_streaming_with_dict(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"generated_text": "done"}
        response.perf_ns = 999
        result = endpoint._parse_non_streaming(response)
        endpoint.make_text_response_data.assert_called_once_with("done")
        assert isinstance(result, ParsedResponse)

    def test_parse_non_streaming_no_text(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"foo": "bar"}
        result = endpoint._parse_non_streaming(response)
        assert result is None
        endpoint.debug.assert_called()

    def test_parse_non_streaming_empty_json(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = None
        assert endpoint._parse_non_streaming(response) is None

    def test_parse_streaming_basic_tokens(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"token": {"text": "hi there"}}
        response.perf_ns = 321

        result = endpoint._parse_streaming(response)

        assert isinstance(result, ParsedResponse)
        endpoint.make_text_response_data.assert_called_once_with("hi there")

    def test_parse_streaming_generated_text_field(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"generated_text": "final"}
        response.perf_ns = 123

        result = endpoint._parse_streaming(response)

        assert isinstance(result, ParsedResponse)
        endpoint.make_text_response_data.assert_called_once_with("final")

    def test_parse_streaming_skips_non_data_and_bad_json(self, endpoint):
        bad_packet = Mock()
        bad_packet.name = "data"
        bad_packet.value = "{bad_json"
        other_packet = Mock()
        other_packet.name = "event"
        other_packet.value = "ignored"
        response = Mock(
            spec=InferenceServerResponse,
            packets=[bad_packet, other_packet],
            perf_ns=555,
        )

        result = endpoint._parse_streaming(response)
        assert result is None
        endpoint.debug.assert_called()

    def test_parse_streaming_empty_packets(self, endpoint):
        response = Mock(spec=InferenceServerResponse, packets=[], perf_ns=1)
        result = endpoint._parse_streaming(response)
        assert result is None
        endpoint.debug.assert_called()

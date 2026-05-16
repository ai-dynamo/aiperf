# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.models import ParsedResponse
from aiperf.common.models.dataset_models import Text
from aiperf.common.models.record_models import (
    InferenceServerResponse,
    TextResponseData,
    Turn,
)
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import _wrap_run, create_config, create_request_info


class TestHuggingFaceGenerateEndpoint:
    """Unit tests for HuggingFaceGenerateEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        return create_config(
            EndpointType.HUGGINGFACE_GENERATE,
            base_url="http://localhost:8081",
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        ep = HuggingFaceGenerateEndpoint(run=_wrap_run(model_endpoint))
        ep.debug = Mock()
        ep.make_text_response_data = Mock(return_value=TextResponseData(text="parsed"))
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
        turn = Turn(texts=[Text(contents=["Hello world"])])
        request_info = create_request_info(config=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        assert payload["inputs"] == "Hello world"
        assert payload["parameters"] == {}

    def test_format_payload_with_max_tokens_and_extra(self):
        cfg = create_config(
            EndpointType.HUGGINGFACE_GENERATE,
            base_url="http://localhost:8081",
            extra={"temperature": 0.9},
        )
        ep = HuggingFaceGenerateEndpoint(run=_wrap_run(cfg))
        ep.debug = Mock()
        turn = Turn(texts=[Text(contents=["hi"])], max_tokens=25)
        request_info = create_request_info(config=cfg, turns=[turn])

        payload = ep.format_payload(request_info)
        assert payload["parameters"]["max_new_tokens"] == 25
        assert payload["parameters"]["temperature"] == 0.9

    def test_format_payload_multiple_turns_raises(self, endpoint, model_endpoint):
        turn1 = Turn(texts=[Text(contents=["a"])])
        turn2 = Turn(texts=[Text(contents=["b"])])
        request_info = create_request_info(config=model_endpoint, turns=[turn1, turn2])
        with pytest.raises(ValueError):
            endpoint.format_payload(request_info)

    def test_parse_response_streaming_calls_streaming(self):
        cfg = create_config(
            EndpointType.HUGGINGFACE_GENERATE,
            base_url="http://localhost:8081",
            streaming=True,
        )
        ep = HuggingFaceGenerateEndpoint(run=_wrap_run(cfg))
        response = Mock(spec=InferenceServerResponse)
        ep._parse_streaming = Mock(return_value="stream_result")
        result = ep.parse_response(response)
        assert result == "stream_result"
        ep._parse_streaming.assert_called_once_with(response)

    def test_parse_response_non_streaming_calls_non_streaming(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
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

    def test_parse_streaming_generated_text_only_returns_none(self, endpoint):
        """Events with only generated_text but no token are ignored."""
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"generated_text": "final"}
        response.perf_ns = 123

        result = endpoint._parse_streaming(response)

        assert result is None

    def test_parse_streaming_final_event_ignores_generated_text(self, endpoint):
        """TGI final events have both token.text and generated_text; only use token.text."""
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {
            "token": {"text": "world"},
            "generated_text": "hello world",
        }
        response.perf_ns = 456

        result = endpoint._parse_streaming(response)

        assert isinstance(result, ParsedResponse)
        endpoint.make_text_response_data.assert_called_once_with("world")

    def test_parse_streaming_bad_json_returns_none(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = None
        response.perf_ns = 555

        result = endpoint._parse_streaming(response)
        assert result is None
        endpoint.debug.assert_called()

    def test_parse_streaming_no_text_fields(self, endpoint):
        response = Mock(spec=InferenceServerResponse)
        response.get_json.return_value = {"some_other_field": "value"}
        response.perf_ns = 1
        result = endpoint._parse_streaming(response)
        assert result is None
        endpoint.debug.assert_called()

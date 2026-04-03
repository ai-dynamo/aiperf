# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models.record_models import (
    EmbeddingResponseData,
    RankingsResponseData,
    TextResponseData,
)
from aiperf.endpoints.raw_endpoint import RawEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def raw_model_endpoint():
    return create_model_endpoint(EndpointType.RAW)


@pytest.fixture
def raw_endpoint(raw_model_endpoint):
    return create_endpoint_with_mock_transport(RawEndpoint, raw_model_endpoint)


class TestRawEndpointFormatPayload:
    def test_format_payload_raises(self, raw_endpoint, raw_model_endpoint):
        with pytest.raises(NotImplementedError, match="does not format payloads"):
            raw_endpoint.format_payload(
                create_request_info(model_endpoint=raw_model_endpoint)
            )


class TestRawEndpointParseResponse:
    @pytest.mark.parametrize(
        "json_data,expected_text",
        [
            ({"choices": [{"message": {"content": "Hello"}}]}, "Hello"),
            ({"choices": [{"delta": {"content": "chunk"}}]}, "chunk"),
            ({"choices": [{"text": "completion"}]}, "completion"),
            ({"text": "simple"}, "simple"),
            ({"content": "direct"}, "direct"),
        ],
    )
    def test_auto_detect_text(self, raw_endpoint, json_data, expected_text):
        parsed = raw_endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == expected_text

    def test_auto_detect_embeddings(self, raw_endpoint):
        json_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "object": "embedding"},
                {"embedding": [0.4, 0.5, 0.6], "object": "embedding"},
            ]
        }
        parsed = raw_endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, EmbeddingResponseData)
        assert len(parsed.data.embeddings) == 2

    def test_auto_detect_rankings(self, raw_endpoint):
        json_data = {"results": [{"index": 0, "score": 0.9}]}
        parsed = raw_endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, RankingsResponseData)

    def test_plain_text_fallback(self, raw_endpoint):
        parsed = raw_endpoint.parse_response(
            create_mock_response(json_data=None, text="Plain text response")
        )

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Plain text response"

    @pytest.mark.parametrize(
        "json_data,text",
        [
            ({"status": "ok"}, None),
            (None, None),
            (None, ""),
        ],
    )
    def test_empty_response_returns_none(self, raw_endpoint, json_data, text):
        parsed = raw_endpoint.parse_response(
            create_mock_response(json_data=json_data, text=text)
        )

        assert parsed is None

    def test_jmespath_response_field(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RAW,
            extra=[("response_field", "data[0].text")],
        )
        endpoint = create_endpoint_with_mock_transport(RawEndpoint, model_endpoint)

        json_data = {"data": [{"text": "extracted"}]}
        parsed = endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "extracted"

    def test_jmespath_falls_back_to_auto_detect(self):
        model_endpoint = create_model_endpoint(
            EndpointType.RAW,
            extra=[("response_field", "nonexistent.path")],
        )
        endpoint = create_endpoint_with_mock_transport(RawEndpoint, model_endpoint)

        json_data = {"text": "auto-detected"}
        parsed = endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "auto-detected"

    def test_jmespath_non_string_response_field_logs_error(self):
        """Non-string response_field must not crash jmespath.compile."""
        model_endpoint = create_model_endpoint(
            EndpointType.RAW,
            extra=[("response_field", 42)],
        )
        endpoint = create_endpoint_with_mock_transport(RawEndpoint, model_endpoint)
        assert endpoint._compiled_jmespath is None


def test_metadata():
    metadata = plugins.get_endpoint_metadata(EndpointType.RAW)
    assert isinstance(metadata, EndpointMetadata)
    assert metadata.endpoint_path is None
    assert metadata.supports_streaming is True
    assert metadata.produces_tokens is True
    assert metadata.tokenizes_input is True
    assert metadata.metrics_title == "LLM Metrics"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import InvalidStateError
from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.record_models import TextResponseData
from aiperf.endpoints.template_endpoint import TemplateEndpoint
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
def simple_template_str():
    """Simple template string."""
    return '{"text": {{ texts|tojson }}, "model": "{{ model }}"}'


@pytest.fixture
def template_model_endpoint(simple_template_str):
    """Model endpoint with simple template."""
    return create_model_endpoint(
        EndpointType.TEMPLATE,
        extra=[("payload_template", simple_template_str)],
    )


@pytest.fixture
def template_endpoint(template_model_endpoint):
    """Create a TemplateEndpoint instance."""
    return create_endpoint_with_mock_transport(
        TemplateEndpoint, template_model_endpoint
    )


class TestTemplateEndpointFormatPayload:
    """Tests for template payload formatting."""

    def test_simple_text(self, template_endpoint, template_model_endpoint):
        """Test basic text formatting."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        payload = template_endpoint.format_payload(
            create_request_info(model_endpoint=template_model_endpoint, turns=[turn])
        )

        assert payload["text"] == ["Hello"]
        assert payload["model"] == "test-model"

    def test_multiple_text_contents(self, template_endpoint, template_model_endpoint):
        """Test flattening of multiple text contents."""
        turn = Turn(
            texts=[Text(contents=["A", "B"]), Text(contents=["C"])],
            model="test-model",
        )
        payload = template_endpoint.format_payload(
            create_request_info(model_endpoint=template_model_endpoint, turns=[turn])
        )

        assert payload["text"] == ["A", "B", "C"]

    @pytest.mark.parametrize(
        "template,extra_vars,expected",
        [
            ('{"images": {{ images|tojson }}}', [], {"images": ["img1"]}),
            (
                '{"base": "value"}',
                [("temperature", 0.7)],
                {"base": "value", "temperature": 0.7},
            ),
            ('{"max_tokens": {{ max_tokens }}}', [], {"max_tokens": 100}),
        ],
    )
    def test_template_variables(self, template, extra_vars, expected):
        """Test various template variables and extra updating payload."""
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)] + extra_vars,
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["test"])],
            images=[Image(contents=["img1"])],
            max_tokens=100,
            model="test-model",
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert all(payload.get(k) == v for k, v in expected.items())

    def test_model_fallback(self, template_endpoint, template_model_endpoint):
        """Test model fallback to endpoint default."""
        turn = Turn(texts=[Text(contents=["Test"])], model=None)
        payload = template_endpoint.format_payload(
            create_request_info(model_endpoint=template_model_endpoint, turns=[turn])
        )

        assert payload["model"] == template_model_endpoint.primary_model_name

    def test_named_template_nv_embedqa(self):
        """Test nv-embedqa named template."""
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", "nv-embedqa")],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(texts=[Text(contents=["What is AI?"])], model="test-model")
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload == {"text": ["What is AI?"]}

    def test_named_template_nv_embedqa_multiple_texts(self):
        """Test nv-embedqa with multiple texts (batch embedding)."""
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", "nv-embedqa")],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["text1", "text2", "text3"])], model="test-model"
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload == {"text": ["text1", "text2", "text3"]}

    def test_named_template_not_found_uses_as_inline(self):
        """Test that unknown named template is treated as inline template."""
        template = '{"custom": {{ texts|tojson }}}'
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["custom"] == ["Test"]

    def test_invalid_json_template(self):
        """Test invalid template raises error."""
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", "Not valid JSON")],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        with pytest.raises(ValueError, match="did not render valid JSON"):
            endpoint.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_no_turns_raises_error(self, template_endpoint, template_model_endpoint):
        """Test error when no turns provided."""
        with pytest.raises(ValueError, match="requires at least one turn"):
            template_endpoint.format_payload(
                create_request_info(model_endpoint=template_model_endpoint, turns=[])
            )

    def test_missing_template_raises_error(self):
        """Test initialization fails without template."""
        model_endpoint = create_model_endpoint(EndpointType.TEMPLATE)
        with pytest.raises(InvalidStateError, match="requires 'payload_template'"):
            create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

    def test_access_full_turn_object(self):
        """Test accessing the full turn object for advanced use cases."""
        template = '{"text": "{{ turn.texts[0].contents[0] }}", "name": "{{ turn.texts[0].name }}"}'
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["Hello"], name="User")],
            model="test-model",
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["text"] == "Hello"
        assert payload["name"] == "User"

    def test_access_request_info_metadata(self):
        """Test accessing request_info for correlation IDs and metadata."""
        template = """
        {
            "text": {{ texts|tojson }},
            "correlation_id": "{{ request_info.x_correlation_id }}",
            "conversation_id": "{{ request_info.conversation_id }}"
        }
        """
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            x_correlation_id="corr-123",
            conversation_id="conv-456",
        )
        payload = endpoint.format_payload(request_info)

        assert payload["correlation_id"] == "corr-123"
        assert payload["conversation_id"] == "conv-456"

    def test_singular_convenience_variables(self):
        """Test singular variables for single item access."""
        template = '{"text": "{{ text }}", "image": "{{ image }}"}'
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["Single text"])],
            images=[Image(contents=["img.png"])],
            model="test-model",
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["text"] == "Single text"
        assert payload["image"] == "img.png"

    def test_named_text_types_query_and_passage(self):
        """Test query/passage categorization for embedding models."""
        template = '{"query": "{{ query }}", "passages": {{ passages|tojson }}}'
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[
                Text(contents=["What is AI?"], name="query"),
                Text(
                    contents=[
                        "AI is artificial intelligence",
                        "AI systems learn from data",
                    ],
                    name="passages",
                ),
            ],
            model="test-model",
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == "What is AI?"
        assert payload["passages"] == [
            "AI is artificial intelligence",
            "AI systems learn from data",
        ]

    def test_generic_name_dictionaries(self):
        """Test generic name-based dictionaries for custom naming schemes."""
        template = """
        {
            "system": {{ texts_by_name.system|tojson }},
            "user_input": {{ texts_by_name.user|tojson }},
            "context": {{ texts_by_name.context|tojson }},
            "main_image": {{ images_by_name.main|tojson }}
        }
        """
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", template)],
        )
        endpoint = create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

        turn = Turn(
            texts=[
                Text(contents=["You are a helpful assistant"], name="system"),
                Text(contents=["What's in this image?"], name="user"),
                Text(contents=["Previous context here"], name="context"),
            ],
            images=[
                Image(contents=["main_img.png"], name="main"),
            ],
            model="test-model",
        )
        payload = endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["system"] == ["You are a helpful assistant"]
        assert payload["user_input"] == ["What's in this image?"]
        assert payload["context"] == ["Previous context here"]
        assert payload["main_image"] == ["main_img.png"]


class TestTemplateEndpointParseResponse:
    """Tests for template response parsing."""

    @pytest.fixture
    def endpoint(self):
        """Endpoint for parse tests."""
        model_endpoint = create_model_endpoint(
            EndpointType.TEMPLATE,
            extra=[("payload_template", '{"text": {{ texts|tojson }}}')],
        )
        return create_endpoint_with_mock_transport(TemplateEndpoint, model_endpoint)

    @pytest.mark.parametrize(
        "json_data,expected_text",
        [
            ({"text": "Response"}, "Response"),
            ({"content": "Response"}, "Response"),
            ({"response": "Response"}, "Response"),
            ({"output": "Response"}, "Response"),
            ({"result": "Response"}, "Response"),
        ],
    )
    def test_json_field_extraction(self, endpoint, json_data, expected_text):
        """Test extraction from various JSON fields."""
        parsed = endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == expected_text

    def test_list_text_joined(self, endpoint):
        """Test list of strings is joined."""
        parsed = endpoint.parse_response(
            create_mock_response(json_data={"text": ["A.", "B.", "C."]})
        )

        assert parsed.data.text == "A.B.C."

    @pytest.mark.parametrize(
        "json_data",
        [
            {"choices": [{"text": "Response"}]},
            {"choices": [{"message": {"content": "Response"}}]},
            {"choices": [{"delta": {"content": "Response"}}]},
        ],
    )
    def test_openai_format(self, endpoint, json_data):
        """Test OpenAI-style response formats (completions, chat, and streaming)."""
        parsed = endpoint.parse_response(create_mock_response(json_data=json_data))

        assert parsed.data.text == "Response"

    def test_plain_text_fallback(self, endpoint):
        """Test plain text response parsing."""
        parsed = endpoint.parse_response(
            create_mock_response(json_data=None, text="Plain text")
        )

        assert parsed.data.text == "Plain text"

    @pytest.mark.parametrize(
        "json_data,text",
        [
            ({"status": "ok"}, None),
            (None, None),
            (None, ""),
            ({"choices": []}, None),
        ],
    )
    def test_no_content_returns_none(self, endpoint, json_data, text):
        """Test None returned when no extractable content."""
        parsed = endpoint.parse_response(
            create_mock_response(json_data=json_data, text=text)
        )

        assert parsed is None


def test_metadata():
    """Test endpoint metadata."""
    metadata = plugins.get_endpoint_metadata(EndpointType.TEMPLATE)
    assert isinstance(metadata, EndpointMetadata)
    assert metadata.endpoint_path is None
    assert metadata.supports_streaming is True
    assert metadata.produces_tokens is True
    assert metadata.supports_audio is True
    assert metadata.supports_images is True
    assert metadata.supports_videos is True
    assert metadata.metrics_title == "LLM Metrics"

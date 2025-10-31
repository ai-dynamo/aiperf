# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.openai_chat import ChatEndpoint


class TestChatEndpoint:
    """Test ChatEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
            ),
        )

    def test_format_payload_basic(self, model_endpoint, sample_conversations):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "messages": [
                {
                    "role": turn.role or "user",
                    "name": turn.texts[0].name,
                    "content": turn.texts[0].contents[0],
                }
            ],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_format_payload_with_max_tokens_and_streaming(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 42
        model_endpoint.endpoint.streaming = True
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "messages": [
                {
                    "role": turn.role or "user",
                    "name": turn.texts[0].name,
                    "content": turn.texts[0].contents[0],
                }
            ],
            "model": "test-model",
            "stream": True,
            "max_completion_tokens": 42,
        }
        assert payload == expected_payload

    def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.extra = {"ignore_eos": True, "temperature": 0.7}
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "messages": [
                {
                    "role": turn.role or "user",
                    "name": turn.texts[0].name,
                    "content": turn.texts[0].contents[0],
                }
            ],
            "model": "test-model",
            "stream": False,
            "ignore_eos": True,
            "temperature": 0.7,
        }
        assert payload == expected_payload

    def test_format_payload_multiple_turns_with_text_and_image(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        # Create a turn with both text and image
        turns = sample_conversations["session_1"].turns
        turns[0].images = type("ImageList", (), {})()
        turns[0].images = [
            type("Image", (), {"contents": ["http://image.url/img1.png"]})()
        ]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "messages": [
                {
                    "role": turns[0].role or "user",
                    "content": [
                        {"type": "text", "text": "Hello, world!"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://image.url/img1.png"},
                        },
                    ],
                },
                {
                    "role": turns[1].role,
                    "name": turns[1].texts[0].name,
                    "content": turns[1].texts[0].contents[0],
                },
            ],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_format_payload_with_audio(self, model_endpoint, sample_conversations):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turn.audios = [type("Audio", (), {"contents": ["mp3,ZmFrZV9hdWRpbw=="]})()]
        turns = [turn]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "messages": [
                {
                    "role": turn.role or "user",
                    "content": [
                        {"type": "text", "text": turn.texts[0].contents[0]},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "ZmFrZV9hdWRpbw==",
                                "format": "mp3",
                            },
                        },
                    ],
                }
            ],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_create_messages_hotfix(self, model_endpoint, sample_conversations):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        messages = endpoint._create_messages(turns)
        assert messages[0]["role"] == (turn.role or "user")
        assert messages[0]["name"] == turn.texts[0].name
        assert messages[0]["content"] == turn.texts[0].contents[0]

    def test_create_messages_with_empty_content(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turn.texts[0].contents = [""]
        turns = [turn]
        messages = endpoint._create_messages(turns)
        assert messages[0]["role"] == (turn.role or "user")
        assert messages[0]["name"] == turn.texts[0].name
        assert messages[0]["content"] == ""

    def test_create_messages_audio_format_error(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turn.audios = [type("Audio", (), {"contents": ["not_base64_audio"]})()]
        turns = [turn]
        with pytest.raises(ValueError):
            endpoint._create_messages(turns)

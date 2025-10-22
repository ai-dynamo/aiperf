# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums.endpoints_enums import EndpointType
from aiperf.common.enums.model_enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_chat import OpenAIChatCompletionRequestConverter


class TestOpenAIChatCompletionRequestConverter:
    """Test OpenAIChatCompletionRequestConverter."""

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

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, model_endpoint, sample_conversations):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        payload = await converter.format_payload(model_endpoint, turns)
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

    @pytest.mark.asyncio
    async def test_format_payload_with_max_tokens_and_streaming(
        self, model_endpoint, sample_conversations
    ):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 42
        model_endpoint.endpoint.streaming = True
        payload = await converter.format_payload(model_endpoint, turns)
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

    @pytest.mark.asyncio
    async def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.extra = {"ignore_eos": True, "temperature": 0.7}
        payload = await converter.format_payload(model_endpoint, turns)
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

    @pytest.mark.asyncio
    async def test_format_payload_multiple_turns_with_text_and_image(
        self, model_endpoint, sample_conversations
    ):
        converter = OpenAIChatCompletionRequestConverter()
        # Create a turn with both text and image
        turns = sample_conversations["session_1"].turns
        turns[0].images = type("ImageList", (), {})()
        turns[0].images = [
            type("Image", (), {"contents": ["http://image.url/img1.png"]})()
        ]
        payload = await converter.format_payload(model_endpoint, turns)
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

    @pytest.mark.asyncio
    async def test_format_payload_with_audio(
        self, model_endpoint, sample_conversations
    ):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turn.audios = [type("Audio", (), {"contents": ["mp3,ZmFrZV9hdWRpbw=="]})()]
        turns = [turn]
        payload = await converter.format_payload(model_endpoint, turns)
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

    def test_create_messages_hotfix(self, sample_conversations):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        messages = converter._create_messages(turns)
        assert messages[0]["role"] == (turn.role or "user")
        assert messages[0]["name"] == turn.texts[0].name
        assert messages[0]["content"] == turn.texts[0].contents[0]

    def test_create_messages_with_empty_content(self, sample_conversations):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turn.texts[0].contents = [""]
        turns = [turn]
        messages = converter._create_messages(turns)
        assert messages[0]["role"] == (turn.role or "user")
        assert messages[0]["name"] == turn.texts[0].name
        assert messages[0]["content"] == ""

    def test_create_messages_audio_format_error(self, sample_conversations):
        converter = OpenAIChatCompletionRequestConverter()
        turn = sample_conversations["session_1"].turns[0]
        turn.audios = [type("Audio", (), {"contents": ["not_base64_audio"]})()]
        turns = [turn]
        with pytest.raises(ValueError):
            converter._create_messages(turns)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_request_info


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
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
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
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
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
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
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
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
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
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
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
        messages = endpoint._create_messages(turns, None, None)
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
        messages = endpoint._create_messages(turns, None, None)
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
            endpoint._create_messages(turns, None, None)

    @pytest.mark.parametrize(
        "streaming,use_server_token_count,user_extra,expected_stream_options",
        [
            # Auto-add when both flags enabled
            (True, True, None, {"include_usage": True}),
            # Don't add when not streaming
            (False, True, None, None),
            # Don't add when flag disabled
            (True, False, None, None),
            # Don't add when neither enabled
            (False, False, None, None),
            # Preserve user's include_usage=False
            (True, True, {"stream_options": {"include_usage": False}}, {"include_usage": False}),
            # Merge with user's other options
            (True, True, {"stream_options": {"continuous_updates": True}}, {"continuous_updates": True, "include_usage": True}),
        ],
    )  # fmt: skip
    def test_stream_options_auto_configuration(
        self,
        model_endpoint,
        sample_conversations,
        streaming,
        use_server_token_count,
        user_extra,
        expected_stream_options,
    ):
        """Verify stream_options.include_usage is automatically configured based on flags and user settings."""
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.streaming = streaming
        model_endpoint.endpoint.use_server_token_count = use_server_token_count
        if user_extra:
            model_endpoint.endpoint.extra = user_extra
        request_info = create_request_info(turns=turns, model_endpoint=model_endpoint)
        payload = endpoint.format_payload(request_info)

        if expected_stream_options is None:
            assert "stream_options" not in payload
        else:
            assert "stream_options" in payload
            assert payload["stream_options"] == expected_stream_options
            endpoint._create_messages(turns, None, None)

    def test_create_messages_with_system_message(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        system_message = "You are a helpful AI assistant."
        messages = endpoint._create_messages(turns, system_message, None)

        # First message should be the system message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_message
        # Second message should be the turn
        assert messages[1]["role"] == (turn.role or "user")
        assert messages[1]["content"] == turn.texts[0].contents[0]

    def test_create_messages_with_user_context_message(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        user_context = "The user is working on a Python project."
        messages = endpoint._create_messages(turns, None, user_context)

        # First message should be the user context
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == user_context
        # Second message should be the turn
        assert messages[1]["role"] == (turn.role or "user")
        assert messages[1]["content"] == turn.texts[0].contents[0]

    def test_create_messages_with_both_context_messages(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        system_message = "You are a helpful AI assistant."
        user_context = "The user is working on a Python project."
        messages = endpoint._create_messages(turns, system_message, user_context)

        # First message should be system
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_message
        # Second message should be user context
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == user_context
        # Third message should be the turn
        assert messages[2]["role"] == (turn.role or "user")
        assert messages[2]["content"] == turn.texts[0].contents[0]

    def test_create_messages_with_context_and_multiple_turns(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turns = sample_conversations["session_1"].turns
        system_message = "You are a helpful AI assistant."
        user_context = "The user is working on a Python project."
        messages = endpoint._create_messages(turns, system_message, user_context)

        # Should have system + user context + 2 turns = 4 messages
        assert len(messages) == 4
        # First message should be system
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_message
        # Second message should be user context
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == user_context
        # Third and fourth messages should be the turns
        assert messages[2]["role"] == (turns[0].role or "user")
        assert messages[3]["role"] == turns[1].role

    def test_format_payload_with_context_messages(
        self, model_endpoint, sample_conversations
    ):
        endpoint = ChatEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        system_message = "You are a helpful AI assistant."
        user_context = "The user is working on a Python project."

        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=turns,
            system_message=system_message,
            user_context_message=user_context,
        )
        payload = endpoint.format_payload(request_info)

        # Verify payload structure
        assert "messages" in payload
        assert len(payload["messages"]) == 3
        # First message should be system
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == system_message
        # Second message should be user context
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == user_context
        # Third message should be the turn
        assert payload["messages"][2]["role"] == (turn.role or "user")

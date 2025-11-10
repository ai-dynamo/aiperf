# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models module."""

import pytest
from aiperf_mock_server.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatMessage,
    ChatStreamChoice,
    ChatStreamCompletionResponse,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    Ranking,
    RankingRequest,
    RankingResponse,
    TextChoice,
    TextCompletionResponse,
    TextStreamChoice,
    TextStreamCompletionResponse,
    Usage,
)


class TestBaseCompletionRequest:
    """Tests for BaseCompletionRequest model."""

    @pytest.mark.parametrize(
        "stream_options,expected",
        [
            (None, False),
            ({"include_usage": True}, True),
            ({"include_usage": False}, False),
        ],
    )
    def test_include_usage(self, stream_options, expected):
        req = CompletionRequest(
            model="test", prompt="test", stream_options=stream_options
        )
        assert req.include_usage is expected


class TestCompletionRequest:
    """Tests for CompletionRequest model."""

    def test_list_prompt_filters_empty(self):
        req = CompletionRequest(model="test", prompt=["Line 1", "", "Line 2"])
        assert req.prompt_text == "Line 1\nLine 2"


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    @pytest.mark.parametrize(
        "max_completion_tokens,max_tokens,expected",
        [
            (100, None, 100),
            (None, 50, 50),
            (100, 50, 100),
        ],
    )
    def test_max_output_tokens(self, max_completion_tokens, max_tokens, expected):
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hi")],
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
        )
        assert req.max_output_tokens == expected


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model."""

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            ("text", ["text"]),
            (["text1", "text2"], ["text1", "text2"]),
        ],
    )
    def test_inputs_property(self, input_data, expected):
        req = EmbeddingRequest(model="test", input=input_data)
        assert req.inputs == expected


class TestRankingRequest:
    """Tests for RankingRequest model."""

    def test_passage_texts(self):
        req = RankingRequest(
            model="test",
            query={"text": "query"},
            passages=[
                {"text": "passage 1"},
                {"text": "passage 2"},
            ],
        )
        assert req.passage_texts == ["passage 1", "passage 2"]


class TestUsage:
    """Tests for Usage model."""

    def test_completion_tokens_details(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details={"reasoning_tokens": 3},
        )
        assert usage.completion_tokens_details["reasoning_tokens"] == 3


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_chat_completion_response(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hi"


class TestTextCompletionResponse:
    """Tests for TextCompletionResponse model."""

    def test_text_completion_response(self):
        response = TextCompletionResponse(
            id="cmpl-123",
            created=1234567890,
            model="test-model",
            choices=[
                TextChoice(
                    index=0,
                    text="Generated text",
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        assert response.id == "cmpl-123"
        assert response.object == "text_completion"
        assert len(response.choices) == 1


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_chat_message_basic(self):
        message = ChatMessage(role="assistant", content="Hello!")
        assert message.role == "assistant"
        assert message.content == "Hello!"
        assert message.reasoning_content is None

    def test_chat_message_with_reasoning(self):
        message = ChatMessage(
            role="assistant", content="Answer", reasoning_content="Thinking..."
        )
        assert message.role == "assistant"
        assert message.content == "Answer"
        assert message.reasoning_content == "Thinking..."


class TestChatDelta:
    """Tests for ChatDelta model."""

    def test_chat_delta_with_role_and_content(self):
        delta = ChatDelta(role="assistant", content="Hello")
        assert delta.role == "assistant"
        assert delta.content == "Hello"
        assert delta.reasoning_content is None

    def test_chat_delta_content_only(self):
        delta = ChatDelta(content=" world")
        assert delta.role is None
        assert delta.content == " world"

    def test_chat_delta_reasoning_content(self):
        delta = ChatDelta(reasoning_content="Thinking")
        assert delta.role is None
        assert delta.content is None
        assert delta.reasoning_content == "Thinking"


class TestChatStreamChoice:
    """Tests for ChatStreamChoice model."""

    def test_chat_stream_choice(self):
        delta = ChatDelta(role="assistant", content="Hello")
        choice = ChatStreamChoice(index=0, finish_reason=None, delta=delta)
        assert choice.index == 0
        assert choice.finish_reason is None
        assert choice.delta.role == "assistant"
        assert choice.delta.content == "Hello"

    def test_chat_stream_choice_final(self):
        delta = ChatDelta(content="!")
        choice = ChatStreamChoice(index=0, finish_reason="stop", delta=delta)
        assert choice.finish_reason == "stop"


class TestTextStreamChoice:
    """Tests for TextStreamChoice model."""

    def test_text_stream_choice(self):
        choice = TextStreamChoice(index=0, finish_reason=None, text="Hello")
        assert choice.index == 0
        assert choice.finish_reason is None
        assert choice.text == "Hello"


class TestChatStreamCompletionResponse:
    """Tests for ChatStreamCompletionResponse model."""

    def test_chat_stream_completion_response(self):
        delta = ChatDelta(role="assistant", content="Hi")
        choice = ChatStreamChoice(index=0, finish_reason=None, delta=delta)
        response = ChatStreamCompletionResponse(
            id="chatcmpl-stream-123",
            created=1234567890,
            model="test-model",
            choices=[choice],
        )
        assert response.id == "chatcmpl-stream-123"
        assert response.object == "chat.completion.chunk"
        assert len(response.choices) == 1
        assert response.choices[0].delta.content == "Hi"


class TestTextStreamCompletionResponse:
    """Tests for TextStreamCompletionResponse model."""

    def test_text_stream_completion_response(self):
        choice = TextStreamChoice(index=0, finish_reason=None, text="Hello")
        response = TextStreamCompletionResponse(
            id="cmpl-stream-123",
            created=1234567890,
            model="test-model",
            choices=[choice],
        )
        assert response.id == "cmpl-stream-123"
        assert response.object == "text_completion.chunk"
        assert len(response.choices) == 1
        assert response.choices[0].text == "Hello"


class TestRankingResponse:
    """Tests for RankingResponse model."""

    def test_ranking_response(self):
        response = RankingResponse(
            id="rank-123",
            model="test-model",
            rankings=[
                Ranking(index=0, relevance_score=0.95),
                Ranking(index=1, relevance_score=0.85),
            ],
            usage=Usage(prompt_tokens=10, total_tokens=10),
        )
        assert response.object == "rankings"
        assert response.id == "rank-123"
        assert len(response.rankings) == 2

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for models module."""

import pytest
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    RankingRequest,
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

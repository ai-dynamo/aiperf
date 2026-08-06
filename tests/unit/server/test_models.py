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
    ResponsesRequest,
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

    @pytest.mark.parametrize(
        "prompt,expected_text",
        [
            ([11, 22, 33], "11 22 33"),
            ([[11, 22], [33]], "11 22 33"),
        ],
    )
    def test_token_id_prompt_is_accepted(self, prompt, expected_text):
        req = CompletionRequest(model="test", prompt=prompt)
        assert req.prompt == prompt
        assert req.prompt_text == expected_text


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


class TestResponsesRequest:
    """Tests for ResponsesRequest model and its `prompt_text` shape-flattener."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("hello world", "hello world"),
            (["alpha", "beta"], "alpha\nbeta"),
            (
                [{"role": "user", "content": "single string content"}],
                "single string content",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "first"},
                            {"type": "input_text", "text": "second"},
                        ],
                    }
                ],
                "first\nsecond",
            ),
            (
                [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "hi"},
                ],
                "policy\nhi",
            ),
        ],
    )
    def test_prompt_text_flattens_input_shapes(self, input_value, expected):
        req = ResponsesRequest(model="m", input=input_value)
        assert req.prompt_text == expected

    def test_unmodeled_fields_are_preserved_via_extras(self):
        """`tools`, `instructions`, custom keys flow through the recorder
        because BaseModel has `extra="allow"`."""
        req = ResponsesRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "max_output_tokens": 64,
                "stream": True,
                "instructions": "be brief",
                "tools": [{"type": "function", "name": "foo"}],
            }
        )
        assert req.max_output_tokens == 64
        assert req.stream is True
        # Extras land on the model instance for downstream inspection.
        assert req.instructions == "be brief"
        assert req.tools == [{"type": "function", "name": "foo"}]

    def test_defaults_are_safe(self):
        req = ResponsesRequest(model="m")
        assert req.input == ""
        assert req.max_output_tokens is None
        assert req.stream is False
        assert req.reasoning_effort is None
        assert req.prompt_text == ""

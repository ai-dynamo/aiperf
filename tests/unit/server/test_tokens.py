# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tokens module."""

from aiperf_mock_server.models import ChatCompletionRequest, CompletionRequest, Message
from aiperf_mock_server.tokens import (
    TokenizedText,
    Tokenizer,
    _tokenize,
)


class TestTokenize:
    """Tests for _tokenize function."""

    def test_tokenize_consistent(self):
        text = "Hello world!"
        result1 = _tokenize(text)
        result2 = _tokenize(text)
        assert result1 == result2


class TestTokenizedText:
    """Tests for TokenizedText model."""

    def test_reasoning_content_with_tokens(self):
        tokenized = TokenizedText(
            text="test",
            tokens=["t", "est"],
            prompt_token_count=5,
            reasoning_content_tokens=["rea", "son"],
        )
        assert tokenized.reasoning_content == "reason"

    def test_create_usage_without_reasoning(self):
        tokenized = TokenizedText(
            text="test", tokens=["a", "b", "c"], prompt_token_count=10
        )
        usage = tokenized.create_usage()
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 13
        assert usage.completion_tokens_details is None

    def test_create_usage_with_reasoning(self):
        tokenized = TokenizedText(
            text="test",
            tokens=["a", "b"],
            prompt_token_count=5,
            reasoning_tokens=10,
        )
        usage = tokenized.create_usage()
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 2
        assert usage.total_tokens == 7
        assert usage.completion_tokens_details == {"reasoning_tokens": 10}


class TestTokenizer:
    """Tests for Tokenizer class."""

    def test_tokenize(self):
        text = "Hello world"
        result = Tokenizer.tokenize(text)
        assert len(result) > 0
        assert isinstance(result, tuple)

    def test_count_tokens(self):
        text = "Hello world"
        count = Tokenizer.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_tokenize_completion_request(self):
        req = CompletionRequest(model="test-model", prompt="Hello world", max_tokens=10)
        result = Tokenizer.tokenize_request(req)
        assert isinstance(result, TokenizedText)
        assert result.prompt_token_count > 0
        assert result.count > 0

    def test_tokenize_chat_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_completion_tokens=10,
        )
        result = Tokenizer.tokenize_request(req)
        assert isinstance(result, TokenizedText)
        assert result.prompt_token_count > 0
        assert result.count > 0

    def test_tokenize_with_reasoning(self):
        req = ChatCompletionRequest(
            model="gpt-oss-120b",
            messages=[Message(role="user", content="Solve this problem")],
            reasoning_effort="high",
            max_completion_tokens=600,
        )
        result = Tokenizer.tokenize_request(req)
        assert result.reasoning_tokens > 0
        assert len(result.reasoning_content_tokens) > 0

    def test_ignore_eos(self):
        req = CompletionRequest(
            model="test-model", prompt="Test", max_tokens=10, ignore_eos=True
        )
        result = Tokenizer.tokenize_request(req)
        assert result.finish_reason == "length"

    def test_min_tokens(self):
        req = CompletionRequest(
            model="test-model", prompt="Test", max_tokens=100, min_tokens=50
        )
        result = Tokenizer.tokenize_request(req)
        assert result.count >= 50

    def test_deterministic_output(self):
        req = CompletionRequest(model="test-model", prompt="Same prompt", max_tokens=20)
        result1 = Tokenizer.tokenize_request(req)
        result2 = Tokenizer.tokenize_request(req)
        assert result1.count == result2.count
        assert result1.tokens == result2.tokens

    def test_chat_with_multiple_messages(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
                Message(role="user", content="How are you?"),
            ],
            max_completion_tokens=10,
        )
        result = Tokenizer.tokenize_request(req)
        assert result.prompt_token_count > 0

    def test_chat_with_multimodal_content(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "url": "http://example.com/img.jpg"},
                    ],
                )
            ],
            max_completion_tokens=10,
        )
        result = Tokenizer.tokenize_request(req)
        assert result.prompt_token_count > 0

    def test_completion_with_list_prompt(self):
        req = CompletionRequest(
            model="test-model", prompt=["Line 1", "Line 2"], max_tokens=10
        )
        result = Tokenizer.tokenize_request(req)
        assert result.prompt_token_count > 0

    def test_finish_reason_with_high_max_tokens(self):
        req = CompletionRequest(model="test-model", prompt="Short", max_tokens=1000)
        result = Tokenizer.tokenize_request(req)
        # With very high max_tokens, should finish naturally with "stop"
        assert result.finish_reason == "stop"
        assert result.count < 1000

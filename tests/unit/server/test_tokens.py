# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tokens module."""

import pytest
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
)
from aiperf_mock_server.tokens import (
    TokenizedText,
    _tokenize,
    count_tokens,
    tokenize,
    tokenize_request,
)


class TestTokenize:
    """Tests for _tokenize function."""

    def test_tokenize_consistent(self):
        text = "Hello world!"
        result1 = _tokenize(text)
        result2 = _tokenize(text)
        assert result1 == result2


class TestTokenizedText:
    """Tests for TokenizedText dataclass."""

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
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 13
        assert "completion_tokens_details" not in usage

    def test_create_usage_with_reasoning(self):
        tokenized = TokenizedText(
            text="test",
            tokens=["a", "b"],
            prompt_token_count=5,
            reasoning_tokens=10,
        )
        usage = tokenized.create_usage()
        assert usage["prompt_tokens"] == 5
        # completion_tokens includes both content (2) and reasoning (10)
        assert usage["completion_tokens"] == 12
        assert usage["total_tokens"] == 17
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 10}


class TestTokenizerFunctions:
    """Tests for tokenizer module functions."""

    def test_tokenize(self):
        text = "Hello world"
        result = tokenize(text)
        assert len(result) > 0
        assert isinstance(result, tuple)

    def test_count_tokens(self):
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_tokenize_completion_request(self):
        req = CompletionRequest(model="test-model", prompt="Hello world", max_tokens=10)
        result = tokenize_request(req)
        assert isinstance(result, TokenizedText)
        assert result.prompt_token_count > 0
        assert result.count > 0

    def test_tokenize_chat_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_completion_tokens=10,
        )
        result = tokenize_request(req)
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
        result = tokenize_request(req)
        assert result.reasoning_tokens > 0
        assert len(result.reasoning_content_tokens) > 0

    def test_ignore_eos(self):
        req = CompletionRequest(
            model="test-model", prompt="Test", max_tokens=10, ignore_eos=True
        )
        result = tokenize_request(req)
        assert result.finish_reason == "length"

    def test_min_tokens(self):
        req = CompletionRequest(
            model="test-model", prompt="Test", max_tokens=100, min_tokens=50
        )
        result = tokenize_request(req)
        assert result.count >= 50

    def test_deterministic_output(self):
        req = CompletionRequest(model="test-model", prompt="Same prompt", max_tokens=20)
        result1 = tokenize_request(req)
        result2 = tokenize_request(req)
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
        result = tokenize_request(req)
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
        result = tokenize_request(req)
        assert result.prompt_token_count > 0

    def test_completion_with_list_prompt(self):
        req = CompletionRequest(
            model="test-model", prompt=["Line 1", "Line 2"], max_tokens=10
        )
        result = tokenize_request(req)
        assert result.prompt_token_count > 0

    def test_finish_reason_with_high_max_tokens(self):
        req = CompletionRequest(model="test-model", prompt="Short", max_tokens=1000)
        result = tokenize_request(req)
        # With very high max_tokens, should finish naturally with "stop"
        assert result.finish_reason == "stop"
        assert result.count < 1000


class TestMaxCompletionTokens:
    """Tests for max_completion_tokens handling."""

    @pytest.mark.parametrize("max_tokens,prompt_len", [
        (50, 1000),
        (100, 500),
        (10, 200),
    ])  # fmt: skip
    def test_max_completion_tokens_respected(self, max_tokens, prompt_len):
        """max_completion_tokens should limit output tokens."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="A" * prompt_len)],
            max_completion_tokens=max_tokens,
        )
        result = tokenize_request(req)
        assert result.count <= max_tokens

    @pytest.mark.parametrize("max_tokens", [0, 1, 5])
    def test_max_tokens_small_values(self, max_tokens):
        """Small max_tokens values should be respected (not fallback to default)."""
        req = CompletionRequest(
            model="test-model",
            prompt="Hello world this is a longer prompt",
            max_tokens=max_tokens,
        )
        result = tokenize_request(req)
        assert result.count <= max_tokens

    @pytest.mark.parametrize("model,max_tokens,effort", [
        ("qwen-reasoning", 100, "medium"),
        ("gpt-oss-model", 150, "low"),
        ("qwen-test", 200, "high"),
    ])  # fmt: skip
    def test_reasoning_model_respects_max_completion_tokens(
        self, model, max_tokens, effort
    ):
        """Reasoning models should respect max_completion_tokens for total output."""
        req = ChatCompletionRequest(
            model=model,
            messages=[Message(role="user", content="A" * 500)],
            max_completion_tokens=max_tokens,
            reasoning_effort=effort,
        )
        result = tokenize_request(req)
        total_output = result.reasoning_tokens + result.count
        assert total_output <= max_tokens

    @pytest.mark.parametrize("max_tokens,effort,expected_reasoning", [
        (50, "medium", 50),   # medium=250, capped to 50
        (80, "low", 80),      # low=100, capped to 80
        (100, "low", 100),    # low=100, exactly fits
    ])  # fmt: skip
    def test_reasoning_consumes_budget_leaves_remainder_for_content(
        self, max_tokens, effort, expected_reasoning
    ):
        """Reasoning tokens consume budget, remainder goes to content."""
        req = ChatCompletionRequest(
            model="qwen-test",
            messages=[Message(role="user", content="Test prompt")],
            max_completion_tokens=max_tokens,
            reasoning_effort=effort,
        )
        result = tokenize_request(req)
        assert result.reasoning_tokens == expected_reasoning
        assert result.count == max_tokens - expected_reasoning


class TestIgnoreEos:
    """Tests for ignore_eos parameter."""

    @pytest.mark.parametrize("max_tokens", [50, 100, 200])
    def test_ignore_eos_generates_exact_max_tokens(self, max_tokens):
        """ignore_eos=True should generate exactly max_tokens."""
        req = CompletionRequest(
            model="test-model",
            prompt="Test",
            max_tokens=max_tokens,
            ignore_eos=True,
        )
        result = tokenize_request(req)
        assert result.count == max_tokens
        assert result.finish_reason == "length"

    def test_ignore_eos_false_may_stop_early(self):
        """ignore_eos=False (default) may stop before max_tokens."""
        req = CompletionRequest(
            model="test-model",
            prompt="Test",
            max_tokens=1000,
            ignore_eos=False,
        )
        result = tokenize_request(req)
        assert result.count < 1000

    @pytest.mark.parametrize("max_tokens", [50, 75, 100])
    def test_ignore_eos_with_chat_completion(self, max_tokens):
        """ignore_eos should work with chat completions too."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_completion_tokens=max_tokens,
            ignore_eos=True,
        )
        result = tokenize_request(req)
        assert result.count == max_tokens
        assert result.finish_reason == "length"


class TestMinTokens:
    """Tests for min_tokens parameter."""

    @pytest.mark.parametrize("min_tokens,max_tokens", [
        (50, 200),
        (100, 200),
        (150, 200),
    ])  # fmt: skip
    def test_min_tokens_enforced(self, min_tokens, max_tokens):
        """Output should be at least min_tokens."""
        req = CompletionRequest(
            model="test-model",
            prompt="X",
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        result = tokenize_request(req)
        assert result.count >= min_tokens

    @pytest.mark.parametrize("min_tokens,max_tokens", [
        (100, 50),
        (200, 100),
        (500, 10),
    ])  # fmt: skip
    def test_min_tokens_capped_by_max_tokens(self, min_tokens, max_tokens):
        """min_tokens cannot exceed max_tokens."""
        req = CompletionRequest(
            model="test-model",
            prompt="Test",
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        result = tokenize_request(req)
        assert result.count <= max_tokens

    @pytest.mark.parametrize("min_tokens,max_tokens", [
        (50, 200),
        (80, 200),
        (100, 150),
    ])  # fmt: skip
    def test_min_and_max_tokens_together(self, min_tokens, max_tokens):
        """Output should be in range [min_tokens, max_tokens]."""
        req = CompletionRequest(
            model="test-model",
            prompt="Test prompt here",
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        result = tokenize_request(req)
        assert min_tokens <= result.count <= max_tokens

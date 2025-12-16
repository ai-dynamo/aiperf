# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for utils module."""

import time

import pytest
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
)
from aiperf_mock_server.utils import (
    LatencySimulator,
    make_ctx,
    stream_chat_completion,
    stream_text_completion,
    with_error_injection,
)
from fastapi import HTTPException


class TestRequestId:
    """Tests for request ID generation via make_ctx."""

    @pytest.mark.parametrize(
        "req,expected_prefix",
        [
            (CompletionRequest(model="test", prompt="test"), "cmpl-"),
            (
                ChatCompletionRequest(
                    model="test", messages=[Message(role="user", content="test")]
                ),
                "chatcmpl-",
            ),
        ],
    )
    def test_request_id_format(self, req, expected_prefix):
        ctx = make_ctx(req, "/test", time.perf_counter())
        assert ctx.request_id.startswith(expected_prefix)
        assert len(ctx.request_id) > 10


class TestWithErrorInjection:
    """Tests for with_error_injection decorator."""

    @pytest.mark.asyncio
    async def test_with_error_injection_no_error(self, monkeypatch):
        monkeypatch.setattr("aiperf_mock_server.utils.server_config.error_rate", 0)

        @with_error_injection
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_error_injection_triggers_error(self, monkeypatch):
        monkeypatch.setattr("aiperf_mock_server.utils.server_config.error_rate", 100)
        monkeypatch.setattr("aiperf_mock_server.utils.random.random", lambda: 0.1)

        @with_error_injection
        async def test_func():
            return "success"

        with pytest.raises(HTTPException) as exc_info:
            await test_func()
        assert exc_info.value.status_code == 500
        assert "Simulated error" in exc_info.value.detail


class TestRequestCtx:
    """Tests for RequestCtx class."""

    def test_request_ctx_creation(self):
        req = CompletionRequest(model="test", prompt="Hello world")
        ctx = make_ctx(req, "/v1/completions", time.perf_counter())

        assert ctx.model == "test"
        assert ctx.request_id.startswith("cmpl-")
        assert isinstance(ctx.latency_sim, LatencySimulator)
        assert ctx.tokenized is not None

    def test_request_ctx_properties(self):
        req = CompletionRequest(model="test", prompt="Hello world")
        ctx = make_ctx(req, "/v1/completions", time.perf_counter())

        assert isinstance(ctx.tokens, list)
        assert isinstance(ctx.content, str)
        assert ctx.finish_reason in ("stop", "length")
        assert isinstance(ctx.usage, dict)


class TestStreamTextCompletion:
    """Tests for stream_text_completion function."""

    @pytest.mark.asyncio
    async def test_stream_text_completion_basic(self):
        req = CompletionRequest(model="test", prompt="test")
        ctx = make_ctx(req, "/v1/completions", time.perf_counter())

        chunks = []
        async for chunk in stream_text_completion(ctx, "/v1/completions", False):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert any(b"data:" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_text_completion_with_usage(self):
        req = CompletionRequest(
            model="test", prompt="test", stream_options={"include_usage": True}
        )
        ctx = make_ctx(req, "/v1/completions", time.perf_counter())

        chunks = []
        async for chunk in stream_text_completion(ctx, "/v1/completions", True):
            chunks.append(chunk)

        assert any(b"usage" in chunk for chunk in chunks)


class TestStreamChatCompletion:
    """Tests for stream_chat_completion function."""

    @pytest.mark.asyncio
    async def test_stream_chat_completion_basic(self):
        req = ChatCompletionRequest(
            model="test", messages=[Message(role="user", content="Hi")]
        )
        ctx = make_ctx(req, "/v1/chat/completions", time.perf_counter())

        chunks = []
        async for chunk in stream_chat_completion(ctx, "/v1/chat/completions", False):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert any(b"data:" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_reasoning(self):
        req = ChatCompletionRequest(
            model="gpt-oss-120b",
            messages=[Message(role="user", content="Solve")],
            reasoning_effort="high",
        )
        ctx = make_ctx(req, "/v1/chat/completions", time.perf_counter())

        chunks = []
        async for chunk in stream_chat_completion(ctx, "/v1/chat/completions", False):
            chunks.append(chunk)

        assert any(b"reasoning_content" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_usage(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hi")],
            stream_options={"include_usage": True},
        )
        ctx = make_ctx(req, "/v1/chat/completions", time.perf_counter())

        chunks = []
        async for chunk in stream_chat_completion(ctx, "/v1/chat/completions", True):
            chunks.append(chunk)

        assert any(b"usage" in chunk for chunk in chunks)

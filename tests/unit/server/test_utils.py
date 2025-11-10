# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for utils module."""

import pytest
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
)
from aiperf_mock_server.utils import (
    LatencySimulator,
    RequestContext,
    create_request_id,
    stream_chat_completion,
    stream_text_completion,
    with_error_injection,
)
from fastapi import HTTPException


class TestGetRequestId:
    """Tests for create_request_id function."""

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
        req_id = create_request_id(req)
        assert req_id.startswith(expected_prefix)
        assert len(req_id) > 10


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


class TestRequestContext:
    """Tests for RequestContext class."""

    def test_request_context_creation(self):
        req = CompletionRequest(model="test", prompt="Hello world")
        ctx = RequestContext(req)

        assert ctx.request == req
        assert ctx.request_id.startswith("cmpl-")
        assert isinstance(ctx.latency_sim, LatencySimulator)
        assert ctx.tokenized is not None


class TestStreamTextCompletion:
    """Tests for stream_text_completion function."""

    @pytest.mark.asyncio
    async def test_stream_text_completion_basic(self):
        req = CompletionRequest(model="test", prompt="test")
        ctx = RequestContext(req)

        chunks = []
        async for chunk in stream_text_completion(ctx):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"
        assert any("data:" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_text_completion_with_usage(self):
        req = CompletionRequest(
            model="test", prompt="test", stream_options={"include_usage": True}
        )
        ctx = RequestContext(req)

        chunks = []
        async for chunk in stream_text_completion(ctx):
            chunks.append(chunk)

        assert any("usage" in chunk for chunk in chunks)


class TestStreamChatCompletion:
    """Tests for stream_chat_completion function."""

    @pytest.mark.asyncio
    async def test_stream_chat_completion_basic(self):
        req = ChatCompletionRequest(
            model="test", messages=[Message(role="user", content="Hi")]
        )
        ctx = RequestContext(req)

        chunks = []
        async for chunk in stream_chat_completion(ctx):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"
        assert any("data:" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_reasoning(self):
        req = ChatCompletionRequest(
            model="gpt-oss-120b",
            messages=[Message(role="user", content="Solve")],
            reasoning_effort="high",
        )
        ctx = RequestContext(req)

        chunks = []
        async for chunk in stream_chat_completion(ctx):
            chunks.append(chunk)

        assert any("reasoning_content" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_usage(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[Message(role="user", content="Hi")],
            stream_options={"include_usage": True},
        )
        ctx = RequestContext(req)

        chunks = []
        async for chunk in stream_chat_completion(ctx):
            chunks.append(chunk)

        assert any("usage" in chunk for chunk in chunks)

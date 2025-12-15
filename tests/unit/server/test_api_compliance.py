# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive API compliance tests for the AIPerf Mock Server."""

import pytest
from aiperf_mock_server.app import asgi_app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    """Create async test client with timing middleware."""
    async with AsyncClient(
        transport=ASGITransport(app=asgi_app), base_url="http://test"
    ) as c:
        yield c


# ============================================================================
# Chat Completions API Compliance
# ============================================================================


class TestChatCompletions:
    """Test /v1/chat/completions endpoint for OpenAI API compliance."""

    async def test_basic_chat_completion(self, client: AsyncClient):
        """Test basic non-streaming chat completion."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields per OpenAI spec
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["model"] == "test-model"

        # Choices validation
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "finish_reason" in choice
        assert choice["finish_reason"] in ("stop", "length")

        # Message validation
        assert "message" in choice
        message = choice["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert isinstance(message["content"], str)

        # Usage validation
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    async def test_chat_completion_with_max_tokens(self, client: AsyncClient):
        """Test chat completion with max_tokens parameter."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["completion_tokens"] <= 10

    async def test_chat_completion_streaming(self, client: AsyncClient):
        """Test streaming chat completion."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                import orjson

                chunk = orjson.loads(data_str)
                chunks.append(chunk)

        assert len(chunks) > 0

        # Validate first chunk
        first = chunks[0]
        assert first["object"] == "chat.completion.chunk"
        assert "id" in first
        assert first["id"].startswith("chatcmpl-")
        assert "created" in first
        assert "model" in first
        assert "choices" in first
        assert len(first["choices"]) == 1
        assert "delta" in first["choices"][0]
        assert first["choices"][0]["delta"].get("role") == "assistant"

        # Validate last chunk has finish_reason
        last = chunks[-1]
        assert last["choices"][0].get("finish_reason") in ("stop", "length")

    async def test_chat_completion_streaming_with_usage(self, client: AsyncClient):
        """Test streaming chat completion with include_usage option."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200

        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                import orjson

                chunk = orjson.loads(data_str)
                chunks.append(chunk)

        # Last chunk before [DONE] should have usage
        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) == 1
        usage = usage_chunks[0]["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    async def test_chat_completion_reasoning_model(self, client: AsyncClient):
        """Test chat completion with reasoning model (qwen)."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-reasoning",
                "messages": [{"role": "user", "content": "Hello world"}],
                "reasoning_effort": "low",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Reasoning models may include reasoning_content
        message = data["choices"][0]["message"]
        assert "content" in message
        # reasoning_content is optional
        if "reasoning_content" in message:
            assert isinstance(message["reasoning_content"], str)

        # Check for completion_tokens_details with reasoning_tokens
        if "completion_tokens_details" in data["usage"]:
            details = data["usage"]["completion_tokens_details"]
            if "reasoning_tokens" in details:
                assert isinstance(details["reasoning_tokens"], int)

    async def test_chat_completion_multimodal_content(self, client: AsyncClient):
        """Test chat completion with multimodal content array."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc123"},
                            },
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data


# ============================================================================
# Text Completions API Compliance
# ============================================================================


class TestTextCompletions:
    """Test /v1/completions endpoint for OpenAI API compliance."""

    async def test_basic_text_completion(self, client: AsyncClient):
        """Test basic non-streaming text completion."""
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Once upon a time",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields per OpenAI spec
        assert "id" in data
        assert data["id"].startswith("cmpl-")
        assert data["object"] == "text_completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["model"] == "test-model"

        # Choices validation
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "finish_reason" in choice
        assert choice["finish_reason"] in ("stop", "length")
        assert "text" in choice
        assert isinstance(choice["text"], str)

        # Usage validation
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    async def test_text_completion_array_prompt(self, client: AsyncClient):
        """Test text completion with array prompt."""
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": ["Hello", "World"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data

    async def test_text_completion_streaming(self, client: AsyncClient):
        """Test streaming text completion."""
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                import orjson

                chunk = orjson.loads(data_str)
                chunks.append(chunk)

        assert len(chunks) > 0

        # Validate chunk structure
        first = chunks[0]
        assert first["object"] == "text_completion"
        assert "id" in first
        assert "choices" in first
        assert "text" in first["choices"][0]


# ============================================================================
# Embeddings API Compliance
# ============================================================================


class TestEmbeddings:
    """Test /v1/embeddings endpoint for OpenAI API compliance."""

    async def test_basic_embedding(self, client: AsyncClient):
        """Test basic embedding request."""
        resp = await client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": "Hello world",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields per OpenAI spec
        assert data["object"] == "list"
        assert data["model"] == "text-embedding-ada-002"

        # Data validation
        assert "data" in data
        assert len(data["data"]) == 1
        embedding = data["data"][0]
        assert embedding["object"] == "embedding"
        assert embedding["index"] == 0
        assert "embedding" in embedding
        assert isinstance(embedding["embedding"], list)
        assert len(embedding["embedding"]) == 768  # Mock uses 768 dimensions
        assert all(isinstance(v, float) for v in embedding["embedding"])

        # Usage validation
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]

    async def test_embedding_array_input(self, client: AsyncClient):
        """Test embedding with array input."""
        resp = await client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": ["Hello", "World", "Test"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["data"]) == 3
        for i, emb in enumerate(data["data"]):
            assert emb["index"] == i
            assert emb["object"] == "embedding"

    async def test_embedding_deterministic(self, client: AsyncClient):
        """Test that embeddings are deterministic for same input."""
        input_text = "Test deterministic embedding"

        resp1 = await client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": input_text},
        )
        resp2 = await client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": input_text},
        )

        emb1 = resp1.json()["data"][0]["embedding"]
        emb2 = resp2.json()["data"][0]["embedding"]
        assert emb1 == emb2


# ============================================================================
# Rankings API Compliance (NIM)
# ============================================================================


class TestNIMRankings:
    """Test /v1/ranking endpoint for NVIDIA NIM API compliance."""

    async def test_basic_ranking(self, client: AsyncClient):
        """Test basic ranking request."""
        resp = await client.post(
            "/v1/ranking",
            json={
                "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                "query": {"text": "What is machine learning?"},
                "passages": [
                    {"text": "Machine learning is a subset of AI."},
                    {"text": "The weather is nice today."},
                    {"text": "Deep learning uses neural networks."},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Required fields per NIM spec
        assert "id" in data
        assert data["id"].startswith("rank-")
        assert data["object"] == "rankings"
        assert data["model"] == "nvidia/nv-rerankqa-mistral-4b-v3"

        # Rankings validation
        assert "rankings" in data
        assert len(data["rankings"]) == 3

        for ranking in data["rankings"]:
            assert "index" in ranking
            assert "relevance_score" in ranking
            assert isinstance(ranking["index"], int)
            assert isinstance(ranking["relevance_score"], float)
            assert 0 <= ranking["relevance_score"] <= 1

        # Rankings should be sorted by score descending
        scores = [r["relevance_score"] for r in data["rankings"]]
        assert scores == sorted(scores, reverse=True)

        # Usage validation
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]


# ============================================================================
# HuggingFace TEI Rerank API Compliance
# ============================================================================


class TestHFTEIRerank:
    """Test /rerank endpoint for HuggingFace TEI API compliance."""

    async def test_basic_rerank_with_texts(self, client: AsyncClient):
        """Test basic rerank request with texts parameter."""
        resp = await client.post(
            "/rerank",
            json={
                "query": "What is machine learning?",
                "texts": [
                    "Machine learning is a subset of AI.",
                    "The weather is nice today.",
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # HF TEI format
        assert "results" in data
        assert len(data["results"]) == 2

        for result in data["results"]:
            assert "index" in result
            assert "score" in result
            assert isinstance(result["index"], int)
            assert isinstance(result["score"], float)

    async def test_rerank_with_documents(self, client: AsyncClient):
        """Test rerank request with documents parameter (alternative format)."""
        resp = await client.post(
            "/rerank",
            json={
                "query": "What is deep learning?",
                "documents": [
                    "Deep learning uses neural networks.",
                    "Cats are cute animals.",
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2


# ============================================================================
# Cohere Rerank API Compliance
# ============================================================================


class TestCohereRerank:
    """Test /v2/rerank endpoint for Cohere API compliance."""

    async def test_basic_cohere_rerank(self, client: AsyncClient):
        """Test basic Cohere rerank request."""
        resp = await client.post(
            "/v2/rerank",
            json={
                "model": "rerank-english-v3.0",
                "query": "What is the capital of France?",
                "documents": [
                    "Paris is the capital of France.",
                    "Berlin is the capital of Germany.",
                    "Madrid is the capital of Spain.",
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Cohere format uses relevance_score
        assert "results" in data
        assert len(data["results"]) == 3

        for result in data["results"]:
            assert "index" in result
            assert "relevance_score" in result
            assert isinstance(result["index"], int)
            assert isinstance(result["relevance_score"], float)


# ============================================================================
# TGI Generate API Compliance
# ============================================================================


class TestTGIGenerate:
    """Test /generate and /generate_stream endpoints for HuggingFace TGI compliance."""

    async def test_basic_generate(self, client: AsyncClient):
        """Test basic TGI generate request."""
        resp = await client.post(
            "/generate",
            json={
                "inputs": "Once upon a time",
                "parameters": {"max_new_tokens": 20},
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # TGI format
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)

    async def test_generate_stream(self, client: AsyncClient):
        """Test TGI streaming generate request."""
        resp = await client.post(
            "/generate_stream",
            json={
                "inputs": "Hello",
                "parameters": {"max_new_tokens": 10},
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                import orjson

                chunk = orjson.loads(data_str)
                chunks.append(chunk)

        assert len(chunks) > 0

        # Validate chunk structure
        for chunk in chunks:
            assert "index" in chunk
            assert "token" in chunk
            token = chunk["token"]
            assert "id" in token
            assert "text" in token
            assert "logprob" in token
            assert "special" in token

        # Last chunk should have generated_text
        assert "generated_text" in chunks[-1]


# ============================================================================
# Health & Info Endpoints
# ============================================================================


class TestHealthEndpoints:
    """Test health and info endpoints."""

    async def test_health_endpoint(self, client: AsyncClient):
        """Test /health endpoint."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "config" in data

    async def test_root_endpoint(self, client: AsyncClient):
        """Test / root endpoint."""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "version" in data
        assert "config" in data


# ============================================================================
# Custom Multimodal Endpoint
# ============================================================================


class TestCustomMultimodal:
    """Test /v1/custom-multimodal endpoint."""

    async def test_custom_multimodal(self, client: AsyncClient):
        """Test custom multimodal endpoint."""
        resp = await client.post(
            "/v1/custom-multimodal",
            json={
                "modality_bundle": {
                    "text_fragments": ["Hello", "World"],
                    "visual_assets": {
                        "images": ["base64data1"],
                        "videos": [],
                    },
                    "audio_streams": [],
                },
                "inference_params": {
                    "model_id": "multimodal-model",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Custom format response
        assert "completion" in data
        assert "generated_text" in data["completion"]
        assert "metadata" in data["completion"]
        assert "tokens_used" in data["completion"]["metadata"]
        tokens = data["completion"]["metadata"]["tokens_used"]
        assert "input" in tokens
        assert "output" in tokens
        assert "total" in tokens


# ============================================================================
# Response Format Tests (orjson compliance)
# ============================================================================


class TestResponseFormat:
    """Test that responses are valid JSON and properly formatted."""

    async def test_response_content_type(self, client: AsyncClient):
        """Test that non-streaming responses have correct content type."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert "application/json" in resp.headers["content-type"]

    async def test_streaming_content_type(self, client: AsyncClient):
        """Test that streaming responses have correct content type."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_json_valid_utf8(self, client: AsyncClient):
        """Test that responses handle UTF-8 correctly."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello \u4e2d\u6587 \U0001f600"}
                ],
            },
        )
        assert resp.status_code == 200
        # Should not raise on JSON parsing
        data = resp.json()
        assert "choices" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_empty_messages(self, client: AsyncClient):
        """Test chat completion with empty messages."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )
        # Should still return valid response (mock server behavior)
        assert resp.status_code == 200

    async def test_empty_prompt(self, client: AsyncClient):
        """Test text completion with empty prompt."""
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Empty prompt should result in empty output
        assert data["usage"]["completion_tokens"] == 0

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures for mock server tests."""

from unittest.mock import patch

import pytest
from aiperf_mock_server.app import app
from aiperf_mock_server.config import MockServerConfig, set_server_config
from aiperf_mock_server.dcgm_faker import GPU_CONFIGS, DCGMFaker
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    RankingRequest,
)
from aiperf_mock_server.tokens import TokenizedText, Tokenizer
from fastapi.testclient import TestClient

# ============================================================================
# Auto-use Fixtures (Applied to all tests)
# ============================================================================


@pytest.fixture(autouse=True)
def reset_config():
    """Reset server config before each test to ensure isolation."""
    config = MockServerConfig(error_rate=0.0, random_seed=42)
    set_server_config(config)
    yield
    # Cleanup after test
    config = MockServerConfig()
    set_server_config(config)


# ============================================================================
# Core Component Fixtures
# ============================================================================


@pytest.fixture
def tokenizer():
    """Create a Tokenizer instance."""
    return Tokenizer()


@pytest.fixture
def dcgm_faker():
    """Create a DCGMFaker instance with default settings."""
    return DCGMFaker(gpu_name="h200", num_gpus=2, seed=42, initial_load=0.7)


@pytest.fixture
def gpu_config():
    """Get H200 GPU configuration."""
    return GPU_CONFIGS["h200"]


@pytest.fixture
def test_client():
    """Create FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_server_config():
    """Mock server config with error_rate=0."""
    with patch("aiperf_mock_server.utils.server_config") as config:
        config.error_rate = 0.0
        config.ttft = 20.0
        config.itl = 5.0
        yield config


# ============================================================================
# Request Fixtures
# ============================================================================


@pytest.fixture
def sample_completion_request():
    """Create a sample text completion request."""
    return CompletionRequest(
        model="test-model",
        prompt="Hello world",
        max_tokens=20,
    )


@pytest.fixture
def sample_chat_request():
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="Hello, how are you?")],
        max_completion_tokens=50,
    )


@pytest.fixture
def sample_chat_request_with_reasoning():
    """Create a chat request with reasoning enabled."""
    return ChatCompletionRequest(
        model="gpt-oss-120b",
        messages=[Message(role="user", content="Solve this problem")],
        reasoning_effort="high",
        max_completion_tokens=100,
    )


@pytest.fixture
def sample_embedding_request():
    """Create a sample embedding request."""
    return EmbeddingRequest(
        model="test-embed",
        input=["test text 1", "test text 2"],
    )


@pytest.fixture
def sample_ranking_request():
    """Create a sample ranking request."""
    return RankingRequest(
        model="test-rank",
        query={"text": "machine learning algorithms"},
        passages=[
            {"text": "Machine learning is a subset of artificial intelligence"},
            {"text": "Deep learning uses neural networks"},
            {"text": "Natural language processing handles text"},
        ],
    )


# ============================================================================
# Response/Data Fixtures
# ============================================================================


@pytest.fixture
def sample_tokenized_text():
    """Create sample tokenized text."""
    return TokenizedText(
        text="Hello world",
        tokens=["Hello", " ", "world"],
        prompt_token_count=3,
        finish_reason="stop",
    )


@pytest.fixture
def sample_tokenized_text_with_reasoning():
    """Create tokenized text with reasoning tokens."""
    return TokenizedText(
        text="Answer to problem",
        tokens=["Answer", " to", " problem"],
        prompt_token_count=10,
        reasoning_tokens=50,
        reasoning_content_tokens=["Let", " me", " think", "..."],
        finish_reason="stop",
    )


# ============================================================================
# Parametrize Helpers
# ============================================================================


@pytest.fixture(params=["low", "medium", "high"])
def reasoning_effort(request):
    """Parametrize reasoning effort levels."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def gpu_count(request):
    """Parametrize GPU counts."""
    return request.param


@pytest.fixture(params=["rtx6000", "a100", "h100", "h100-sxm", "h200", "b200", "gb200"])
def gpu_model(request):
    """Parametrize GPU models."""
    return request.param

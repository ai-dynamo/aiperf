#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.clients.openai.openai_aiohttp import OpenAIClientAioHttp
from aiperf.clients.openai.openai_convert import (
    OpenAIChatCompletionRequestConverter,
    OpenAICompletionRequestConverter,
    OpenAIEmbeddingsRequestConverter,
    OpenAIResponsesRequestConverter,
)
from aiperf.common.dataset_models import Text
from aiperf.common.enums import EndpointType
from aiperf.common.record_models import (
    SSEField,
    SSEMessage,
)


# OpenAI Request Converter Fixtures
@pytest.fixture
def openai_chat_converter():
    """OpenAI Chat Completion request converter for testing."""
    return OpenAIChatCompletionRequestConverter()


@pytest.fixture
def openai_completion_converter():
    """OpenAI Completion request converter for testing."""
    return OpenAICompletionRequestConverter()


@pytest.fixture
def openai_embeddings_converter():
    """OpenAI Embeddings request converter for testing."""
    return OpenAIEmbeddingsRequestConverter()


@pytest.fixture
def openai_responses_converter():
    """OpenAI Responses request converter for testing."""
    return OpenAIResponsesRequestConverter()


# OpenAI Client Fixtures
@pytest.fixture
def openai_client(sample_model_endpoint_info):
    """OpenAI aiohttp client for testing."""
    return OpenAIClientAioHttp(sample_model_endpoint_info)


@pytest.fixture
def openai_client_streaming(sample_model_endpoint_info):
    """OpenAI aiohttp client with streaming enabled for testing."""
    streaming_endpoint = sample_model_endpoint_info.model_copy(deep=True)
    streaming_endpoint.endpoint.streaming = True
    return OpenAIClientAioHttp(streaming_endpoint)


# Endpoint-specific Model Endpoint Info Fixtures
@pytest.fixture
def chat_completions_endpoint_info():
    """ModelEndpointInfo for OpenAI Chat Completions endpoint."""
    return ModelEndpointInfo.from_user_config(
        Mock(
            model_names=["gpt-4"],
            endpoint=Mock(
                type=EndpointType.OPENAI_CHAT_COMPLETIONS,
                url="https://api.openai.com",
                custom=None,
                streaming=False,
                timeout=30.0,
                api_key="test-api-key",
                model_selection_strategy="round_robin",
            ),
            input=Mock(
                headers={"Custom-Header": "test-value"}, extra={"temperature": 0.7}
            ),
        )
    )


@pytest.fixture
def completions_endpoint_info():
    """ModelEndpointInfo for OpenAI Completions endpoint."""
    return ModelEndpointInfo.from_user_config(
        Mock(
            model_names=["gpt-3.5-turbo-instruct"],
            endpoint=Mock(
                type=EndpointType.OPENAI_COMPLETIONS,
                url="https://api.openai.com",
                custom=None,
                streaming=False,
                timeout=30.0,
                api_key="test-api-key",
                model_selection_strategy="round_robin",
            ),
            input=Mock(
                headers={"Custom-Header": "test-value"}, extra={"max_tokens": 100}
            ),
        )
    )


@pytest.fixture
def embeddings_endpoint_info():
    """ModelEndpointInfo for OpenAI Embeddings endpoint."""
    return ModelEndpointInfo.from_user_config(
        Mock(
            model_names=["text-embedding-ada-002"],
            endpoint=Mock(
                type=EndpointType.OPENAI_EMBEDDINGS,
                url="https://api.openai.com",
                custom=None,
                streaming=False,
                timeout=30.0,
                api_key="test-api-key",
                model_selection_strategy="round_robin",
                url_params={"dimensions": 1536, "encoding_format": "float"},
            ),
            input=Mock(headers={"Custom-Header": "test-value"}, extra={}),
        )
    )


@pytest.fixture
def responses_endpoint_info():
    """ModelEndpointInfo for OpenAI Responses endpoint."""
    return ModelEndpointInfo.from_user_config(
        Mock(
            model_names=["gpt-4"],
            endpoint=Mock(
                type=EndpointType.OPENAI_RESPONSES,
                url="https://api.openai.com",
                custom=None,
                streaming=False,
                timeout=30.0,
                api_key="test-api-key",
                model_selection_strategy="round_robin",
                url_params={"max_output_tokens": 1000},
            ),
            input=Mock(headers={"Custom-Header": "test-value"}, extra={}),
        )
    )


# OpenAI Response Fixtures
@pytest.fixture
def openai_chat_response():
    """Sample OpenAI Chat Completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18},
    }


@pytest.fixture
def openai_completion_response():
    """Sample OpenAI Completion response."""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "text": "Hello! How can I help you today?",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
    }


@pytest.fixture
def openai_embeddings_response():
    """Sample OpenAI Embeddings response."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def openai_responses_response():
    """Sample OpenAI Responses response."""
    return {
        "id": "resp-123",
        "object": "response",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"content": "Hello! How can I help you today?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
    }


# OpenAI Streaming Response Fixtures
@pytest.fixture
def openai_chat_streaming_chunks():
    """Sample OpenAI Chat Completion streaming chunks."""
    return [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "! How can I help you today?"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


@pytest.fixture
def openai_streaming_sse_messages(openai_chat_streaming_chunks):
    """Sample OpenAI streaming SSE messages."""
    messages = []
    for chunk in openai_chat_streaming_chunks:
        messages.append(
            SSEMessage(
                perf_ns=1677652288000000000,
                packets=[SSEField(name="data", value=json.dumps(chunk))],
            )
        )
    # Add [DONE] message
    messages.append(
        SSEMessage(
            perf_ns=1677652288000000000, packets=[SSEField(name="data", value="[DONE]")]
        )
    )
    return messages


# Error Response Fixtures
@pytest.fixture
def openai_error_response():
    """Sample OpenAI error response."""
    return {
        "error": {
            "message": "Invalid API key provided",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }


@pytest.fixture
def openai_rate_limit_error():
    """Sample OpenAI rate limit error response."""
    return {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "param": None,
            "code": "rate_limit_exceeded",
        }
    }


# Mock HTTP Response Fixtures
@pytest.fixture
def mock_successful_response(openai_chat_response):
    """Mock successful HTTP response."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content_type = "application/json"
    mock_response.text = AsyncMock(return_value=json.dumps(openai_chat_response))
    mock_response.reason = "OK"
    return mock_response


@pytest.fixture
def mock_error_response(openai_error_response):
    """Mock error HTTP response."""
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.content_type = "application/json"
    mock_response.text = AsyncMock(return_value=json.dumps(openai_error_response))
    mock_response.reason = "Unauthorized"
    return mock_response


@pytest.fixture
def mock_rate_limit_response(openai_rate_limit_error):
    """Mock rate limit HTTP response."""
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.content_type = "application/json"
    mock_response.text = AsyncMock(return_value=json.dumps(openai_rate_limit_error))
    mock_response.reason = "Too Many Requests"
    return mock_response


@pytest.fixture
def mock_timeout_response():
    """Mock timeout HTTP response."""
    mock_response = AsyncMock()
    mock_response.status = 408
    mock_response.content_type = "application/json"
    mock_response.text = AsyncMock(
        return_value='{"error": {"message": "Request timeout"}}'
    )
    mock_response.reason = "Request Timeout"
    return mock_response


# Mock aiohttp Session with OpenAI Responses
@pytest.fixture
def mock_openai_session(mock_successful_response):
    """Mock aiohttp session with OpenAI responses."""
    mock_session = AsyncMock()
    mock_session.post.return_value.__aenter__.return_value = mock_successful_response
    return mock_session


@pytest.fixture
def mock_openai_streaming_session(openai_streaming_sse_messages):
    """Mock aiohttp session with OpenAI streaming responses."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content_type = "text/event-stream"

    # Mock the SSE stream reader
    with patch(
        "aiperf.clients.openai.openai_aiohttp.AioHttpSSEStreamReader"
    ) as mock_reader:
        mock_reader.return_value.read_complete_stream = AsyncMock(
            return_value=openai_streaming_sse_messages
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response
        yield mock_session


# Expected Payload Fixtures
@pytest.fixture
def expected_chat_payload():
    """Expected payload for chat completions."""
    return {
        "messages": [
            {"role": "user", "name": "text", "content": "Hello, how are you?"}
        ],
        "model": "gpt-4",
        "stream": False,
        "temperature": 0.7,
    }


@pytest.fixture
def expected_completion_payload():
    """Expected payload for completions."""
    return {
        "prompt": [Text(name="text", role="user", content=["Hello, how are you?"])],
        "model": "gpt-4",
        "stream": False,
        "temperature": 0.7,
    }


@pytest.fixture
def expected_embeddings_payload():
    """Expected payload for embeddings."""
    return {
        "input": [Text(name="text", role="user", content=["Hello, how are you?"])],
        "model": "gpt-4",
        "dimensions": 1536,
        "encoding_format": "float",
        "user": "",
        "stream": False,
        "temperature": 0.7,
    }


@pytest.fixture
def expected_responses_payload():
    """Expected payload for responses."""
    return {
        "input": [Text(name="text", role="user", content=["Hello, how are you?"])],
        "model": "gpt-4",
        "max_output_tokens": 1000,
        "stream": False,
        "temperature": 0.7,
    }


# URL and Header Fixtures
@pytest.fixture
def expected_openai_headers():
    """Expected headers for OpenAI requests."""
    return {
        "User-Agent": "aiperf/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test-api-key",
        "Custom-Header": "test-value",
    }


@pytest.fixture
def expected_openai_streaming_headers():
    """Expected headers for OpenAI streaming requests."""
    return {
        "User-Agent": "aiperf/1.0",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": "Bearer test-api-key",
        "Custom-Header": "test-value",
    }


@pytest.fixture
def expected_openai_urls():
    """Expected URLs for different OpenAI endpoints."""
    return {
        EndpointType.OPENAI_CHAT_COMPLETIONS: "https://api.openai.com/v1/chat/completions",
        EndpointType.OPENAI_COMPLETIONS: "https://api.openai.com/v1/completions",
        EndpointType.OPENAI_EMBEDDINGS: "https://api.openai.com/v1/embeddings",
        EndpointType.OPENAI_RESPONSES: "https://api.openai.com/v1/responses",
    }


# Test Data Combinations
@pytest.fixture
def openai_endpoint_converter_combinations():
    """Combinations of endpoint types and their corresponding converters."""
    return [
        (EndpointType.OPENAI_CHAT_COMPLETIONS, OpenAIChatCompletionRequestConverter),
        (EndpointType.OPENAI_COMPLETIONS, OpenAICompletionRequestConverter),
        (EndpointType.OPENAI_EMBEDDINGS, OpenAIEmbeddingsRequestConverter),
        (EndpointType.OPENAI_RESPONSES, OpenAIResponsesRequestConverter),
    ]

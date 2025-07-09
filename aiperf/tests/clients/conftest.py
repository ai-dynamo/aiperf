#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import time
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.config.config_defaults import EndPointDefaults
from aiperf.common.config.endpoint.endpoint_config import EndPointConfig
from aiperf.common.config.input.input_config import InputConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.dataset_models import Audio, Conversation, Image, Text, Turn
from aiperf.common.enums import EndpointType, Modality, ModelSelectionStrategy
from aiperf.common.record_models import (
    ErrorDetails,
    RequestRecord,
    ResponseData,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.tokenizer import Tokenizer


# Model Info Fixtures
@pytest.fixture
def sample_model_info():
    """Sample ModelInfo for testing."""
    return ModelInfo(name="gpt-4", version="2024-01-01", modality=Modality.TEXT)


@pytest.fixture
def sample_model_list_info(sample_model_info):
    """Sample ModelListInfo for testing."""
    return ModelListInfo(
        models=[sample_model_info],
        model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
    )


@pytest.fixture
def sample_endpoint_info():
    """Sample EndpointInfo for testing."""
    return EndpointInfo(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS,
        base_url="https://api.openai.com",
        streaming=False,
        api_key="test-api-key",
        timeout=EndPointDefaults.TIMEOUT,
        headers={"Custom-Header": "test-value"},
        extra={"temperature": 0.7},
    )


@pytest.fixture
def sample_model_endpoint_info(sample_model_list_info, sample_endpoint_info):
    """Sample ModelEndpointInfo for testing."""
    return ModelEndpointInfo(
        models=sample_model_list_info, endpoint=sample_endpoint_info
    )


# User Config Fixtures
@pytest.fixture
def sample_user_config():
    """Sample UserConfig for testing."""
    return UserConfig(
        model_names=["gpt-4"],
        endpoint=EndPointConfig(
            type="chat",
            url="https://api.openai.com",
            api_key="test-api-key",
            streaming=False,
            timeout=30.0,
            model_selection_strategy="round_robin",
        ),
        input=InputConfig(
            headers={"Custom-Header": "test-value"}, extra={"temperature": 0.7}
        ),
    )


# Dataset Fixtures
@pytest.fixture
def sample_text():
    """Sample Text for testing."""
    return Text(name="text", role="user", content=["Hello, how are you?"])


@pytest.fixture
def sample_image():
    """Sample Image for testing."""
    return Image(name="image_url", content=["https://example.com/image.jpg"])


@pytest.fixture
def sample_audio():
    """Sample Audio for testing."""
    return Audio(
        name="input_audio",
        content=["data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEA..."],
    )


@pytest.fixture
def sample_turn(sample_text):
    """Sample Turn for testing."""
    return Turn(timestamp=int(time.time() * 1000), delay=100, text=[sample_text])


@pytest.fixture
def sample_multimodal_turn(sample_text, sample_image):
    """Sample Turn with multimodal content for testing."""
    return Turn(
        timestamp=int(time.time() * 1000),
        delay=100,
        text=[sample_text],
        image=[sample_image],
    )


@pytest.fixture
def sample_conversation(sample_turn):
    """Sample Conversation for testing."""
    return Conversation(session_id="test-session-123", turns=[sample_turn])


# Record Fixtures
@pytest.fixture
def sample_request_record():
    """Sample RequestRecord for testing."""
    return RequestRecord(
        request={"messages": [{"role": "user", "content": "Hello"}]},
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns() + 1000000,
        status=200,
        responses=[
            TextResponse(
                perf_ns=time.perf_counter_ns() + 500000,
                content_type="application/json",
                text='{"choices": [{"message": {"content": "Hello! How can I help you?"}}]}',
            )
        ],
    )


@pytest.fixture
def sample_error_record():
    """Sample RequestRecord with error for testing."""
    return RequestRecord(
        request={"messages": [{"role": "user", "content": "Hello"}]},
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns() + 1000000,
        status=500,
        error=ErrorDetails(
            code=500, type="InternalServerError", message="Internal server error"
        ),
    )


@pytest.fixture
def sample_streaming_record():
    """Sample RequestRecord with streaming responses for testing."""
    return RequestRecord(
        request={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns() + 2000000,
        status=200,
        responses=[
            SSEMessage(
                perf_ns=time.perf_counter_ns() + 500000,
                packets=[
                    SSEField(
                        name="data",
                        value='{"choices": [{"delta": {"content": "Hello"}}]}',
                    )
                ],
            ),
            SSEMessage(
                perf_ns=time.perf_counter_ns() + 1000000,
                packets=[
                    SSEField(
                        name="data",
                        value='{"choices": [{"delta": {"content": " there!"}}]}',
                    )
                ],
            ),
            SSEMessage(
                perf_ns=time.perf_counter_ns() + 1500000,
                packets=[SSEField(name="data", value="[DONE]")],
            ),
        ],
    )


@pytest.fixture
def sample_response_data():
    """Sample ResponseData for testing."""
    return ResponseData(
        perf_ns=time.perf_counter_ns(),
        raw_text=["Hello! How can I help you?"],
        parsed_text=["Hello! How can I help you?"],
        token_count=6,
        metadata={
            "model": "gpt-4",
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        },
    )


# Mock Fixtures
@pytest.fixture
def mock_tokenizer():
    """Mock Tokenizer for testing."""
    mock = Mock(spec=Tokenizer)
    mock.encode = Mock(return_value=[1, 2, 3, 4])
    mock.decode = Mock(return_value="Hello world")
    mock.count_tokens = Mock(return_value=4)
    return mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content_type = "application/json"
    mock_response.text = AsyncMock(
        return_value='{"choices": [{"message": {"content": "Hello!"}}]}'
    )
    mock_session.post.return_value.__aenter__.return_value = mock_response
    return mock_session


@pytest.fixture
def mock_streaming_response():
    """Mock streaming response for testing."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.content_type = "text/event-stream"
    mock_response.content.at_eof.side_effect = [False, False, False, True]
    mock_response.content.read.side_effect = [
        b"d",
        b"e",
        b"d",
    ]
    mock_response.content.readuntil.side_effect = [
        b'ata: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'ata: {"choices": [{"delta": {"content": " there!"}}]}\n\n',
        b"ata: [DONE]\n\n",
    ]
    return mock_response


# Endpoint Type Fixtures
@pytest.fixture(
    params=[
        EndpointType.OPENAI_CHAT_COMPLETIONS,
        EndpointType.OPENAI_COMPLETIONS,
        EndpointType.OPENAI_EMBEDDINGS,
        EndpointType.OPENAI_RESPONSES,
    ]
)
def endpoint_type(request):
    """Parametrized endpoint type fixture for testing multiple endpoint types."""
    return request.param


@pytest.fixture
def all_endpoint_types():
    """All supported OpenAI endpoint types for testing."""
    return [
        EndpointType.OPENAI_CHAT_COMPLETIONS,
        EndpointType.OPENAI_COMPLETIONS,
        EndpointType.OPENAI_EMBEDDINGS,
        EndpointType.OPENAI_RESPONSES,
    ]


# Validation Fixtures
@pytest.fixture
def valid_payload_data():
    """Valid payload data for different endpoint types."""
    return {
        EndpointType.OPENAI_CHAT_COMPLETIONS: {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": False,
        },
        EndpointType.OPENAI_COMPLETIONS: {
            "prompt": "Hello",
            "model": "gpt-4",
            "stream": False,
        },
        EndpointType.OPENAI_EMBEDDINGS: {
            "input": "Hello",
            "model": "text-embedding-ada-002",
            "dimensions": 1536,
            "encoding_format": "float",
            "user": "",
            "stream": False,
        },
        EndpointType.OPENAI_RESPONSES: {
            "input": "Hello",
            "model": "gpt-4",
            "max_output_tokens": 1000,
            "stream": False,
        },
    }


@pytest.fixture
def invalid_payload_data():
    """Invalid payload data for testing error scenarios."""
    return {
        "missing_required_field": {},
        "invalid_model": {"model": "", "messages": []},
        "invalid_messages": {"model": "gpt-4", "messages": "not a list"},
        "invalid_stream": {"model": "gpt-4", "messages": [], "stream": "not boolean"},
    }

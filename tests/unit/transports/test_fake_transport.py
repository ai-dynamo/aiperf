# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FakeTransport."""

import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import RequestInfo, RequestRecord, SSEMessage, TextResponse
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType
from tests.harness.fake_transport import FakeTransport as FakeTransport


def create_model_endpoint(
    endpoint_type: EndpointType = EndpointType.CHAT,
    base_url: str = "mock://localhost:8000",
    streaming: bool = True,
    model_name: str = "test-model",
) -> ModelEndpointInfo:
    """Create a ModelEndpointInfo for testing."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_url=base_url,
            streaming=streaming,
        ),
    )


def create_request_info(
    model_endpoint: ModelEndpointInfo,
    x_request_id: str = "test-request-id",
) -> RequestInfo:
    """Create a RequestInfo for testing."""
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=x_request_id,
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation-id",
    )


class TestFakeTransportInit:
    """Test FakeTransport initialization."""

    def test_init_with_custom_config(self):
        """Test transport uses provided config."""
        model_endpoint = create_model_endpoint()
        custom_config = MockServerConfig(ttft=50, itl=10)
        transport = FakeTransport(model_endpoint=model_endpoint, config=custom_config)
        assert transport.config is custom_config
        assert transport.config.ttft == 50
        assert transport.config.itl == 10


class TestFakeTransportChat:
    """Test FakeTransport chat completion handlers."""

    @pytest.fixture
    def fast_config(self):
        """Config with minimal latency for fast tests."""
        return MockServerConfig(ttft=1, itl=1)

    @pytest.fixture
    def streaming_endpoint(self):
        """Create streaming chat endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.CHAT,
            streaming=True,
        )

    @pytest.fixture
    def non_streaming_endpoint(self):
        """Create non-streaming chat endpoint."""
        return create_model_endpoint(
            endpoint_type=EndpointType.CHAT,
            streaming=False,
        )

    @pytest.mark.asyncio
    async def test_streaming_chat_returns_sse_messages(
        self, streaming_endpoint, fast_config
    ):
        """Test streaming chat returns SSEMessage responses."""
        transport = FakeTransport(model_endpoint=streaming_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.status == 200
        assert len(record.responses) > 0
        # Streaming returns SSEMessage
        assert all(isinstance(r, SSEMessage) for r in record.responses)

    @pytest.mark.asyncio
    async def test_non_streaming_chat_returns_text_response(
        self, non_streaming_endpoint, fast_config
    ):
        """Test non-streaming chat returns TextResponse."""
        transport = FakeTransport(
            model_endpoint=non_streaming_endpoint, config=fast_config
        )
        await transport.initialize()

        request_info = create_request_info(non_streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        record = await transport.send_request(request_info, payload)

        assert isinstance(record, RequestRecord)
        assert record.status == 200
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)
        assert "chat.completion" in record.responses[0].text

    @pytest.mark.asyncio
    async def test_streaming_chat_with_first_token_callback(
        self, streaming_endpoint, fast_config
    ):
        """Test FirstTokenCallback is fired during streaming."""
        transport = FakeTransport(model_endpoint=streaming_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(streaming_endpoint)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        callback_fired = False
        callback_ttft_ns = None

        async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal callback_fired, callback_ttft_ns
            callback_fired = True
            callback_ttft_ns = ttft_ns
            return True  # Mark as meaningful

        record = await transport.send_request(
            request_info, payload, first_token_callback=first_token_callback
        )

        assert callback_fired
        assert callback_ttft_ns is not None
        assert callback_ttft_ns > 0
        assert record.status == 200


class TestFakeTransportEmbedding:
    """Test FakeTransport embedding handlers."""

    @pytest.mark.asyncio
    async def test_embedding_multiple_inputs(self):
        """Test embedding with multiple inputs."""
        fast_config = MockServerConfig(
            ttft=1, itl=1, embedding_base_latency=1, embedding_per_input_latency=0
        )
        embedding_endpoint = create_model_endpoint(
            endpoint_type=EndpointType.EMBEDDINGS
        )
        transport = FakeTransport(model_endpoint=embedding_endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(embedding_endpoint)
        payload = {
            "model": "test-model",
            "input": ["Hello", "World", "Test"],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        # Response should contain all 3 embeddings
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert len(response_data["data"]) == 3


class TestFakeTransportRanking:
    """Test FakeTransport ranking handlers."""

    @pytest.mark.asyncio
    async def test_nim_ranking(self):
        """Test NIM ranking endpoint."""
        fast_config = MockServerConfig(
            ttft=1, itl=1, ranking_base_latency=1, ranking_per_passage_latency=0
        )
        endpoint = create_model_endpoint(endpoint_type=EndpointType.NIM_RANKINGS)
        transport = FakeTransport(model_endpoint=endpoint, config=fast_config)
        await transport.initialize()

        request_info = create_request_info(endpoint)
        payload = {
            "model": "test-model",
            "query": {"text": "What is AI?"},
            "passages": [
                {"text": "AI is artificial intelligence"},
                {"text": "Machine learning is a subset"},
            ],
        }

        record = await transport.send_request(request_info, payload)

        assert record.status == 200
        import orjson

        response_data = orjson.loads(record.responses[0].text)
        assert "rankings" in response_data

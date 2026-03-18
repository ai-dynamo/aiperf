# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests ensuring API keys are NOT redacted on the wire path.

Redaction must only happen at storage/serialization points, never on data
being sent to inference servers. These tests verify the real API key flows
through every layer that constructs or passes headers to the HTTP client.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models.dataset_models import Text
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord, Turn
from aiperf.common.redact import REDACTED_VALUE
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.plugin.enums import TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.aiohttp_transport import AioHttpTransport
from aiperf.transports.base_transports import BaseTransport
from aiperf.workers.inference_client import InferenceClient

API_KEY = "sk-real-secret-key-12345"
BEARER = f"Bearer {API_KEY}"


class _StubEndpoint(BaseEndpoint):
    """Minimal endpoint stub for testing header construction."""

    def format_payload(self, ri):
        return {}

    def parse_response(self, resp):
        return None


@pytest.fixture
def model_endpoint_with_key():
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            base_urls=["http://localhost:8000"],
            api_key=API_KEY,
        ),
    )


@pytest.fixture
def request_info(model_endpoint_with_key):
    return RequestInfo(
        model_endpoint=model_endpoint_with_key,
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-1",
        x_correlation_id="corr-1",
        conversation_id="conv-1",
    )


# =============================================================================
# EndpointInfo: api_key accessible at runtime
# =============================================================================


class TestEndpointInfoApiKeyAccessible:
    """EndpointInfo.api_key must remain accessible even though it's excluded
    from serialization. Code paths that construct headers read it directly."""

    def test_api_key_attribute_returns_real_value(self, model_endpoint_with_key):
        assert model_endpoint_with_key.endpoint.api_key == API_KEY

    def test_api_key_survives_model_copy(self, model_endpoint_with_key):
        copy = model_endpoint_with_key.model_copy(deep=True)
        assert copy.endpoint.api_key == API_KEY


# =============================================================================
# BaseEndpoint.get_endpoint_headers: real key in Authorization header
# =============================================================================


class TestGetEndpointHeadersNotRedacted:
    """BaseEndpoint.get_endpoint_headers must produce real Authorization values."""

    def test_authorization_header_contains_real_key(
        self, model_endpoint_with_key, request_info
    ):
        endpoint = _StubEndpoint(model_endpoint=model_endpoint_with_key)
        headers = endpoint.get_endpoint_headers(request_info)
        assert headers["Authorization"] == BEARER

    def test_no_redaction_marker_in_headers(
        self, model_endpoint_with_key, request_info
    ):
        endpoint = _StubEndpoint(model_endpoint=model_endpoint_with_key)
        headers = endpoint.get_endpoint_headers(request_info)
        assert REDACTED_VALUE not in headers.get("Authorization", "")

    def test_custom_headers_combined_with_auth(self, request_info):
        """User-supplied --header values must also pass through unredacted."""
        request_info.model_endpoint.endpoint.headers = [("X-Custom", "my-value")]
        endpoint = _StubEndpoint(model_endpoint=request_info.model_endpoint)
        headers = endpoint.get_endpoint_headers(request_info)
        assert headers["Authorization"] == BEARER
        assert headers["X-Custom"] == "my-value"


# =============================================================================
# BaseTransport.build_headers: real key flows into composed headers
# =============================================================================


class TestBuildHeadersNotRedacted:
    """BaseTransport.build_headers must include the real Authorization
    value so the HTTP client sends it to the server."""

    def test_build_headers_preserves_real_auth(
        self, model_endpoint_with_key, request_info
    ):
        class FakeTransport(BaseTransport):
            @classmethod
            def metadata(cls):
                return TransportMetadata(
                    transport_type=TransportType.HTTP,
                    url_schemes=["http", "https"],
                )

            def get_url(self, ri):
                return "http://localhost:8000"

            async def send_request(self, ri, payload, **kw):
                return RequestRecord()

        transport = FakeTransport(model_endpoint=model_endpoint_with_key)
        request_info.endpoint_headers = {"Authorization": BEARER}
        headers = transport.build_headers(request_info)

        assert headers["Authorization"] == BEARER
        assert REDACTED_VALUE not in str(headers)


# =============================================================================
# AioHttpTransport.send_request: real headers reach aiohttp_client
# =============================================================================


class TestAioHttpTransportWireHeaders:
    """The headers passed to aiohttp_client.post_request must be unredacted.
    Only the stored record.request_headers should be redacted."""

    @pytest.mark.asyncio
    async def test_real_headers_sent_to_aiohttp_client(
        self, model_endpoint_with_key, request_info
    ):
        transport = AioHttpTransport(model_endpoint=model_endpoint_with_key)
        request_info.endpoint_headers = {"Authorization": BEARER}

        mock_record = RequestRecord()
        mock_client = AsyncMock()
        mock_client.post_request = AsyncMock(return_value=mock_record)
        transport.aiohttp_client = mock_client

        await transport.send_request(request_info, payload={"model": "test"})

        call_args = mock_client.post_request.call_args
        sent_headers = call_args[0][2]  # 3rd positional arg is headers
        assert sent_headers["Authorization"] == BEARER
        assert REDACTED_VALUE not in str(sent_headers)

    @pytest.mark.asyncio
    async def test_stored_record_headers_are_redacted(
        self, model_endpoint_with_key, request_info
    ):
        transport = AioHttpTransport(model_endpoint=model_endpoint_with_key)
        request_info.endpoint_headers = {"Authorization": BEARER}

        mock_record = RequestRecord()
        mock_client = AsyncMock()
        mock_client.post_request = AsyncMock(return_value=mock_record)
        transport.aiohttp_client = mock_client

        record = await transport.send_request(request_info, payload={"model": "test"})
        assert record.request_headers["Authorization"] == REDACTED_VALUE


# =============================================================================
# InferenceClient: real headers flow to transport, redacted in record
# =============================================================================


class TestInferenceClientWireHeaders:
    """InferenceClient must pass real headers to the transport layer
    and only redact when enriching the stored RequestRecord."""

    @pytest.mark.asyncio
    async def test_transport_receives_real_headers(
        self, model_endpoint_with_key, request_info
    ):
        client = InferenceClient(
            model_endpoint=model_endpoint_with_key, service_id="test"
        )
        await client.initialize()

        captured_headers = None

        async def fake_send(ri, payload, **kw):
            nonlocal captured_headers
            captured_headers = dict(ri.endpoint_headers)
            return RequestRecord()

        client.transport = MagicMock()
        client.transport.send_request = fake_send

        request_info.turns = [Turn(texts=[Text(contents=["hello"])])]
        await client.send_request(request_info)

        assert captured_headers is not None
        assert captured_headers["Authorization"] == BEARER

    @pytest.mark.asyncio
    async def test_record_headers_redacted_after_enrichment(
        self, model_endpoint_with_key, request_info
    ):
        client = InferenceClient(
            model_endpoint=model_endpoint_with_key, service_id="test"
        )
        await client.initialize()

        async def fake_send(ri, payload, **kw):
            return RequestRecord()

        client.transport = MagicMock()
        client.transport.send_request = fake_send

        request_info.turns = [Turn(texts=[Text(contents=["hello"])])]
        record = await client.send_request(request_info)

        assert record.request_headers is not None
        assert record.request_headers["Authorization"] == REDACTED_VALUE

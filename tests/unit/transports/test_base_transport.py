# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata as importlib_metadata

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport

AIPERF_USER_AGENT = f"aiperf/{importlib_metadata.version('aiperf')}"


def _ensure_scheme(url: str) -> str:
    """Ensure URL has a scheme prefix."""
    if not url.startswith(("http://", "https://")):
        return f"http://{url}"
    return url


class FakeTransport(BaseTransport):
    """Concrete implementation of BaseTransport for testing."""

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )

    def get_url(self, request_info: RequestInfo) -> str:
        endpoint_info = request_info.model_endpoint.endpoint
        base_url = (
            _ensure_scheme(endpoint_info.base_url) if endpoint_info.base_url else ""
        )
        if endpoint_info.custom_endpoint:
            return f"{base_url}{endpoint_info.custom_endpoint}"
        return base_url

    async def send_request(
        self, request_info: RequestInfo, payload: dict
    ) -> RequestRecord:
        return RequestRecord()


class TestBaseTransport:
    """Comprehensive tests for BaseTransport functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=["http://localhost:8000"],
                custom_endpoint="/v1/chat/completions",
            ),
        )

    @pytest.fixture
    def transport(self, model_endpoint):
        """Create a FakeTransport instance."""
        return FakeTransport(model_endpoint=model_endpoint)

    @pytest.fixture
    def request_info(self, model_endpoint):
        """Create a basic RequestInfo."""
        return RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={},
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-request-id",
            x_correlation_id="test-correlation-id",
            conversation_id="test-conversation-id",
        )

    def test_metadata(self, transport):
        """Test metadata method returns correct information."""
        metadata = plugins.get_transport_metadata(TransportType.HTTP)
        assert isinstance(metadata, TransportMetadata)
        assert metadata.transport_type == TransportType.HTTP
        assert "http" in metadata.url_schemes
        assert "https" in metadata.url_schemes

    def test_get_transport_headers_default(self, transport, request_info):
        """Test default get_transport_headers returns empty dict."""
        headers = transport.get_transport_headers(request_info)
        assert headers == {}

    @pytest.mark.parametrize(
        "x_request_id,x_correlation_id,expected_headers",
        [
            (None, None, {"User-Agent": AIPERF_USER_AGENT}),
            (
                "req-123456",
                None,
                {"User-Agent": AIPERF_USER_AGENT, "X-Request-ID": "req-123456"},
            ),
            (
                None,
                "corr-789",
                {"User-Agent": AIPERF_USER_AGENT, "X-Correlation-ID": "corr-789"},
            ),
            (
                "req-123",
                "corr-456",
                {
                    "User-Agent": AIPERF_USER_AGENT,
                    "X-Request-ID": "req-123",
                    "X-Correlation-ID": "corr-456",
                },
            ),
        ],
    )
    def test_build_headers_universal_headers(
        self, transport, request_info, x_request_id, x_correlation_id, expected_headers
    ):
        """Test that build_headers includes universal headers."""
        request_info.x_request_id = x_request_id
        request_info.x_correlation_id = x_correlation_id
        headers = transport.build_headers(request_info)

        for key, value in expected_headers.items():
            assert headers[key] == value

    def test_build_headers_merges_endpoint_headers(self, transport, request_info):
        """Test that endpoint headers are merged into final headers."""
        request_info.endpoint_headers = {
            "Authorization": "Bearer token123",
            "Custom-Header": "custom-value",
        }
        headers = transport.build_headers(request_info)
        assert headers["Authorization"] == "Bearer token123"
        assert headers["Custom-Header"] == "custom-value"
        assert headers["User-Agent"] == AIPERF_USER_AGENT

    def test_build_headers_transport_headers_override(self, request_info):
        """Test that transport headers can override endpoint headers."""

        class CustomTransport(FakeTransport):
            def get_transport_headers(
                self, request_info: RequestInfo
            ) -> dict[str, str]:
                return {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }

        transport = CustomTransport(model_endpoint=request_info.model_endpoint)
        request_info.endpoint_headers = {"Content-Type": "text/plain"}

        headers = transport.build_headers(request_info)
        # Transport headers should override endpoint headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "text/event-stream"

    def test_build_headers_priority_order(self, request_info):
        """Test header merge priority: universal < endpoint < transport."""

        class CustomTransport(FakeTransport):
            def get_transport_headers(
                self, request_info: RequestInfo
            ) -> dict[str, str]:
                return {"X-Priority": "transport", "Content-Type": "application/json"}

        transport = CustomTransport(model_endpoint=request_info.model_endpoint)
        request_info.endpoint_headers = {
            "X-Priority": "endpoint",
            "Authorization": "Bearer token",
        }

        headers = transport.build_headers(request_info)
        assert headers["User-Agent"] == AIPERF_USER_AGENT  # Universal
        assert headers["Authorization"] == "Bearer token"  # Endpoint
        assert headers["X-Priority"] == "transport"  # Transport wins
        assert headers["Content-Type"] == "application/json"  # Transport

    def test_build_url_simple(self, transport, request_info):
        """Test build_url with no query parameters."""
        request_info.endpoint_params = {}
        url = transport.build_url(request_info)
        assert url == "http://localhost:8000/v1/chat/completions"

    def test_build_url_with_endpoint_params(self, transport, request_info):
        """Test build_url adds endpoint params as query string."""
        request_info.endpoint_params = {"api-version": "2024-10-01", "timeout": "30"}
        url = transport.build_url(request_info)
        assert "api-version=2024-10-01" in url
        assert "timeout=30" in url
        assert url.startswith("http://localhost:8000/v1/chat/completions?")

    def test_build_url_preserves_existing_params(self, transport):
        """Test that existing URL params are preserved."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=["http://localhost:8000/v1/chat/completions?existing=param"],
                custom_endpoint=None,
            ),
        )
        transport = FakeTransport(model_endpoint=model_endpoint)
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={"new": "value"},
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-request-id",
            x_correlation_id="test-correlation-id",
            conversation_id="test-conversation-id",
        )

        url = transport.build_url(request_info)
        assert "existing=param" in url
        assert "new=value" in url

    def test_build_url_endpoint_params_override_existing(self, transport):
        """Test that endpoint params override existing URL params."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=["http://localhost:8000/v1/chat/completions?key=original"],
                custom_endpoint=None,
            ),
        )
        transport = FakeTransport(model_endpoint=model_endpoint)
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={"key": "overridden"},
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-request-id",
            x_correlation_id="test-correlation-id",
            conversation_id="test-conversation-id",
        )

        url = transport.build_url(request_info)
        assert "key=overridden" in url
        assert "key=original" not in url

    def test_build_url_empty_param_value(self, transport, request_info):
        """Test build_url handles empty parameter values."""
        request_info.endpoint_params = {"empty": "", "normal": "value"}
        url = transport.build_url(request_info)
        assert "empty=" in url  # Empty values should be preserved
        assert "normal=value" in url

    def test_build_url_special_characters_encoded(self, transport, request_info):
        """Test that special characters in params are URL encoded."""
        request_info.endpoint_params = {"filter": "name=test&status=active"}
        url = transport.build_url(request_info)
        # Should be URL encoded
        assert "filter=name%3Dtest%26status%3Dactive" in url

    def test_build_url_no_params_preserves_clean_url(self, transport, request_info):
        """Test that URLs without params remain clean (no trailing ?)."""
        request_info.endpoint_params = {}
        url = transport.build_url(request_info)
        assert "?" not in url
        assert url == "http://localhost:8000/v1/chat/completions"

    def test_build_url_complex_query_string(self, transport):
        """Test complex query string handling."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_urls=["http://localhost:8000/api?a=1&b=2&c=3"],
                custom_endpoint=None,
            ),
        )
        transport = FakeTransport(model_endpoint=model_endpoint)
        request_info = RequestInfo(
            model_endpoint=model_endpoint,
            turns=[],
            endpoint_headers={},
            endpoint_params={"d": "4", "b": "overridden"},  # Override 'b'
            turn_index=0,
            credit_num=1,
            credit_phase=CreditPhase.PROFILING,
            x_request_id="test-request-id",
            x_correlation_id="test-correlation-id",
            conversation_id="test-conversation-id",
        )

        url = transport.build_url(request_info)
        assert "a=1" in url
        assert "b=overridden" in url  # Should override
        assert "b=2" not in url
        assert "c=3" in url
        assert "d=4" in url

    @pytest.mark.asyncio
    async def test_send_request_abstract_implemented(self, transport, request_info):
        """Test that send_request can be called on concrete implementation."""
        record = await transport.send_request(request_info, {"test": "payload"})
        assert isinstance(record, RequestRecord)

    def test_get_url_abstract_implemented(self, transport, request_info):
        """Test that get_url can be called on concrete implementation."""
        url = transport.get_url(request_info)
        assert isinstance(url, str)
        assert url == "http://localhost:8000/v1/chat/completions"


class TestTransportMetadata:
    """Tests for TransportMetadata model."""

    def test_transport_metadata_creation(self):
        """Test creating TransportMetadata instance."""
        metadata = TransportMetadata(
            transport_type=TransportType.HTTP, url_schemes=["http", "https"]
        )
        assert metadata.transport_type == TransportType.HTTP
        assert metadata.url_schemes == ["http", "https"]

    def test_transport_metadata_single_scheme(self):
        """Test metadata with single URL scheme."""
        metadata = TransportMetadata(
            transport_type=TransportType.HTTP, url_schemes=["grpc"]
        )
        assert len(metadata.url_schemes) == 1
        assert "grpc" in metadata.url_schemes


class TestBaseTransportAbstractMethods:
    """Test that BaseTransport enforces abstract methods."""

    def test_cannot_instantiate_base_transport(self):
        """Test that BaseTransport cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTransport()

    def test_must_implement_metadata(self):
        """Test that subclasses must implement metadata()."""

        class IncompleteTransport(BaseTransport):
            def get_url(self, request_info: RequestInfo) -> str:
                return ""

            async def send_request(
                self, request_info: RequestInfo, payload: dict
            ) -> RequestRecord:
                return RequestRecord()

        with pytest.raises(TypeError):
            IncompleteTransport()

    def test_must_implement_get_url(self):
        """Test that subclasses must implement get_url()."""

        class IncompleteTransport(BaseTransport):
            @classmethod
            def metadata(cls) -> TransportMetadata:
                return TransportMetadata(
                    transport_type=TransportType.HTTP, url_schemes=["http"]
                )

            async def send_request(
                self, request_info: RequestInfo, payload: dict
            ) -> RequestRecord:
                return RequestRecord()

        with pytest.raises(TypeError):
            IncompleteTransport()

    def test_must_implement_send_request(self):
        """Test that subclasses must implement send_request()."""

        class IncompleteTransport(BaseTransport):
            @classmethod
            def metadata(cls) -> TransportMetadata:
                return TransportMetadata(
                    transport_type=TransportType.HTTP, url_schemes=["http"]
                )

            def get_url(self, request_info: RequestInfo) -> str:
                return ""

        with pytest.raises(TypeError):
            IncompleteTransport()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.plugin import plugins
from aiperf.plugin.enums import TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport

AIPERF_USER_AGENT = f"aiperf/{importlib_metadata.version('aiperf')}"

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000"],
        "path": "/v1/chat/completions",
    },
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def _make_config(**overrides: Any) -> BenchmarkConfig:
    """Create a BenchmarkConfig with minimal defaults."""
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _make_run(**overrides: Any) -> BenchmarkRun:
    """Create a BenchmarkRun wrapping a BenchmarkConfig with minimal defaults."""
    cfg = _make_config(**overrides)
    return BenchmarkRun(benchmark_id="test", cfg=cfg, artifact_dir=Path("/tmp/test"))


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
        ep = self.run.cfg.endpoint
        base_url = _ensure_scheme(ep.urls[0]) if ep.urls else ""
        if ep.path:
            return f"{base_url}{ep.path}"
        return base_url

    async def send_request(
        self, request_info: RequestInfo, payload: dict
    ) -> RequestRecord:
        return RequestRecord()


def _make_request_info(cfg: BenchmarkConfig, **overrides) -> RequestInfo:
    """Create a basic RequestInfo for transport tests."""
    defaults = {
        "turns": [],
        "endpoint_headers": {},
        "endpoint_params": {},
        "turn_index": 0,
        "credit_num": 1,
        "credit_phase": "profiling",
        "x_request_id": "test-request-id",
        "x_correlation_id": "test-correlation-id",
        "conversation_id": "test-conversation-id",
    }
    defaults.update(overrides)
    return RequestInfo(**defaults)


class TestBaseTransport:
    """Comprehensive tests for BaseTransport functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test BenchmarkRun."""
        return _make_run()

    @pytest.fixture
    def transport(self, model_endpoint):
        """Create a FakeTransport instance."""
        return FakeTransport(run=model_endpoint)

    @pytest.fixture
    def request_info(self, model_endpoint):
        """Create a basic RequestInfo."""
        return _make_request_info(model_endpoint.cfg)

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

    def test_build_headers_transport_headers_override(
        self, request_info, model_endpoint
    ):
        """Test that transport headers can override endpoint headers."""

        class CustomTransport(FakeTransport):
            def get_transport_headers(
                self, request_info: RequestInfo
            ) -> dict[str, str]:
                return {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }

        transport = CustomTransport(run=model_endpoint)
        request_info.endpoint_headers = {"Content-Type": "text/plain"}

        headers = transport.build_headers(request_info)
        # Transport headers should override endpoint headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "text/event-stream"

    def test_build_headers_priority_order(self, request_info, model_endpoint):
        """Test header merge priority: universal < endpoint < transport."""

        class CustomTransport(FakeTransport):
            def get_transport_headers(
                self, request_info: RequestInfo
            ) -> dict[str, str]:
                return {"X-Priority": "transport", "Content-Type": "application/json"}

        transport = CustomTransport(run=model_endpoint)
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
        run = _make_run(
            endpoint={
                "type": "chat",
                "urls": ["http://localhost:8000/v1/chat/completions?existing=param"],
            }
        )
        transport = FakeTransport(run=run)
        request_info = _make_request_info(run.cfg, endpoint_params={"new": "value"})

        url = transport.build_url(request_info)
        assert "existing=param" in url
        assert "new=value" in url

    def test_build_url_endpoint_params_override_existing(self, transport):
        """Test that endpoint params override existing URL params."""
        run = _make_run(
            endpoint={
                "type": "chat",
                "urls": ["http://localhost:8000/v1/chat/completions?key=original"],
            }
        )
        transport = FakeTransport(run=run)
        request_info = _make_request_info(
            run.cfg, endpoint_params={"key": "overridden"}
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
        run = _make_run(
            endpoint={
                "type": "chat",
                "urls": ["http://localhost:8000/api?a=1&b=2&c=3"],
            }
        )
        transport = FakeTransport(run=run)
        request_info = _make_request_info(
            run.cfg, endpoint_params={"d": "4", "b": "overridden"}
        )

        url = transport.build_url(request_info)
        assert "a=1" in url
        assert "b=overridden" in url  # Should override
        assert "b=2" not in url
        assert "c=3" in url
        assert "d=4" in url

    @pytest.mark.asyncio
    async def test_send_request_abstract_implemented(self, transport, request_info):
        """Test that send_request is callable on concrete implementation."""
        record = await transport.send_request(request_info, {"test": "payload"})
        assert isinstance(record, RequestRecord)

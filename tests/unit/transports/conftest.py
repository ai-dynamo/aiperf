# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and shared fixtures for the aiohttp client test suite."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.models import RequestRecord, SSEMessage, TextResponse
from aiperf.config import AIPerfConfig, BenchmarkConfig, BenchmarkRun
from aiperf.plugin.enums import EndpointType
from aiperf.transports.aiohttp_client import AioHttpClient, create_tcp_connector

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000"],
        "path": "/v1/chat/completions",
        "streaming": False,
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


class MockStreamReader:
    """Async generator wrapper that mimics aiohttp StreamReader.iter_any()."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk

    def __aiter__(self):
        return self.iter_any()


def create_config(
    base_url: str = "http://localhost:8000",
    custom_endpoint: str | None = "/v1/chat/completions",
    streaming: bool = False,
    model_name: str = "test-model",
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    connection_reuse: ConnectionReuseStrategy | None = None,
    connection_reuse_strategy: ConnectionReuseStrategy | None = None,
) -> BenchmarkConfig:
    """Factory function to create BenchmarkConfig instances for transport tests."""
    endpoint: dict[str, Any] = {
        "type": "chat",
        "urls": [base_url],
        "streaming": streaming,
        "connection_reuse": connection_reuse_strategy
        or connection_reuse
        or ConnectionReuseStrategy.POOLED,
    }
    if custom_endpoint is not None:
        endpoint["path"] = custom_endpoint
    if api_key is not None:
        endpoint["api_key"] = api_key
    if headers:
        endpoint["headers"] = headers
    return BenchmarkConfig(
        **{**_MINIMAL_CONFIG_KWARGS, "models": [model_name], "endpoint": endpoint}
    )


def make_run(cfg: BenchmarkConfig | None = None, **config_kwargs: Any) -> BenchmarkRun:
    """Wrap a BenchmarkConfig in a BenchmarkRun for transport tests."""
    if cfg is None:
        cfg = create_config(**config_kwargs)
    return BenchmarkRun(benchmark_id="test", cfg=cfg, artifact_dir=Path("/tmp/test"))


@pytest.fixture
def complex_sse_message_data() -> dict[str, str]:
    """Fixture providing complex SSE message test data."""
    return {
        "openai_chunk": """event: message
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk"}
data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
id: msg_123
retry: 5000""",
        "with_comments": """data: first line
: This is a comment
data: second line
: Another comment
event: custom""",
        "mixed_format": """data: {"json": "value"}
custom-header: custom-value
field-without-value
: comment line
retry: 3000""",
        "empty_and_whitespace": """data: test

event: message

: comment
data: final""",
    }


@pytest.fixture
def edge_case_inputs() -> dict[str, str]:
    """Fixture providing edge case inputs for robust testing."""
    return {
        "empty_string": "",
        "only_newlines": "\n\n\n",
        "only_whitespace": "   \t   ",
        "mixed_whitespace": "  \n  \t  \n  ",
        "single_colon": ":",
        "multiple_colons": "data: value: with: many: colons",
        "unicode_content": "data: 你好世界 🚀💻",
        "special_chars": "data: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "very_long_value": f"data: {'x' * 1000}",
    }


@pytest.fixture
def user_config() -> AIPerfConfig:
    """Fixture providing a sample AIPerfConfig for transports testing."""
    return AIPerfConfig(
        models=["gpt-4"],
        endpoint={
            "type": EndpointType.CHAT,
            "urls": ["http://localhost:8000"],
            "timeout": 600,
            "api_key": "test-api-key",
        },
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
    )


@pytest.fixture
async def aiohttp_client():
    """Fixture providing an AioHttpClient instance."""
    client = AioHttpClient(timeout=600.0)
    yield client
    await client.close()


def create_mock_response(
    status: int = 200,
    reason: str = "OK",
    content_type: str = "application/json",
    text_content: str = '{"success": true}',
) -> Mock:
    """Create a standardized mock aiohttp.ClientResponse."""
    return Mock(
        spec=aiohttp.ClientResponse,
        status=status,
        reason=reason,
        content_type=content_type,
        text=AsyncMock(return_value=text_content),
        content=Mock(
            spec=aiohttp.StreamReader,
            at_eof=AsyncMock(return_value=False),
            read=AsyncMock(),
            readuntil=AsyncMock(),
        ),
    )


@pytest.fixture
def mock_aiohttp_response() -> Mock:
    """Fixture providing a mock aiohttp.ClientResponse."""
    return create_mock_response()


@pytest.fixture
def mock_sse_response() -> Mock:
    """Fixture providing a mock SSE response."""
    return Mock(
        spec=aiohttp.ClientResponse,
        status=200,
        reason="OK",
        content_type="text/event-stream",
        content=Mock(),
    )


@pytest.fixture
def socket_factory_setup():
    """Fixture providing common socket factory test setup."""

    def _setup():
        with patch("aiohttp.TCPConnector") as mock_connector_class:
            create_tcp_connector()
            socket_factory = mock_connector_class.call_args[1]["socket_factory"]
            return mock_connector_class, socket_factory

    return _setup


def create_mock_error_response(status: int, reason: str, error_text: str) -> Mock:
    """Create a mock response for HTTP error testing."""
    return create_mock_response(status=status, reason=reason, text_content=error_text)


def assert_successful_request_record(
    record: RequestRecord,
    expected_status: int = 200,
    expected_response_count: int = 1,
    expected_response_type: type = TextResponse,
) -> None:
    """Assert that a RequestRecord represents a successful request."""
    assert isinstance(record, RequestRecord)
    assert record.status == expected_status
    assert record.error is None
    assert len(record.responses) == expected_response_count
    assert record.start_perf_ns is not None
    assert record.end_perf_ns is not None

    if expected_response_count > 0:
        if expected_response_type == TextResponse:
            assert all(isinstance(resp, TextResponse) for resp in record.responses)
        elif expected_response_type == SSEMessage:
            assert all(isinstance(resp, SSEMessage) for resp in record.responses)


def assert_error_request_record(
    record: RequestRecord,
    expected_error_code: int | None = None,
    expected_error_type: str | None = None,
    expected_error_message: str | None = None,
) -> None:
    """Assert that a RequestRecord represents a failed request."""
    assert isinstance(record, RequestRecord)
    assert record.error is not None
    assert len(record.responses) == 0

    if expected_error_code is not None:
        assert record.error.code == expected_error_code
    if expected_error_type is not None:
        assert record.error.type == expected_error_type
    if expected_error_message is not None:
        # The error message is formatted as repr(exception), e.g., "ValueError('message')"
        # Check if the expected message is contained in the actual message
        assert expected_error_message in record.error.message


def setup_mock_session(
    mock_session_class: Mock,
    mock_response: Mock,
    methods: list[str] | None = None,
) -> AsyncMock:
    """Simplified helper to set up aiohttp session mocks with proper async context manager support."""
    if methods is None:
        methods = [
            "request",
            "get",
            "post",
            "put",
            "patch",
            "delete",
            "head",
            "options",
        ]

    mock_session = AsyncMock()

    # Create a factory function that accepts the ClientSession parameters and returns the context manager
    def session_factory(*args, **kwargs):
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        return mock_session_context

    # Set the session class to return our factory
    mock_session_class.side_effect = session_factory

    # Setup context managers for all HTTP methods
    for method in methods:
        mock_method_context = AsyncMock()
        mock_method_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_method_context.__aexit__ = AsyncMock(return_value=None)
        setattr(mock_session, method, Mock(return_value=mock_method_context))

    return mock_session


def create_aiohttp_exception(
    exception_class: type[Exception], message: str
) -> Exception:
    """Create aiohttp-specific exceptions with proper initialization."""
    if exception_class == aiohttp.ClientConnectorError:
        return aiohttp.ClientConnectorError(
            connection_key=MagicMock(), os_error=OSError(message)
        )
    elif exception_class == aiohttp.ClientResponseError:
        mock_request_info = Mock(url="http://test.com", method="POST")
        return aiohttp.ClientResponseError(
            request_info=mock_request_info,
            history=(),
            status=500,
            message=message,
        )
    else:
        return exception_class(message)


@pytest.fixture
def config_non_streaming():
    """Fixture providing non-streaming BenchmarkRun."""
    return make_run(streaming=False)


@pytest.fixture
def config_streaming():
    """Fixture providing streaming BenchmarkRun."""
    return make_run(streaming=True)

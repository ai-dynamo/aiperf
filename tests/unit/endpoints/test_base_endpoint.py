# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import ParsedResponse, TextResponse, TextResponseData
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.record_models import (
    InferenceServerResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class MockEndpoint(BaseEndpoint):
    """Concrete implementation of BaseEndpoint for testing."""

    def format_payload(self, request_info: RequestInfo) -> dict:
        return {"test": "payload"}

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        if (json_obj := response.get_json()) and (text := json_obj.get("text")):
            return ParsedResponse(
                perf_ns=response.perf_ns, data=TextResponseData(text=text)
            )
        return None


class TestBaseEndpoint:
    """Comprehensive tests for BaseEndpoint functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return create_model_endpoint(
            EndpointType.CHAT, base_url="http://localhost:8000/v1/test"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a MockEndpoint instance."""
        return create_endpoint_with_mock_transport(MockEndpoint, model_endpoint)

    @pytest.mark.parametrize(
        "api_key,custom_headers,expected_headers",
        [
            (None, None, {}),
            ("test-api-key-123", None, {"Authorization": "Bearer test-api-key-123"}),
            (
                None,
                [
                    ("X-Custom-Header", "custom-value"),
                    ("X-Another-Header", "another-value"),
                ],
                {
                    "X-Custom-Header": "custom-value",
                    "X-Another-Header": "another-value",
                },
            ),
            (
                "secret-key",
                [("Content-Language", "en-US"), ("X-Client-Version", "1.0.0")],
                {
                    "Authorization": "Bearer secret-key",
                    "Content-Language": "en-US",
                    "X-Client-Version": "1.0.0",
                },
            ),
        ],
    )
    def test_get_endpoint_headers(
        self, endpoint, model_endpoint, api_key, custom_headers, expected_headers
    ):
        """Test get_endpoint_headers with various combinations."""
        model_endpoint.endpoint.api_key = api_key
        model_endpoint.endpoint.headers = custom_headers
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        headers = endpoint.get_endpoint_headers(request_info)

        for key, value in expected_headers.items():
            assert headers[key] == value

    def test_get_endpoint_headers_merges_per_turn_headers(
        self, endpoint, model_endpoint
    ):
        """Per-turn headers are merged on top of endpoint headers."""
        model_endpoint.endpoint.api_key = None
        model_endpoint.endpoint.headers = [("X-Static", "static-value")]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(headers={"x-session-token": "tok-A"})],
        )

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["X-Static"] == "static-value"
        assert headers["x-session-token"] == "tok-A"

    def test_get_endpoint_headers_per_turn_overrides_endpoint(
        self, endpoint, model_endpoint
    ):
        """On key conflict, per-turn headers win over endpoint config headers."""
        model_endpoint.endpoint.api_key = None
        model_endpoint.endpoint.headers = [("baggage", "from-config")]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(headers={"baggage": "from-trace"})],
        )

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["baggage"] == "from-trace"

    def test_get_endpoint_headers_no_turns(self, endpoint, model_endpoint):
        """Empty turns list does not break header construction."""
        model_endpoint.endpoint.api_key = "k"
        model_endpoint.endpoint.headers = None
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers == {"Authorization": "Bearer k"}

    def test_get_endpoint_headers_turn_without_headers(self, endpoint, model_endpoint):
        """A Turn with headers=None is a no-op for the merge."""
        model_endpoint.endpoint.api_key = None
        model_endpoint.endpoint.headers = [("X-Static", "v")]
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[Turn(headers=None)]
        )

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers == {"X-Static": "v"}

    def test_get_endpoint_headers_uses_current_turn_in_multi_turn_session(
        self, endpoint, model_endpoint
    ):
        """Multi-turn DELTAS_WITHOUT_RESPONSES sessions accumulate prior turns
        in `turns`; the merge must read headers from the current (last) turn.
        """
        model_endpoint.endpoint.api_key = None
        model_endpoint.endpoint.headers = None
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[
                Turn(headers={"x-session-token": "tok-A"}),  # earlier user turn
                Turn(headers=None),  # accumulated assistant response
                Turn(headers={"x-session-token": "tok-B"}),  # current user turn
            ],
        )

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["x-session-token"] == "tok-B"

    def test_get_endpoint_headers_per_turn_overrides_endpoint_case_insensitively(
        self, endpoint, model_endpoint
    ):
        """A per-turn header with different casing replaces the endpoint-config
        header rather than producing two duplicate wire headers (HTTP header
        names are case-insensitive per RFC 7230).
        """
        model_endpoint.endpoint.api_key = "secret"
        model_endpoint.endpoint.headers = [("Baggage", "from-config")]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[
                Turn(headers={"baggage": "from-trace", "authorization": "Bearer t"})
            ],
        )

        headers = endpoint.get_endpoint_headers(request_info)

        # Only one entry per case-insensitive key, and the trace value/casing wins.
        keys_lower = [k.lower() for k in headers]
        assert keys_lower.count("baggage") == 1
        assert keys_lower.count("authorization") == 1
        assert headers["baggage"] == "from-trace"
        assert headers["authorization"] == "Bearer t"
        assert "Baggage" not in headers
        assert "Authorization" not in headers

    @pytest.mark.parametrize(
        "url_params,expected_params",
        [
            (None, {}),
            ({}, {}),
            (
                {"api-version": "2024-10-01", "timeout": "60"},
                {"api-version": "2024-10-01", "timeout": "60"},
            ),
        ],
    )
    def test_get_endpoint_params(
        self, endpoint, model_endpoint, url_params, expected_params
    ):
        """Test get_endpoint_params with various URL parameters."""
        model_endpoint.endpoint.url_params = url_params
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        params = endpoint.get_endpoint_params(request_info)

        assert params == expected_params

    @pytest.mark.asyncio
    async def test_extract_response_data_single_response(self, endpoint):
        """Test extract_response_data with single valid response."""
        response = TextResponse(
            perf_ns=123456789,
            text='{"text": "Hello, world!"}',
            content_type="application/json",
        )

        record = RequestRecord(
            responses=[response],
            start_perf_ns=100000000,
            end_perf_ns=123456789,
        )

        results = endpoint.extract_response_data(record)

        assert len(results) == 1
        assert results[0].perf_ns == 123456789
        assert results[0].data.text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_extract_response_data_multiple_responses(self, endpoint):
        """Test extract_response_data with multiple responses."""
        responses = []
        for i in range(3):
            response = TextResponse(
                perf_ns=100000000 + i,
                text=f'{{"text": "Response {i}"}}',
                content_type="application/json",
            )
            responses.append(response)

        record = RequestRecord(
            responses=responses,
            start_perf_ns=50000000,
            end_perf_ns=100000002,
        )

        results = endpoint.extract_response_data(record)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.data.text == f"Response {i}"

    @pytest.mark.asyncio
    async def test_extract_response_data_filters_none(self, endpoint):
        """Test that None responses are filtered out."""
        response1 = TextResponse(
            perf_ns=100,
            text='{"text": "Valid"}',
            content_type="application/json",
        )

        response2 = TextResponse(
            perf_ns=200,
            text="{}",  # Will return None from parse
            content_type="application/json",
        )

        response3 = TextResponse(
            perf_ns=300,
            text='{"text": "Also valid"}',
            content_type="application/json",
        )

        record = RequestRecord(
            responses=[response1, response2, response3],
            start_perf_ns=50,
            end_perf_ns=300,
        )

        results = endpoint.extract_response_data(record)

        assert len(results) == 2
        assert results[0].data.text == "Valid"
        assert results[1].data.text == "Also valid"

    @pytest.mark.asyncio
    async def test_extract_response_data_empty_record(self, endpoint):
        """Test extract_response_data with no responses."""
        record = RequestRecord(
            responses=[],
            start_perf_ns=100,
            end_perf_ns=200,
        )
        results = endpoint.extract_response_data(record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_format_payload_called(self, endpoint, model_endpoint):
        """Test that format_payload is implemented and callable."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])
        payload = endpoint.format_payload(request_info)
        assert payload == {"test": "payload"}

    def test_parse_response_called(self, endpoint):
        """Test that parse_response is implemented and callable."""
        response = TextResponse(
            perf_ns=12345,
            text='{"text": "Hello"}',
            content_type="application/json",
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.text == "Hello"
        assert parsed.perf_ns == 12345


class TestBaseEndpointAbstractMethods:
    """Test that BaseEndpoint enforces abstract methods."""

    @pytest.fixture
    def test_model_endpoint(self):
        """Create a test ModelEndpointInfo for abstract method tests."""
        return create_model_endpoint(EndpointType.CHAT, base_url="http://localhost")

    def test_cannot_instantiate_base_endpoint(self, test_model_endpoint):
        """Test that BaseEndpoint cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEndpoint(model_endpoint=test_model_endpoint)

    def test_must_implement_format_payload(self, test_model_endpoint):
        """Test that subclasses must implement format_payload()."""

        class IncompleteEndpoint(BaseEndpoint):
            def parse_response(
                self, response: InferenceServerResponse
            ) -> ParsedResponse | None:
                return None

        with pytest.raises(TypeError):
            IncompleteEndpoint(model_endpoint=test_model_endpoint)

    def test_must_implement_parse_response(self, test_model_endpoint):
        """Test that subclasses must implement parse_response()."""

        class IncompleteEndpoint(BaseEndpoint):
            @classmethod
            def format_payload(self, request_info: RequestInfo) -> dict:
                return {}

        with pytest.raises(TypeError):
            IncompleteEndpoint(model_endpoint=test_model_endpoint)

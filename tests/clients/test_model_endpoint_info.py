# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ModelEndpointInfo.url property."""

import pytest
from pytest import param

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.enums import EndpointType, ModelSelectionStrategy


@pytest.fixture
def base_model_list_info():
    """Create a basic ModelListInfo for testing."""
    return ModelListInfo(
        models=[ModelInfo(name="test-model")],
        model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
    )


@pytest.fixture
def make_model_endpoint(base_model_list_info):
    """Factory fixture to create ModelEndpointInfo instances."""

    def _make(endpoint_type=EndpointType.CHAT, base_url=None, custom_endpoint=None):
        endpoint = EndpointInfo(
            type=endpoint_type,
            base_url=base_url,
            custom_endpoint=custom_endpoint,
        )
        return ModelEndpointInfo(
            models=base_model_list_info,
            endpoint=endpoint,
        )

    return _make


class TestModelEndpointInfoUrl:
    """Test cases for the ModelEndpointInfo.url property."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            param(
                "http://localhost:8000",
                "http://localhost:8000/v1/chat/completions",
                id="basic",
            ),
            param(
                "http://localhost:8000/",
                "http://localhost:8000/v1/chat/completions",
                id="single_trailing_slash",
            ),
            param(
                "http://localhost:8000///",
                "http://localhost:8000/v1/chat/completions",
                id="multiple_trailing_slashes",
            ),
            param(
                "https://api.example.com",
                "https://api.example.com/v1/chat/completions",
                id="https_with_subdomain",
            ),
            param(
                "http://localhost:9000",
                "http://localhost:9000/v1/chat/completions",
                id="http_with_port",
            ),
            param(
                "http://localhost:8000/api/inference",
                "http://localhost:8000/api/inference/v1/chat/completions",
                id="with_path_in_base_url",
            ),
            param(None, "/v1/chat/completions", id="none_base_url"),
            param("", "/v1/chat/completions", id="empty_string_base_url"),
        ],
    )
    def test_url_construction_base_url_variations(
        self, make_model_endpoint, base_url, expected
    ):
        """Test URL construction with various base URL configurations."""
        model_endpoint = make_model_endpoint(base_url=base_url)
        assert model_endpoint.url == expected

    @pytest.mark.parametrize(
        "base_url,custom_endpoint,expected",
        [
            param(
                "http://localhost:8000",
                "/custom/path",
                "http://localhost:8000/custom/path",
                id="with_leading_slash",
            ),
            param(
                "http://localhost:8000",
                "///custom/path",
                "http://localhost:8000/custom/path",
                id="with_multiple_leading_slashes",
            ),
            param(
                "http://localhost:8000",
                "custom/path",
                "http://localhost:8000/custom/path",
                id="without_leading_slash",
            ),
            param(None, "/custom/path", "/custom/path", id="no_base_url"),
        ],
    )
    def test_url_with_custom_endpoint(
        self, make_model_endpoint, base_url, custom_endpoint, expected
    ):
        """Test URL construction with custom endpoint variations."""
        model_endpoint = make_model_endpoint(
            base_url=base_url, custom_endpoint=custom_endpoint
        )
        assert model_endpoint.url == expected

    @pytest.mark.parametrize(
        "endpoint_type,expected_path",
        [
            (EndpointType.CHAT, "http://localhost:8000/v1/chat/completions"),
            (EndpointType.COMPLETIONS, "http://localhost:8000/v1/completions"),
            (EndpointType.EMBEDDINGS, "http://localhost:8000/v1/embeddings"),
            (EndpointType.RANKINGS, "http://localhost:8000/v1/ranking"),
        ],
    )
    def test_url_with_v1_deduplication(
        self, make_model_endpoint, endpoint_type, expected_path
    ):
        """Test URL construction deduplicates v1 for all endpoint types."""
        model_endpoint = make_model_endpoint(
            endpoint_type=endpoint_type, base_url="http://localhost:8000/v1"
        )
        assert model_endpoint.url == expected_path

    @pytest.mark.parametrize(
        "base_url,custom_endpoint,expected",
        [
            param(
                "http://localhost:8000/v1/extra",
                None,
                "http://localhost:8000/v1/extra/v1/chat/completions",
                id="v1_in_middle_of_base_url",
            ),
            param(
                "http://localhost:8000/v1",
                "v1/custom/endpoint",
                "http://localhost:8000/v1/v1/custom/endpoint",
                id="custom_endpoint_no_v1_deduplication",
            ),
            param(
                "http://localhost:8000/v1",
                "/api/v1/custom",
                "http://localhost:8000/v1/api/v1/custom",
                id="v1_in_middle_of_custom_endpoint",
            ),
            param(
                "http://localhost:8000/v1",
                "v1/",
                "http://localhost:8000/v1/v1/",
                id="custom_endpoint_only_v1_no_deduplication",
            ),
        ],
    )
    def test_url_v1_deduplication_edge_cases(
        self, make_model_endpoint, base_url, custom_endpoint, expected
    ):
        """Test edge cases for v1 deduplication logic.

        Note: v1 deduplication only applies when no custom endpoint is set.
        When custom_endpoint is provided, v1 is NOT deduplicated.
        """
        model_endpoint = make_model_endpoint(
            base_url=base_url, custom_endpoint=custom_endpoint
        )
        assert model_endpoint.url == expected

    @pytest.mark.parametrize(
        "endpoint_type,expected_path",
        [
            (EndpointType.CHAT, "/v1/chat/completions"),
            (EndpointType.COMPLETIONS, "/v1/completions"),
            (EndpointType.EMBEDDINGS, "/v1/embeddings"),
            (EndpointType.RANKINGS, "/v1/ranking"),
        ],
    )
    def test_url_different_endpoint_types(
        self, make_model_endpoint, endpoint_type, expected_path
    ):
        """Test URL construction with different endpoint types."""
        base_url = "http://localhost:8000"
        model_endpoint = make_model_endpoint(
            endpoint_type=endpoint_type, base_url=base_url
        )
        assert model_endpoint.url == f"{base_url}{expected_path}"

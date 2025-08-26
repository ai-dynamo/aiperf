# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.enums import EndpointServiceKind, EndpointType, EndpointTypeInfo


class TestEndpointInfo:
    """Test class for EndpointInfo Pydantic model."""

    @pytest.mark.parametrize(
        "tag,supports_streaming,produces_tokens,endpoint_path,metrics_title",
        [
            ("chat", True, True, "/v1/chat/completions", "LLM Metrics"),
            ("embeddings", False, False, "/v1/embeddings", "Embeddings Metrics"),
            ("custom", True, False, None, None),
            ("test", False, True, "/test", "Test Metrics"),
        ],
    )
    def test_valid_endpoint_info_creation(
        self, tag, supports_streaming, produces_tokens, endpoint_path, metrics_title
    ):
        """Test creating EndpointInfo with valid parameters."""
        endpoint_info = EndpointTypeInfo(
            tag=tag,
            service_kind=EndpointServiceKind.OPENAI,
            supports_streaming=supports_streaming,
            produces_tokens=produces_tokens,
            endpoint_path=endpoint_path,
            metrics_title=metrics_title,
        )

        assert endpoint_info.tag == tag
        assert endpoint_info.supports_streaming == supports_streaming
        assert endpoint_info.produces_tokens == produces_tokens
        assert endpoint_info.endpoint_path == endpoint_path
        assert endpoint_info.metrics_title == metrics_title

    @pytest.mark.parametrize(
        "invalid_data,expected_error_field",
        [
            ({"supports_streaming": True, "produces_tokens": True}, "tag"),
            ({"tag": "test", "produces_tokens": True}, "supports_streaming"),
            ({"tag": "test", "supports_streaming": True}, "produces_tokens"),
        ],
    )
    def test_invalid_endpoint_info_creation(self, invalid_data, expected_error_field):
        """Test EndpointInfo validation with invalid data."""
        with pytest.raises(ValidationError) as exc_info:
            EndpointTypeInfo(**invalid_data)

        if expected_error_field:
            assert any(
                error["loc"][0] == expected_error_field
                for error in exc_info.value.errors()
            )

    def test_endpoint_info_does_not_accept_empty_string(self):
        """Test that EndpointInfo does not accept empty string for tag."""
        with pytest.raises(ValidationError):
            EndpointTypeInfo(
                tag="",
                service_kind=EndpointServiceKind.OPENAI,
                supports_streaming=True,
                produces_tokens=True,
                endpoint_path=None,
                metrics_title=None,
            )


class TestEndpointType:
    """Test class for EndpointType enum."""

    @pytest.mark.parametrize(
        "endpoint_type,expected_tag,expected_streaming,expected_tokens,expected_path",
        [
            (
                EndpointType.OPENAI_CHAT_COMPLETIONS,
                "chat",
                True,
                True,
                "/v1/chat/completions",
            ),
            (
                EndpointType.OPENAI_COMPLETIONS,
                "completions",
                True,
                True,
                "/v1/completions",
            ),
            (
                EndpointType.OPENAI_EMBEDDINGS,
                "embeddings",
                False,
                False,
                "/v1/embeddings",
            ),
            (
                EndpointType.RANKING,
                "ranking",
                False,
                False,
                "/v1/ranking",
            ),
            (EndpointType.OPENAI_RESPONSES, "responses", True, True, "/v1/responses"),
        ],
    )
    def test_endpoint_type_properties(
        self,
        endpoint_type,
        expected_tag,
        expected_streaming,
        expected_tokens,
        expected_path,
    ):
        """Test EndpointType enum properties and methods."""
        assert str(endpoint_type) == expected_tag
        assert endpoint_type.value == expected_tag
        assert endpoint_type.supports_streaming == expected_streaming
        assert endpoint_type.produces_tokens == expected_tokens
        assert endpoint_type.endpoint_path == expected_path

    @pytest.mark.parametrize(
        "endpoint_type,expected_title",
        [
            (EndpointType.OPENAI_CHAT_COMPLETIONS, "LLM Metrics"),
            (EndpointType.OPENAI_COMPLETIONS, "LLM Metrics"),
            (EndpointType.OPENAI_EMBEDDINGS, "Embeddings Metrics"),
            (EndpointType.RANKING, "Ranking Metrics"),
            (EndpointType.OPENAI_RESPONSES, "LLM Metrics"),
        ],
    )
    def test_metrics_title_property(self, endpoint_type, expected_title):
        """Test metrics_title property returns correct titles."""
        assert endpoint_type.metrics_title == expected_title

    def test_endpoint_type_info_property(self):
        """Test that info property returns EndpointInfo instance."""
        endpoint_type = EndpointType.OPENAI_CHAT_COMPLETIONS
        info = endpoint_type.info

        assert isinstance(info, EndpointTypeInfo)
        assert info.tag == "chat"
        assert info.supports_streaming is True
        assert info.produces_tokens is True

    @pytest.mark.parametrize(
        "tag_value",
        ["chat", "completions", "embeddings", "responses"],
    )
    def test_enum_string_comparison(self, tag_value):
        """Test that enum values can be compared with strings."""
        endpoint_type = EndpointType(tag_value)
        assert endpoint_type == tag_value
        assert str(endpoint_type) == tag_value
        assert endpoint_type.info.tag == tag_value

    def test_endpoint_type_case_insensitive(self):
        """Test case insensitive enum behavior."""
        assert EndpointType("CHAT") == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert EndpointType("Chat") == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert EndpointType("chat") == EndpointType.OPENAI_CHAT_COMPLETIONS

    def test_all_endpoint_types_have_valid_info(self):
        """Test that all endpoint types have valid EndpointInfo objects."""
        for endpoint_type in EndpointType:
            info = endpoint_type.info
            assert isinstance(info, EndpointTypeInfo)
            assert isinstance(info.tag, str)
            assert len(info.tag) > 0
            assert isinstance(info.supports_streaming, bool)
            assert isinstance(info.produces_tokens, bool)
            assert endpoint_type.metrics_title is not None
            assert len(endpoint_type.metrics_title) > 0

    def test_endpoint_type_metrics_title_fallback(self):
        """Test metrics_title returns default when None."""
        # This tests the fallback logic in the metrics_title property
        endpoint_type = EndpointType.OPENAI_CHAT_COMPLETIONS

        # Temporarily modify the info to test fallback
        original_title = endpoint_type.info.metrics_title
        endpoint_type.info.metrics_title = None

        try:
            # Check that the metrics_title returns the default value
            assert endpoint_type.metrics_title == "Metrics"
        finally:
            # Restore original value
            endpoint_type.info.metrics_title = original_title

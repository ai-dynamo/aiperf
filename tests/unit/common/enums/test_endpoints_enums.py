# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType


class TestEndpointType:
    """Tests for EndpointType enum and its associated metadata."""

    @pytest.mark.parametrize(
        "endpoint_type,expected_tag,expected_streaming,expected_tokens,expected_path,expected_title",
        [
            (
                EndpointType.CHAT,
                "chat",
                True,
                True,
                "/v1/chat/completions",
                "LLM Metrics",
            ),
            (
                EndpointType.COMPLETIONS,
                "completions",
                True,
                True,
                "/v1/completions",
                "LLM Metrics",
            ),
            (
                EndpointType.EMBEDDINGS,
                "embeddings",
                False,
                False,
                "/v1/embeddings",
                "Embeddings Metrics",
            ),
            (
                EndpointType.NIM_RANKINGS,
                "nim_rankings",
                False,
                False,
                "/v1/ranking",
                "Rankings Metrics",
            ),
            (
                EndpointType.HF_TEI_RANKINGS,
                "hf_tei_rankings",
                False,
                False,
                "/rerank",
                "Ranking Metrics",
            ),
            (
                EndpointType.COHERE_RANKINGS,
                "cohere_rankings",
                False,
                False,
                "/v2/rerank",
                "Ranking Metrics",
            ),
            (
                EndpointType.HUGGINGFACE_GENERATE,
                "huggingface_generate",
                True,
                True,
                "/generate",
                "LLM Metrics",
            ),
        ],
    )
    def test_endpoint_type_metadata(
        self,
        endpoint_type,
        expected_tag,
        expected_streaming,
        expected_tokens,
        expected_path,
        expected_title,
    ):
        """Verify EndpointType metadata returned by factory."""
        assert str(endpoint_type) == expected_tag
        assert endpoint_type.value == expected_tag

        metadata = plugins.get_endpoint_metadata(endpoint_type)
        assert metadata.endpoint_path == expected_path
        assert metadata.supports_streaming == expected_streaming
        assert metadata.produces_tokens == expected_tokens
        assert metadata.metrics_title == expected_title

        # Optional sanity check for completeness
        assert hasattr(metadata, "endpoint_path")
        assert hasattr(metadata, "supports_streaming")
        assert hasattr(metadata, "produces_tokens")
        assert hasattr(metadata, "metrics_title")

    @pytest.mark.parametrize(
        "tag_value",
        [
            "chat",
            "completions",
            "embeddings",
            "nim_rankings",
            "hf_tei_rankings",
            "cohere_rankings",
            "huggingface_generate",
        ],
    )
    def test_enum_string_comparison(self, tag_value):
        """Ensure string comparisons and casts work correctly."""
        endpoint_type = EndpointType(tag_value)
        assert endpoint_type == tag_value
        assert str(endpoint_type) == tag_value
        assert EndpointType(tag_value).value == tag_value

    def test_endpoint_type_case_insensitive(self):
        """Verify EndpointType is case-insensitive."""
        assert EndpointType("CHAT") == EndpointType.CHAT
        assert EndpointType("Chat") == EndpointType.CHAT
        assert EndpointType("chat") == EndpointType.CHAT

    def test_all_endpoint_types_have_valid_metadata(self):
        """Ensure all EndpointType members have valid metadata via factory."""
        for endpoint_type in EndpointType:
            metadata = plugins.get_endpoint_metadata(endpoint_type)
            assert isinstance(metadata.supports_streaming, bool)
            assert isinstance(metadata.produces_tokens, bool)
            assert metadata.metrics_title
            assert isinstance(metadata.metrics_title, str)

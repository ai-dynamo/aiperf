# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consolidated tests for usage field parsing across endpoints."""

import pytest

from aiperf.common.models.record_models import ReasoningResponseData
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
)


@pytest.mark.parametrize(
    "endpoint_type,endpoint_class,object_type,data_key",
    [
        (EndpointType.CHAT, ChatEndpoint, "chat.completion", "message"),
        (EndpointType.CHAT, ChatEndpoint, "chat.completion.chunk", "delta"),
        (EndpointType.COMPLETIONS, CompletionsEndpoint, "completion", None),
    ],
)
class TestUsageParsing:
    """Parameterized tests for usage parsing across endpoints."""

    @pytest.fixture
    def endpoint(self, endpoint_type, endpoint_class):
        """Create endpoint instance."""
        model_endpoint = create_model_endpoint(endpoint_type)
        return create_endpoint_with_mock_transport(endpoint_class, model_endpoint)

    def test_parse_with_standard_usage(
        self, endpoint, object_type, data_key, endpoint_class
    ):
        """Test parsing response with standard usage fields."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test response"}}
        else:
            content_data = {"text": "Test response"}

        mock_response = create_mock_response(
            12345,
            {
                "object": object_type,
                "choices": [content_data],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 5
        assert parsed.usage.total_tokens == 15

    def test_parse_without_usage(self, endpoint, object_type, data_key, endpoint_class):
        """Test parsing response without usage field."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test"}}
        else:
            content_data = {"text": "Test"}

        mock_response = create_mock_response(
            12345, {"object": object_type, "choices": [content_data]}
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is None

    def test_parse_with_empty_usage(
        self, endpoint, object_type, data_key, endpoint_class
    ):
        """Test parsing response with empty usage dict."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test"}}
        else:
            content_data = {"text": "Test"}

        mock_response = create_mock_response(
            12345, {"object": object_type, "choices": [content_data], "usage": {}}
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is None  # Empty dict treated as None


class TestChatEndpointUsageSpecific:
    """Chat-specific usage parsing tests."""

    @pytest.fixture
    def endpoint(self):
        """Create a ChatEndpoint instance."""
        model_endpoint = create_model_endpoint(EndpointType.CHAT)
        return create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

    def test_parse_with_nested_reasoning_tokens(self, endpoint):
        """Test parsing response with nested reasoning tokens (o1/o3 models)."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Thinking...",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "total_tokens": 80,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.usage["completion_tokens_details"]["reasoning_tokens"] == 50
        assert parsed.usage.reasoning_tokens == 50

    def test_parse_with_modern_naming(self, endpoint):
        """Test parsing with input_tokens/output_tokens naming."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "Response"}}],
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 15,
                    "total_tokens": 40,
                    "output_tokens_details": {"reasoning_tokens": 5},
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage["input_tokens"] == 25
        assert parsed.usage["output_tokens"] == 15
        assert parsed.usage.reasoning_tokens == 5

    @pytest.mark.parametrize(
        "usage_data,expected_prompt,expected_completion,expected_total",
        [
            ({"prompt_tokens": 10}, 10, None, None),
            ({"completion_tokens": 5}, None, 5, None),
            ({"prompt_tokens": 10, "total_tokens": 10}, 10, None, 10),
            (
                {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                100,
                200,
                300,
            ),
        ],
    )
    def test_parse_partial_usage(
        self,
        endpoint,
        usage_data,
        expected_prompt,
        expected_completion,
        expected_total,
    ):
        """Test parsing responses with partial usage data."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "Test"}}],
                "usage": usage_data,
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.get("prompt_tokens") == expected_prompt
        assert parsed.usage.get("completion_tokens") == expected_completion
        assert parsed.usage.get("total_tokens") == expected_total


class TestUsageModelProperties:
    """Test Usage model helper properties for various provider formats."""

    @pytest.mark.parametrize(
        "usage_data,expected",
        [
            # OpenAI standard format
            (
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": None,
                },
            ),
            # Anthropic naming (input/output tokens)
            (
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": None,
                },
            ),
            # OpenAI with reasoning tokens
            (
                {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "total_tokens": 80,
                    "completion_tokens_details": {
                        "reasoning_tokens": 50,
                    },
                },
                {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "reasoning_tokens": 50,
                    "total_tokens": 80,
                },
            ),
        ],
    )
    def test_provider_specific_fields(self, usage_data, expected):
        """Test extraction of provider-specific usage fields."""
        usage = Usage(usage_data)

        assert usage.prompt_tokens == expected["prompt_tokens"]
        assert usage.completion_tokens == expected["completion_tokens"]
        assert usage.total_tokens == expected["total_tokens"]
        assert usage.reasoning_tokens == expected["reasoning_tokens"]

    @pytest.mark.parametrize(
        "usage_data,missing_fields",
        [
            # No special fields at all
            (
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                [
                    "reasoning_tokens",
                    "prompt_cache_read_tokens",
                    "prompt_cache_write_tokens",
                    "prompt_audio_tokens",
                    "completion_audio_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                ],
            ),
        ],
    )
    def test_missing_fields_return_none(self, usage_data, missing_fields):
        """Test that missing optional fields return None."""
        usage = Usage(usage_data)

        for field in missing_fields:
            assert getattr(usage, field) is None

    @pytest.mark.parametrize(
        "details_key,field,prop",
        [
            ("prompt_tokens_details", "cached_tokens", "prompt_cache_read_tokens"),
            ("prompt_tokens_details", "audio_tokens", "prompt_audio_tokens"),
            ("input_tokens_details", "cached_tokens", "prompt_cache_read_tokens"),
            ("input_tokens_details", "audio_tokens", "prompt_audio_tokens"),
            ("completion_tokens_details", "audio_tokens", "completion_audio_tokens"),
            (
                "completion_tokens_details",
                "accepted_prediction_tokens",
                "accepted_prediction_tokens",
            ),
            (
                "completion_tokens_details",
                "rejected_prediction_tokens",
                "rejected_prediction_tokens",
            ),
            ("output_tokens_details", "audio_tokens", "completion_audio_tokens"),
            (
                "output_tokens_details",
                "accepted_prediction_tokens",
                "accepted_prediction_tokens",
            ),
            (
                "output_tokens_details",
                "rejected_prediction_tokens",
                "rejected_prediction_tokens",
            ),
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ],
    )
    def test_detail_token_properties(self, details_key, field, prop):
        """Test extraction of token detail sub-fields from both naming conventions."""
        usage = Usage({"prompt_tokens": 10, details_key: {field: 42}})
        assert getattr(usage, prop) == 42

    @pytest.mark.parametrize(
        "top_level_field,prop",
        [
            ("cache_read_input_tokens", "prompt_cache_read_tokens"),
            ("cache_creation_input_tokens", "prompt_cache_write_tokens"),
        ],
    )
    def test_anthropic_top_level_cache_fields(self, top_level_field, prop):
        """Test that Anthropic-shape top-level cache fields are extracted."""
        usage = Usage({"input_tokens": 100, top_level_field: 256})
        assert getattr(usage, prop) == 256

    @pytest.mark.parametrize(
        "top_level_field,prop",
        [
            ("cache_read_input_tokens", "prompt_cache_read_tokens"),
            ("cache_creation_input_tokens", "prompt_cache_write_tokens"),
        ],
    )
    def test_anthropic_top_level_cache_fields_zero_not_skipped(
        self, top_level_field, prop
    ):
        """Test that Anthropic-shape top-level cache zero is returned, not skipped."""
        usage = Usage({"input_tokens": 100, top_level_field: 0})
        assert getattr(usage, prop) == 0

    def test_openai_nested_takes_precedence_for_cache_read(self):
        """OpenAI-style nested cached_tokens wins over Anthropic top-level
        cache_read_input_tokens when both happen to be present (defensive).
        """
        usage = Usage(
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 7},
                "cache_read_input_tokens": 99,
            }
        )
        assert usage.prompt_cache_read_tokens == 7

    @pytest.mark.parametrize(
        "details_key,field,prop",
        [
            ("prompt_tokens_details", "cached_tokens", "prompt_cache_read_tokens"),
            ("prompt_tokens_details", "audio_tokens", "prompt_audio_tokens"),
            ("completion_tokens_details", "audio_tokens", "completion_audio_tokens"),
            (
                "completion_tokens_details",
                "accepted_prediction_tokens",
                "accepted_prediction_tokens",
            ),
            (
                "completion_tokens_details",
                "rejected_prediction_tokens",
                "rejected_prediction_tokens",
            ),
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ],
    )
    def test_detail_token_properties_zero_not_skipped(self, details_key, field, prop):
        """Test that zero values are returned, not treated as missing."""
        usage = Usage({"prompt_tokens": 10, details_key: {field: 0}})
        assert getattr(usage, prop) == 0

    def test_prompt_tokens_zero_not_skipped(self):
        """Test that prompt_tokens=0 is returned, not falling through to input_tokens."""
        usage = Usage({"prompt_tokens": 0, "input_tokens": 99})
        assert usage.prompt_tokens == 0

    def test_completion_tokens_zero_not_skipped(self):
        """Test that completion_tokens=0 is returned, not falling through to output_tokens."""
        usage = Usage({"completion_tokens": 0, "output_tokens": 99})
        assert usage.completion_tokens == 0


class TestUsageVendorEnvelopes:
    """Coverage of vendor-specific Usage envelopes and synonym keys.

    Each vendor reports usage with a slightly different shape; Usage.__init__
    normalizes the recognized envelopes (Gemini's `usageMetadata`, Cohere's
    `meta`) so all properties read from the top level uniformly.
    """

    def test_gemini_camelcase_basic_tokens(self):
        """Gemini wraps usage in usageMetadata with camelCase token fields."""
        usage = Usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                }
            }
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_gemini_thoughts_token_count_maps_to_reasoning(self):
        """Gemini's thoughtsTokenCount surfaces as the reasoning_tokens property."""
        usage = Usage({"usageMetadata": {"thoughtsTokenCount": 200}})
        assert usage.reasoning_tokens == 200

    def test_gemini_cached_content_maps_to_cache_read(self):
        """Gemini's cachedContentTokenCount surfaces as prompt_cache_read_tokens."""
        usage = Usage({"usageMetadata": {"cachedContentTokenCount": 80}})
        assert usage.prompt_cache_read_tokens == 80

    def test_gemini_tool_use_prompt_token_count(self):
        """Gemini surfaces tool/function-call input tokens separately."""
        usage = Usage({"usageMetadata": {"toolUsePromptTokenCount": 30}})
        assert usage.tool_use_prompt_tokens == 30

    def test_bedrock_camelcase_basic_tokens(self):
        """AWS Bedrock uses camelCase top-level fields like inputTokens."""
        usage = Usage({"inputTokens": 100, "outputTokens": 50, "totalTokens": 150})
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_bedrock_camelcase_cache_fields(self):
        """Bedrock surfaces cache reads/writes as cacheReadInputTokens / cacheWriteInputTokens."""
        usage = Usage(
            {
                "inputTokens": 100,
                "cacheReadInputTokens": 80,
                "cacheWriteInputTokens": 1024,
            }
        )
        assert usage.prompt_cache_read_tokens == 80
        assert usage.prompt_cache_write_tokens == 1024

    def test_deepseek_cache_hit_maps_to_cache_read(self):
        """DeepSeek's prompt_cache_hit_tokens surfaces as prompt_cache_read_tokens."""
        usage = Usage(
            {
                "prompt_tokens": 1600,
                "completion_tokens": 50,
                "prompt_cache_hit_tokens": 1280,
                "prompt_cache_miss_tokens": 320,
            }
        )
        assert usage.prompt_cache_read_tokens == 1280
        assert usage.prompt_cache_miss_tokens == 320

    def test_deepseek_cache_miss_zero_not_skipped(self):
        """A 0-miss DeepSeek response (full cache hit) returns 0, not None."""
        usage = Usage({"prompt_cache_miss_tokens": 0})
        assert usage.prompt_cache_miss_tokens == 0

    def test_cohere_meta_tokens_raw_counts(self):
        """Cohere wraps raw token counts under meta.tokens; we unwrap it.

        The Cohere-specific `meta.billed_units` distinction (billed vs raw)
        is intentionally NOT modelled as a separate property — the raw
        count is what the model actually processed (and what every other
        vendor reports), so `prompt_tokens` stays consistent across
        vendors. Callers that need billing reconciliation can still read
        `usage["meta"]["billed_units"]` directly.
        """
        usage = Usage(
            {
                "meta": {
                    "billed_units": {"input_tokens": 100, "output_tokens": 50},
                    "tokens": {"input_tokens": 105, "output_tokens": 52},
                }
            }
        )
        assert usage.prompt_tokens == 105
        assert usage.completion_tokens == 52
        # Underlying dict is preserved verbatim for advanced consumers.
        assert usage["meta"]["billed_units"] == {
            "input_tokens": 100,
            "output_tokens": 50,
        }

    def test_cohere_billed_only_passes_through(self):
        """A Cohere response with only meta.billed_units (no meta.tokens) leaves
        prompt_tokens/completion_tokens unset but the dict is still preserved."""
        usage = Usage(
            {"meta": {"billed_units": {"input_tokens": 12, "output_tokens": 8}}}
        )
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage["meta"]["billed_units"] == {
            "input_tokens": 12,
            "output_tokens": 8,
        }

    def test_mistral_prompt_audio_seconds(self):
        """Mistral surfaces prompt audio duration in seconds, not tokens."""
        usage = Usage(
            {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "prompt_audio_seconds": 12.5,
            }
        )
        assert usage.prompt_audio_seconds == 12.5

    def test_mistral_prompt_audio_seconds_int_coerced_to_float(self):
        """Even if the API reports an integer, the property returns float."""
        usage = Usage({"prompt_audio_seconds": 12})
        assert usage.prompt_audio_seconds == 12.0
        assert isinstance(usage.prompt_audio_seconds, float)

    def test_normalization_does_not_overwrite_existing_top_level(self):
        """If a top-level key exists, it wins over the same key in a wrapper."""
        usage = Usage(
            {
                "promptTokenCount": 999,
                "usageMetadata": {"promptTokenCount": 10},
            }
        )
        assert usage.prompt_tokens == 999

    def test_unrecognized_fields_pass_through(self):
        """Usage preserves the underlying dict so unmodelled fields are accessible."""
        usage = Usage({"prompt_tokens": 10, "vendor_specific_field": "foo"})
        assert usage["vendor_specific_field"] == "foo"

    def test_synonym_precedence_for_prompt_tokens(self):
        """When multiple synonyms are present, PROMPT_TOKENS_KEYS order wins."""
        # prompt_tokens (1st) beats input_tokens (2nd) beats promptTokenCount (3rd)
        usage = Usage(
            {
                "prompt_tokens": 1,
                "input_tokens": 2,
                "promptTokenCount": 3,
                "inputTokens": 4,
            }
        )
        assert usage.prompt_tokens == 1
        # Same precedence test with the 1st absent
        usage = Usage({"input_tokens": 2, "promptTokenCount": 3, "inputTokens": 4})
        assert usage.prompt_tokens == 2

    @pytest.mark.parametrize(
        "shape,expected",
        [
            # OpenAI-style nested
            (
                {"prompt_tokens_details": {"cached_tokens": 50}},
                50,
            ),
            # Anthropic top-level
            ({"cache_read_input_tokens": 60}, 60),
            # DeepSeek top-level
            ({"prompt_cache_hit_tokens": 70}, 70),
            # Gemini camelCase top-level
            ({"cachedContentTokenCount": 80}, 80),
            # Bedrock camelCase top-level
            ({"cacheReadInputTokens": 90}, 90),
        ],
    )
    def test_cache_read_recognizes_all_vendors(self, shape, expected):
        """prompt_cache_read_tokens unifies all five vendor shapes."""
        usage = Usage({"prompt_tokens": 100, **shape})
        assert usage.prompt_cache_read_tokens == expected

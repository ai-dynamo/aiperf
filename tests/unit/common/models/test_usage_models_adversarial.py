# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial / edge-case tests for the Usage model.

These tests exercise the Usage dict subclass under conditions that the
happy-path tests in test_usage_parsing.py and test_usage_metrics.py don't
cover:

- Real verbatim payloads from each supported vendor (vendor fixture replay).
- Envelope normalization edge cases: empty wrappers, wrong-typed wrappers,
  collisions with existing top-level keys, nested wrappers.
- Type pollution: None values, wrong types in nested fields, sentinel-like
  values, very large numbers.
- Synonym precedence rules under all permutations of multiple keys.
- Mutability: post-construction mutation, copy semantics, JSON / pickle
  round-trips, construction-from-Usage.
- Property determinism: repeated reads return identical values; mutation
  propagates without caching.
- Streaming metric behavior with mixed-shape chunks.

If a test in this file fails, the failure is intentional adversarial coverage
— either the Usage model has a real bug, or a contract changed and the test
needs updating to match new behavior. Do not silence these tests by adding
defensive shims to Usage; investigate the failure first.
"""

import copy
import json
import pickle

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import param

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.models.usage_models import Usage
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.usage_cache_metrics import (
    UsagePromptCacheMissTokensMetric,
    UsagePromptCacheReadTokensMetric,
    UsagePromptCacheWriteTokensMetric,
)
from aiperf.metrics.types.usage_extras_metrics import (
    UsagePromptAudioSecondsMetric,
    UsageToolUsePromptTokensMetric,
)
from aiperf.metrics.types.usage_metrics import (
    UsageCompletionTokensMetric,
    UsagePromptTokensMetric,
    UsageReasoningTokensMetric,
    UsageTotalTokensMetric,
)

# Verbatim usage payloads from each supported vendor's API documentation,
# trimmed to the `usage` field of a real response. These exercise the full
# normalization + property pipeline against shapes the model encounters in
# production rather than in synthetic dict literals.
VENDOR_FIXTURES: dict[str, dict] = {
    "openai_gpt4o_basic": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    },
    "openai_gpt4o_with_caching": {
        "prompt_tokens": 2006,
        "completion_tokens": 300,
        "total_tokens": 2306,
        "prompt_tokens_details": {"cached_tokens": 1920, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    "openai_o1_reasoning": {
        "prompt_tokens": 50,
        "completion_tokens": 1500,
        "total_tokens": 1550,
        "completion_tokens_details": {"reasoning_tokens": 1024},
    },
    "openai_predicted_outputs": {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 150,
            "rejected_prediction_tokens": 30,
        },
    },
    "anthropic_claude_with_caching": {
        "input_tokens": 100,
        "cache_creation_input_tokens": 1024,
        "cache_read_input_tokens": 200,
        "output_tokens": 50,
    },
    "anthropic_claude_no_caching": {
        "input_tokens": 100,
        "output_tokens": 50,
    },
    "deepseek_v3_chat": {
        "prompt_tokens": 1600,
        "completion_tokens": 100,
        "total_tokens": 1700,
        "prompt_cache_hit_tokens": 1280,
        "prompt_cache_miss_tokens": 320,
    },
    "gemini_2_flash_basic": {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        }
    },
    "gemini_with_thinking": {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 200,
            "totalTokenCount": 260,
        }
    },
    "gemini_with_tools": {
        "usageMetadata": {
            "promptTokenCount": 100,
            "toolUsePromptTokenCount": 30,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 80,
            "totalTokenCount": 180,
        }
    },
    "bedrock_converse_basic": {
        "inputTokens": 100,
        "outputTokens": 50,
        "totalTokens": 150,
    },
    "bedrock_converse_with_caching": {
        "inputTokens": 100,
        "outputTokens": 50,
        "totalTokens": 1174,
        "cacheReadInputTokens": 200,
        "cacheWriteInputTokens": 1024,
    },
    "cohere_command_r_chat": {
        # Cohere v1 envelope: response root has a `meta` field. If the parser
        # passes the response root to Usage(), this is what arrives.
        "meta": {
            "billed_units": {"input_tokens": 100, "output_tokens": 50},
            "tokens": {"input_tokens": 105, "output_tokens": 52},
        }
    },
    "cohere_v2_chat": {
        # Cohere v2 envelope: `usage` field on the response root has
        # billed_units, tokens, and cached_tokens at its top level (no `meta`
        # wrapper). The parser passes that `usage` dict directly to Usage().
        "billed_units": {"input_tokens": 100, "output_tokens": 50},
        "tokens": {"input_tokens": 105, "output_tokens": 52},
        "cached_tokens": 30,
    },
    "mistral_audio_input": {
        "prompt_tokens": 24,
        "completion_tokens": 27,
        "total_tokens": 51,
        "prompt_audio_seconds": 12.5,
    },
    "vllm_openai_compatible": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": None,  # vLLM may emit explicit None
        "completion_tokens_details": None,
    },
}


class TestUsageRealVendorFixtures:
    """Replay verbatim API response fixtures from each supported vendor.

    The dicts in VENDOR_FIXTURES are trimmed from each vendor's actual
    documented response shape, including their quirks (camelCase vs snake,
    nested vs top-level, explicit nulls, alternate field names).
    """

    @pytest.mark.parametrize(
        "fixture_name,prompt,completion,total",
        [
            param("openai_gpt4o_basic", 100, 50, 150, id="openai_gpt4o_basic"),
            param(
                "openai_gpt4o_with_caching", 2006, 300, 2306, id="openai_gpt4o_caching"
            ),
            param("openai_o1_reasoning", 50, 1500, 1550, id="openai_o1_reasoning"),
            param("anthropic_claude_with_caching", 100, 50, None, id="anthropic_cached"),
            param("anthropic_claude_no_caching", 100, 50, None, id="anthropic_basic"),
            param("deepseek_v3_chat", 1600, 100, 1700, id="deepseek_v3"),
            param("gemini_2_flash_basic", 10, 20, 30, id="gemini_flash"),
            param("gemini_with_thinking", 10, 50, 260, id="gemini_thinking"),
            param("gemini_with_tools", 100, 50, 180, id="gemini_tools"),
            param("bedrock_converse_basic", 100, 50, 150, id="bedrock_basic"),
            param(
                "bedrock_converse_with_caching", 100, 50, 1174, id="bedrock_caching"
            ),
            param("cohere_command_r_chat", 105, 52, None, id="cohere_command_r"),
            param("cohere_v2_chat", 105, 52, None, id="cohere_v2"),
            param("mistral_audio_input", 24, 27, 51, id="mistral_audio"),
            param("vllm_openai_compatible", 100, 50, 150, id="vllm_compat"),
        ],
    )  # fmt: skip
    def test_basic_token_counts_extract(self, fixture_name, prompt, completion, total):
        usage = Usage(VENDOR_FIXTURES[fixture_name])
        assert usage.prompt_tokens == prompt
        assert usage.completion_tokens == completion
        assert usage.total_tokens == total

    def test_openai_o1_reasoning_extracted(self):
        usage = Usage(VENDOR_FIXTURES["openai_o1_reasoning"])
        assert usage.reasoning_tokens == 1024

    def test_openai_predicted_outputs_extracted(self):
        usage = Usage(VENDOR_FIXTURES["openai_predicted_outputs"])
        assert usage.accepted_prediction_tokens == 150
        assert usage.rejected_prediction_tokens == 30

    def test_anthropic_caching_extracted(self):
        usage = Usage(VENDOR_FIXTURES["anthropic_claude_with_caching"])
        assert usage.prompt_cache_read_tokens == 200
        assert usage.prompt_cache_write_tokens == 1024

    def test_anthropic_no_caching_returns_none(self):
        usage = Usage(VENDOR_FIXTURES["anthropic_claude_no_caching"])
        assert usage.prompt_cache_read_tokens is None
        assert usage.prompt_cache_write_tokens is None

    def test_deepseek_cache_split_extracted(self):
        usage = Usage(VENDOR_FIXTURES["deepseek_v3_chat"])
        assert usage.prompt_cache_read_tokens == 1280
        assert usage.prompt_cache_miss_tokens == 320
        # DeepSeek invariant: prompt_tokens == hit + miss
        assert (
            usage.prompt_tokens
            == usage.prompt_cache_read_tokens + usage.prompt_cache_miss_tokens
        )

    def test_gemini_thinking_extracted(self):
        usage = Usage(VENDOR_FIXTURES["gemini_with_thinking"])
        assert usage.reasoning_tokens == 200

    def test_gemini_tools_and_caching_extracted(self):
        usage = Usage(VENDOR_FIXTURES["gemini_with_tools"])
        assert usage.tool_use_prompt_tokens == 30
        assert usage.prompt_cache_read_tokens == 80

    def test_bedrock_caching_extracted(self):
        usage = Usage(VENDOR_FIXTURES["bedrock_converse_with_caching"])
        assert usage.prompt_cache_read_tokens == 200
        assert usage.prompt_cache_write_tokens == 1024

    def test_cohere_billed_preserved_on_underlying_dict(self):
        """Cohere's billed_units is intentionally not modelled as a property,
        but the underlying dict still carries it for billing reconciliation."""
        usage = Usage(VENDOR_FIXTURES["cohere_command_r_chat"])
        assert usage["meta"]["billed_units"] == {
            "input_tokens": 100,
            "output_tokens": 50,
        }

    def test_cohere_v2_top_level_envelope_unwrapped(self):
        """Cohere v2 has `tokens` and `billed_units` at the top level of the
        usage dict (no `meta` wrapper). The top-level `tokens` sub-dict is
        unwrapped so input_tokens / output_tokens resolve via the standard
        synonym list."""
        usage = Usage(VENDOR_FIXTURES["cohere_v2_chat"])
        assert usage.prompt_tokens == 105
        assert usage.completion_tokens == 52
        # billed_units stays accessible on the underlying dict
        assert usage["billed_units"] == {"input_tokens": 100, "output_tokens": 50}

    def test_cohere_v2_cached_tokens_resolves_as_cache_read(self):
        """Cohere v2 emits `cached_tokens` at the top level of the usage dict;
        we treat it as a synonym for prompt_cache_read_tokens."""
        usage = Usage(VENDOR_FIXTURES["cohere_v2_chat"])
        assert usage.prompt_cache_read_tokens == 30

    def test_cohere_v1_meta_cached_tokens_resolves_as_cache_read(self):
        """Cohere v1's ApiMeta also carries `cached_tokens` as a scalar at
        the meta level (verified against the cohere-python SDK ApiMeta type).
        We lift it during normalization so the standard cache-read lookup
        finds it."""
        usage = Usage(
            {
                "meta": {
                    "billed_units": {"input_tokens": 100, "output_tokens": 50},
                    "tokens": {"input_tokens": 105, "output_tokens": 52},
                    "cached_tokens": 25,
                }
            }
        )
        assert usage.prompt_cache_read_tokens == 25
        # And the standard prompt/completion synonyms still resolve.
        assert usage.prompt_tokens == 105
        assert usage.completion_tokens == 52

    def test_watsonx_input_token_count_resolves_as_prompt_tokens(self):
        """IBM watsonx uses `input_token_count` / `generated_token_count` as
        response-root fields (no `usage` envelope). When passed to Usage(),
        these resolve via the appended synonyms in PROMPT/COMPLETION_TOKENS_KEYS."""
        usage = Usage(
            {
                "generated_text": "...",
                "input_token_count": 100,
                "generated_token_count": 50,
                "stop_reason": "eos_token",
            }
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    def test_watsonx_synonyms_lose_precedence_to_openai_shape(self):
        """If a payload has both OpenAI shape AND watsonx _count fields
        (defensive, e.g. a translating proxy), OpenAI synonyms win since
        they're listed first in the keys."""
        usage = Usage(
            {
                "prompt_tokens": 1,
                "input_token_count": 999,
                "completion_tokens": 2,
                "generated_token_count": 888,
            }
        )
        assert usage.prompt_tokens == 1
        assert usage.completion_tokens == 2

    def test_mistral_audio_seconds_extracted(self):
        usage = Usage(VENDOR_FIXTURES["mistral_audio_input"])
        assert usage.prompt_audio_seconds == 12.5
        assert isinstance(usage.prompt_audio_seconds, float)

    def test_mistral_audio_seconds_empty_dict_sentinel_returns_none(self):
        """Mistral emits `prompt_audio_seconds: {}` when there's no audio
        in the prompt. We treat any non-numeric value as "no audio" and
        return None — must not crash trying to coerce {} to float."""
        usage = Usage(
            {
                "prompt_tokens": 24,
                "completion_tokens": 27,
                "total_tokens": 51,
                "prompt_audio_seconds": {},
            }
        )
        assert usage.prompt_audio_seconds is None

    @pytest.mark.parametrize(
        "value",
        [
            param({}, id="empty_dict"),
            param([], id="empty_list"),
            param("12.5", id="string_number"),
            param("not-a-number", id="string_garbage"),
            param(None, id="none_value"),
        ],
    )  # fmt: skip
    def test_mistral_audio_seconds_non_numeric_returns_none(self, value):
        usage = Usage({"prompt_audio_seconds": value})
        assert usage.prompt_audio_seconds is None

    def test_vllm_explicit_none_details_does_not_crash(self):
        """vLLM may emit details fields explicitly set to None; the property
        must treat that as "no nested field" and return None, not crash."""
        usage = Usage(VENDOR_FIXTURES["vllm_openai_compatible"])
        assert usage.reasoning_tokens is None
        assert usage.prompt_cache_read_tokens is None
        assert usage.completion_audio_tokens is None


class TestUsageEnvelopeEdgeCases:
    """Wrapper / envelope normalization under malformed or sparse input."""

    @pytest.mark.parametrize(
        "envelope_value",
        [
            param(None, id="none"),
            param("not-a-dict", id="string"),
            param(["list", "items"], id="list"),
            param(42, id="int"),
            param(3.14, id="float"),
            param({}, id="empty_dict"),
        ],
    )  # fmt: skip
    def test_gemini_envelope_with_wrong_type_does_not_crash(self, envelope_value):
        usage = Usage({"usageMetadata": envelope_value, "prompt_tokens": 5})
        assert usage.prompt_tokens == 5  # falls through to top-level

    @pytest.mark.parametrize(
        "envelope_value",
        [
            param(None, id="none"),
            param("not-a-dict", id="string"),
            param([], id="list"),
            param(42, id="int"),
            param({}, id="empty_dict"),
        ],
    )  # fmt: skip
    def test_cohere_envelope_with_wrong_type_does_not_crash(self, envelope_value):
        usage = Usage({"meta": envelope_value, "prompt_tokens": 7})
        assert usage.prompt_tokens == 7

    def test_meta_with_no_recognized_subfields(self):
        """A meta envelope with neither tokens nor billed_units is a no-op."""
        usage = Usage({"meta": {"random_other_field": 42}, "prompt_tokens": 99})
        assert usage.prompt_tokens == 99
        # Original meta still preserved
        assert usage["meta"] == {"random_other_field": 42}

    def test_meta_tokens_wrong_type(self):
        """meta.tokens that is not a dict must not crash unwrap."""
        usage = Usage({"meta": {"tokens": "not-a-dict"}, "prompt_tokens": 5})
        assert usage.prompt_tokens == 5

    def test_gemini_envelope_keys_do_not_overwrite_top_level(self):
        """If a top-level key already exists, the envelope's same-named key
        loses (setdefault semantics)."""
        usage = Usage(
            {
                "promptTokenCount": 999,
                "usageMetadata": {"promptTokenCount": 10},
            }
        )
        assert usage.prompt_tokens == 999

    def test_cohere_meta_tokens_does_not_overwrite_top_level(self):
        """Same rule for Cohere: if input_tokens is already top-level, keep it."""
        usage = Usage(
            {
                "input_tokens": 999,
                "meta": {"tokens": {"input_tokens": 10}},
            }
        )
        # input_tokens at top-level is the FIRST synonym in PROMPT_TOKENS_KEYS
        # for Cohere shape (after prompt_tokens), so 999 wins.
        assert usage.prompt_tokens == 999

    def test_both_envelopes_present(self):
        """A response that somehow carries both Gemini and Cohere envelopes
        must unwrap both without error."""
        usage = Usage(
            {
                "usageMetadata": {"promptTokenCount": 10},
                "meta": {"tokens": {"output_tokens": 5}},
            }
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    def test_doubly_nested_envelope_is_not_recursively_unwrapped(self):
        """If usageMetadata contains another usageMetadata, we only unwrap
        the outer one (single pass at __init__). This documents intentional
        non-recursion."""
        usage = Usage(
            {
                "usageMetadata": {
                    "usageMetadata": {"promptTokenCount": 999},
                    "promptTokenCount": 10,
                }
            }
        )
        # Outer usageMetadata.promptTokenCount lifts to top → 10
        assert usage.prompt_tokens == 10

    def test_empty_usage_dict(self):
        usage = Usage({})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.total_tokens is None
        assert usage.reasoning_tokens is None
        assert usage.prompt_cache_read_tokens is None

    def test_usage_constructed_from_none_raises(self):
        """Usage(None) is not a valid construction (dict(None) raises)."""
        with pytest.raises(TypeError):
            Usage(None)


class TestUsageTypePollution:
    """Adversarial type combinations in fields we don't strictly validate."""

    def test_explicit_none_value_for_top_level_returns_none(self):
        """If a vendor explicitly sets prompt_tokens to null, the property
        returns None (key is present but its value is None — distinct from
        key-missing)."""
        usage = Usage({"prompt_tokens": None, "input_tokens": 99})
        # First-present-key semantics: prompt_tokens IS present (value None);
        # returns None. We do NOT fall through to input_tokens here.
        assert usage.prompt_tokens is None

    def test_negative_token_count_passes_through(self):
        """We don't validate non-negative — surface what the API said."""
        usage = Usage({"prompt_tokens": -5})
        assert usage.prompt_tokens == -5

    def test_string_value_for_token_count_passes_through_unchanged(self):
        """No coercion: if the API misformatted, the caller sees the bug."""
        usage = Usage({"prompt_tokens": "100"})
        assert usage.prompt_tokens == "100"  # type: ignore[comparison-overlap]

    def test_float_value_for_token_count_passes_through(self):
        usage = Usage({"prompt_tokens": 100.0})
        assert usage.prompt_tokens == 100.0

    def test_bool_value_for_token_count_passes_through(self):
        """Python bool is an int subclass; we don't mask that quirk."""
        usage = Usage({"prompt_tokens": True})
        assert usage.prompt_tokens is True

    def test_very_large_token_count(self):
        """No overflow: Python ints are arbitrary precision."""
        usage = Usage({"prompt_tokens": 10**18})
        assert usage.prompt_tokens == 10**18

    @pytest.mark.parametrize(
        "details_value",
        [
            param(None, id="none"),
            param("not-a-dict", id="string"),
            param(["list-not-dict"], id="list"),
            param(42, id="int"),
            param({}, id="empty_dict"),
            param({"unrelated_field": 1}, id="dict_no_known_keys"),
        ],
    )  # fmt: skip
    def test_prompt_tokens_details_wrong_type_or_empty(self, details_value):
        """isinstance(details, dict) guard prevents crashes on bad shapes."""
        usage = Usage({"prompt_tokens": 10, "prompt_tokens_details": details_value})
        assert usage.prompt_cache_read_tokens is None
        assert usage.prompt_audio_tokens is None

    def test_inner_field_explicit_none_returns_none(self):
        """If `cached_tokens` is explicitly None inside details, return None
        (the key IS in the dict, even if the value is None)."""
        usage = Usage(
            {
                "prompt_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": None},
            }
        )
        assert usage.prompt_cache_read_tokens is None

    def test_unrecognized_top_level_keys_pass_through_unchanged(self):
        """Usage preserves the original dict contents verbatim."""
        usage = Usage(
            {
                "prompt_tokens": 10,
                "vendor_specific_field": "foo",
                "future_field_we_dont_know_about": [1, 2, 3],
            }
        )
        assert usage["vendor_specific_field"] == "foo"
        assert usage["future_field_we_dont_know_about"] == [1, 2, 3]

    def test_dict_methods_still_work(self):
        """Subclassing dict shouldn't break standard dict operations."""
        usage = Usage({"prompt_tokens": 10})
        assert "prompt_tokens" in usage
        assert len(usage) == 1
        assert list(usage.keys()) == ["prompt_tokens"]
        assert list(usage.values()) == [10]


class TestUsageSynonymPrecedence:
    """When multiple synonyms coexist, the FIRST present key in *_KEYS wins."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            # PROMPT_TOKENS_KEYS order: prompt_tokens > input_tokens > promptTokenCount > inputTokens
            param({"prompt_tokens": 1, "input_tokens": 2}, 1, id="prompt_beats_input"),
            param({"input_tokens": 2, "promptTokenCount": 3}, 2, id="input_beats_camel"),
            param({"promptTokenCount": 3, "inputTokens": 4}, 3, id="camel_beats_bedrock"),
            param({"inputTokens": 4}, 4, id="bedrock_alone"),
            param({"prompt_tokens": 1, "inputTokens": 4}, 1, id="prompt_skips_to_bedrock"),
        ],
    )  # fmt: skip
    def test_prompt_tokens_precedence(self, data, expected):
        assert Usage(data).prompt_tokens == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            param({"completion_tokens": 1, "output_tokens": 2}, 1, id="completion_first"),
            param({"output_tokens": 2, "candidatesTokenCount": 3}, 2, id="output_second"),
            param({"candidatesTokenCount": 3, "outputTokens": 4}, 3, id="gemini_third"),
            param({"outputTokens": 4}, 4, id="bedrock_fallback"),
        ],
    )  # fmt: skip
    def test_completion_tokens_precedence(self, data, expected):
        assert Usage(data).completion_tokens == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            # CACHE_READ_TOP_LEVEL_KEYS order:
            # cache_read_input_tokens > prompt_cache_hit_tokens > cachedContentTokenCount > cacheReadInputTokens
            param({"cache_read_input_tokens": 1, "prompt_cache_hit_tokens": 2}, 1,
                  id="anthropic_beats_deepseek"),
            param({"prompt_cache_hit_tokens": 2, "cachedContentTokenCount": 3}, 2,
                  id="deepseek_beats_gemini"),
            param({"cachedContentTokenCount": 3, "cacheReadInputTokens": 4}, 3,
                  id="gemini_beats_bedrock"),
        ],
    )  # fmt: skip
    def test_cache_read_top_level_precedence(self, data, expected):
        assert Usage(data).prompt_cache_read_tokens == expected

    def test_nested_cache_read_beats_top_level(self):
        """OpenAI nested prompt_tokens_details.cached_tokens beats every
        top-level synonym (nested wins for backwards-compat with the existing
        OpenAI-shape contract)."""
        usage = Usage(
            {
                "prompt_tokens_details": {"cached_tokens": 7},
                "cache_read_input_tokens": 99,
                "prompt_cache_hit_tokens": 88,
                "cachedContentTokenCount": 77,
                "cacheReadInputTokens": 66,
            }
        )
        assert usage.prompt_cache_read_tokens == 7

    def test_nested_input_tokens_details_beats_completion_tokens_details_for_prompt(
        self,
    ):
        """input_tokens_details (Anthropic-style nested) is in PROMPT_DETAILS_KEYS
        and wins for prompt_audio_tokens lookup."""
        usage = Usage(
            {
                "prompt_tokens_details": {"audio_tokens": 1},
                "input_tokens_details": {"audio_tokens": 99},
            }
        )
        # prompt_tokens_details is FIRST in PROMPT_DETAILS_KEYS
        assert usage.prompt_audio_tokens == 1

    def test_completion_details_precedence_for_reasoning(self):
        usage = Usage(
            {
                "completion_tokens_details": {"reasoning_tokens": 1},
                "output_tokens_details": {"reasoning_tokens": 99},
            }
        )
        assert usage.reasoning_tokens == 1

    def test_nested_reasoning_beats_gemini_top_level(self):
        usage = Usage(
            {
                "completion_tokens_details": {"reasoning_tokens": 5},
                "thoughtsTokenCount": 200,
            }
        )
        assert usage.reasoning_tokens == 5


class TestUsageMutability:
    """Behavior under post-construction mutation, copies, and serialization."""

    def test_mutation_after_construction_propagates_to_property(self):
        usage = Usage({"prompt_tokens": 10})
        usage["prompt_tokens"] = 20
        assert usage.prompt_tokens == 20

    def test_post_hoc_added_synonym_picked_up(self):
        usage = Usage({})
        assert usage.prompt_tokens is None
        usage["promptTokenCount"] = 100
        assert usage.prompt_tokens == 100

    def test_post_hoc_envelope_mutation_does_NOT_re_normalize(self):
        """Normalization is one-shot at __init__. Adding usageMetadata after
        construction does NOT lift its keys — document this contract."""
        usage = Usage({})
        usage["usageMetadata"] = {"promptTokenCount": 10}
        # promptTokenCount was never lifted to top-level
        assert "promptTokenCount" not in usage
        # And the property won't find it because the synonym list reads top-level
        assert usage.prompt_tokens is None

    def test_construct_from_another_usage(self):
        original = Usage({"prompt_tokens": 10, "completion_tokens": 5})
        derived = Usage(original)
        assert derived.prompt_tokens == 10
        assert derived.completion_tokens == 5
        # Mutation isolation — derived is a separate dict
        derived["prompt_tokens"] = 999
        assert original.prompt_tokens == 10

    def test_construct_from_usage_re_runs_normalization(self):
        """If the source Usage was constructed from a Gemini envelope, the
        derived Usage re-normalizes — but since the source was already
        normalized, this is a no-op."""
        source = Usage({"usageMetadata": {"promptTokenCount": 10}})
        derived = Usage(source)
        assert derived.prompt_tokens == 10

    def test_dict_copy_returns_dict_not_usage(self):
        """`.copy()` on dict subclasses returns a plain dict — known Python
        behavior. Properties are LOST on the copy."""
        usage = Usage({"prompt_tokens": 10})
        plain = usage.copy()
        assert type(plain) is dict
        assert plain == {"prompt_tokens": 10}

    def test_copy_module_copy_preserves_type(self):
        """copy.copy uses __class__ correctly for dict subclasses."""
        usage = Usage({"prompt_tokens": 10})
        cloned = copy.copy(usage)
        assert isinstance(cloned, Usage)
        assert cloned.prompt_tokens == 10

    def test_deepcopy_preserves_type_and_isolates_nested(self):
        original = Usage(
            {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 5}}
        )
        cloned = copy.deepcopy(original)
        assert isinstance(cloned, Usage)
        # Mutate nested in clone — original must not change
        cloned["prompt_tokens_details"]["cached_tokens"] = 999
        assert original["prompt_tokens_details"]["cached_tokens"] == 5

    def test_pickle_round_trip_preserves_type(self):
        original = Usage(VENDOR_FIXTURES["openai_gpt4o_with_caching"])
        round_tripped = pickle.loads(pickle.dumps(original))
        assert isinstance(round_tripped, Usage)
        assert round_tripped.prompt_tokens == 2006
        assert round_tripped.prompt_cache_read_tokens == 1920

    def test_json_round_trip_loses_type_but_preserves_content(self):
        """json.loads returns a plain dict; document this so callers know to
        re-wrap as Usage if they need the properties."""
        original = Usage({"prompt_tokens": 10})
        round_tripped = json.loads(json.dumps(original))
        assert type(round_tripped) is dict
        assert round_tripped == {"prompt_tokens": 10}
        # Re-wrapping restores the properties
        assert Usage(round_tripped).prompt_tokens == 10

    def test_json_serializable_with_orjson_compatible_payloads(self):
        """All vendor fixtures must round-trip through json.dumps without
        errors — a regression here means we accidentally added a non-JSON
        type to the dict."""
        for name, payload in VENDOR_FIXTURES.items():
            usage = Usage(payload)
            # Must not raise
            serialized = json.dumps(usage)
            assert isinstance(serialized, str), f"failed: {name}"


class TestUsagePropertyDeterminism:
    """Properties are pure functions of the dict; no caching, no side effects."""

    def test_repeated_reads_return_same_value(self):
        usage = Usage({"prompt_tokens": 10})
        results = [usage.prompt_tokens for _ in range(100)]
        assert all(r == 10 for r in results)

    def test_property_is_not_memoized(self):
        usage = Usage({"prompt_tokens": 10})
        first = usage.prompt_tokens
        usage["prompt_tokens"] = 99
        second = usage.prompt_tokens
        assert first == 10
        assert second == 99

    def test_reading_property_does_not_mutate_dict(self):
        usage = Usage({"prompt_tokens": 10})
        keys_before = set(usage.keys())
        _ = usage.prompt_tokens
        _ = usage.completion_tokens
        _ = usage.prompt_cache_read_tokens
        _ = usage.tool_use_prompt_tokens
        keys_after = set(usage.keys())
        assert keys_before == keys_after


class TestUsageCrossVendorMixedShapes:
    """Defensive coverage for response payloads that mix vendor shapes
    (e.g., a proxy that translates between providers and emits both forms)."""

    def test_openai_nested_and_anthropic_top_level_for_cache_read(self):
        usage = Usage(
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 7},
                "cache_read_input_tokens": 99,
            }
        )
        # Nested wins (OpenAI shape is the historical baseline)
        assert usage.prompt_cache_read_tokens == 7

    def test_camelcase_and_snake_case_for_basic_tokens(self):
        usage = Usage(
            {
                "prompt_tokens": 1,
                "promptTokenCount": 2,
                "inputTokens": 3,
            }
        )
        assert usage.prompt_tokens == 1

    def test_anthropic_top_level_with_openai_details_for_writes(self):
        """If a payload has Anthropic-style writes top-level AND OpenAI-style
        nested, write reads top-level (OpenAI never has writes)."""
        usage = Usage(
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 50},
                "cache_creation_input_tokens": 1024,
            }
        )
        assert usage.prompt_cache_write_tokens == 1024
        assert usage.prompt_cache_read_tokens == 50  # nested wins for reads


# Streaming metric coverage. Streaming responses report cumulative usage
# fields per chunk; AIPerf metrics walk responses backwards and take the
# first (last-emitted) non-None usage value.


def _record_with_response_usages(*usages) -> ParsedResponseRecord:
    """Build a ParsedResponseRecord whose responses carry the given usages,
    in order. Pass a ParsedResponse-compatible dict or None per chunk."""
    request = RequestRecord(
        conversation_id="test",
        turn_index=0,
        model_name="m",
        start_perf_ns=100,
        timestamp_ns=100,
        end_perf_ns=200,
    )
    responses = []
    for i, usage_dict in enumerate(usages):
        responses.append(
            ParsedResponse(
                perf_ns=100 + i,
                data=TextResponseData(text=f"chunk{i}"),
                usage=Usage(usage_dict) if usage_dict is not None else None,
            )
        )
    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(input=0, output=0, reasoning=0),
    )


class TestStreamingMetricEdgeCases:
    """Adversarial streaming behavior across mixed-shape chunks."""

    def test_only_last_chunk_has_usage(self):
        record = _record_with_response_usages(None, None, None, {"prompt_tokens": 50})
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 50

    def test_only_middle_chunk_has_usage(self):
        """The "last non-None" walks backwards, so a middle-only chunk wins."""
        record = _record_with_response_usages(None, {"prompt_tokens": 42}, None, None)
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 42

    def test_all_chunks_none_raises(self):
        record = _record_with_response_usages(None, None, None)
        with pytest.raises(NoMetricValue):
            UsagePromptTokensMetric().parse_record(record, MetricRecordDict())

    def test_no_responses_raises(self):
        record = _record_with_response_usages()
        with pytest.raises(NoMetricValue):
            UsagePromptTokensMetric().parse_record(record, MetricRecordDict())

    def test_cumulative_increasing_returns_last(self):
        """Streaming chunks typically report cumulative totals; we take the
        last (largest)."""
        record = _record_with_response_usages(
            {"prompt_tokens": 10},
            {"prompt_tokens": 20},
            {"prompt_tokens": 30},
        )
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 30

    def test_last_chunk_decreasing_value_is_returned_as_is(self):
        """If a vendor reports DECREASING values across chunks (invalid but
        not impossible), we still take the last — we don't validate."""
        record = _record_with_response_usages(
            {"prompt_tokens": 100},
            {"prompt_tokens": 50},
        )
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 50

    def test_mixed_vendor_shapes_across_chunks(self):
        """If a hypothetical proxy emits OpenAI shape then Anthropic shape
        across chunks, `record.final_usage` returns the LAST non-empty chunk's
        Usage object as-is (no merging). The metric reads through synonym
        precedence on that last chunk only, so chunk 2's `input_tokens=20`
        resolves as `usage.prompt_tokens=20`.

        Real vendors don't change shape mid-stream, so this is purely
        documenting the contract for the synthetic translating-proxy case.
        """
        record = _record_with_response_usages(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"input_tokens": 20, "output_tokens": 7},
        )
        # Last chunk wins; its `input_tokens` resolves via synonym to prompt_tokens.
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 20
        assert (
            UsageCompletionTokensMetric().parse_record(record, MetricRecordDict()) == 7
        )

    def test_late_chunk_zero_is_preferred_over_earlier_nonzero(self):
        """0 is a valid value, not a "missing" sentinel — the last chunk wins
        even when it's 0."""
        record = _record_with_response_usages(
            {"prompt_tokens": 100},
            {"prompt_tokens": 0},
        )
        assert UsagePromptTokensMetric().parse_record(record, MetricRecordDict()) == 0

    def test_explicit_none_in_last_chunk_does_not_fall_back(self):
        """If the last chunk explicitly sets prompt_tokens=None (a synthetic
        case no real vendor produces), the metric raises NoMetricValue —
        `record.final_usage` returns the last non-empty Usage as-is, and
        that Usage has prompt_tokens=None. We do NOT walk back per-field
        looking for a non-None value in earlier chunks.

        Documenting this as a contract: vendors don't null fields they had
        previously set; the simpler "last non-empty chunk wins" semantic is
        what we ship.
        """
        record = _record_with_response_usages(
            {"prompt_tokens": 50},
            {"prompt_tokens": None, "completion_tokens": 5},
        )
        with pytest.raises(NoMetricValue):
            UsagePromptTokensMetric().parse_record(record, MetricRecordDict())
        # completion_tokens IS present and non-None on the last chunk → 5.
        assert (
            UsageCompletionTokensMetric().parse_record(record, MetricRecordDict()) == 5
        )

    def test_cache_read_streaming_with_shape_change(self):
        """Cache read should be detected even if the LAST chunk's shape
        differs from earlier chunks' shapes."""
        record = _record_with_response_usages(
            {"prompt_tokens": 100},  # no caching info
            {"prompt_tokens_details": {"cached_tokens": 50}},  # OpenAI nested
        )
        assert (
            UsagePromptCacheReadTokensMetric().parse_record(record, MetricRecordDict())
            == 50
        )


class TestMetricsAcrossAllFixtures:
    """End-to-end: every vendor fixture must produce sensible metric values
    or correctly raise NoMetricValue."""

    @pytest.mark.parametrize(
        "fixture_name",
        list(VENDOR_FIXTURES.keys()),
        ids=list(VENDOR_FIXTURES.keys()),
    )
    def test_total_tokens_metric_is_either_extractable_or_absent(self, fixture_name):
        """Every fixture either has total_tokens or raises NoMetricValue
        — never crashes with anything else."""
        record = _record_with_response_usages(VENDOR_FIXTURES[fixture_name])
        try:
            value = UsageTotalTokensMetric().parse_record(record, MetricRecordDict())
            assert value is not None
        except NoMetricValue:
            pass

    def test_anthropic_no_caching_cache_metrics_all_raise(self):
        record = _record_with_response_usages(
            VENDOR_FIXTURES["anthropic_claude_no_caching"]
        )
        for metric_cls in (
            UsagePromptCacheReadTokensMetric,
            UsagePromptCacheWriteTokensMetric,
            UsagePromptCacheMissTokensMetric,
        ):
            with pytest.raises(NoMetricValue):
                metric_cls().parse_record(record, MetricRecordDict())

    def test_openai_basic_audio_seconds_raises(self):
        """OpenAI doesn't surface prompt_audio_seconds — Mistral-only field."""
        record = _record_with_response_usages(VENDOR_FIXTURES["openai_gpt4o_basic"])
        with pytest.raises(NoMetricValue):
            UsagePromptAudioSecondsMetric().parse_record(record, MetricRecordDict())

    def test_openai_basic_tool_use_raises(self):
        """OpenAI folds tool definitions into prompt_tokens; no separate field."""
        record = _record_with_response_usages(VENDOR_FIXTURES["openai_gpt4o_basic"])
        with pytest.raises(NoMetricValue):
            UsageToolUsePromptTokensMetric().parse_record(record, MetricRecordDict())

    def test_gemini_basic_reasoning_raises(self):
        """Plain Gemini Flash without thinking has no thoughtsTokenCount."""
        record = _record_with_response_usages(VENDOR_FIXTURES["gemini_2_flash_basic"])
        with pytest.raises(NoMetricValue):
            UsageReasoningTokensMetric().parse_record(record, MetricRecordDict())

    def test_gemini_thinking_reasoning_extracts(self):
        record = _record_with_response_usages(VENDOR_FIXTURES["gemini_with_thinking"])
        assert (
            UsageReasoningTokensMetric().parse_record(record, MetricRecordDict()) == 200
        )

    def test_deepseek_invariant_prompt_equals_hit_plus_miss(self):
        """DeepSeek's prompt_tokens should equal cache_hit + cache_miss
        — verify our metrics compose correctly."""
        record = _record_with_response_usages(VENDOR_FIXTURES["deepseek_v3_chat"])
        prompt = UsagePromptTokensMetric().parse_record(record, MetricRecordDict())
        hit = UsagePromptCacheReadTokensMetric().parse_record(
            record, MetricRecordDict()
        )
        miss = UsagePromptCacheMissTokensMetric().parse_record(
            record, MetricRecordDict()
        )
        assert prompt == hit + miss


class TestFinalUsageDirectAccess:
    """Direct tests on `record.final_usage` without going through metrics."""

    def test_no_responses_returns_none(self):
        record = _record_with_response_usages()
        assert record.final_usage is None

    def test_single_response_with_usage_returns_it(self):
        record = _record_with_response_usages({"prompt_tokens": 10})
        assert record.final_usage is not None
        assert record.final_usage["prompt_tokens"] == 10

    def test_single_response_without_usage_returns_none(self):
        record = _record_with_response_usages(None)
        assert record.final_usage is None

    def test_all_chunks_none_returns_none(self):
        record = _record_with_response_usages(None, None, None, None, None)
        assert record.final_usage is None

    def test_returns_last_non_empty_chunk(self):
        record = _record_with_response_usages(
            {"prompt_tokens": 1},
            {"prompt_tokens": 2},
            {"prompt_tokens": 3},
        )
        assert record.final_usage["prompt_tokens"] == 3

    def test_returns_last_chunk_even_when_earlier_were_richer(self):
        """If a richer chunk precedes a sparser non-empty chunk, the sparser
        one still wins — we don't merge."""
        record = _record_with_response_usages(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            {"prompt_tokens": 10},
        )
        usage = record.final_usage
        assert usage["prompt_tokens"] == 10
        assert "completion_tokens" not in usage
        assert "total_tokens" not in usage

    def test_skips_trailing_none_chunks(self):
        record = _record_with_response_usages(
            {"prompt_tokens": 42},
            None,
            None,
            None,
        )
        assert record.final_usage["prompt_tokens"] == 42

    def test_skips_only_trailing_nones_and_finds_middle(self):
        record = _record_with_response_usages(
            None,
            {"prompt_tokens": 99},
            None,
        )
        assert record.final_usage["prompt_tokens"] == 99

    def test_returns_a_usage_instance(self):
        record = _record_with_response_usages({"prompt_tokens": 1})
        assert isinstance(record.final_usage, Usage)

    def test_empty_usage_dict_treated_as_no_usage(self):
        """`Usage({})` is falsy (empty dict), so the walkback skips it
        as if it had no usage at all. Document this contract."""
        record = _record_with_response_usages({})
        assert record.final_usage is None

    def test_empty_usage_among_nonempty_skipped(self):
        """An empty Usage is not 'a chunk that reported usage' — it's skipped."""
        record = _record_with_response_usages(
            {"prompt_tokens": 50},
            {},
            {},
        )
        assert record.final_usage["prompt_tokens"] == 50


class TestFinalUsageCaching:
    """`final_usage` is a `@cached_property` — verify caching contract."""

    def test_repeated_access_returns_same_object(self):
        record = _record_with_response_usages({"prompt_tokens": 1})
        first = record.final_usage
        second = record.final_usage
        # Identity, not just equality — cached_property stores in __dict__.
        assert first is second

    def test_cached_value_persists_across_100_reads(self):
        record = _record_with_response_usages({"prompt_tokens": 1})
        results = [record.final_usage for _ in range(100)]
        assert all(r is results[0] for r in results)

    def test_cache_does_not_recompute_after_responses_mutated(self):
        """cached_property snapshots on first access; later mutations to
        `responses` are NOT reflected. Document this contract."""
        record = _record_with_response_usages({"prompt_tokens": 1})
        first = record.final_usage
        # Mutate the responses list after caching
        record.responses.append(
            ParsedResponse(
                perf_ns=999,
                data=TextResponseData(text="late"),
                usage=Usage({"prompt_tokens": 999}),
            )
        )
        assert record.final_usage is first
        assert record.final_usage["prompt_tokens"] == 1

    def test_cache_invalidation_via_dict_pop(self):
        """The standard cached_property invalidation pattern is
        `del instance.attr` or `del instance.__dict__["attr"]`. Verify it
        recomputes on the next access."""
        record = _record_with_response_usages({"prompt_tokens": 1})
        _ = record.final_usage
        record.responses.append(
            ParsedResponse(
                perf_ns=999,
                data=TextResponseData(text="late"),
                usage=Usage({"prompt_tokens": 999}),
            )
        )
        # Force invalidation
        del record.__dict__["final_usage"]
        assert record.final_usage["prompt_tokens"] == 999

    def test_cached_none_value_does_not_recompute(self):
        record = _record_with_response_usages(None, None)
        assert record.final_usage is None
        record.responses.append(
            ParsedResponse(
                perf_ns=999,
                data=TextResponseData(text="late"),
                usage=Usage({"prompt_tokens": 5}),
            )
        )
        # Still None — cache holds the first computed value, even though it was None.
        assert record.final_usage is None


class TestFinalUsageCrossRecordIsolation:
    """Each ParsedResponseRecord has its own cached final_usage."""

    def test_two_records_compute_independently(self):
        a = _record_with_response_usages({"prompt_tokens": 1})
        b = _record_with_response_usages({"prompt_tokens": 2})
        assert a.final_usage["prompt_tokens"] == 1
        assert b.final_usage["prompt_tokens"] == 2

    def test_caching_one_record_does_not_affect_another(self):
        a = _record_with_response_usages({"prompt_tokens": 1})
        b = _record_with_response_usages({"prompt_tokens": 2})
        # Read a first
        _ = a.final_usage
        # Now read b — must still be 2, not borrowed from a's cache
        assert b.final_usage["prompt_tokens"] == 2

    def test_records_with_same_data_have_distinct_cached_objects(self):
        a = _record_with_response_usages({"prompt_tokens": 1})
        b = _record_with_response_usages({"prompt_tokens": 1})
        assert a.final_usage is not b.final_usage
        assert a.final_usage == b.final_usage  # Equal dicts


class TestUsageInheritance:
    """Usage is a dict subclass; user-defined subclasses should still work."""

    def test_simple_subclass_construction(self):
        class MyUsage(Usage):
            pass

        u = MyUsage({"prompt_tokens": 10})
        assert isinstance(u, Usage)
        assert isinstance(u, MyUsage)
        assert u.prompt_tokens == 10

    def test_subclass_can_add_property(self):
        class MyUsage(Usage):
            @property
            def custom_field(self) -> int | None:
                return self.get("custom")

        u = MyUsage({"prompt_tokens": 10, "custom": 42})
        assert u.prompt_tokens == 10
        assert u.custom_field == 42

    def test_subclass_envelope_normalization_inherited(self):
        class MyUsage(Usage):
            pass

        u = MyUsage({"usageMetadata": {"promptTokenCount": 7}})
        assert u.prompt_tokens == 7


class TestUsageDictSemantics:
    """Usage is a dict — verify standard dict equality/repr/hash behavior."""

    def test_equality_with_plain_dict(self):
        assert Usage({"a": 1}) == {"a": 1}
        assert Usage({"a": 1}) == {"a": 1}

    def test_equality_with_other_usage(self):
        assert Usage({"a": 1}) == Usage({"a": 1})

    def test_inequality(self):
        assert Usage({"a": 1}) != Usage({"a": 2})
        assert Usage({"a": 1}) != Usage({"b": 1})

    def test_equality_after_envelope_normalization(self):
        """A Gemini envelope and an already-flattened equivalent compare equal
        on the post-normalization dict — but the original Usage retains the
        envelope key so they're NOT equal as plain dicts."""
        wrapped = Usage({"usageMetadata": {"promptTokenCount": 10}})
        flat = Usage({"promptTokenCount": 10})
        # wrapped also retains its envelope key
        assert "usageMetadata" in wrapped
        assert wrapped != flat

    def test_repr_is_dict_like(self):
        u = Usage({"prompt_tokens": 10})
        # Just verify repr doesn't crash and contains the data
        r = repr(u)
        assert "prompt_tokens" in r
        assert "10" in r

    def test_not_hashable(self):
        """dict subclasses inherit dict's unhashability — Usage is not hashable."""
        with pytest.raises(TypeError):
            hash(Usage({"prompt_tokens": 10}))

    def test_iteration_order_preserved(self):
        """Python dicts preserve insertion order; Usage inherits this."""
        u = Usage({"z": 1, "a": 2, "m": 3})
        assert list(u.keys()) == ["z", "a", "m"]

    def test_iteration_includes_envelope_lifted_keys(self):
        """Lifted keys appear after originals in iteration order."""
        u = Usage({"existing": 1, "usageMetadata": {"promptTokenCount": 10}})
        # 'existing' and 'usageMetadata' are original; promptTokenCount was lifted.
        keys = list(u.keys())
        assert "existing" in keys
        assert "usageMetadata" in keys
        assert "promptTokenCount" in keys


class TestRealJSONRoundTrip:
    """Round-trip from raw JSON bytes (as a wire format) through orjson + Usage."""

    @pytest.mark.parametrize(
        "fixture_name",
        list(VENDOR_FIXTURES.keys()),
        ids=list(VENDOR_FIXTURES.keys()),
    )
    def test_round_trip_via_orjson(self, fixture_name):
        import orjson

        original = VENDOR_FIXTURES[fixture_name]
        raw = orjson.dumps(original)
        decoded = orjson.loads(raw)
        usage = Usage(decoded)
        # The full original dict is preserved
        assert dict(usage)  # not None or empty
        # And token counts (where present) match
        for top_level_key in ("prompt_tokens", "input_tokens"):
            if top_level_key in original:
                assert usage.prompt_tokens == original[top_level_key]
                break

    def test_unicode_keys_pass_through(self):
        """Unicode in keys/values must not break envelope normalization."""
        usage = Usage(
            {
                "prompt_tokens": 100,
                "用户标签": "测试",  # arbitrary unicode key/value
                "metadata": {"emoji_field_😀": "value"},
            }
        )
        assert usage.prompt_tokens == 100
        assert usage["用户标签"] == "测试"

    def test_very_deeply_nested_user_metadata_passes_through(self):
        """Usage doesn't recursively touch unrecognized fields, so deeply
        nested user metadata survives intact."""
        deep = {"a": {"b": {"c": {"d": {"e": [1, 2, 3]}}}}}
        usage = Usage({"prompt_tokens": 10, "metadata": deep})
        assert usage["metadata"] == deep


class TestMoreVendorVariants:
    """Additional vendor variants beyond the core fixture set."""

    def test_openai_realtime_audio_io(self):
        """OpenAI Realtime API: both prompt audio AND completion audio,
        plus standard text token counts."""
        usage = Usage(
            {
                "total_tokens": 250,
                "prompt_tokens": 100,
                "completion_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "text_tokens": 60,
                    "audio_tokens": 40,
                },
                "completion_tokens_details": {
                    "text_tokens": 100,
                    "audio_tokens": 50,
                },
            }
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 150
        assert usage.prompt_audio_tokens == 40
        assert usage.completion_audio_tokens == 50

    def test_openai_batch_api_shape(self):
        """OpenAI batch responses have the same usage shape as sync."""
        usage = Usage(
            {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275}
        )
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 75
        assert usage.total_tokens == 275

    def test_anthropic_with_streaming_message_delta(self):
        """Anthropic streaming emits usage in `message_delta` events; the
        usage dict still has the same shape."""
        usage = Usage(
            {
                "input_tokens": 0,  # often 0 in deltas; full count in initial event
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        )
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 50
        # 0 cache values are valid (not missing)
        assert usage.prompt_cache_read_tokens == 0
        assert usage.prompt_cache_write_tokens == 0

    def test_groq_openai_compatible(self):
        """Groq's OpenAI-compatible API; adds queue_time / prompt_time fields
        that we should preserve verbatim."""
        usage = Usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "queue_time": 0.0123,
                "prompt_time": 0.045,
                "completion_time": 0.789,
                "total_time": 0.846,
            }
        )
        assert usage.prompt_tokens == 100
        assert usage["queue_time"] == 0.0123
        assert usage["completion_time"] == 0.789

    def test_together_ai_openai_compatible(self):
        """Together AI uses OpenAI-compatible shape."""
        usage = Usage(
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        assert usage.total_tokens == 30

    def test_fireworks_openai_compatible(self):
        """Fireworks uses OpenAI-compatible shape."""
        usage = Usage(
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        assert usage.total_tokens == 30

    def test_azure_openai_passthrough(self):
        """Azure OpenAI mirrors OpenAI's shape exactly."""
        usage = Usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 30},
            }
        )
        assert usage.prompt_cache_read_tokens == 30

    def test_tgi_input_output_tokens(self):
        """TGI (Hugging Face Text Generation Inference) emits OpenAI-like
        with input_tokens/output_tokens as the modern field names."""
        usage = Usage({"input_tokens": 42, "output_tokens": 17, "total_tokens": 59})
        assert usage.prompt_tokens == 42
        assert usage.completion_tokens == 17
        assert usage.total_tokens == 59

    def test_vllm_with_explicit_none_prompt_logprobs(self):
        """vLLM may include `prompt_logprobs: null` alongside usage; that key
        is preserved on the dict but doesn't affect token-count properties."""
        usage = Usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_logprobs": None,
            }
        )
        assert usage.prompt_tokens == 100
        assert "prompt_logprobs" in usage

    def test_anthropic_messages_count_tokens_endpoint(self):
        """Anthropic's `count_tokens` helper endpoint returns just one field."""
        usage = Usage({"input_tokens": 100})
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens is None  # not present
        assert usage.total_tokens is None

    def test_gemini_with_modality_breakdown(self):
        """Gemini's response can include `*ModalityTokenCount` arrays for
        multimodal inputs. They pass through unmodelled but accessible."""
        usage = Usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 80},
                        {"modality": "IMAGE", "tokenCount": 20},
                    ],
                }
            }
        )
        assert usage.prompt_tokens == 100
        # The list is preserved on the dict for advanced consumers
        assert usage["promptTokensDetails"][0]["modality"] == "TEXT"

    def test_replicate_openai_proxy(self):
        """Replicate's OpenAI proxy mirrors OpenAI shape with the addition
        of `id` and other Replicate-specific fields."""
        usage = Usage(
            {
                "prompt_tokens": 25,
                "completion_tokens": 30,
                "total_tokens": 55,
            }
        )
        assert usage.prompt_tokens == 25


class TestPropertyBasedInvariants:
    """Hypothesis-driven property tests over random Usage shapes.

    These don't replace the explicit fixtures — they catch surprises in
    interactions between envelope normalization, synonym precedence, and
    nested-dict lookups that wouldn't occur to a human writing examples.
    """

    @given(value=st.integers(min_value=-(2**62), max_value=2**62))
    def test_prompt_tokens_returns_what_was_set(self, value):
        usage = Usage({"prompt_tokens": value})
        assert usage.prompt_tokens == value

    @given(value=st.integers(min_value=0, max_value=10**12))
    def test_cache_read_either_synonym_resolves(self, value):
        # OpenAI nested
        u_nested = Usage({"prompt_tokens_details": {"cached_tokens": value}})
        assert u_nested.prompt_cache_read_tokens == value
        # Anthropic top-level
        u_top = Usage({"cache_read_input_tokens": value})
        assert u_top.prompt_cache_read_tokens == value
        # DeepSeek top-level
        u_ds = Usage({"prompt_cache_hit_tokens": value})
        assert u_ds.prompt_cache_read_tokens == value
        # Gemini envelope
        u_gem = Usage({"usageMetadata": {"cachedContentTokenCount": value}})
        assert u_gem.prompt_cache_read_tokens == value
        # Bedrock camelCase
        u_br = Usage({"cacheReadInputTokens": value})
        assert u_br.prompt_cache_read_tokens == value

    @given(
        prompt=st.integers(min_value=0, max_value=10**6),
        completion=st.integers(min_value=0, max_value=10**6),
    )
    def test_envelope_unwrap_preserves_token_counts(self, prompt, completion):
        usage = Usage(
            {
                "usageMetadata": {
                    "promptTokenCount": prompt,
                    "candidatesTokenCount": completion,
                }
            }
        )
        assert usage.prompt_tokens == prompt
        assert usage.completion_tokens == completion

    @given(
        extras=st.dictionaries(
            keys=st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
                min_size=1,
                max_size=20,
            ),
            values=st.integers(min_value=-100, max_value=100),
            max_size=10,
        )
    )
    def test_random_top_level_keys_dont_affect_known_properties(self, extras):
        # Strip any known synonym keys to keep the test honest
        synonym_keys = (
            Usage.PROMPT_TOKENS_KEYS
            + Usage.COMPLETION_TOKENS_KEYS
            + Usage.TOTAL_TOKENS_KEYS
            + Usage.CACHE_READ_TOP_LEVEL_KEYS
            + Usage.CACHE_WRITE_TOP_LEVEL_KEYS
            + Usage.CACHE_MISS_TOP_LEVEL_KEYS
            + Usage.REASONING_TOP_LEVEL_KEYS
            + Usage.TOOL_USE_PROMPT_KEYS
            + Usage.PROMPT_AUDIO_SECONDS_KEYS
        )
        clean = {k: v for k, v in extras.items() if k not in synonym_keys}
        usage = Usage({"prompt_tokens": 42, **clean})
        # Random extras don't affect the known property
        assert usage.prompt_tokens == 42
        # And every random key is preserved
        for k, v in clean.items():
            assert usage[k] == v

    @given(usage_dict=st.dictionaries(keys=st.text(), values=st.integers(), max_size=5))
    def test_construction_never_raises_on_str_int_dicts(self, usage_dict):
        """Any str→int dict must construct without raising."""
        Usage(usage_dict)  # must not raise

    @given(
        chunk_count=st.integers(min_value=1, max_value=10),
        last_value=st.integers(min_value=0, max_value=10**6),
    )
    def test_final_usage_returns_last_chunks_value(self, chunk_count, last_value):
        """For any chunk count ≥ 1 with all-non-empty cumulative chunks,
        `final_usage` reflects the last chunk's value."""
        chunks = [{"prompt_tokens": i} for i in range(chunk_count)]
        chunks[-1] = {"prompt_tokens": last_value}
        record = _record_with_response_usages(*chunks)
        assert record.final_usage["prompt_tokens"] == last_value


class TestTotalMetricEndToEnd:
    """Verify DerivedSumMetric totals correctly aggregate across multiple
    records, each with its own merged final_usage."""

    def _make_records(self, prompt_values: list[int]) -> list[ParsedResponseRecord]:
        return [
            _record_with_response_usages({"prompt_tokens": v}) for v in prompt_values
        ]

    def test_total_aggregates_correctly_across_records(self):
        records = self._make_records([10, 20, 30])
        # Per-record metric values
        per_record = [
            UsagePromptTokensMetric().parse_record(r, MetricRecordDict())
            for r in records
        ]
        assert per_record == [10, 20, 30]
        assert sum(per_record) == 60

    def test_total_handles_record_with_missing_field(self):
        """A record where the metric raises NoMetricValue should not crash
        the per-record extract — and the consumer (DerivedSumMetric) is
        expected to skip those records."""
        good = _record_with_response_usages({"prompt_tokens": 10})
        missing = _record_with_response_usages(None)
        assert UsagePromptTokensMetric().parse_record(good, MetricRecordDict()) == 10
        with pytest.raises(NoMetricValue):
            UsagePromptTokensMetric().parse_record(missing, MetricRecordDict())

    def test_metric_extraction_uses_cached_final_usage(self):
        """Verify the cached_property is the one being read — re-extracting
        doesn't re-walk responses. Indirect proof: the same Usage instance
        is reused across metric calls on the same record."""
        record = _record_with_response_usages(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        # First metric access — caches final_usage
        first_usage = record.final_usage
        # Run several different metrics
        UsagePromptTokensMetric().parse_record(record, MetricRecordDict())
        UsageCompletionTokensMetric().parse_record(record, MetricRecordDict())
        UsageTotalTokensMetric().parse_record(record, MetricRecordDict())
        # Same cached object
        assert record.final_usage is first_usage


class TestSpecificPropertyEdges:
    """Targeted edge cases per Usage property."""

    def test_total_tokens_does_not_fall_through_to_sum(self):
        """`total_tokens` doesn't compute prompt + completion when missing —
        it just returns None. We don't synthesize values."""
        usage = Usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert usage.total_tokens is None

    def test_reasoning_tokens_nested_takes_precedence_over_top_level(self):
        """When both nested and Gemini-style top-level are present, nested
        (the OpenAI baseline) wins."""
        usage = Usage(
            {
                "completion_tokens_details": {"reasoning_tokens": 10},
                "thoughtsTokenCount": 999,
            }
        )
        assert usage.reasoning_tokens == 10

    def test_prompt_audio_seconds_returns_float_for_int_input(self):
        usage = Usage({"prompt_audio_seconds": 5})
        result = usage.prompt_audio_seconds
        assert result == 5.0
        assert isinstance(result, float)

    def test_prompt_audio_seconds_returns_none_if_missing(self):
        usage = Usage({"prompt_tokens": 10})
        assert usage.prompt_audio_seconds is None

    def test_tool_use_prompt_tokens_only_via_top_level(self):
        """tool_use_prompt_tokens has no nested fallback — only Gemini's
        toolUsePromptTokenCount counts."""
        usage_top = Usage({"toolUsePromptTokenCount": 5})
        assert usage_top.tool_use_prompt_tokens == 5
        # No fallback from prompt_tokens or anywhere else
        usage_no = Usage({"prompt_tokens": 100})
        assert usage_no.tool_use_prompt_tokens is None

    def test_cache_miss_only_via_top_level(self):
        """cache_miss is DeepSeek-only and has no nested fallback."""
        usage = Usage({"prompt_cache_miss_tokens": 25})
        assert usage.prompt_cache_miss_tokens == 25
        assert Usage({"prompt_tokens": 100}).prompt_cache_miss_tokens is None

    def test_cache_write_only_via_top_level(self):
        usage = Usage({"cache_creation_input_tokens": 1024})
        assert usage.prompt_cache_write_tokens == 1024
        # Bedrock variant
        usage_br = Usage({"cacheWriteInputTokens": 2048})
        assert usage_br.prompt_cache_write_tokens == 2048

    def test_completion_audio_tokens_under_output_tokens_details(self):
        """The Anthropic-style output_tokens_details synonym path must work
        for completion_audio_tokens."""
        usage = Usage({"output_tokens_details": {"audio_tokens": 50}})
        assert usage.completion_audio_tokens == 50

    def test_accepted_prediction_tokens_under_output_tokens_details(self):
        usage = Usage({"output_tokens_details": {"accepted_prediction_tokens": 100}})
        assert usage.accepted_prediction_tokens == 100

    def test_rejected_prediction_tokens_under_output_tokens_details(self):
        usage = Usage({"output_tokens_details": {"rejected_prediction_tokens": 30}})
        assert usage.rejected_prediction_tokens == 30


class TestRecordResponsesShapeEdges:
    """Edge cases on the responses list itself."""

    def test_record_with_zero_responses(self):
        record = _record_with_response_usages()
        assert record.final_usage is None
        with pytest.raises(NoMetricValue):
            UsagePromptTokensMetric().parse_record(record, MetricRecordDict())

    def test_record_with_one_hundred_chunks_only_last_has_usage(self):
        """Walkback from the end is fast — a long chain of None chunks
        followed by one with usage finds it on the first iteration."""
        chunks = [None] * 99 + [{"prompt_tokens": 42}]
        record = _record_with_response_usages(*chunks)
        assert record.final_usage["prompt_tokens"] == 42

    def test_record_with_one_hundred_chunks_only_first_has_usage(self):
        """The opposite: first chunk has usage, rest are None. Walkback
        traverses all 99 None responses before finding it."""
        chunks = [{"prompt_tokens": 42}] + [None] * 99
        record = _record_with_response_usages(*chunks)
        assert record.final_usage["prompt_tokens"] == 42

    def test_alternating_chunks_returns_last_non_empty(self):
        """Alternating pattern: last non-empty in iteration order wins."""
        chunks = [
            {"prompt_tokens": 1},
            None,
            {"prompt_tokens": 2},
            None,
            {"prompt_tokens": 3},
            None,
        ]
        record = _record_with_response_usages(*chunks)
        assert record.final_usage["prompt_tokens"] == 3

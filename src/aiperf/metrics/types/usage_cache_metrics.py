# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-record API usage field prompt-cache token metrics.

These track prompt tokens that participate in the API's prompt-caching
mechanism. The vendors expose this differently:

- OpenAI surfaces only cache reads, nested under
  prompt_tokens_details.cached_tokens (writes are transparent and free).
- Anthropic surfaces both reads and writes at the top level of usage,
  as cache_read_input_tokens and cache_creation_input_tokens
  (writes are billed at a +25% premium, reads at -90%).
- DeepSeek surfaces both reads (`prompt_cache_hit_tokens`) AND a separate
  miss count (`prompt_cache_miss_tokens`) at the top level — they bill
  hits and misses at different rates so the split is first-class.
- Google Gemini surfaces only reads, top-level as `cachedContentTokenCount`.
- AWS Bedrock mirrors Anthropic shape with camelCase top-level
  `cacheReadInputTokens` / `cacheWriteInputTokens`.

`Usage` normalizes the read / write / miss synonyms via the
`prompt_cache_read_tokens` / `prompt_cache_write_tokens` /
`prompt_cache_miss_tokens` properties. Each metric here is a thin
declarative subclass of `BaseUsageRecordMetric` reading one of those
properties from `record.final_usage`. Aggregated (sum-across-requests)
variants live in `usage_total_metrics.py`.
"""

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.metrics.base_usage_record_metric import BaseUsageRecordMetric


class UsagePromptCacheReadTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field prompt cache-read token count metric.

    Counts prompt tokens served from cache (cache hits). OpenAI surfaces this
    as prompt_tokens_details.cached_tokens (writes are transparent). Anthropic
    surfaces it at the top level as cache_read_input_tokens; cache writes are
    a separate metric (UsagePromptCacheWriteTokensMetric).

    Formula:
        Usage Prompt Cache Read Tokens = response.usage.prompt_cache_read_tokens (last non-None)
    """

    tag = "usage_prompt_cache_read_tokens"
    header = "Usage Prompt Cache Read Tokens"
    short_header = "Usage Prompt Cache Read"
    display_order = 1010
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_cache_read_tokens"
    missing_message = (
        "Usage prompt cache-read token count not available: no response had "
        "`prompt_tokens_details.cached_tokens`, "
        "`input_tokens_details.cached_tokens`, "
        "or top-level `cache_read_input_tokens`."
    )


class UsagePromptCacheWriteTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field prompt cache-write (cache creation) token count metric.

    Counts prompt tokens written to cache. Reported only by APIs that bill
    cache writes separately — Anthropic surfaces this at the top level as
    cache_creation_input_tokens. OpenAI does not surface writes, so this
    metric raises NoMetricValue for OpenAI-shaped responses.

    LARGER_IS_BETTER is intentionally omitted: writes cost more than ordinary
    input tokens but enable cheap reads on subsequent requests, so larger is
    not unambiguously better.

    Formula:
        Usage Prompt Cache Write Tokens = response.usage.prompt_cache_write_tokens (last non-None)
    """

    tag = "usage_prompt_cache_write_tokens"
    header = "Usage Prompt Cache Write Tokens"
    short_header = "Usage Prompt Cache Write"
    display_order = 1015
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_cache_write_tokens"
    missing_message = (
        "Usage prompt cache-write token count not available: no response "
        "had top-level `cache_creation_input_tokens` "
        "(this field is Anthropic-specific; OpenAI does not surface writes)."
    )


class UsagePromptCacheMissTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field prompt cache-miss token count metric.

    Counts prompt tokens that missed cache (and required fresh processing).
    DeepSeek surfaces this directly as top-level prompt_cache_miss_tokens —
    they bill hits and misses at different rates, so the split is first-class.
    Other vendors do not surface a separate miss count (it can be derived
    from prompt_tokens - prompt_cache_read_tokens, but not as its own field).

    Formula:
        Usage Prompt Cache Miss Tokens = response.usage.prompt_cache_miss_tokens (last non-None)
    """

    tag = "usage_prompt_cache_miss_tokens"
    header = "Usage Prompt Cache Miss Tokens"
    short_header = "Usage Prompt Cache Miss"
    display_order = 1017
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_cache_miss_tokens"
    missing_message = (
        "Usage prompt cache-miss token count not available: no response "
        "had top-level `prompt_cache_miss_tokens` "
        "(DeepSeek-specific; other vendors do not surface a separate miss count)."
    )

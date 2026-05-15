# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-record API usage metrics for vendor-specific concepts.

These metrics wrap fields that don't have an OpenAI-shape baseline equivalent
— each is currently surfaced by exactly one provider, but they're worth
modeling as first-class metrics so cross-provider benchmark comparisons can
include them where present:

- UsageToolUsePromptTokensMetric: Gemini's toolUsePromptTokenCount — tokens
  consumed by tool / function-call declarations, separate from user-content
  prompt tokens.
- UsagePromptAudioSecondsMetric: Mistral's prompt_audio_seconds — a duration,
  not a token count. Uses MetricTimeUnit.SECONDS, distinct from
  UsagePromptAudioTokensMetric.

Each metric is a thin declarative subclass of `BaseUsageRecordMetric` that
reads one property from `record.final_usage`. Aggregated (sum-across-requests)
variants live in `usage_total_metrics.py`.
"""

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    MetricTimeUnit,
)
from aiperf.metrics.base_usage_record_metric import BaseUsageRecordMetric


class UsageToolUsePromptTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field tool-use (function-call) prompt token count metric.

    Tokens spent on tool / function-call declarations sent in the request,
    separate from the user-content prompt tokens. Currently surfaced only by
    Google Gemini as top-level toolUsePromptTokenCount in usageMetadata.
    Other vendors fold tool definition tokens into the regular prompt_tokens
    count, so this metric raises NoMetricValue for OpenAI / Anthropic / etc.

    Formula:
        Usage Tool Use Prompt Tokens = response.usage.tool_use_prompt_tokens (last non-None)
    """

    tag = "usage_tool_use_prompt_tokens"
    header = "Usage Tool Use Prompt Tokens"
    short_header = "Usage Tool Prompt"
    display_order = 1030
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "tool_use_prompt_tokens"
    missing_message = (
        "Usage tool-use prompt token count not available: no response had "
        "`toolUsePromptTokenCount` (Gemini-specific; other vendors fold "
        "tool definitions into regular prompt_tokens)."
    )


class UsagePromptAudioSecondsMetric(BaseUsageRecordMetric[float]):
    """
    API usage field prompt audio duration metric (seconds, not tokens).

    Mistral surfaces audio-input duration as top-level prompt_audio_seconds —
    a duration, not a token count. Coexists with prompt_audio_tokens for
    frameworks that report both. This metric uses MetricTimeUnit.SECONDS;
    do NOT confuse with UsagePromptAudioTokensMetric.

    Formula:
        Usage Prompt Audio Seconds = response.usage.prompt_audio_seconds (last non-None)
    """

    tag = "usage_prompt_audio_seconds"
    header = "Usage Prompt Audio Seconds"
    short_header = "Usage Prompt Audio Sec"
    display_order = 1040
    unit = MetricTimeUnit.SECONDS
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.SUPPORTS_AUDIO_ONLY
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_audio_seconds"
    missing_message = (
        "Usage prompt audio seconds not available: no response had "
        "top-level `prompt_audio_seconds` "
        "(Mistral-specific; this is a duration, not a token count)."
    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated (sum-across-requests) API usage field token metrics.

These metrics derive from the per-record metrics in `usage_metrics.py` by
summing each metric's value across every request in the benchmark run.
"""

from aiperf.common.enums import MetricConsoleGroup
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
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
    UsageAcceptedPredictionTokensMetric,
    UsageCompletionAudioTokensMetric,
    UsageCompletionTokensMetric,
    UsagePromptAudioTokensMetric,
    UsagePromptTokensMetric,
    UsageReasoningTokensMetric,
    UsageRejectedPredictionTokensMetric,
    UsageTotalTokensMetric,
)


class TotalUsagePromptTokensMetric(DerivedSumMetric[int, UsagePromptTokensMetric]):
    """
    Total API-reported prompt tokens across all requests.

    Formula:
        ```
        Total Usage Prompt Tokens = Sum(Usage Prompt Tokens)
        ```
    """

    tag = "total_usage_prompt_tokens"
    header = "Total Usage Prompt Tokens"
    short_header = "Total Usage Prompt"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2000


class TotalUsageCompletionTokensMetric(
    DerivedSumMetric[int, UsageCompletionTokensMetric]
):
    """
    Total API-reported completion tokens across all requests.

    Formula:
        ```
        Total Usage Completion Tokens = Sum(Usage Completion Tokens)
        ```
    """

    tag = "total_usage_completion_tokens"
    header = "Total Usage Completion Tokens"
    short_header = "Total Usage Completion"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2100


class TotalUsageTokensMetric(DerivedSumMetric[int, UsageTotalTokensMetric]):
    """
    Total API-reported total tokens across all requests.

    Formula:
        ```
        Total Usage Total Tokens = Sum(Usage Total Tokens)
        ```
    """

    tag = "total_usage_total_tokens"
    header = "Total Usage Total Tokens"
    short_header = "Total Usage Total"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2200


class TotalUsageReasoningTokensMetric(
    DerivedSumMetric[int, UsageReasoningTokensMetric]
):
    """
    Total API-reported reasoning tokens across all requests.

    This sums the values reported in each response's `usage.reasoning_tokens`
    field. For the parser-derived equivalent (computed from
    `record.token_counts.reasoning`), see `TotalReasoningTokensMetric` in
    `metrics/types/reasoning_token_count.py`. The two will diverge whenever
    the server's reported usage disagrees with our own per-chunk counting.

    Formula:
        ```
        Total Usage Reasoning Tokens = Sum(Usage Reasoning Tokens)
        ```
    """

    tag = "total_usage_reasoning_tokens"
    header = "Total Usage Reasoning Tokens"
    short_header = "Total Usage Reasoning"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2110


class TotalUsagePromptCacheReadTokensMetric(
    DerivedSumMetric[int, UsagePromptCacheReadTokensMetric]
):
    """
    Total API-reported prompt cache-read tokens across all requests.

    Sums the per-request cache-read counts (OpenAI prompt_tokens_details
    .cached_tokens or Anthropic top-level cache_read_input_tokens).

    Formula:
        ```
        Total Usage Prompt Cache Read Tokens = Sum(Usage Prompt Cache Read Tokens)
        ```
    """

    tag = "total_usage_prompt_cache_read_tokens"
    header = "Total Usage Prompt Cache Read Tokens"
    short_header = "Total Usage Prompt Cache Read"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2010


class TotalUsagePromptCacheWriteTokensMetric(
    DerivedSumMetric[int, UsagePromptCacheWriteTokensMetric]
):
    """
    Total API-reported prompt cache-write (cache creation) tokens across all
    requests.

    Sums the per-request cache-write counts (Anthropic top-level
    cache_creation_input_tokens). Will be empty for OpenAI workloads since
    OpenAI does not surface cache writes.

    Formula:
        ```
        Total Usage Prompt Cache Write Tokens = Sum(Usage Prompt Cache Write Tokens)
        ```
    """

    tag = "total_usage_prompt_cache_write_tokens"
    header = "Total Usage Prompt Cache Write Tokens"
    short_header = "Total Usage Prompt Cache Write"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2015


class TotalUsagePromptAudioTokensMetric(
    DerivedSumMetric[int, UsagePromptAudioTokensMetric]
):
    """
    Total API-reported prompt audio tokens across all requests.

    Formula:
        ```
        Total Usage Prompt Audio Tokens = Sum(Usage Prompt Audio Tokens)
        ```
    """

    tag = "total_usage_prompt_audio_tokens"
    header = "Total Usage Prompt Audio Tokens"
    short_header = "Total Usage Prompt Audio"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2020


class TotalUsageCompletionAudioTokensMetric(
    DerivedSumMetric[int, UsageCompletionAudioTokensMetric]
):
    """
    Total API-reported completion audio tokens across all requests.

    Formula:
        ```
        Total Usage Completion Audio Tokens = Sum(Usage Completion Audio Tokens)
        ```
    """

    tag = "total_usage_completion_audio_tokens"
    header = "Total Usage Completion Audio Tokens"
    short_header = "Total Usage Comp Audio"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2120


class TotalUsageAcceptedPredictionTokensMetric(
    DerivedSumMetric[int, UsageAcceptedPredictionTokensMetric]
):
    """
    Total API-reported accepted prediction tokens across all requests.

    Formula:
        ```
        Total Usage Accepted Prediction Tokens = Sum(Usage Accepted Prediction Tokens)
        ```
    """

    tag = "total_usage_accepted_prediction_tokens"
    header = "Total Usage Accepted Prediction Tokens"
    short_header = "Total Usage Accepted Pred"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2130


class TotalUsageRejectedPredictionTokensMetric(
    DerivedSumMetric[int, UsageRejectedPredictionTokensMetric]
):
    """
    Total API-reported rejected prediction tokens across all requests.

    Formula:
        ```
        Total Usage Rejected Prediction Tokens = Sum(Usage Rejected Prediction Tokens)
        ```
    """

    tag = "total_usage_rejected_prediction_tokens"
    header = "Total Usage Rejected Prediction Tokens"
    short_header = "Total Usage Rejected Pred"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2140


class TotalUsagePromptCacheMissTokensMetric(
    DerivedSumMetric[int, UsagePromptCacheMissTokensMetric]
):
    """
    Total API-reported prompt cache-miss tokens across all requests
    (DeepSeek-specific).

    Sums the per-request cache-miss counts (DeepSeek's top-level
    prompt_cache_miss_tokens). Empty for vendors that don't surface a
    separate miss field.

    Formula:
        ```
        Total Usage Prompt Cache Miss Tokens = Sum(Usage Prompt Cache Miss Tokens)
        ```
    """

    tag = "total_usage_prompt_cache_miss_tokens"
    header = "Total Usage Prompt Cache Miss Tokens"
    short_header = "Total Usage Prompt Cache Miss"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2017


class TotalUsageToolUsePromptTokensMetric(
    DerivedSumMetric[int, UsageToolUsePromptTokensMetric]
):
    """
    Total API-reported tool-use prompt tokens across all requests
    (Gemini-specific).

    Sums the per-request tool-use prompt counts (Gemini's
    toolUsePromptTokenCount). Empty for vendors that fold tool definitions
    into regular prompt_tokens.

    Formula:
        ```
        Total Usage Tool Use Prompt Tokens = Sum(Usage Tool Use Prompt Tokens)
        ```
    """

    tag = "total_usage_tool_use_prompt_tokens"
    header = "Total Usage Tool Use Prompt Tokens"
    short_header = "Total Usage Tool Prompt"
    short_header_hide_unit = True
    console_group = MetricConsoleGroup.USAGE
    display_order = 2030


class TotalUsagePromptAudioSecondsMetric(
    DerivedSumMetric[float, UsagePromptAudioSecondsMetric]
):
    """
    Total API-reported prompt audio duration across all requests, in seconds
    (Mistral-specific).

    Sums the per-request audio durations (Mistral's prompt_audio_seconds).
    Unit is seconds, not tokens.

    Formula:
        ```
        Total Usage Prompt Audio Seconds = Sum(Usage Prompt Audio Seconds)
        ```
    """

    tag = "total_usage_prompt_audio_seconds"
    header = "Total Usage Prompt Audio Seconds"
    short_header = "Total Usage Prompt Audio Sec"
    console_group = MetricConsoleGroup.USAGE
    display_order = 2040

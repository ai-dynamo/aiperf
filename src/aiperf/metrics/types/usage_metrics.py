# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-record API usage field token metrics.

These metrics track token counts as reported in the API response's usage field
for each individual request. Cache-related metrics live in
`usage_cache_metrics.py`; vendor-specific outliers (tool-use, audio
seconds) live in `usage_extras_metrics.py`. Aggregated (summed) variants live
in `usage_total_metrics.py`.

Each metric is a thin declarative subclass of `BaseUsageRecordMetric`,
which reads a single field from `ParsedResponseRecord.final_usage` (the
streaming-merged Usage). The streaming walk-back loop lives once on the
record, not redundantly per metric.
"""

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.metrics.base_usage_record_metric import BaseUsageRecordMetric


class UsagePromptTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field prompt token count metric.

    This represents the number of prompt/input tokens as reported in the
    API response's usage field for a single request, recognized across all
    supported vendor naming conventions (OpenAI prompt_tokens, Anthropic
    input_tokens, Gemini promptTokenCount, AWS Bedrock inputTokens).

    Formula:
        Usage Prompt Tokens = response.usage.prompt_tokens (last non-None)
    """

    tag = "usage_prompt_tokens"
    header = "Usage Prompt Tokens"
    short_header = "Usage Prompt"
    display_order = 1000
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.TOKENIZES_INPUT_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_tokens"
    missing_message = "Usage prompt token count is not available in the record."


class UsageCompletionTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field completion token count metric.

    This represents the number of completion/output tokens as reported in
    the API response's usage field for a single request, recognized across
    all supported vendor naming conventions.

    Formula:
        Usage Completion Tokens = response.usage.completion_tokens (last non-None)
    """

    tag = "usage_completion_tokens"
    header = "Usage Completion Tokens"
    short_header = "Usage Completion"
    display_order = 1100
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "completion_tokens"
    missing_message = "Usage completion token count is not available in the record."


class UsageTotalTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field total token count metric.

    This represents the total number of tokens (prompt + completion) as
    reported in the API response's usage field for a single request.

    Formula:
        Usage Total Tokens = response.usage.total_tokens (last non-None)
    """

    tag = "usage_total_tokens"
    header = "Usage Total Tokens"
    short_header = "Usage Total"
    display_order = 1200
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "total_tokens"
    missing_message = "Usage total token count is not available in the record."


class UsageReasoningTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field reasoning token count metric.

    This represents the number of reasoning tokens as reported in the
    API response's usage field (for models that support reasoning).
    Recorded for reference and comparison.

    Formula:
        Usage Reasoning Tokens = response.usage.completion_tokens_details.reasoning_tokens (last non-None)
    """

    tag = "usage_reasoning_tokens"
    header = "Usage Reasoning Tokens"
    short_header = "Usage Reasoning"
    display_order = 1110
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "reasoning_tokens"
    missing_message = "Usage reasoning token count is not available in the record."


class UsagePromptAudioTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field prompt audio token count metric.

    This represents the number of audio tokens from prompt_tokens_details
    as reported in the API response's usage field.

    Formula:
        Usage Prompt Audio Tokens = response.usage.prompt_tokens_details.audio_tokens (last non-None)
    """

    tag = "usage_prompt_audio_tokens"
    header = "Usage Prompt Audio Tokens"
    short_header = "Usage Prompt Audio"
    display_order = 1020
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.SUPPORTS_AUDIO_ONLY
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "prompt_audio_tokens"
    missing_message = (
        "Usage prompt audio token count not available: no response had "
        "`prompt_tokens_details.audio_tokens` "
        "(or `input_tokens_details.audio_tokens`)."
    )


class UsageCompletionAudioTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field completion audio token count metric.

    This represents the number of audio tokens from completion_tokens_details
    as reported in the API response's usage field (for audio output models).

    Formula:
        Usage Completion Audio Tokens = response.usage.completion_tokens_details.audio_tokens (last non-None)
    """

    tag = "usage_completion_audio_tokens"
    header = "Usage Completion Audio Tokens"
    short_header = "Usage Completion Audio"
    display_order = 1120
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.SUPPORTS_AUDIO_ONLY
    )
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "completion_audio_tokens"
    missing_message = (
        "Usage completion audio token count not available: no response had "
        "`completion_tokens_details.audio_tokens` "
        "(or `output_tokens_details.audio_tokens`)."
    )


class UsageAcceptedPredictionTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field accepted prediction token count metric.

    This represents the number of accepted prediction tokens from
    completion_tokens_details as reported in the API response's usage field.
    These are tokens from a predicted completion that the model used.

    Formula:
        Usage Accepted Prediction Tokens = response.usage.completion_tokens_details.accepted_prediction_tokens (last non-None)
    """

    tag = "usage_accepted_prediction_tokens"
    header = "Usage Accepted Prediction Tokens"
    short_header = "Usage Accepted Pred"
    display_order = 1130
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "accepted_prediction_tokens"
    missing_message = (
        "Usage accepted prediction token count not available: no response had "
        "`completion_tokens_details.accepted_prediction_tokens` "
        "(or `output_tokens_details.accepted_prediction_tokens`)."
    )


class UsageRejectedPredictionTokensMetric(BaseUsageRecordMetric[int]):
    """
    API usage field rejected prediction token count metric.

    This represents the number of rejected prediction tokens from
    completion_tokens_details as reported in the API response's usage field.
    These are tokens from a predicted completion that the model did not use.

    Formula:
        Usage Rejected Prediction Tokens = response.usage.completion_tokens_details.rejected_prediction_tokens (last non-None)
    """

    tag = "usage_rejected_prediction_tokens"
    header = "Usage Rejected Prediction Tokens"
    short_header = "Usage Rejected Pred"
    display_order = 1140
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.USAGE
    required_metrics = None

    usage_field = "rejected_prediction_tokens"
    missing_message = (
        "Usage rejected prediction token count not available: no response had "
        "`completion_tokens_details.rejected_prediction_tokens` "
        "(or `output_tokens_details.rejected_prediction_tokens`)."
    )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API usage field token metrics.

These metrics track token counts as reported in the API response's usage field.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class UsagePromptTokensMetric(BaseRecordMetric[int]):
    """
    API usage field prompt token count metric.

    This represents the number of prompt (input) tokens as reported in the
    API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Prompt Tokens = response.usage.prompt_tokens (last non-None)
    """

    tag = "usage_prompt_tokens"
    header = "Usage Prompt Tokens"
    short_header = "Usage Prompt"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported prompt token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide prompt token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                if prompt_tokens is not None:
                    return prompt_tokens

        raise NoMetricValue("Usage prompt token count is not available in the record.")


class UsageCompletionTokensMetric(BaseRecordMetric[int]):
    """
    API usage field completion token count metric.

    This represents the number of completion (output) tokens as reported in the
    API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Completion Tokens = response.usage.completion_tokens (last non-None)
    """

    tag = "usage_completion_tokens"
    header = "Usage Completion Tokens"
    short_header = "Usage Completion"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported completion token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide completion token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                completion_tokens = response.usage.completion_tokens
                if completion_tokens is not None:
                    return completion_tokens

        raise NoMetricValue(
            "Usage completion token count is not available in the record."
        )


class UsageTotalTokensMetric(BaseRecordMetric[int]):
    """
    API usage field total token count metric.

    This represents the total number of tokens (prompt + completion) as reported
    in the API response's usage field. Recorded for reference and comparison.

    Formula:
        Usage Total Tokens = response.usage.total_tokens (last non-None)
    """

    tag = "usage_total_tokens"
    header = "Usage Total Tokens"
    short_header = "Usage Total"
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported total token count from the record.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide total token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                total_tokens = response.usage.total_tokens
                if total_tokens is not None:
                    return total_tokens

        raise NoMetricValue("Usage total token count is not available in the record.")


class UsageReasoningTokensMetric(BaseRecordMetric[int]):
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
    short_header_hide_unit = True
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the API-reported reasoning token count from the record.

        Reasoning tokens are nested in completion_tokens_details.reasoning_tokens
        (or output_tokens_details.reasoning_tokens) per the official OpenAI spec.

        In streaming responses, each chunk reports cumulative totals, so we take
        the last non-None value from the response stream by searching backwards.

        Raises:
            NoMetricValue: If the API did not provide reasoning token count.
        """
        for response in reversed(record.responses):
            if response.usage:
                reasoning = response.usage.reasoning_tokens
                if reasoning is not None:
                    return reasoning

        raise NoMetricValue(
            "Usage reasoning token count is not available in the record."
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

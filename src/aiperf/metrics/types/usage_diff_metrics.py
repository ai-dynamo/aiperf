# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API usage field vs client-computed token count difference metrics.

These metrics calculate the absolute percentage difference between API-reported usage
prompt token counts and client-side computed input token counts. They help identify
discrepancies between API billing metrics and actual tokenization.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.usage_metrics import (
    UsagePromptTokensMetric,
)


class UsagePromptTokensDiffMetric(BaseRecordMetric[float]):
    """
    Absolute percentage difference between API-reported and client-computed prompt tokens.

    This metric compares the API's usage field prompt token count with the
    client-side tokenized input token count. Discrepancies can indicate:
    - Different tokenization algorithms
    - API preprocessing or special tokens
    - Billing vs actual token count differences

    Formula:
        Diff % = abs((API Prompt Tokens - Client Input Tokens) / Client Input Tokens) * 100

    Example:
        If API reports 105 tokens and client computed 100 tokens:
        Diff % = abs((105 - 100) / 100) * 100 = 5.0%

        If API reports 95 tokens and client computed 100 tokens:
        Diff % = abs((95 - 100) / 100) * 100 = 5.0%
    """

    tag = "usage_prompt_tokens_diff_pct"
    header = "Usage Prompt Diff"
    short_header = "Prompt Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.TOKENIZES_INPUT_ONLY | MetricFlags.NO_CONSOLE
    required_metrics = {
        UsagePromptTokensMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between API and client prompt tokens.

        Reads client-side input count from record.token_counts.input_local
        (not from InputSequenceLengthMetric, which prefers server values).

        Raises:
            NoMetricValue: If either metric is not available or client tokens is zero.
        """
        usage_prompt_tokens = record_metrics.get_or_raise(UsagePromptTokensMetric)

        if record.token_counts is None or record.token_counts.input_local is None:
            raise NoMetricValue(
                "Client-side input token count is not available for the record."
            )
        client_input_tokens = record.token_counts.input_local

        if client_input_tokens == 0:
            raise NoMetricValue(
                "Cannot calculate prompt token difference with zero client tokens."
            )

        diff_pct = (
            abs(usage_prompt_tokens - client_input_tokens) / client_input_tokens
        ) * 100.0

        return diff_pct


class UsageCompletionTokensDiffMetric(BaseRecordMetric[float]):
    """
    Absolute percentage difference between server-reported and client-computed output tokens.

    This metric compares the server-reported output token count with the
    client-side tokenized output token count. Requires ``--tokenize-output``
    to populate both values.

    Formula:
        Diff % = abs((Server Output Tokens - Client Output Tokens) / Client Output Tokens) * 100

    Example:
        If server reports 105 tokens and client computed 100 tokens:
        Diff % = abs((105 - 100) / 100) * 100 = 5.0%
    """

    tag = "usage_completion_tokens_diff_pct"
    header = "Usage Completion Diff"
    short_header = "Completion Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.NO_CONSOLE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between server and client output tokens.

        Raises:
            NoMetricValue: If either value is not available or client tokens is zero.
        """
        if record.token_counts is None:
            raise NoMetricValue("Token counts are not available for the record.")

        server = record.token_counts.output
        client = record.token_counts.output_local

        if server is None or client is None:
            raise NoMetricValue(
                "Both server and client output token counts are required for diff calculation."
            )

        if client == 0:
            raise NoMetricValue(
                "Cannot calculate output token difference with zero client tokens."
            )

        return abs((server - client) / client) * 100.0


class UsageReasoningTokensDiffMetric(BaseRecordMetric[float]):
    """
    Absolute percentage difference between server-reported and client-computed reasoning tokens.

    This metric compares the server-reported reasoning token count with the
    client-side tokenized reasoning token count. Requires ``--tokenize-output``
    to populate both values.

    Formula:
        Diff % = abs((Server Reasoning Tokens - Client Reasoning Tokens) / Client Reasoning Tokens) * 100

    Example:
        If server reports 55 tokens and client computed 50 tokens:
        Diff % = abs((55 - 50) / 50) * 100 = 10.0%
    """

    tag = "usage_reasoning_tokens_diff_pct"
    header = "Usage Reasoning Diff"
    short_header = "Reasoning Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between server and client reasoning tokens.

        Raises:
            NoMetricValue: If either value is not available or client tokens is zero.
        """
        if record.token_counts is None:
            raise NoMetricValue("Token counts are not available for the record.")

        server = record.token_counts.reasoning
        client = record.token_counts.reasoning_local

        if server is None or client is None:
            raise NoMetricValue(
                "Both server and client reasoning token counts are required for diff calculation."
            )

        if client == 0:
            raise NoMetricValue(
                "Cannot calculate reasoning token difference with zero client tokens."
            )

        return abs((server - client) / client) * 100.0


class UsageDiscrepancyCountMetric(BaseAggregateCounterMetric[int]):
    """
    Count of records where the prompt token usage difference exceeds the configured threshold.

    This aggregate counter metric increments by 1 for each record where the
    prompt token diff metric exceeds the threshold defined in
    Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD. The final result is the
    total count of records with significant discrepancies.

    Note: With NO_INDIVIDUAL_RECORDS flag, only the aggregate count is stored, not
    per-record values.

    Use this metric to quantify how many requests have significant tokenization
    differences between API-reported and client-computed token counts. The threshold
    can be configured via AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD (default: 10%).

    Formula:
        For each record: increment = 1 if prompt_diff > threshold
        For each record: increment = 0 otherwise

        Total Count = Sum of all increments

    Example:
        With threshold=10.0% and 4 records:
        - Record 1: 5% prompt → increment = 0
        - Record 2: 15% prompt → increment = 1
        - Record 3: 0% prompt → increment = 0
        - Record 4: 20% prompt → increment = 1

        Total Count = 2 (records 2, 4 had discrepancies)
    """

    tag = "usage_discrepancy_count"
    header = "Usage Discrepancy Count"
    short_header = "Discrepancies"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = (
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.NO_CONSOLE
        | MetricFlags.NO_INDIVIDUAL_RECORDS
    )
    required_metrics = {
        UsagePromptTokensDiffMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Return 1 if prompt diff metric exceeds threshold, 0 otherwise.

        Returns:
            1 if prompt diff metric exceeds threshold (to increment count), 0 otherwise
        """
        threshold = Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD

        prompt_diff = record_metrics.get_or_raise(UsagePromptTokensDiffMetric)
        if prompt_diff > threshold:
            return 1

        return 0

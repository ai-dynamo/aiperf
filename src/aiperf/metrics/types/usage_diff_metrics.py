# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API usage field vs client-computed token count difference metrics.

These metrics calculate the absolute percentage difference between API-reported usage
token counts and client-side computed token counts. They can help identify
discrepancies between API billing metrics and actual tokenization.
"""

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.reasoning_token_count import ReasoningTokenCountMetric
from aiperf.metrics.types.usage_metrics import (
    UsageCompletionTokensMetric,
    UsagePromptTokensMetric,
    UsageReasoningTokensMetric,
)


class UsagePromptTokensDiffMetric(BaseRecordMetric[float]):
    """
    Absolute percentage difference between API-reported and client-computed prompt tokens.

    This metric compares the API's usage field prompt token count with the
    client-side tokenized input sequence length. Discrepancies can indicate:
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
        InputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between API and client prompt tokens.

        Raises:
            NoMetricValue: If either metric is not available or client tokens is zero.
        """
        usage_prompt_tokens = record_metrics.get_or_raise(UsagePromptTokensMetric)
        client_input_tokens = record_metrics.get_or_raise(InputSequenceLengthMetric)

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
    Absolute percentage difference between API-reported and client-computed completion tokens.

    This metric compares the API's usage field completion token count (which includes
    reasoning tokens) with the client-side output sequence length (which also includes
    reasoning tokens). Both metrics represent the TOTAL completion tokens.

    Discrepancies can indicate:
    - Different tokenization algorithms
    - API special tokens or formatting
    - Billing vs actual token count differences

    Formula:
        Diff % = abs((API Completion Tokens - Client Output Sequence Length) / Client Output Sequence Length) * 100

    Example:
        If API reports 55 tokens and client computed 50 tokens:
        Diff % = abs((55 - 50) / 50) * 100 = 10.0%

        If API reports 48 tokens and client computed 50 tokens:
        Diff % = abs((48 - 50) / 50) * 100 = 4.0%
    """

    tag = "usage_completion_tokens_diff_pct"
    header = "Usage Completion Diff %"
    short_header = "Completion Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.NO_CONSOLE
    required_metrics = {
        UsageCompletionTokensMetric.tag,
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between API and client completion tokens.

        Raises:
            NoMetricValue: If either metric is not available or client tokens is zero.
        """
        usage_completion_tokens = record_metrics.get_or_raise(
            UsageCompletionTokensMetric
        )
        client_output_seq_length = record_metrics.get_or_raise(
            OutputSequenceLengthMetric
        )

        if client_output_seq_length == 0:
            raise NoMetricValue(
                "Cannot calculate completion token difference with zero client tokens."
            )

        diff_pct = (
            abs(usage_completion_tokens - client_output_seq_length)
            / client_output_seq_length
        ) * 100.0

        return diff_pct


class UsageReasoningTokensDiffMetric(BaseRecordMetric[float]):
    """
    Absolute percentage difference between API-reported and client-computed reasoning tokens.

    This metric compares the API's usage field reasoning token count with the
    client-side tokenized reasoning token count (for reasoning-capable models).
    Discrepancies can indicate:
    - Different reasoning token identification logic
    - API-specific reasoning token counting
    - Billing vs actual reasoning token differences

    Formula:
        Diff % = abs((API Reasoning Tokens - Client Reasoning Tokens) / Client Reasoning Tokens) * 100

    Example:
        If API reports 52 tokens and client computed 50 tokens:
        Diff % = abs((52 - 50) / 50) * 100 = 4.0%

        If API reports 48 tokens and client computed 50 tokens:
        Diff % = abs((48 - 50) / 50) * 100 = 4.0%
    """

    tag = "usage_reasoning_tokens_diff_pct"
    header = "Usage Reasoning Diff %"
    short_header = "Reasoning Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = {
        UsageReasoningTokensMetric.tag,
        ReasoningTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between API and client reasoning tokens.

        Raises:
            NoMetricValue: If either metric is not available or client tokens is zero.
        """
        usage_reasoning_tokens = record_metrics.get_or_raise(UsageReasoningTokensMetric)
        client_reasoning_tokens = record_metrics.get_or_raise(ReasoningTokenCountMetric)

        if client_reasoning_tokens == 0:
            raise NoMetricValue(
                "Cannot calculate reasoning token difference with zero client tokens."
            )

        diff_pct = (
            abs(usage_reasoning_tokens - client_reasoning_tokens)
            / client_reasoning_tokens
        ) * 100.0

        return diff_pct


class UsageDiscrepancyCountMetric(BaseAggregateCounterMetric[int]):
    """
    Count of records where usage difference metrics exceed the configured threshold.

    This aggregate counter metric increments by 1 for each record where ANY of the
    usage difference metrics (prompt, completion, or reasoning) exceed the threshold
    defined in Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD. The final result is the
    total count of records with significant discrepancies.

    Note: With NO_INDIVIDUAL_RECORDS flag, only the aggregate count is stored, not
    per-record values.

    Use this metric to quantify how many requests have significant tokenization
    differences between API-reported and client-computed token counts. The threshold
    can be configured via AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD (default: 10%).

    Formula:
        For each record: increment = 1 if (prompt_diff > threshold OR completion_diff > threshold OR reasoning_diff > threshold)
        For each record: increment = 0 otherwise

        Total Count = Sum of all increments

    Example:
        With threshold=10.0% and 6 records:
        - Record 1: 5% prompt, 3% completion → increment = 0
        - Record 2: 15% prompt, 3% completion → increment = 1
        - Record 3: 5% prompt, 12% completion → increment = 1
        - Record 4: 0% prompt, 0% completion → increment = 0
        - Record 5: 20% prompt, 15% completion → increment = 1
        - Record 6: 3% prompt, 2% completion → increment = 0

        Total Count = 3 (records 2, 3, 5 had discrepancies)
    """

    tag = "usage_discrepancy_count"
    header = "Usage Discrepancy Count"
    short_header = "Discrepancies"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.NO_CONSOLE | MetricFlags.NO_INDIVIDUAL_RECORDS
    # Required metrics ensures dependency ordering. We require prompt and completion
    # which are always available, and opportunistically check reasoning in _parse_record
    required_metrics = {
        UsagePromptTokensDiffMetric.tag,
        UsageCompletionTokensDiffMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Return 1 if any usage diff metric exceeds threshold, 0 otherwise.

        Checks prompt, completion, and reasoning diff metrics. If ANY exceed the
        threshold, increments the counter. Reasoning tokens are optional since not
        all models support them.

        Returns:
            1 if any diff metric exceeds threshold (to increment count), 0 otherwise
        """
        threshold = Environment.METRICS.USAGE_PCT_DIFF_THRESHOLD

        # Check required metrics (prompt and completion)
        prompt_diff = record_metrics.get_or_raise(UsagePromptTokensDiffMetric)
        if prompt_diff > threshold:
            return 1

        completion_diff = record_metrics.get_or_raise(UsageCompletionTokensDiffMetric)
        if completion_diff > threshold:
            return 1

        # Check optional reasoning metric (may not be available)
        reasoning_diff = record_metrics.get(UsageReasoningTokensDiffMetric.tag)
        if reasoning_diff is not None and reasoning_diff > threshold:
            return 1

        return 0

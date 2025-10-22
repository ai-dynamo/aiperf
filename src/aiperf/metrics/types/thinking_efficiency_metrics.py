# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseDerivedMetric, BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.output_token_count import (
    OutputTokenCountMetric,
    TotalOutputTokensMetric,
)
from aiperf.metrics.types.reasoning_token_count import (
    ReasoningTokenCountMetric,
    TotalReasoningTokensMetric,
)


class ThinkingEfficiencyMetric(BaseRecordMetric[float]):
    """
    This is the ratio of the reasoning tokens to the output tokens for a single record.

    Formula:
        ```
        Thinking Efficiency = Total Reasoning Tokens / Total Output Tokens
        ```

    References:
        @misc{lrm_token_economy_2025,
            title={Measuring Thinking Efficiency in Reasoning Models: The Missing Benchmark},
            author={TSB},
            year={2025},
            month={August},
            url={https://github.com/cpldcpu/LRMTokenEconomy}
        }
    """

    tag = "thinking_efficiency"
    header = "Thinking Efficiency"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = {
        ReasoningTokenCountMetric.tag,
        OutputTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        reasoning_token_count = record_metrics.get_or_raise(ReasoningTokenCountMetric)
        output_token_count = record_metrics.get_or_raise(OutputTokenCountMetric)

        return reasoning_token_count / output_token_count  # type: ignore


class OverallThinkingEfficiencyMetric(BaseDerivedMetric[float]):
    """
    This is the ratio of the total reasoning tokens to the total output tokens across all records.

    This is different from the individual thinking efficiency metric in that the value gets normalized
    by the total number of tokens produced.

    Formula:
        ```
        Overall Thinking Efficiency = Total Reasoning Tokens / Total Output Tokens
        ```

    References:
        @misc{lrm_token_economy_2025,
            title={Measuring Thinking Efficiency in Reasoning Models: The Missing Benchmark},
            author={TSB},
            year={2025},
            month={August},
            url={https://github.com/cpldcpu/LRMTokenEconomy}
        }
    """

    tag = "overall_thinking_efficiency"
    header = "Overall Thinking Efficiency"
    short_header = "Overall Eff."
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = {
        TotalOutputTokensMetric.tag,
        TotalReasoningTokensMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_reasoning_tokens = metric_results.get_or_raise(TotalReasoningTokensMetric)
        total_output_tokens = metric_results.get_or_raise(TotalOutputTokensMetric)

        return total_reasoning_tokens / total_output_tokens  # type: ignore

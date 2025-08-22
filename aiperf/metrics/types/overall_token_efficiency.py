# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.total_output_tokens import TotalOutputTokensMetric
from aiperf.metrics.types.total_reasoning_tokens import TotalReasoningTokensMetric


class OverallTokenEfficiencyMetric(BaseDerivedMetric[float]):
    """
    Post-processor for calculating Overall Token Efficiency metrics from records.

    Formula:
        Overall Token Efficiency = Total Reasoning Tokens / Total Output Tokens
    """

    tag = "overall_token_efficiency"
    header = "Overall Token Efficiency"
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
        total_output_tokens = metric_results.get(TotalOutputTokensMetric.tag)
        if not total_output_tokens:
            raise ValueError("Total output tokens is missing in the metrics.")

        total_reasoning_tokens = metric_results.get(TotalReasoningTokensMetric.tag)
        if not total_reasoning_tokens:
            raise ValueError("Total reasoning tokens is missing in the metrics.")

        return total_reasoning_tokens / total_output_tokens  # type: ignore

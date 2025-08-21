# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.reasoning_token_count import ReasoningTokenCountMetric


class TotalReasoningTokensMetric(BaseAggregateMetric[int]):
    """
    Post-processor for calculating Total Reasoning Tokens metrics from records.

    Formula:
        Total Reasoning Tokens = Sum(Reasoning Tokens)
    """

    tag = "total_reasoning_tokens"
    header = "Total Reasoning Tokens"
    short_header = "Total Reasoning"
    short_header_hide_unit = False
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = {
        ReasoningTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        reasoning_token_count = record_metrics.get(ReasoningTokenCountMetric.tag)
        if not reasoning_token_count:
            raise ValueError("Reasoning token count is missing in the record.")
        return reasoning_token_count  # type: ignore

    def _aggregate_value(self, value: int) -> None:
        """
        Aggregate the reasoning token count.
        """
        self._value += value

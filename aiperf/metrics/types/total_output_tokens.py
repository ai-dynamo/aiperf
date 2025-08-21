# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_token_count import OutputTokenCountMetric


class TotalOutputTokensMetric(BaseAggregateMetric[int]):
    """
    Post-processor for calculating Total Output Tokens metrics from records.

    Formula:
        Total Output Tokens = Sum(Output Tokens)
    """

    tag = "total_output_tokens"
    header = "Total Output Tokens"
    short_header = "Total Output"
    short_header_hide_unit = False
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = {
        OutputTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        output_token_count = record_metrics.get(OutputTokenCountMetric.tag)
        if not output_token_count:
            raise ValueError("Output token count is missing in the record.")
        return output_token_count  # type: ignore

    def _aggregate_value(self, value: int) -> None:
        """
        Aggregate the output token count.
        """
        self._value += value

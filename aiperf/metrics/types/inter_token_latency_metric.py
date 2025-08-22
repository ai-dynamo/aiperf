# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class InterTokenLatencyMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Inter Token Latency (ITL) metric.

    Formula:
        Inter Token Latency = (Request Latency - Time to First Token) / (Output Sequence Length - 1)
    """

    tag = "inter_token_latency"
    header = "Inter Token Latency"
    short_header = "ITL"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 400
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        RequestLatencyMetric.tag,
        TTFTMetric.tag,
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculates the Inter Token Latency (ITL) metric.
        """
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        if osl < 2:  # type: ignore
            raise NoMetricValue(f"Output sequence length must be at least 2, got {osl}")

        ttft = record_metrics.get_or_raise(TTFTMetric)
        request_latency = record_metrics.get_or_raise(RequestLatencyMetric)

        return (request_latency - ttft) / (osl - 1)  # type: ignore

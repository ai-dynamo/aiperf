# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class E2EOutputTokenThroughputMetric(BaseRecordMetric[float]):
    """
    Per-request output token throughput based on end-to-end request latency.

    Unlike OutputTokenThroughputPerUserMetric (1/ITL), this includes TTFT,
    queuing, and all other overhead in the denominator.

    Formula:
        E2E Output Token Throughput = Output Sequence Length / Request Latency (seconds)
    """

    tag = "e2e_output_token_throughput"
    header = "E2E Output Token Throughput"
    short_header = "E2E Output TPS/User"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    display_order = 510
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        OutputSequenceLengthMetric.tag,
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        latency_seconds = record_metrics.get_converted_or_raise(
            RequestLatencyMetric,
            self.unit.time_unit,  # type: ignore
        )
        if latency_seconds == 0:
            raise NoMetricValue(
                "Request latency is zero, cannot calculate E2E output token throughput"
            )
        return osl / latency_seconds

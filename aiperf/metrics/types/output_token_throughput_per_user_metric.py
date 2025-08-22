# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric


class OutputTokenThroughputPerUserMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Output Token Throughput Per User Metric.

    Formula:
        Output Token Throughput Per User = 1 / Inter-Token Latency (seconds)
    """

    tag = "output_token_throughput_per_user"
    header = "Output Token Throughput Per User"
    short_header = "Output TPS/User"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    display_order = 500
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        InterTokenLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """This method calculates the output token throughput per user by computing the inverse of the inter-token latency."""
        itl = record_metrics.get(InterTokenLatencyMetric.tag)
        if not itl:
            raise NoMetricValue(
                "Inter-token latency is not available, cannot compute output token throughput per user."
            )
        converted_itl = record_metrics.get_converted_or_raise(
            InterTokenLatencyMetric,
            self.unit.time_unit,  # type: ignore
        )
        return 1 / converted_itl

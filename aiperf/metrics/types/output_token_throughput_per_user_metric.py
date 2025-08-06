# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.models.record_models import ParsedResponseRecord
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
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        InterTokenLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """This method calculates the output inference speed by computing the inverse of the inter-token latency."""
        itl = record_metrics[InterTokenLatencyMetric.tag]
        if itl is None or itl == 0:
            raise ValueError(
                "Inter-token latency is 0, cannot compute output token throughput per user."
            )
        converted_itl = record_metrics.get_converted(
            InterTokenLatencyMetric,
            self.unit.time_unit,  # type: ignore
        )
        return 1 / converted_itl

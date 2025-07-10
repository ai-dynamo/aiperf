#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class OutputTokenThroughputPerUserMetric(BaseMetric):
    """
    Post Processor for calculating Output Token Throughput per user metrics from records.
    """

    tag = "output_token_throughput_per_user"
    unit = MetricTimeType.SECONDS
    larger_is_better = True
    header = "Output Token Throughput Per User"
    type = MetricType.METRIC_OF_METRICS
    streaming_only = True

    def __init__(self):
        self.metric: list[float] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ):
        inter_token_latencies = metrics["inter_token_latency"].values()
        for inter_token_latency in inter_token_latencies:
            inter_token_latency_s = inter_token_latency / NANOS_PER_SECOND
            self.metric.append(1 / inter_token_latency_s)

    def values(self):
        """
        Returns the list of Output Token Throughput Per User metrics.
        """
        return self.metric

    def _check_record(self, record):
        pass

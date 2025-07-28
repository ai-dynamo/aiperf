# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricTag, MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class OutputTokenThroughputPerUserMetric(BaseMetric):
    """
    Post Processor for calculating Output Token Throughput per user metrics from records.
    """

    tag = MetricTag.OUTPUT_TOKEN_THROUGHPUT_PER_USER
    unit = MetricTimeType.SECONDS
    larger_is_better = True
    header = "Output Token Throughput Per User"
    type = MetricType.METRIC_OF_METRICS
    streaming_only = True
    required_metrics = {MetricTag.INTER_TOKEN_LATENCY}

    def __init__(self):
        self.metric: list[float] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[MetricTagT, "BaseMetric"] | None = None,
    ):
        self._check_metrics(metrics)
        # Clear the current value because we re-compute it each time
        self.metric.clear()
        inter_token_latencies = metrics[MetricTag.INTER_TOKEN_LATENCY].values()
        for inter_token_latency in inter_token_latencies:
            inter_token_latency_s = inter_token_latency / NANOS_PER_SECOND
            if inter_token_latency_s <= 0:
                raise ValueError("Inter-token latency must be greater than 0.")
            self.metric.append(1 / inter_token_latency_s)

    def values(self):
        """
        Returns the list of Output Token Throughput Per User metrics.
        """
        return self.metric

    def _check_record(self, record):
        pass

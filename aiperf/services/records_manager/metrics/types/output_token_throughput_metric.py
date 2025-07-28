# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricTag, MetricType
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class OutputTokenThroughputMetric(BaseMetric):
    """
    Post Processor for calculating Output Token Throughput Metric.
    """

    tag = MetricTag.OUTPUT_TOKEN_THROUGHPUT
    unit = None
    larger_is_better = True
    header = "Output Token Throughput (tokens/sec)"
    type = MetricType.METRIC_OF_METRICS
    required_metrics = {
        MetricTag.OUTPUT_TOKEN_COUNT,
        MetricTag.BENCHMARK_DURATION,
    }

    def __init__(self):
        self.metric: float = 0.0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[MetricTagT, "BaseMetric"] | None = None,
    ):
        self._check_metrics(metrics)
        tokens = metrics[MetricTag.OUTPUT_TOKEN_COUNT].values()
        total_tokens = sum(tokens)

        duration_ns = metrics[MetricTag.BENCHMARK_DURATION].values()
        self.metric = total_tokens / (duration_ns / NANOS_PER_SECOND)

    def values(self) -> float:
        """
        Returns the OutputTokenThroughput metric.
        """
        return self.metric

    def _check_record(self, record):
        pass

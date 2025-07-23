# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricType
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.benchmark_duration_metric import (
    BenchmarkDurationMetric,
)
from aiperf.services.records_manager.metrics.types.output_token_count_metric import (
    OutputTokenCountMetric,
)


class OutputTokenThroughputMetric(BaseMetric):
    """
    Post Processor for calculating Output Token Throughput Metric.
    """

    tag = "output_token_throughput"
    unit = None
    larger_is_better = True
    header = "Output Token Throughput (tokens/sec)"
    type = MetricType.METRIC_OF_METRICS
    required_metrics = {
        OutputTokenCountMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def __init__(self):
        self.metric: float = 0.0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ):
        self._check_metrics(metrics)
        tokens = metrics[OutputTokenCountMetric.tag].values()
        total_tokens = sum(tokens)

        duration_ns = metrics[BenchmarkDurationMetric.tag].values()
        self.metric = total_tokens / (duration_ns / NANOS_PER_SECOND)

    def values(self) -> float:
        """
        Returns the OutputTokenThroughput metric.
        """
        return self.metric

    def _check_record(self, record):
        pass

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.max_response_metric import (
    MaxResponseMetric,
)
from aiperf.services.records_manager.metrics.types.min_request_metric import (
    MinRequestMetric,
)


class BenchmarkDurationMetric(BaseMetric):
    """
    Post-processor for calculating the Benchmark Duration metric.
    """

    tag = "benchmark_duration"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Benchmark Duration"
    type = MetricType.METRIC_OF_METRICS
    required_metrics = {MinRequestMetric.tag, MaxResponseMetric.tag}

    def __init__(self):
        self.metric: float = 0.0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ) -> None:
        self._check_metrics(metrics)
        min_req_time = metrics[MinRequestMetric.tag].values()
        max_res_time = metrics[MaxResponseMetric.tag].values()
        benchmark_duration = max_res_time - min_req_time
        self.metric = benchmark_duration

    def values(self) -> float:
        """
        Returns the BenchmarkDuration metric.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        pass

#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class RequestThroughputMetric(BaseMetric):
    """
    Post Processor for calculating Request throughput metrics from records.
    """

    tag = "request_throughput"
    unit = MetricTimeType.SECONDS
    larger_is_better = True
    header = "Request Throughput"
    type = MetricType.METRIC_OF_BOTH
    streaming_only = False

    def __init__(self):
        self.total_requests: int = 0
        self.metric: float = 0.0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        if record:
            self.total_requests += 1

        else:
            benchmark_duration = metrics["benchmark_duration"].values()
            self.metric = self.total_requests / (benchmark_duration / 1e9)

    def values(self) -> float:
        """
        Returns the Request Throughput metric.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        pass

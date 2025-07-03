#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.benchmark_duration_metric import (
    BenchmarkDurationMetric,
)
from aiperf.services.records_manager.metrics.types.token_count_metric import (
    TokenCountMetric,
)


class OutputTokenThroughputMetric(BaseMetric):
    """
    Post-processor for calculating Output Token Throughput metrics from records.
    """

    tag = "output_token_throughput"
    unit = None
    larger_is_better = False
    header = "Output Token Throughput"
    type = MetricType.METRIC_OF_METRICS

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Processes metrics to calculate Output Token Throughput.

        """
        token_count = metrics[TokenCountMetric.tag].values()
        benchmark_duration = metrics[BenchmarkDurationMetric.tag].values()

        for i in range(len(token_count)):
            throughput = token_count[i] / benchmark_duration
            self.metric.append(throughput)

    def values(self) -> list[int]:
        """
        Returns the list of Inter-Token Latency (ITL) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        """
        Checks if the record is valid for TTFT calculation.

        Raises:
            ValueError: If the record does not have at least one response.
        """
        pass

#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.output_token_metric import (
    OutputTokenMetric,
)
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric
from aiperf.services.records_manager.records import Record


class InterTokenLatencyMetric(BaseMetric):
    """
    Post-processor for calculating the Inter Token Latency metric.
    """

    # request_latency - ttft / total output tokens

    tag = "inter_token_latency"
    unit = MetricTimeType.NANOSECONDS
    type = MetricType.METRIC_OF_METRICS
    larger_is_better = False
    header = "Inter Token Latency"

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self, record: Record | None = None, metrics: dict["BaseMetric"] | None = None
    ) -> None:
        request_latency = metrics[RequestLatencyMetric].values()
        ttft = metrics[TTFTMetric.tag].values()
        number_of_output_tokens = metrics[OutputTokenMetric.tag].values()
        if len(request_latency) != len(ttft) or len(request_latency) != len(
            number_of_output_tokens
        ):
            raise ValueError(
                "All metrics must have the same number of records to calculate Inter Token Latency."
            )
        for i in range(len(request_latency)):
            if number_of_output_tokens[i] == 0:
                inter_token_latency = 0
            else:
                # Calculate inter-token latency for each record
                inter_token_latency = (
                    request_latency[i] - ttft[i]
                ) / number_of_output_tokens[i]
            self.metric.append(inter_token_latency)

    def values(self) -> float:
        """
        Returns the Inter Token Latency for the requests.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        pass

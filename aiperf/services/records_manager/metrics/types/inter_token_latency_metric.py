#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.token_count_metric import (
    TokenCountMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


class InterTokenLatencyMetric(BaseMetric):
    """
    Post-processor for calculating Inter-Token Latency (ITL) metrics from records.
    """

    tag = "itl"
    unit = None
    larger_is_better = False
    header = "Inter-Token Latency (ITL)"
    type = MetricType.METRIC_OF_METRICS

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Processes metrics to calculate Inter-Token Latency (ITL).

        """
        request_latency = metrics[RequestLatencyMetric.tag].values()
        ttft = metrics[TTFTMetric.tag].values()
        token_count = metrics[TokenCountMetric.tag].values()

        for i in range(len(request_latency)):
            inter_token_latency = (request_latency[i] - ttft[i]) / token_count[i]
            self.metric.append(inter_token_latency)

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

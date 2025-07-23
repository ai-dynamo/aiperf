# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.output_token_count_metric import (
    OutputTokenCountMetric,
)
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


class InterTokenLatencyMetric(BaseMetric):
    """
    Post Processor for calculating Inter Token Latency (ITL) from validated metrics.

    ITL = (Request Latency - TTFT) / (Output Token Count - 1)
    """

    tag = "inter_token_latency"
    unit = MetricTimeType.MILLISECONDS
    larger_is_better = False
    header = "Inter Token Latency (ITL)"
    type = MetricType.METRIC_OF_METRICS
    streaming_only = True
    required_metrics = {
        RequestLatencyMetric.tag,
        TTFTMetric.tag,
        OutputTokenCountMetric.tag,
    }

    def __init__(self):
        self.metric: list[float] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ):
        self._check_metrics(metrics)

        latencies = metrics[RequestLatencyMetric.tag].values()
        ttfts = metrics[TTFTMetric.tag].values()
        output_token_counts = metrics[OutputTokenCountMetric.tag].values()

        for _, (latency, ttft, output_tokens) in enumerate(
            zip(latencies, ttfts, output_token_counts, strict=False)
        ):
            itl = (latency - ttft) / (output_tokens - 1)
            self.metric.append(itl)

    def values(self) -> list[float]:
        return self.metric

    def _check_record(self, record):
        pass

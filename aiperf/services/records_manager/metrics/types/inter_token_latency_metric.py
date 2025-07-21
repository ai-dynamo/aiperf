# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


class InterTokenLatencyMetric(BaseMetric):
    """
    Post Processor for calculating Inter Token Latency Metric from records.
    """

    tag = "inter_token_latency"
    unit = MetricTimeType.MILLISECONDS
    larger_is_better = False
    header = "Inter Token Latency (ITL)"
    type = MetricType.METRIC_OF_BOTH
    streaming_only = True
    required_metrics: set[str] = {RequestLatencyMetric.tag, TTFTMetric.tag}

    def __init__(self):
        self.metric: list[float] = []
        self._current_index: int = 0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ):
        self._check_record(record)
        self._check_metrics(metrics)
        request_latency = metrics[RequestLatencyMetric.tag].values()[
            self._current_index
        ]
        ttft = metrics[TTFTMetric.tag].values()[self._current_index]
        output_tokens = record.output_token_count
        itl = (request_latency - ttft) / (output_tokens - 1)
        self.metric.append(itl)
        self._current_index = self._current_index + 1

    def values(self) -> list[float]:
        """
        Returns the list of Inter Token Latency (ITL) metrics.
        """
        return self.metric

    def _check_record(self, record):
        self._require_valid_record(record)
        if record.output_token_count is None:
            raise ValueError("Output token count is not available for the record.")
        if record.output_token_count <= 1:
            raise ValueError(
                "Output token count must be greater than 1 for ITL calculation."
            )

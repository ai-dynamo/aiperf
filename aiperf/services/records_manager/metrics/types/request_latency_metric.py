# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class RequestLatencyMetric(BaseMetric):
    """
    Post-processor for calculating Request Latency metrics from records.
    """

    tag = "request_latency"
    unit = MetricTimeType.NANOSECONDS
    type = MetricType.METRIC_OF_RECORDS
    larger_is_better = False
    header = "Request Latency"
    required_metrics: set[str] = set()

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Adds a new record and calculates the Request Latency metric.

        This method extracts the request and last response timestamps, calculates the differences in time, and
        appends the result to the metric list.
        """
        self._check_record(record)
        request_ts = record.start_perf_ns
        final_response_ts = record.responses[-1].perf_ns
        request_latency = final_response_ts - request_ts
        self.metric.append(request_latency)

    def values(self) -> list[int]:
        """
        Returns the list of Request Latency metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        if not record or not record.valid:
            raise ValueError("Invalid Record")

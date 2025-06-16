#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.records import Record


class RequestLatencyMetric(BaseMetric):
    """
    Post-processor for calculating Request Latency metrics from records.
    """

    tag = "request_latency"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Request Latency"

    def __init__(self):
        self.metric: list[int] = []

    def update_value(self, record: Record) -> None:
        """
        Adds a new record and calculates the Request Latencies metric.

        This method extracts the request and last response timestamps, calculates the differences in time, and
        appends the result to the metric list.
        """
        self._check_record(record)
        request_ts = record.request.timestamp
        final_response_ts = record.responses[-1].timestamp
        request_latency = final_response_ts - request_ts
        self.metric.append(request_latency)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (Request Latencies) metrics.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        if not record.request or not record.request.timestamp:
            raise ValueError("Record must have a valid request with a timestamp.")
        if len(record.responses) < 1:
            raise ValueError("Record must have at least one response.")

        request_ts = record.request.timestamp
        response_ts = record.responses[-1].timestamp

        if request_ts < 0 or response_ts < 0:
            raise ValueError("Timestamps must be positive values.")

        if response_ts < request_ts:
            raise ValueError(
                "Response timestamp must be greater than or equal to request timestamp."
            )

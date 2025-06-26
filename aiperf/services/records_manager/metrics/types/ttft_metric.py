#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.record_models import RequestRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class TTFTMetric(BaseMetric):
    """
    Post-processor for calculating Time to First Token (TTFT) metrics from records.
    """

    tag = "ttft"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Time to First Token (TTFT)"
    type = MetricType.METRIC_OF_RECORDS

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: RequestRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Adds a new record and calculates the Time To First Token (TTFT) metric.

        This method extracts the timestamp from the request and the first response in the given
        RequestRecord object, computes the difference (TTFT), and appends the result to the metric list.
        """
        self._check_record(record)
        request_ts = record.recv_start_perf_ns
        response_ts = record.responses[0].perf_ns
        ttft = response_ts - request_ts
        self.metric.append(ttft)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: RequestRecord) -> None:
        """
        Checks if the record is valid for TTFT calculation.

        Raises:
            ValueError: If the record does not have at least one response.
        """
        if not record or not record.recv_start_perf_ns:
            raise ValueError("Record must have a valid request with a timestamp.")
        if not record.responses or len(record.responses) < 1:
            raise ValueError(
                "Record must have at least one response to calculate TTFT."
            )
        if record.responses[0].perf_ns < record.recv_start_perf_ns:
            raise ValueError(
                "Response timestamp must be greater than or equal to request timestamp."
            )

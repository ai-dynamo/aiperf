#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType
from aiperf.services.records_manager.metrics.metric import Metric
from aiperf.services.records_manager.records import Record


class TTFTMetric(Metric):
    """
    Post-processor for calculating Time to First Token (TTFT) metrics from records.
    """

    tag = "ttft"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Time to First Token (TTFT)"

    def __init__(self):
        self.metric: list[int] = []

    def add_record(self, record: Record) -> None:
        """
        Adds a new record and calculates the Time To First Token (TTFT) metric.

        This method extracts the timestamp from the request and the first response in the given
        Record object, computes the difference (TTFT), and appends the result to the metric list.
        """
        TTFTMetric._check_record(record)
        request_ts = record.request.timestamp
        response_ts = record.responses[0].timestamp
        ttft = response_ts - request_ts
        self.metric.append(ttft)

    def get_metrics(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    @staticmethod
    def _check_record(record: Record) -> None:
        """
        Checks if the record is valid for TTFT calculation.

        Raises:
            ValueError: If the record does not have at least one response.
        """
        if not record.request or not record.request.timestamp:
            raise ValueError("Record must have a valid request with a timestamp.")
        if not record.responses or len(record.responses) < 1:
            raise ValueError(
                "Record must have at least one response to calculate TTFT."
            )
        if not record.responses[0].timestamp >= record.request.timestamp:
            raise ValueError(
                "Response timestamp must be greater than or equal to request timestamp."
            )

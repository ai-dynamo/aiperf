#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.records import Record


class MaxResponseMetric(BaseMetric):
    """
    Post-processor for calculating the maximum response time stamp metric from records.
    """

    tag = "max_response"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Maximum Response Timestamp"

    def __init__(self):
        self.metric: float = 0

    def add_record(self, record: Record) -> None:
        """
        Adds a new record and calculates the maximum response timestamp metric.

        """
        self._check_record(record)
        if record.responses[-1].timestamp > self.metric:
            self.metric = record.responses[-1].timestamp

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        """
        Checks if the record is valid for calculations.

        """
        if not record.responses or not record.responses[-1].timestamp:
            raise ValueError("Record must have valid responses with timestamps.")

#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.records import Record


class TTSTMetric(BaseMetric):
    """
    Post-processor for calculating Time to Second Token (TTST) metrics from records.
    """

    tag = "ttst"
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Time to Second Token (TTST)"
    type = MetricType.METRIC_OF_RECORDS

    def __init__(self):
        self.metric: list[int] = []

    def update_value(self, record: Record) -> None:
        """
        Adds a new record and calculates the Time To Second Token (TTST) metric.

        This method extracts the timestamp from the first and second response in the given
        Record object, computes the difference (TTST), and appends the result to the metric list.
        """
        self._check_record(record)
        first_reponse_ts = record.responses[0].timestamp
        second_response_ts = record.responses[1].timestamp
        TTST = second_response_ts - first_reponse_ts
        self.metric.append(TTST)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTST) metrics.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        """
        Checks if the record is valid for TTST calculation.

        Raises:
            ValueError: If the record does not have at least two responses.
        """
        if not record.request or not record.request.timestamp:
            raise ValueError("Record must have a valid request with a timestamp.")
        if not record.responses or len(record.responses) < 2:
            raise ValueError(
                "Record must have at least two responses to calculate TTST."
            )
        if record.responses[1].timestamp < record.responses[0].timestamp:
            raise ValueError(
                "Second response timestamp must be greater than or equal to the first response timestamp."
            )

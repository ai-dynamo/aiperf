# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTag, MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.metrics.base_metric import BaseMetric


class TTSTMetric(BaseMetric):
    """
    Post-processor for calculating Time to Second Token (TTST) metrics from records.
    """

    tag = MetricTag.TTST
    unit = MetricTimeType.NANOSECONDS
    larger_is_better = False
    header = "Time to Second Token (TTST)"
    type = MetricType.METRIC_OF_RECORDS
    streaming_only = True
    required_metrics = set()

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[MetricTagT, "BaseMetric"] | None = None,
    ) -> None:
        """
        Adds a new record and calculates the Time To Second Token (TTST) metric.

        This method extracts the timestamp from the first and second response in the given
        Record object, computes the difference (TTST), and appends the result to the metric list.
        """
        self._check_record(record)
        first_reponse_ts = record.responses[0].perf_ns
        second_response_ts = record.responses[1].perf_ns
        ttst = second_response_ts - first_reponse_ts
        self.metric.append(ttst)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTST) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        """
        Checks if the record is valid for TTST calculation.

        Raises:
            ValueError: If the record does not have at least two responses.
        """
        self._require_valid_record(record)
        if len(record.responses) < 2:
            raise ValueError(
                "Record must have at least two responses to calculate TTST."
            )
        if record.responses[1].perf_ns < record.responses[0].perf_ns:
            raise ValueError(
                "Second response timestamp must be greater than or equal to the first response timestamp."
            )

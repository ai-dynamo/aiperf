# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
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
    streaming_only = True
    required_metrics: set[str] = set()

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Adds a new record and calculates the Time To First Token (TTFT) metric.

        This method extracts the timestamp from the request and the first response in the given
        RequestRecord object, computes the difference (TTFT), and appends the result to the metric list.
        """
        self._check_record(record)
        request_ts = record.request.start_perf_ns
        response_ts = record.responses[0].perf_ns
        ttft = response_ts - request_ts
        self.metric.append(ttft)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        """
        Checks if the record is valid for TTFT calculation.

        Raises:
            ValueError: If record is None or record is not valid
        """
        if not record or not record.valid:
            raise ValueError("Invalid Record")

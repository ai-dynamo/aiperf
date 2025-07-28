# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTag, MetricTimeType, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class MaxResponseMetric(BaseMetric):
    """
    Post-processor for calculating the maximum response time stamp metric from records.
    """

    tag = MetricTag.MAX_RESPONSE
    unit = MetricTimeType.NANOSECONDS
    type = MetricType.METRIC_OF_RECORDS
    larger_is_better = False
    header = "Maximum Response Timestamp"
    required_metrics = set()

    def __init__(self):
        self.metric: float = 0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[MetricTagT, "BaseMetric"] | None = None,
    ) -> None:
        """
        Adds a new record and calculates the maximum response timestamp metric.

        """
        self._check_record(record)
        if record.responses[-1].perf_ns > self.metric:
            self.metric = record.responses[-1].perf_ns

    def values(self) -> float:
        """
        Returns the Max Response Timestamp metric.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        """
        Checks if the record is valid for calculations.
        """
        self._require_valid_record(record)

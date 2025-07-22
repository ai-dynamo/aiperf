# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class RequestCountMetric(BaseMetric):
    """
    Post-processor for counting the number of valid requests.
    """

    tag = "request_count"
    unit = None
    larger_is_better = True
    header = "Request Count"
    type = MetricType.METRIC_OF_RECORDS
    streaming_only = False
    required_metrics: set[str] = set()

    def __init__(self):
        self.metric: int = 0

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ) -> None:
        self._check_record(record)
        self.metric += 1

    def values(self) -> int:
        """
        Returns the Request Count metric.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        self._require_valid_record(record)

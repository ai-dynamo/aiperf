#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class TokenCountMetric(BaseMetric):
    """
    Post-processor for gathering Token Counts into a metric from records.
    """

    tag = "token_count"
    unit = None
    larger_is_better = False
    header = "Token Count"
    type = MetricType.METRIC_OF_RECORDS

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict["BaseMetric"] | None = None,
    ) -> None:
        """
        Stores Token Count in a metric.

        """
        self._check_record(record)
        self.metric.append(record.token_count)

    def values(self) -> list[int]:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord) -> None:
        """
        Checks if the record is valid for TTFT calculation.

        Raises:
            ValueError: If the record does not have at least one response.
        """
        pass

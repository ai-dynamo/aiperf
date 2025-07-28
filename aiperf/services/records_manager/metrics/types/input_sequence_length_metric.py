#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricTag, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class InputSequenceLengthMetric(BaseMetric):
    """
    Post-processor for calculating Input Sequence Length (ISL) metrics from records.
    """

    tag = MetricTag.ISL
    unit = None
    larger_is_better = False
    header = "Input Sequence Length"
    type = MetricType.METRIC_OF_RECORDS
    streaming_only = False
    required_metrics = set()

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[MetricTagT, "BaseMetric"] | None = None,
    ):
        self._check_record(record)
        input_token_count = record.input_token_count
        self.metric.append(input_token_count)

    def values(self) -> list[int]:
        """
        Returns the list of Input Sequence Length (ISL) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord):
        """
        Checks if the record is valid for ISL calculation.

        Raises:
            ValueError: If the record is not valid or doesn't have input_token_count.
        """
        self._require_valid_record(record)
        if record.input_token_count is None:
            raise ValueError("Input Token Count is not available for the record.")

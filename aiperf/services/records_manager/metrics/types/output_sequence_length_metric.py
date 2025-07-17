# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricType
from aiperf.common.record_models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class OutputSequenceLengthMetric(BaseMetric):
    """
    Post-processor for calculating Output Sequence Length (OSL) metrics from records.
    """

    tag = "osl"
    unit = None
    larger_is_better = False
    header = "Output Sequence Length"
    type = MetricType.METRIC_OF_RECORDS
    streaming_only = False

    def __init__(self):
        self.metric: list[int] = []

    def update_value(
        self,
        record: ParsedResponseRecord | None = None,
        metrics: dict[str, "BaseMetric"] | None = None,
    ):
        self._check_record(record)
        if record.token_count is not None:
            self.metric.append(record.token_count)

    def values(self):
        """
        Returns the list of Output Sequence Length (OSL) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord):
        """
        Checks if the record is valid for OSL calculation.

        Raises:
            ValueError: If the record is not valid
        """
        if not record or not record.valid:
            raise ValueError("Invalid Record")

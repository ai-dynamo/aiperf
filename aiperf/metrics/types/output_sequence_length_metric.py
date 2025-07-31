# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricTag, MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.metrics.base_metric import BaseMetric


class OutputSequenceLengthMetric(BaseMetric):
    """
    Post-processor for calculating Output Sequence Length (OSL) metrics from records.
    """

    tag = MetricTag.OSL
    unit = None
    larger_is_better = False
    header = "Output Sequence Length"
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
        self.metric.append(record.output_token_count)

    def values(self):
        """
        Returns the list of Output Sequence Length (OSL) metrics.
        """
        return self.metric

    def _check_record(self, record: ParsedResponseRecord):
        """
        Checks if the record is valid for OSL calculation.

        Raises:
            ValueError: If record is not valid or output_token_count is missing.
        """
        self._require_valid_record(record)
        if record.output_token_count is None:
            raise ValueError("Output token count is missing in the record.")

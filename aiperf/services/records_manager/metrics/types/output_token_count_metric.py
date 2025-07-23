# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricType
from aiperf.common.models import ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


class OutputTokenCountMetric(BaseMetric):
    tag = "output_token_count"
    unit = None
    larger_is_better = True
    header = "Output Token Count"
    type = MetricType.METRIC_OF_RECORDS
    required_metrics = set()

    def __init__(self):
        self.metric: list[int] = []

    def update_value(self, record: ParsedResponseRecord, metrics=None):
        self._check_record(record)
        self.metric.append(record.output_token_count)

    def values(self) -> list[int]:
        return self.metric

    def _check_record(self, record: ParsedResponseRecord):
        self._require_valid_record(record)
        if record.output_token_count is None or record.output_token_count < 0:
            raise ValueError("Record has invalid output token count.")

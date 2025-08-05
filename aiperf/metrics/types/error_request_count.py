# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class ErrorRequestCountMetric(BaseAggregateMetric[int]):
    """
    Post-processor for counting the number of error requests.

    This metric is only applicable to error records.

    Formula:
        Error Request Count = Sum(Error Requests)
    """

    tag = "error_request_count"
    header = "Error Request Count"
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.ERROR_ONLY | MetricFlags.HIDE_IF_ZERO
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        # We are guaranteed that the record is an error record, so we can return 1.
        return 1

    def _aggregate_value(self, value: int) -> None:
        """Aggregate the metric value. For this metric, we just sum the values from the different processes."""
        self._value += value

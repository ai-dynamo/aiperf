# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class RequestCountMetric(BaseAggregateMetric[int]):
    """
    Post-processor for counting the number of valid requests.

    Formula:
        Request Count = Sum(Valid Requests)
    """

    tag = "request_count"
    header = "Request Count"
    unit = GenericMetricUnit.REQUESTS
    display_order = 1000
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        # NOTE: We don't need to update the value here, because we are just counting the number of requests.
        #       The value is updated in the ResultsProcessor via the `_aggregate_value` method.
        return 1

    def _aggregate_value(self, value: int) -> None:
        """Aggregate the metric value. For this metric, we just sum the values from the different processes."""
        self._value += value

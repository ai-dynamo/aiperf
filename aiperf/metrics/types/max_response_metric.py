# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricDateTimeUnit, MetricFlags, MetricTimeUnit
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class MaxResponseTimestampMetric(BaseAggregateMetric[int]):
    """
    Post-processor for calculating the maximum response time stamp metric from records.

    Formula:
        Maximum Response Timestamp = Max(Final Response Timestamps)
    """

    tag = "max_response_timestamp"
    header = "Maximum Response Timestamp"
    short_header = "Max Resp"
    short_header_hide_unit = True
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricDateTimeUnit.DATE_TIME
    flags = MetricFlags.HIDDEN
    required_metrics = {
        RequestLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Updates the maximum response timestamp metric.
        """
        # Compute the final response timestamp by adding the request latency to the request timestamp.
        # We do this because we want wall-clock timestamps, and the only one we have that is wall-clock
        # time is the timestamp_ns for the start of the request, so we need to use that and work from there.
        request_latency: int = record_metrics.get_or_raise(RequestLatencyMetric)  # type: ignore
        final_response_ts = record.timestamp_ns + request_latency
        return final_response_ts

    def _aggregate_value(self, value: int) -> None:
        """Aggregate the metric value. For this metric, we just take the max of the values from the different processes."""
        if value > self._value:
            self._value = value

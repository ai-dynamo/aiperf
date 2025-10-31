# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class RequestLatencyMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Request Latency metrics from records.

    Formula:
        Request Latency = Final Response Timestamp - Request Start Timestamp
    """

    tag = "request_latency"
    header = "Request Latency"
    short_header = "Req Latency"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 300
    flags = MetricFlags.NONE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the request and last response timestamps, and calculates the differences in time.

        Note: Uses the last content response (with actual data), not usage-only chunks.
        """
        request_ts: int = record.start_perf_ns

        # Use content_responses to get last response with actual content
        if not record.content_responses:
            raise NoMetricValue(
                "Request latency requires at least 1 non-empty content response."
            )
        final_response_ts = record.content_responses[-1].perf_ns

        if final_response_ts < request_ts:
            raise ValueError("Final response timestamp is less than request timestamp.")
        return final_response_ts - request_ts

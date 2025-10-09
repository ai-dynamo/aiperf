# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TTFTMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Time to First Token (TTFT) metrics from records.

    Formula:
        TTFT = First Response Timestamp - Request Start Timestamp
    """

    tag = "time_to_first_token"
    header = "Time to First Token"
    short_header = "TTFT"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 100
    flags = MetricFlags.STREAMING_TOKENS_ONLY
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the timestamps from the request start and the first response in the given
        RequestRecord object, computes the difference (TTFT), and returns the result.

        Raises:
            NoMetricValue: If the record does not have at least one response
            ValueError: If the first response is before the request start timestamp.
        """

        if len(record.responses) < 1:
            raise NoMetricValue(
                "Record must have at least one response to calculate TTFT."
            )

        request_ts: int = record.request.start_perf_ns
        first_response_ts: int = record.responses[0].perf_ns
        if first_response_ts < request_ts:
            raise ValueError(
                "First response timestamp is before request start timestamp, cannot compute TTFT."
            )

        return first_response_ts - request_ts

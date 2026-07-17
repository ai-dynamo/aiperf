# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.constants import STREAMED_REQUEST_TAG
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
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
    )
    required_metrics = {STREAMED_REQUEST_TAG}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the timestamps from the request start and the first response in the given
        RequestRecord object, computes the difference (TTFT), and returns the result.

        Raises:
            NoMetricValue: If the record did not stream, or does not have at least one content response
            ValueError: If the first response is before the request start timestamp.
        """
        if STREAMED_REQUEST_TAG not in record_metrics:
            raise NoMetricValue("record did not stream; streaming metrics skipped")

        if len(record.content_responses) < 1:
            raise NoMetricValue(
                "Record must have at least one content response to calculate TTFT."
            )

        request_ts: int = record.request.start_perf_ns
        first_response_ts: int = record.content_responses[0].perf_ns
        if first_response_ts < request_ts:
            raise ValueError(
                "First response timestamp is before request start timestamp, cannot compute TTFT."
            )

        return first_response_ts - request_ts

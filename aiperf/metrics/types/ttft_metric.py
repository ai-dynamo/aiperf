# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TTFTMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Time to First Token (TTFT) metrics from records.

    Formula:
        TTFT = First Response Timestamp - Request Start Timestamp

    Alternatively, this is equivalent to:
        TTFT = Stream Setup Latency + Stream Prefill Latency
    """

    tag = "ttft"
    header = "Time to First Token"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
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
            ValueError: If the record does not have at least one response.
        """

        if len(record.responses) < 1:
            raise ValueError(
                "Record must have at least one response to calculate TTFT."
            )

        request_ts: int = record.request.start_perf_ns
        first_response_ts: int = record.responses[0].perf_ns
        if first_response_ts < request_ts:
            raise ValueError(
                "First response timestamp is before request start timestamp, cannot compute TTFT."
            )

        return first_response_ts - request_ts

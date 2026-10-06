# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class DecodeDurationMetric(BaseRecordMetric[int]):
    """Client-observed interval from the first to final content response."""

    tag = "decode_duration"
    header = "Decode Duration"
    short_header = "Decode Duration"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 350
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
    )
    required_metrics = {
        RequestLatencyMetric.tag,
        TTFTMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        request_latency = record_metrics.get_or_raise(RequestLatencyMetric)
        ttft = record_metrics.get_or_raise(TTFTMetric)
        decode_duration = request_latency - ttft  # type: ignore
        if decode_duration < 0:
            raise ValueError("Request latency is less than time to first token.")
        return decode_duration


class FullDecodeDurationMetric(BaseRecordMetric[int]):
    """Client-observed interval from first content to full request completion."""

    tag = "full_decode_duration"
    header = "Full Decode Duration"
    short_header = "Full Decode Duration"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 360
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        if not record.content_responses:
            raise NoMetricValue(
                "Full decode duration requires at least 1 non-empty content response."
            )
        if record.request.end_perf_ns is None:
            raise NoMetricValue(
                "Full decode duration requires an explicit request end timestamp."
            )

        first_content_ts = record.content_responses[0].perf_ns
        duration = record.request.end_perf_ns - first_content_ts
        if duration < 0:
            raise ValueError("Request end timestamp is before first content response.")
        return duration

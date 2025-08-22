# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.ttft_metric import TTFTMetric


class StreamSetupLatencyMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Stream Setup Latency metrics from records. This is only applicable to streaming responses.

    This is the time it takes for the client to send the request and receive the 200 OK response from the server,
    before any SSE content is received. It measures the tcp/http connection time, request processing, and stream initialization time.

    Note that not all servers will respond with a 200 OK response as soon as the stream is established.
    For example, some servers will wait for the first token to be ready before sending the 200 OK response.
    In these cases, the stream setup latency will not be meaningful.

    Formula:
        Stream Setup Latency = Stream Start Timestamp - Request Start Timestamp
    """

    tag = "stream_setup_latency"
    header = "Stream Setup Latency"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.STREAMING_ONLY | MetricFlags.EXPERIMENTAL
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """This method extracts the request and receive start timestamps, and calculates the stream setup time."""

        if not record.request.recv_start_perf_ns or not record.start_perf_ns:
            raise NoMetricValue(
                "Stream setup latency metric requires a recv_start_perf_ns and start_perf_ns"
            )

        if record.request.recv_start_perf_ns < record.start_perf_ns:
            raise ValueError("recv_start_perf_ns is less than start_perf_ns")

        return record.request.recv_start_perf_ns - record.start_perf_ns


class StreamPrefillLatencyMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Stream Prefill Latency metrics from records. This is only applicable to streaming responses.

    This is the time it takes for the server to process the input prompt and begin streaming content,
    after the stream has been established (200 OK response received). This is an alternate version of the
    TTFT metric, which removes some of the connection and stream setup overhead, sometimes referred to as "Pure TTFT".

    Note that not all servers will respond with a 200 OK response as soon as the stream is established.
    For example, some servers will wait for the first token to be ready before sending the 200 OK response.
    In these cases, the stream prefill latency will not be meaningful.

    Formula:
        Stream Prefill Latency = Time to First Token - Stream Setup Latency
    """

    tag = "stream_prefill_latency"
    header = "Stream Prefill Latency"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.EXPERIMENTAL
    required_metrics = {
        StreamSetupLatencyMetric.tag,
        TTFTMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """This method calculates the stream prefill latency by subtracting the stream setup latency from the TTFT."""

        stream_setup_latency = record_metrics.get_or_raise(StreamSetupLatencyMetric)
        ttft = record_metrics.get_or_raise(TTFTMetric)

        return ttft - stream_setup_latency  # type: ignore

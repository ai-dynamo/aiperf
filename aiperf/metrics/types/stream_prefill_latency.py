# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.stream_setup_latency import StreamSetupLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


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

        stream_setup_latency = record_metrics.get(StreamSetupLatencyMetric.tag)
        if not stream_setup_latency:
            raise NoMetricValue("Stream setup latency is not available for the record.")

        ttft = record_metrics.get(TTFTMetric.tag)
        if not ttft:
            raise NoMetricValue("Time to first token is not available for the record.")

        return ttft - stream_setup_latency  # type: ignore

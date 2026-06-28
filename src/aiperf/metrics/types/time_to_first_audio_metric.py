# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TimeToFirstAudioMetric(BaseRecordMetric[int]):
    """Time to First Audio (TTFA) for streaming text-to-speech responses.

    The TTS counterpart of Time to First Token: the latency from sending the
    request to receiving the first audio chunk. Only meaningful when
    streaming is enabled, since a non-streaming response delivers the entire
    clip at once.

    Formula:
        TTFA = First Audio Chunk Timestamp - Request Start Timestamp
    """

    tag = "time_to_first_audio"
    header = "Time to First Audio"
    short_header = "TTFA"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 100
    flags = MetricFlags.PRODUCES_AUDIO_ONLY | MetricFlags.STREAMING_ONLY
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        if len(record.content_responses) < 1:
            raise NoMetricValue(
                "Record must have at least one audio response to calculate TTFA."
            )

        request_ts: int = record.request.start_perf_ns
        first_response_ts: int = record.content_responses[0].perf_ns
        if first_response_ts < request_ts:
            raise ValueError(
                "First audio response timestamp is before request start timestamp, "
                "cannot compute TTFA."
            )

        return first_response_ts - request_ts

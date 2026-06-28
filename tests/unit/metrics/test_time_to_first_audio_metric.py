# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    AudioResponseData,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.time_to_first_audio_metric import TimeToFirstAudioMetric


def _record(start_ns: int, chunk_perf_ns: list[int]) -> ParsedResponseRecord:
    request = RequestRecord(
        model_name="tts-1",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=chunk_perf_ns[-1] if chunk_perf_ns else start_ns,
    )
    responses = [
        ParsedResponse(perf_ns=ns, data=AudioResponseData(audio_bytes=b"\x00\x01"))
        for ns in chunk_perf_ns
    ]
    return ParsedResponseRecord(request=request, responses=responses)


class TestTimeToFirstAudioMetric:
    def test_ttfa_basic(self):
        record = _record(start_ns=1000, chunk_perf_ns=[1500, 1800, 2100])
        result = TimeToFirstAudioMetric().parse_record(record, MetricRecordDict())
        assert result == 500

    def test_ttfa_no_responses_raises(self):
        # A record with no content responses is gated as invalid upstream.
        record = _record(start_ns=1000, chunk_perf_ns=[])
        with pytest.raises(NoMetricValue):
            TimeToFirstAudioMetric().parse_record(record, MetricRecordDict())

    def test_metric_properties(self):
        metric = TimeToFirstAudioMetric()
        assert metric.tag == "time_to_first_audio"
        assert metric.short_header == "TTFA"
        assert MetricFlags.PRODUCES_AUDIO_ONLY in metric.flags
        assert MetricFlags.STREAMING_ONLY in metric.flags

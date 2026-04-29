# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import Turn
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.rtfx_metric import RTFxMetric
from tests.unit.metrics.conftest import create_record


def _record_with_audio_duration(audio_duration: float | None) -> "object":
    record = create_record()
    record.request.turns = [Turn(audio_duration_seconds=audio_duration)]
    return record


class TestRTFxMetric:
    def test_rtfx_basic(self):
        """10s audio, 1s latency -> RTFx = 10."""
        record = _record_with_audio_duration(10.0)
        metric = RTFxMetric()
        metric_dict = MetricRecordDict()
        metric_dict[RequestLatencyMetric.tag] = 1_000_000_000  # 1s in ns

        result = metric.parse_record(record, metric_dict)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_rtfx_various_values(self):
        metric = RTFxMetric()
        cases = [
            (5.0, 500_000_000, 10.0),  # 5s audio, 0.5s latency -> 10x
            (60.0, 12_000_000_000, 5.0),  # 60s audio, 12s latency -> 5x
            (
                1.0,
                2_000_000_000,
                0.5,
            ),  # 1s audio, 2s latency -> 0.5x (slower than real-time)
            (30.0, 100_000_000, 300.0),  # 30s audio, 100ms latency -> 300x
        ]
        for audio_dur, latency_ns, expected in cases:
            record = _record_with_audio_duration(audio_dur)
            md = MetricRecordDict()
            md[RequestLatencyMetric.tag] = latency_ns
            assert metric.parse_record(record, md) == pytest.approx(expected, rel=1e-6)

    def test_rtfx_no_audio_duration_raises_no_metric_value(self):
        record = _record_with_audio_duration(None)
        metric = RTFxMetric()
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 1_000_000_000
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, md)

    def test_rtfx_zero_audio_duration_raises_no_metric_value(self):
        record = _record_with_audio_duration(0.0)
        metric = RTFxMetric()
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 1_000_000_000
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, md)

    def test_rtfx_zero_latency_raises_no_metric_value(self):
        record = _record_with_audio_duration(5.0)
        metric = RTFxMetric()
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 0
        with pytest.raises(NoMetricValue, match="latency is zero"):
            metric.parse_record(record, md)

    def test_rtfx_no_turns_raises_no_metric_value(self):
        record = create_record()
        record.request.turns = []
        metric = RTFxMetric()
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 1_000_000_000
        with pytest.raises(NoMetricValue, match="No turns"):
            metric.parse_record(record, md)

    def test_rtfx_metric_properties(self):
        metric = RTFxMetric()
        assert metric.tag == "rtfx"
        assert metric.header == "Real-Time Factor (RTFx)"
        assert metric.short_header == "RTFx"
        assert metric.short_header_hide_unit is True
        assert RequestLatencyMetric.tag in metric.required_metrics

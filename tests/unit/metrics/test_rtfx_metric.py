# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.audio_duration_metric import AudioDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.rtfx_metric import RTFxMetric
from tests.unit.metrics.conftest import create_record


def _metric_dict(audio_duration: float, latency_ns: int) -> MetricRecordDict:
    md = MetricRecordDict()
    md[AudioDurationMetric.tag] = audio_duration
    md[RequestLatencyMetric.tag] = latency_ns
    return md


class TestRTFxMetric:
    def test_rtfx_basic(self):
        """10s audio, 1s latency -> RTFx = 10."""
        record = create_record()
        metric = RTFxMetric()
        md = _metric_dict(10.0, 1_000_000_000)

        result = metric.parse_record(record, md)
        assert result == pytest.approx(10.0, rel=1e-6)

    @pytest.mark.parametrize(
        "audio_duration_s,latency_ns,expected_rtfx",
        [
            param(5.0, 500_000_000, 10.0, id="5s_audio_500ms_latency_10x"),
            param(60.0, 12_000_000_000, 5.0, id="60s_audio_12s_latency_5x"),
            param(1.0, 2_000_000_000, 0.5, id="1s_audio_2s_latency_slower_than_realtime"),
            param(30.0, 100_000_000, 300.0, id="30s_audio_100ms_latency_300x"),
        ],
    )  # fmt: skip
    def test_rtfx_various_values(self, audio_duration_s, latency_ns, expected_rtfx):
        md = _metric_dict(audio_duration_s, latency_ns)
        assert RTFxMetric().parse_record(create_record(), md) == pytest.approx(
            expected_rtfx, rel=1e-6
        )

    def test_rtfx_no_audio_duration_raises_no_metric_value(self):
        record = create_record()
        metric = RTFxMetric()
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 1_000_000_000
        with pytest.raises(NoMetricValue, match="audio_duration"):
            metric.parse_record(record, md)

    def test_rtfx_zero_latency_raises_no_metric_value(self):
        record = create_record()
        metric = RTFxMetric()
        md = _metric_dict(5.0, 0)
        with pytest.raises(NoMetricValue, match="non-positive"):
            metric.parse_record(record, md)

    def test_rtfx_metric_properties(self):
        metric = RTFxMetric()
        assert metric.tag == "rtfx"
        assert metric.header == "Inverse Real-Time Factor (RTFx)"
        assert metric.short_header == "RTFx"
        assert metric.short_header_hide_unit is True
        assert AudioDurationMetric.tag in metric.required_metrics
        assert RequestLatencyMetric.tag in metric.required_metrics
        assert MetricFlags.SUPPORTS_AUDIO_ONLY in metric.flags
        assert MetricFlags.LARGER_IS_BETTER in metric.flags

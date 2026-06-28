# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_audio_duration_metric import OutputAudioDurationMetric
from aiperf.metrics.types.output_rtf_metric import OutputRTFMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record


def _metric_dict(audio_duration_s: float, latency_ns: int) -> MetricRecordDict:
    md = MetricRecordDict()
    md[OutputAudioDurationMetric.tag] = audio_duration_s
    md[RequestLatencyMetric.tag] = latency_ns
    return md


class TestOutputRTFMetric:
    @pytest.mark.parametrize(
        "audio_duration_s,latency_ns,expected_rtf",
        [
            param(10.0, 2_000_000_000, 0.2, id="10s_audio_2s_latency_5x_realtime"),
            param(5.0, 5_000_000_000, 1.0, id="realtime"),
            param(1.0, 2_000_000_000, 2.0, id="slower_than_realtime"),
        ],
    )  # fmt: skip
    def test_rtf_values(self, audio_duration_s, latency_ns, expected_rtf):
        md = _metric_dict(audio_duration_s, latency_ns)
        result = OutputRTFMetric().parse_record(create_record(), md)
        assert result == pytest.approx(expected_rtf, rel=1e-6)

    def test_zero_audio_duration_raises(self):
        md = _metric_dict(0.0, 1_000_000_000)
        with pytest.raises(NoMetricValue, match="non-positive"):
            OutputRTFMetric().parse_record(create_record(), md)

    def test_missing_audio_duration_raises(self):
        md = MetricRecordDict()
        md[RequestLatencyMetric.tag] = 1_000_000_000
        with pytest.raises(NoMetricValue):
            OutputRTFMetric().parse_record(create_record(), md)

    def test_metric_properties(self):
        metric = OutputRTFMetric()
        assert metric.tag == "output_rtf"
        assert metric.short_header == "RTF"
        assert OutputAudioDurationMetric.tag in metric.required_metrics
        assert RequestLatencyMetric.tag in metric.required_metrics
        assert MetricFlags.PRODUCES_AUDIO_ONLY in metric.flags
        # Lower is better -> must NOT be flagged larger-is-better.
        assert MetricFlags.LARGER_IS_BETTER not in metric.flags

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import Turn
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.audio_duration_metric import AudioDurationMetric
from tests.unit.metrics.conftest import create_record


class TestAudioDurationMetric:
    def test_returns_audio_duration(self):
        record = create_record()
        record.request.turns = [Turn(audio_duration_seconds=12.5)]
        metric = AudioDurationMetric()

        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(12.5, rel=1e-6)

    def test_no_turns_raises(self):
        record = create_record()
        record.request.turns = []
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="No turns"):
            metric.parse_record(record, MetricRecordDict())

    def test_no_audio_duration_raises(self):
        record = create_record()
        record.request.turns = [Turn(audio_duration_seconds=None)]
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, MetricRecordDict())

    def test_zero_audio_duration_raises(self):
        record = create_record()
        record.request.turns = [Turn(audio_duration_seconds=0.0)]
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, MetricRecordDict())

    def test_metric_properties(self):
        metric = AudioDurationMetric()
        assert metric.tag == "audio_duration"
        assert metric.header == "Audio Duration"
        assert MetricFlags.NO_CONSOLE in metric.flags
        assert MetricFlags.SUPPORTS_AUDIO_ONLY in metric.flags

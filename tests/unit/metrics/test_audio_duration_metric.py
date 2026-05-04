# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.audio_duration_metric import AudioDurationMetric
from tests.unit.metrics.conftest import create_record


class TestAudioDurationMetric:
    def test_returns_audio_duration(self):
        record = create_record()
        record.request.request_info.audio_duration_seconds = 12.5
        metric = AudioDurationMetric()

        result = metric.parse_record(record, MetricRecordDict())
        assert result == pytest.approx(12.5, rel=1e-6)

    def test_no_request_info_raises(self):
        record = create_record()
        record.request.request_info = None
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="no request_info"):
            metric.parse_record(record, MetricRecordDict())

    def test_no_audio_duration_raises(self):
        record = create_record()
        record.request.request_info.audio_duration_seconds = None
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, MetricRecordDict())

    def test_zero_audio_duration_raises(self):
        record = create_record()
        record.request.request_info.audio_duration_seconds = 0.0
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue, match="ASR requests only"):
            metric.parse_record(record, MetricRecordDict())

    def test_default_text_only_record_raises_no_metric_value(self):
        """Regression: a plain text-only record (no audio fields set anywhere)
        must raise ``NoMetricValue`` — not ``AttributeError`` — so the record
        processor's ``except NoMetricValue`` branch swallows it silently
        instead of logging a per-record warning.
        """
        record = create_record()
        # Default request_info from the fixture has no audio_duration_seconds.
        assert record.request.request_info is not None
        assert record.request.request_info.audio_duration_seconds is None
        metric = AudioDurationMetric()
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())

    def test_metric_properties(self):
        metric = AudioDurationMetric()
        assert metric.tag == "audio_duration"
        assert metric.header == "Audio Duration"
        assert metric.console_group == MetricConsoleGroup.NONE
        assert MetricFlags.SUPPORTS_AUDIO_ONLY in metric.flags

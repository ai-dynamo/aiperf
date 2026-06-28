# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.audio_throughput_metric import AudioThroughputMetric
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.output_audio_duration_metric import (
    TotalOutputAudioDurationMetric,
)


class TestAudioThroughputMetric:
    def test_throughput_calculation(self):
        """10s of audio generated in 2s wall-clock -> 5 audio_sec/sec."""
        metric_results = MetricResultsDict()
        metric_results[TotalOutputAudioDurationMetric.tag] = 10.0
        metric_results[BenchmarkDurationMetric.tag] = 2_000_000_000  # 2s in ns
        result = AudioThroughputMetric().derive_value(metric_results)
        assert result == pytest.approx(5.0, rel=1e-6)

    def test_throughput_exceeds_realtime_under_concurrency(self):
        metric_results = MetricResultsDict()
        metric_results[TotalOutputAudioDurationMetric.tag] = 100.0
        metric_results[BenchmarkDurationMetric.tag] = 2_000_000_000
        result = AudioThroughputMetric().derive_value(metric_results)
        assert result == pytest.approx(50.0, rel=1e-6)

    def test_zero_duration_raises(self):
        metric_results = MetricResultsDict()
        metric_results[TotalOutputAudioDurationMetric.tag] = 10.0
        metric_results[BenchmarkDurationMetric.tag] = 0.0
        with pytest.raises(NoMetricValue):
            AudioThroughputMetric().derive_value(metric_results)

    def test_metric_properties(self):
        metric = AudioThroughputMetric()
        assert metric.tag == "audio_throughput"
        assert TotalOutputAudioDurationMetric.tag in metric.required_metrics
        assert BenchmarkDurationMetric.tag in metric.required_metrics
        assert MetricFlags.PRODUCES_AUDIO_ONLY in metric.flags
        assert MetricFlags.LARGER_IS_BETTER in metric.flags

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.output_audio_duration_metric import (
    TotalOutputAudioDurationMetric,
)


class AudioThroughputMetric(BaseDerivedMetric[float]):
    """Aggregate audio throughput for text-to-speech benchmarks.

    Formula:
        Audio Throughput = Total Output Audio Duration (seconds) / Benchmark Duration (seconds)

    Larger is better. Expressed as seconds-of-audio synthesized per
    wall-clock second across the whole run, so under concurrency it can far
    exceed 1.0 (e.g. 50 means the fleet produces 50s of audio every second).
    """

    tag = "audio_throughput"
    header = "Audio Throughput"
    short_header = "Audio Thpt"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.AUDIO_SECONDS_PER_SECOND
    display_order = 800
    flags = MetricFlags.PRODUCES_AUDIO_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        TotalOutputAudioDurationMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_audio_seconds = metric_results.get_or_raise(
            TotalOutputAudioDurationMetric
        )
        benchmark_duration_converted = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        if benchmark_duration_converted == 0:
            raise NoMetricValue(
                "Benchmark duration is zero, cannot calculate audio throughput metric"
            )
        return total_audio_seconds / benchmark_duration_converted  # type: ignore

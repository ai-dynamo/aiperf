# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated-audio metrics, distinct from ASR input-audio duration and RTFx."""

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    MetricOverTimeUnit,
    MetricTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import AudioResponseData, ParsedResponseRecord
from aiperf.metrics import BaseDerivedMetric, BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class OutputAudioDurationMetric(BaseRecordMetric[float]):
    """Duration of validated, successfully completed generated audio."""

    tag = "output_audio_duration"
    header = "Output Audio Duration"
    short_header = "Output Audio"
    unit = MetricTimeUnit.SECONDS
    display_order = 870
    flags = MetricFlags.PRODUCES_AUDIO_ONLY
    console_group = MetricConsoleGroup.AUDIO
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Convert complete sample frames to seconds without decoding audio again."""
        audio = [
            r.data for r in record.responses if isinstance(r.data, AudioResponseData)
        ]
        if not audio or any(a.frames <= 0 or a.sample_rate <= 0 for a in audio):
            raise NoMetricValue("No valid output audio duration")
        return sum(a.frames / a.sample_rate for a in audio)


class TotalOutputAudioDurationMetric(
    DerivedSumMetric[float, OutputAudioDurationMetric]
):
    """Sum of output audio seconds from successful requests only."""

    tag = "total_output_audio_duration"
    header = "Total Output Audio Duration"
    short_header = "Total Output Audio"
    console_group = MetricConsoleGroup.NONE


class OutputAudioRTFMetric(BaseRecordMetric[float]):
    """Complete-response latency / output audio duration; lower is better."""

    tag = "output_audio_rtf"
    header = "Output Audio Real-Time Factor"
    short_header = "Output Audio RTF"
    unit = GenericMetricUnit.RATIO
    display_order = 880
    flags = MetricFlags.PRODUCES_AUDIO_ONLY
    console_group = MetricConsoleGroup.AUDIO
    required_metrics = {OutputAudioDurationMetric.tag, RequestLatencyMetric.tag}

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> float:
        """Use generated duration rather than the request's input-audio duration."""
        duration = record_metrics.get_or_raise(OutputAudioDurationMetric)
        latency = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, MetricTimeUnit.SECONDS
        )
        if duration <= 0 or latency <= 0:
            raise NoMetricValue(
                "Output audio RTF requires positive duration and latency"
            )
        return latency / duration


class OutputAudioThroughputMetric(BaseDerivedMetric[float]):
    """Successful output audio seconds per observation wall second."""

    tag = "output_audio_throughput"
    header = "Output Audio Throughput"
    short_header = "Output Audio/sec"
    unit = MetricOverTimeUnit.AUDIO_SECONDS_PER_SECOND
    display_order = 890
    flags = MetricFlags.PRODUCES_AUDIO_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.AUDIO
    required_metrics = {TotalOutputAudioDurationMetric.tag, BenchmarkDurationMetric.tag}

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        """Use the common observation window, including time spent on failures."""
        total = metric_results.get_or_raise(TotalOutputAudioDurationMetric)
        duration = metric_results.observation_duration(MetricTimeUnit.SECONDS)
        return total / duration

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_audio_duration_metric import OutputAudioDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class OutputRTFMetric(BaseRecordMetric[float]):
    """Real-Time Factor (RTF) for text-to-speech benchmarks.

    Formula:
        RTF = request_latency_seconds / output_audio_duration_seconds

    Lower is better; the standard TTS quality-of-service metric. RTF < 1.0
    means the server synthesizes audio faster than it plays back (suitable
    for real-time streaming); RTF > 1.0 means it is slower than real-time.
    This is the inverse of the ASR ``RTFx`` convention - TTS literature
    reports RTF (smaller is better), so we follow that here.

    Example:
        A 10s clip produced with 2s of request latency -> RTF = 0.2
        ("5x faster than real-time").

    Requires ``OutputAudioDurationMetric`` and ``RequestLatencyMetric`` to
    be computed first.

    Raises:
        NoMetricValue: when the output audio duration is missing or
            non-positive, so the ratio is undefined.
    """

    tag = "output_rtf"
    header = "Real-Time Factor (RTF)"
    short_header = "RTF"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 850
    flags = MetricFlags.PRODUCES_AUDIO_ONLY
    required_metrics = {OutputAudioDurationMetric.tag, RequestLatencyMetric.tag}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        audio_duration = record_metrics.get_or_raise(OutputAudioDurationMetric)
        if audio_duration <= 0:
            raise NoMetricValue(
                f"Output audio duration is non-positive ({audio_duration}s); RTF undefined."
            )

        latency_seconds = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, MetricTimeUnit.SECONDS
        )
        return latency_seconds / audio_duration

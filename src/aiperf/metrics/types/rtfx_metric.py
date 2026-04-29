# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.audio_duration_metric import AudioDurationMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class RTFxMetric(BaseRecordMetric[float]):
    """Inverse Real-Time Factor (RTFx) for ASR benchmarks.

    Formula:
        RTFx = audio_duration_seconds / request_latency_seconds

    Higher is better; expressed as "Nx faster than real-time." This is the
    industry-standard ASR throughput metric (HuggingFace Open ASR Leaderboard
    requires it; NVIDIA Riva and NeMo use it as headline metric).

    Example:
        10s of input audio transcribed with 1s request latency -> RTFx = 10.0
        ("10x faster than real-time"). RTFx < 1.0 means the server is slower
        than real-time and not suitable for live transcription.

    Requires ``AudioDurationMetric`` and ``RequestLatencyMetric`` to be
    computed first. Non-ASR requests (no audio duration) yield no metric value.

    Raises:
        NoMetricValue: when ``audio_duration`` is missing from the record
            metrics, or the measured request latency is non-positive.
    """

    tag = "rtfx"
    header = "Inverse Real-Time Factor (RTFx)"
    short_header = "RTFx"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 850
    flags = MetricFlags.SUPPORTS_AUDIO_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {AudioDurationMetric.tag, RequestLatencyMetric.tag}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        audio_duration = record_metrics.get_or_raise(AudioDurationMetric)

        latency_seconds = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, MetricTimeUnit.SECONDS
        )
        if latency_seconds <= 0:
            raise NoMetricValue(
                f"Request latency is non-positive ({latency_seconds}s); "
                "RTFx undefined. Likely an upstream measurement bug."
            )

        return audio_duration / latency_seconds

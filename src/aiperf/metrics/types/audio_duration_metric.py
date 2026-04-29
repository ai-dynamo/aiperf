# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class AudioDurationMetric(BaseRecordMetric[float]):
    """Per-request input audio duration in seconds.

    Surfaces the duration set by ASR dataset loaders so it appears in
    JSON / CSV record exports alongside other per-request metrics. Hidden
    from the console summary (the headline is RTFx); aggregate stats
    (avg, p50, p99) are still computed automatically and available for
    characterizing dataset shape.

    Computed only when the request's first turn carries
    ``audio_duration_seconds``. Non-ASR requests yield no metric value.
    """

    tag = "audio_duration"
    header = "Audio Duration"
    short_header = "Audio Dur"
    unit = MetricTimeUnit.SECONDS
    display_order = 860
    flags = MetricFlags.NO_CONSOLE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        turns = record.request.turns
        if not turns:
            raise NoMetricValue("No turns in request; audio duration unavailable.")

        audio_duration = turns[0].audio_duration_seconds
        if audio_duration is None or audio_duration <= 0:
            raise NoMetricValue(
                "Turn has no audio_duration_seconds; metric applies to ASR requests only."
            )

        return audio_duration

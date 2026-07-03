# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricConsoleGroup, MetricFlags, MetricTimeUnit
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

    Example:
        A 12.5s audio clip produces audio_duration = 12.5. Useful for
        correlating latency with clip length and verifying RTFx post-hoc.

    Read from ``record.request.request_info.audio_duration_seconds``,
    which is hoisted off the originating turn at record-enrichment time
    (see ``inference_client._finalize_request_record``). The full ``turns``
    list does not cross the ZMQ hop to the record processor, so reading
    ``record.request.turns`` here would AttributeError on every record.
    Non-ASR requests yield no metric value.

    Raises:
        NoMetricValue: when the record has no ``request_info``, or
            ``request_info.audio_duration_seconds`` is missing or
            non-positive.
    """

    tag = "audio_duration"
    header = "Audio Duration"
    short_header = "Audio Dur"
    unit = MetricTimeUnit.SECONDS
    display_order = 870
    flags = MetricFlags.SUPPORTS_AUDIO_ONLY
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        request_info = record.request.request_info
        if request_info is None:
            raise NoMetricValue(
                "Record has no request_info; audio_duration unavailable."
            )

        audio_duration = request_info.audio_duration_seconds
        if audio_duration is None or audio_duration <= 0:
            raise NoMetricValue(
                "Record has no audio_duration_seconds; audio_duration metric applies to ASR requests only."
            )

        return audio_duration

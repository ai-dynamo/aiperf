# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class RTFxMetric(BaseRecordMetric[float]):
    """Inverse Real-Time Factor (RTFx) for ASR benchmarks.

    Formula:
        RTFx = audio_duration_seconds / request_latency_seconds

    Higher is better; expressed as "Nx faster than real-time." This is the
    industry-standard ASR throughput metric (HuggingFace Open ASR Leaderboard
    requires it; NVIDIA Riva and NeMo use it as headline metric).

    Computed only when the request's first turn carries
    ``audio_duration_seconds``. Non-ASR requests yield no metric value.
    """

    tag = "rtfx"
    header = "Real-Time Factor (RTFx)"
    short_header = "RTFx"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    display_order = 850
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = {RequestLatencyMetric.tag}

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        turns = record.request.turns
        if not turns:
            raise NoMetricValue("No turns in request; cannot compute RTFx.")

        audio_duration = turns[0].audio_duration_seconds
        if audio_duration is None or audio_duration <= 0:
            raise NoMetricValue(
                "Turn has no audio_duration_seconds; RTFx applies to ASR requests only."
            )

        latency_seconds = record_metrics.get_converted_or_raise(
            RequestLatencyMetric, MetricTimeUnit.SECONDS
        )
        if latency_seconds <= 0:
            raise NoMetricValue("Request latency is zero; cannot compute RTFx.")

        return audio_duration / latency_seconds

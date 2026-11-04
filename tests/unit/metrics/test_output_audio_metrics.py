# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output-audio units, failures and observation-window accounting."""

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    AudioResponseData,
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
)
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.output_audio_metrics import (
    OutputAudioDurationMetric,
    OutputAudioRTFMetric,
    OutputAudioThroughputMetric,
    TotalOutputAudioDurationMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record


def audio_record(error: ErrorDetails | None = None) -> ParsedResponseRecord:
    """Two seconds of stereo audio whose complete response takes half a second."""
    record = create_record(start_ns=0, responses=[500_000_000], error=error)
    record.responses = [
        ParsedResponse(
            perf_ns=500_000_000,
            data=AudioResponseData(frames=96000, sample_rate=48000, channels=2),
        )
    ]
    return record


def test_duration_and_rtf_have_output_audio_units() -> None:
    """The channel count does not inflate duration; RTF uses seconds for both terms."""
    record = audio_record()
    metrics = MetricRecordDict()
    metrics[OutputAudioDurationMetric.tag] = OutputAudioDurationMetric().parse_record(
        record, metrics
    )
    metrics[RequestLatencyMetric.tag] = 500_000_000
    assert metrics[OutputAudioDurationMetric.tag] == 2.0
    assert OutputAudioRTFMetric().parse_record(record, metrics) == 0.25


def test_failed_audio_contributes_no_duration() -> None:
    """Even a response with valid samples is excluded if the request failed."""
    record = audio_record(ErrorDetails(code=500, type="ServerError", message="failed"))
    with pytest.raises(NoMetricValue):
        OutputAudioDurationMetric().parse_record(record, MetricRecordDict())


def test_non_audio_record_has_no_output_duration() -> None:
    """Ordinary text responses cannot fabricate audio duration."""
    with pytest.raises(NoMetricValue):
        OutputAudioDurationMetric().parse_record(create_record(), MetricRecordDict())


def test_throughput_uses_audio_sum_over_wall_duration() -> None:
    """Aggregate audio throughput is not an average of per-request inverse RTFs."""
    metrics = MetricResultsDict()
    metrics[OutputAudioDurationMetric.tag] = 12.0
    metrics[TotalOutputAudioDurationMetric.tag] = (
        TotalOutputAudioDurationMetric().derive_value(metrics)
    )
    metrics[BenchmarkDurationMetric.tag] = 3_000_000_000
    assert OutputAudioThroughputMetric().derive_value(metrics) == 4.0


def test_zero_observation_window_has_no_throughput() -> None:
    """An empty measurement window must not yield infinity."""
    metrics = MetricResultsDict()
    metrics[TotalOutputAudioDurationMetric.tag] = 12.0
    metrics[BenchmarkDurationMetric.tag] = 0
    with pytest.raises(NoMetricValue):
        OutputAudioThroughputMetric().derive_value(metrics)


def test_zero_audio_duration_has_no_rtf() -> None:
    """Incomplete output must not yield an infinite ratio."""
    metrics = MetricRecordDict()
    metrics[OutputAudioDurationMetric.tag] = 0.0
    metrics[RequestLatencyMetric.tag] = 500_000_000
    with pytest.raises(NoMetricValue):
        OutputAudioRTFMetric().parse_record(audio_record(), metrics)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.decode_duration_metric import (
    DecodeDurationMetric,
    FullDecodeDurationMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestDecodeDurationMetric:
    def test_decode_duration_uses_first_and_last_content_timestamps(self):
        record = create_record(start_ns=100, responses=[120, 200])

        metric_results = run_simple_metrics_pipeline(
            [record],
            DecodeDurationMetric.tag,
        )

        assert metric_results[DecodeDurationMetric.tag] == [80]

    def test_decode_duration_single_content_response_is_zero(self):
        record = create_record(start_ns=100, responses=[120])

        metric_results = run_simple_metrics_pipeline(
            [record],
            DecodeDurationMetric.tag,
        )

        assert metric_results[DecodeDurationMetric.tag] == [0]

    def test_decode_duration_missing_required_metrics(self):
        record = create_record()

        with pytest.raises(NoMetricValue):
            DecodeDurationMetric().parse_record(record, MetricRecordDict())

    def test_decode_duration_rejects_negative_interval(self):
        record = create_record(start_ns=100, responses=[120])
        record_metrics = MetricRecordDict(
            {
                RequestLatencyMetric.tag: 10,
                TTFTMetric.tag: 20,
            }
        )

        with pytest.raises(
            ValueError,
            match="Request latency is less than time to first token",
        ):
            DecodeDurationMetric().parse_record(record, record_metrics)


class TestFullDecodeDurationMetric:
    def test_uses_first_content_and_explicit_request_end(self) -> None:
        record = create_record(start_ns=100, responses=[120, 200])
        record.request.end_perf_ns = 300

        metric_results = run_simple_metrics_pipeline(
            [record],
            FullDecodeDurationMetric.tag,
        )

        assert metric_results[FullDecodeDurationMetric.tag] == [180]

    def test_requires_content_response(self) -> None:
        record = create_record()
        record.responses = []

        with pytest.raises(NoMetricValue):
            FullDecodeDurationMetric().parse_record(record, MetricRecordDict())

    def test_requires_explicit_request_end(self) -> None:
        record = create_record(start_ns=100, responses=[120])
        record.request.end_perf_ns = None

        with pytest.raises(NoMetricValue, match="explicit request end timestamp"):
            FullDecodeDurationMetric().parse_record(record, MetricRecordDict())

    def test_rejects_request_end_before_first_content(self) -> None:
        record = create_record(start_ns=100, responses=[120])
        record.request.end_perf_ns = 110

        with pytest.raises(
            ValueError,
            match="Request end timestamp is before first content response",
        ):
            FullDecodeDurationMetric().parse_record(record, MetricRecordDict())

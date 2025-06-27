#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)


def test_update_value_and_values():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = RequestRecord(start_perf_ns=100, responses=[SSEMessage(perf_ns=150)])

    metric.update_value(record=record, metrics=None)
    assert metric.values() == [50]


def test_add_multiple_records():
    metric = RequestLatencyMetric()
    metric.metric = []
    records = [
        RequestRecord(
            start_perf_ns=10,
            responses=[SSEMessage(perf_ns=15), SSEMessage(perf_ns=25)],
        ),
        RequestRecord(
            start_perf_ns=20,
            responses=[SSEMessage(perf_ns=25), SSEMessage(perf_ns=35)],
        ),
        RequestRecord(
            start_perf_ns=30,
            responses=[SSEMessage(perf_ns=40), SSEMessage(perf_ns=50)],
        ),
    ]
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == [15, 15, 20]


def test_record_without_responses_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = RequestRecord(start_perf_ns=10)
    with pytest.raises(ValueError, match="at least one response"):
        metric.update_value(record=record, metrics=None)


def test_record_with_no_request_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = None
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)


def test_response_timestamp_less_than_request_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = RequestRecord(start_perf_ns=100, responses=[SSEMessage(perf_ns=90)])
    with pytest.raises(ValueError, match="Response timestamp must be greater"):
        metric.update_value(record=record, metrics=None)


def test_metric_initialization_none():
    metric = RequestLatencyMetric()
    assert metric.metric == []
    record = RequestRecord(start_perf_ns=1, responses=[SSEMessage(perf_ns=2)])
    metric.update_value(record=record, metrics=None)
    assert metric.values() == [1]

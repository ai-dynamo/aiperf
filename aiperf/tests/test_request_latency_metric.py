#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.request_latency_metric import (
    RequestLatencyMetric,
)
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_update_value_and_values():
    metric = RequestLatencyMetric()
    metric.metric = []
    request = MockRequest(timestamp=100)
    response = MockResponse(timestamp=150)
    record = MockRecord(request, [response])

    metric.update_value(record)
    assert metric.values() == [50]


def test_add_multiple_records():
    metric = RequestLatencyMetric()
    metric.metric = []
    records = [
        MockRecord(MockRequest(10), [MockResponse(15), MockResponse(25)]),
        MockRecord(MockRequest(20), [MockResponse(25), MockResponse(35)]),
        MockRecord(MockRequest(30), [MockResponse(40), MockResponse(50)]),
    ]
    for record in records:
        metric.update_value(record)
    assert metric.values() == [15, 15, 20]


def test_record_without_responses_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = MockRecord(MockRequest(10), [])
    with pytest.raises(ValueError, match="at least one response"):
        metric.update_value(record)


def test_record_with_no_request_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    record = MockRecord(None, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record)


def test_record_with_no_request_timestamp_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    request = MockRequest(None)
    record = MockRecord(request, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record)


def test_response_timestamp_less_than_request_raises():
    metric = RequestLatencyMetric()
    metric.metric = []
    request = MockRequest(100)
    response = MockResponse(90)
    record = MockRecord(request, [response])
    with pytest.raises(ValueError, match="Response timestamp must be greater"):
        metric.update_value(record)


def test_metric_initialization_none():
    metric = RequestLatencyMetric()
    assert metric.metric == []
    # After setting to list, works as expected
    metric.metric = []
    request = MockRequest(1)
    response = MockResponse(2)
    record = MockRecord(request, [response])
    metric.update_value(record)
    assert metric.values() == [1]

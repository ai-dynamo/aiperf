#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import MetricTimeType
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_update_value_and_values():
    metric = TTFTMetric()
    metric.metric = []
    request = MockRequest(timestamp=100)
    response = MockResponse(timestamp=150)
    record = MockRecord(request, [response])

    metric.update_value(record)
    assert metric.values() == [50]


def test_add_multiple_records():
    metric = TTFTMetric()
    metric.metric = []
    records = [
        MockRecord(MockRequest(10), [MockResponse(15)]),
        MockRecord(MockRequest(20), [MockResponse(25)]),
        MockRecord(MockRequest(30), [MockResponse(40)]),
    ]
    for record in records:
        metric.update_value(record)
    assert metric.values() == [5, 5, 10]


def test_record_without_responses_raises():
    metric = TTFTMetric()
    metric.metric = []
    record = MockRecord(MockRequest(10), [])
    with pytest.raises(ValueError, match="at least one response"):
        metric.update_value(record)


def test_record_with_no_request_raises():
    metric = TTFTMetric()
    metric.metric = []
    record = MockRecord(None, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record)


def test_record_with_no_request_timestamp_raises():
    metric = TTFTMetric()
    metric.metric = []
    request = MockRequest(None)
    record = MockRecord(request, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record)


def test_response_timestamp_less_than_request_raises():
    metric = TTFTMetric()
    metric.metric = []
    request = MockRequest(100)
    response = MockResponse(90)
    record = MockRecord(request, [response])
    with pytest.raises(ValueError, match="Response timestamp must be greater"):
        metric.update_value(record)


def test_metric_initialization_none():
    metric = TTFTMetric()
    assert metric.metric == []
    # After setting to list, works as expected
    metric.metric = []
    request = MockRequest(1)
    response = MockResponse(2)
    record = MockRecord(request, [response])
    metric.update_value(record)
    assert metric.values() == [1]


def test_convert_metrics():
    metric = TTFTMetric()
    metric.metric = []
    records = [
        MockRecord(MockRequest(10_000_000), [MockResponse(15_000_000)]),
        MockRecord(MockRequest(20_000_000), [MockResponse(25_000_000)]),
        MockRecord(MockRequest(30_000_000), [MockResponse(40_000_000)]),
    ]
    for record in records:
        metric.update_value(record)
    assert metric.get_converted_metrics(unit=MetricTimeType.MILLISECONDS) == [5, 5, 10]

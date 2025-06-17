#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.ttst_metric import TTSTMetric
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_ttst_metric_update_value_and_values():
    metric = TTSTMetric()
    metric.metric = []
    request = MockRequest(timestamp=100)
    response1 = MockResponse(timestamp=150)
    response2 = MockResponse(timestamp=180)
    record = MockRecord(request, [response1, response2])

    metric.update_value(record=record, metrics=None)
    assert metric.values() == [30]  # 180 - 150


def test_ttst_metric_add_multiple_records():
    metric = TTSTMetric()
    metric.metric = []
    records = [
        MockRecord(MockRequest(10), [MockResponse(15), MockResponse(20)]),
        MockRecord(MockRequest(20), [MockResponse(25), MockResponse(35)]),
        MockRecord(MockRequest(30), [MockResponse(40), MockResponse(50)]),
    ]
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == [5, 10, 10]


def test_ttst_metric_with_one_response_raises():
    metric = TTSTMetric()
    metric.metric = []
    record = MockRecord(MockRequest(10), [MockResponse(15)])
    with pytest.raises(ValueError, match="at least two responses"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_with_no_request_raises():
    metric = TTSTMetric()
    metric.metric = []
    record = MockRecord(None, [MockResponse(20), MockResponse(30)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_with_no_request_timestamp_raises():
    metric = TTSTMetric()
    metric.metric = []
    request = MockRequest(None)
    record = MockRecord(request, [MockResponse(20), MockResponse(30)])
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_response_timestamp_order_raises():
    metric = TTSTMetric()
    metric.metric = []
    request = MockRequest(100)
    response1 = MockResponse(150)
    response2 = MockResponse(140)
    record = MockRecord(request, [response1, response2])
    with pytest.raises(
        ValueError,
        match="Second response timestamp must be greater than or equal to the first response timestamp.",
    ):
        metric.update_value(record=record, metrics=None)

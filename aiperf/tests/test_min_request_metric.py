#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.min_request_metric import (
    MinRequestMetric,
)
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_add_record_and_values():
    metric = MinRequestMetric()
    request = MockRequest(timestamp=100)
    response = MockResponse(timestamp=150)
    record = MockRecord(request, [response])

    metric.add_record(record)
    assert metric.values() == 100


def test_add_multiple_records():
    metric = MinRequestMetric()
    records = [
        MockRecord(MockRequest(20), [MockResponse(25)]),
        MockRecord(MockRequest(10), [MockResponse(15)]),
        MockRecord(MockRequest(30), [MockResponse(40)]),
    ]
    for record in records:
        metric.add_record(record)
    assert metric.values() == 10


def test_record_with_no_request_raises():
    metric = MinRequestMetric()
    record = MockRecord(None, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.add_record(record)


def test_record_with_no_request_timestamp_raises():
    metric = MinRequestMetric()
    request = MockRequest(None)
    record = MockRecord(request, [MockResponse(20)])
    with pytest.raises(ValueError, match="valid request"):
        metric.add_record(record)

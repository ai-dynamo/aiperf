#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.max_response_metric import (
    MaxResponseMetric,
)
from aiperf.tests.utils.metric_test_utils import MockRecord, MockRequest, MockResponse


def test_update_value_and_values():
    metric = MaxResponseMetric()
    request = MockRequest(timestamp=100)
    response = MockResponse(timestamp=150)
    record = MockRecord(request, [response])

    metric.update_value(record)
    assert metric.values() == 150


def test_add_multiple_records():
    metric = MaxResponseMetric()
    records = [
        MockRecord(MockRequest(20), [MockResponse(25)]),
        MockRecord(MockRequest(10), [MockResponse(15)]),
        MockRecord(MockRequest(30), [MockResponse(40)]),
    ]
    for record in records:
        metric.update_value(record)
    assert metric.values() == 40


def test_record_with_no_responses_raises():
    metric = MaxResponseMetric()
    record = MockRecord(MockRequest(10), None)
    with pytest.raises(ValueError, match="valid responses"):
        metric.update_value(record)


def test_record_with_no_respone_timestamps_raises():
    metric = MaxResponseMetric()
    response = MockResponse(None)
    record = MockRecord(MockRequest(10), [response])
    with pytest.raises(ValueError, match="valid responses"):
        metric.update_value(record)

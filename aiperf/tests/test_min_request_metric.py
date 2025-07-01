#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.min_request_metric import (
    MinRequestMetric,
)


def test_update_value_and_values():
    metric = MinRequestMetric()
    record = RequestRecord(start_perf_ns=100, responses=[SSEMessage(perf_ns=150)])

    metric.update_value(record=record, metrics=None)
    assert metric.values() == 100


def test_add_multiple_records():
    metric = MinRequestMetric()
    records = [
        RequestRecord(start_perf_ns=20, responses=[SSEMessage(perf_ns=25)]),
        RequestRecord(start_perf_ns=10, responses=[SSEMessage(perf_ns=15)]),
        RequestRecord(start_perf_ns=30, responses=[SSEMessage(perf_ns=40)]),
    ]
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == 10


def test_record_with_no_request_raises():
    metric = MinRequestMetric()
    record = None
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)

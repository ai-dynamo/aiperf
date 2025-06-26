#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.max_response_metric import (
    MaxResponseMetric,
)


def test_update_value_and_values():
    metric = MaxResponseMetric()
    record = RequestRecord(recv_start_perf_ns=100, responses=[SSEMessage(perf_ns=150)])

    metric.update_value(record=record, metrics=None)
    assert metric.values() == 150


def test_add_multiple_records():
    metric = MaxResponseMetric()
    records = [
        RequestRecord(recv_start_perf_ns=20, responses=[SSEMessage(perf_ns=25)]),
        RequestRecord(recv_start_perf_ns=10, responses=[SSEMessage(perf_ns=15)]),
        RequestRecord(recv_start_perf_ns=30, responses=[SSEMessage(perf_ns=40)]),
    ]
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == 40


def test_record_with_no_responses_raises():
    metric = MaxResponseMetric()
    record = RequestRecord(recv_start_perf_ns=10, responses=[])
    with pytest.raises(ValueError, match="valid responses"):
        metric.update_value(record=record, metrics=None)

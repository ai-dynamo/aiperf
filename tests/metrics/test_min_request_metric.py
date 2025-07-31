# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.metrics.types.min_request_metric import (
    MinRequestMetric,
)


def test_update_value_and_values(parsed_response_record_builder):
    metric = MinRequestMetric()
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=150)
        .build()
    )
    metric.update_value(record=record, metrics=None)
    assert metric.values() == 100


def test_add_multiple_records(parsed_response_record_builder):
    metric = MinRequestMetric()
    records = (
        parsed_response_record_builder.with_request_start_time(20)
        .add_response(perf_ns=25)
        .new_record()
        .with_request_start_time(10)
        .add_response(perf_ns=15)
        .new_record()
        .with_request_start_time(30)
        .add_response(perf_ns=40)
        .build_all()
    )
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == 10


def test_record_with_no_request_raises():
    metric = MinRequestMetric()
    record = None
    with pytest.raises(ValueError, match="Invalid Record"):
        metric.update_value(record=record, metrics=None)

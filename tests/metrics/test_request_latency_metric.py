# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


@pytest.skip(reason="TODO: Metric refactor work in progress", allow_module_level=True)
def test_update_value_and_values(parsed_response_record_builder):
    metric = RequestLatencyMetric()
    metric.values = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=150)
        .build()
    )
    metric.update_value(record=record, metrics=None)
    assert metric.values() == [50]


@pytest.skip(reason="TODO: Metric refactor work in progress", allow_module_level=True)
def test_add_multiple_records(parsed_response_record_builder):
    metric = RequestLatencyMetric()
    metric.values = []
    records = (
        parsed_response_record_builder.with_request_start_time(10)
        .add_response(perf_ns=15)
        .add_response(perf_ns=25)
        .new_record()
        .with_request_start_time(20)
        .add_response(perf_ns=25)
        .add_response(perf_ns=35)
        .new_record()
        .with_request_start_time(30)
        .add_response(perf_ns=40)
        .add_response(perf_ns=50)
        .build_all()
    )
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == [15, 15, 20]


@pytest.skip(reason="TODO: Metric refactor work in progress", allow_module_level=True)
def test_response_timestamp_less_than_request_raises(parsed_response_record_builder):
    metric = RequestLatencyMetric()
    metric.values = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=90)
        .build()
    )
    with pytest.raises(ValueError, match="Invalid Record"):
        metric.update_value(record=record, metrics=None)


@pytest.skip(reason="TODO: Metric refactor work in progress", allow_module_level=True)
def test_metric_initialization_none(parsed_response_record_builder):
    metric = RequestLatencyMetric()
    assert metric.values() == []
    record = (
        parsed_response_record_builder.with_request_start_time(1)
        .add_response(perf_ns=2)
        .build()
    )
    metric.update_value(record=record, metrics=None)
    assert metric.values() == [1]

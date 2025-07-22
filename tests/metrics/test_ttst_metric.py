# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.records_manager.metrics.types.ttst_metric import TTSTMetric


def test_ttst_metric_update_value_and_values(parsed_response_record_builder):
    metric = TTSTMetric()
    metric.metric = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=150)
        .add_response(perf_ns=180)
        .build()
    )

    metric.update_value(record=record, metrics=None)
    assert metric.values() == [30]  # 180 - 150


def test_ttst_metric_add_multiple_records(parsed_response_record_builder):
    metric = TTSTMetric()
    metric.metric = []
    records = (
        parsed_response_record_builder.with_request_start_time(10)
        .add_response(perf_ns=15)
        .add_response(perf_ns=20)
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
    assert metric.values() == [5, 10, 10]


def test_ttst_metric_with_one_response_raises(parsed_response_record_builder):
    metric = TTSTMetric()
    metric.metric = []
    record = (
        parsed_response_record_builder.with_request_start_time(10)
        .add_response(perf_ns=15)
        .build()
    )
    with pytest.raises(ValueError, match="at least two responses"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_response_timestamp_order_raises(parsed_response_record_builder):
    metric = TTSTMetric()
    metric.metric = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=150)
        .add_response(perf_ns=140)
        .build()
    )
    with pytest.raises(
        ValueError,
        match="Second response timestamp must be greater than or equal to the first response timestamp.",
    ):
        metric.update_value(record=record, metrics=None)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricTimeType
from aiperf.services.records_manager.metrics.types.ttft_metric import TTFTMetric


def test_single_record(parsed_response_record_builder):
    metric = TTFTMetric()
    metric.metric = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=150)
        .build()
    )

    metric.update_value(record=record, metrics=None)
    assert metric.values() == [50]


def test_add_multiple_records(parsed_response_record_builder):
    metric = TTFTMetric()
    metric.metric = []
    records = (
        parsed_response_record_builder.with_request_start_time(10)
        .add_response(perf_ns=15)
        .new_record()
        .with_request_start_time(20)
        .add_response(perf_ns=25)
        .new_record()
        .with_request_start_time(30)
        .add_response(perf_ns=40)
        .build_all()
    )
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == [5, 5, 10]


def test_record_without_responses_raises(parsed_response_record_builder):
    metric = TTFTMetric()
    metric.metric = []
    record = parsed_response_record_builder.with_request_start_time(10).build()
    with pytest.raises(ValueError, match="at least one response"):
        metric.update_value(record=record, metrics=None)


def test_record_with_no_request_raises():
    metric = TTFTMetric()
    metric.metric = []
    record = None
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)


def test_response_timestamp_less_than_request_raises(parsed_response_record_builder):
    metric = TTFTMetric()
    metric.metric = []
    record = (
        parsed_response_record_builder.with_request_start_time(100)
        .add_response(perf_ns=90)
        .build()
    )
    with pytest.raises(ValueError, match="Response timestamp must be greater"):
        metric.update_value(record=record, metrics=None)


def test_convert_metrics(parsed_response_record_builder):
    metric = TTFTMetric()
    metric.metric = []
    records = (
        parsed_response_record_builder.with_request_start_time(10_000_000)
        .add_response(perf_ns=15_000_000)
        .new_record()
        .with_request_start_time(20_000_000)
        .add_response(perf_ns=25_000_000)
        .new_record()
        .with_request_start_time(30_000_000)
        .add_response(perf_ns=40_000_000)
        .build_all()
    )
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.get_converted_metrics(unit=MetricTimeType.MILLISECONDS) == [5, 5, 10]

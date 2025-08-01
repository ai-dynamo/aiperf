# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.metrics.types.ttft_metric import TTFTMetric


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

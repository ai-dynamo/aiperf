#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.record_models import RequestRecord, SSEMessage
from aiperf.services.records_manager.metrics.types.ttst_metric import TTSTMetric


def test_ttst_metric_update_value_and_values():
    metric = TTSTMetric()
    metric.metric = []
    record = RequestRecord(
        start_perf_ns=100,
        responses=[SSEMessage(perf_ns=150), SSEMessage(perf_ns=180)],
    )

    metric.update_value(record=record, metrics=None)
    assert metric.values() == [30]  # 180 - 150


def test_ttst_metric_add_multiple_records():
    metric = TTSTMetric()
    metric.metric = []
    records = [
        RequestRecord(
            start_perf_ns=10,
            responses=[SSEMessage(perf_ns=15), SSEMessage(perf_ns=20)],
        ),
        RequestRecord(
            start_perf_ns=20,
            responses=[SSEMessage(perf_ns=25), SSEMessage(perf_ns=35)],
        ),
        RequestRecord(
            start_perf_ns=30,
            responses=[SSEMessage(perf_ns=40), SSEMessage(perf_ns=50)],
        ),
    ]
    for record in records:
        metric.update_value(record=record, metrics=None)
    assert metric.values() == [5, 10, 10]


def test_ttst_metric_with_one_response_raises():
    metric = TTSTMetric()
    metric.metric = []
    record = RequestRecord(
        start_perf_ns=15,
        responses=[SSEMessage(perf_ns=15)],
    )
    with pytest.raises(ValueError, match="at least two responses"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_with_no_request_raises():
    metric = TTSTMetric()
    metric.metric = []
    record = None
    with pytest.raises(ValueError, match="valid request"):
        metric.update_value(record=record, metrics=None)


def test_ttst_metric_response_timestamp_order_raises():
    metric = TTSTMetric()
    metric.metric = []
    record = RequestRecord(
        start_perf_ns=100,
        responses=[SSEMessage(perf_ns=150), SSEMessage(perf_ns=140)],
    )
    with pytest.raises(
        ValueError,
        match="Second response timestamp must be greater than or equal to the first response timestamp.",
    ):
        metric.update_value(record=record, metrics=None)

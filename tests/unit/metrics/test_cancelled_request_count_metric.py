# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pytest import approx

from aiperf.common.enums import MetricFlags
from aiperf.common.models import ErrorDetails
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.cancelled_request_count_metric import (
    CancelledRequestCountMetric,
)
from tests.unit.metrics.conftest import create_record


def _cancelled_record():
    """A client-cancelled record: code-499 error plus cancellation_perf_ns set."""
    record = create_record(
        start_ns=100,
        error=ErrorDetails(
            type="RequestCancellationError",
            message="Request cancelled by external signal",
            code=499,
        ),
    )
    record.request.cancellation_perf_ns = 150
    return record


class TestCancelledRequestCountMetric:
    def test_registered_in_metric_registry(self):
        cls = MetricRegistry.get_class("cancelled_request_count")
        assert cls is CancelledRequestCountMetric

    def test_has_cancelled_only_flag_not_error_only(self):
        # CANCELLED_ONLY keeps it off the error path (so it never lands in
        # error_request_count); it must NOT carry ERROR_ONLY.
        assert CancelledRequestCountMetric.has_flags(MetricFlags.CANCELLED_ONLY)
        assert not CancelledRequestCountMetric.has_flags(MetricFlags.ERROR_ONLY)

    def test_counts_cancelled_record_despite_invalid(self):
        # The record is invalid (has an error), but CANCELLED_ONLY lets the
        # counter parse it instead of raising NoMetricValue.
        metric = CancelledRequestCountMetric()
        record = _cancelled_record()
        assert record.request.was_cancelled is True
        assert record.valid is False
        assert metric.parse_record(record, MetricRecordDict()) == 1

    def test_aggregates_across_records(self):
        metric = CancelledRequestCountMetric()
        for _ in range(4):
            metric.aggregate_value(
                metric.parse_record(_cancelled_record(), MetricRecordDict())
            )
        assert metric.current_value == approx(4)

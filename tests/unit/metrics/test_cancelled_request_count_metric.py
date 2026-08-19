# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import AggregationKind, MetricFlags
from aiperf.common.models import ErrorDetails
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.cancelled_request_count_metric import (
    CancelledRequestCountMetric,
)
from tests.unit.metrics.conftest import create_record


def test_cancelled_request_count_registered_and_counts_invalid_cancellation() -> None:
    record = create_record(
        error=ErrorDetails(
            type="RequestCancellationError",
            message="Request cancelled by external signal",
            code=499,
        )
    )
    record.request.cancellation_perf_ns = 150

    assert (
        MetricRegistry.get_class("cancelled_request_count")
        is CancelledRequestCountMetric
    )
    assert CancelledRequestCountMetric.has_flags(MetricFlags.CANCELLED_ONLY)
    assert not CancelledRequestCountMetric.has_flags(MetricFlags.ERROR_ONLY)
    assert CancelledRequestCountMetric.aggregation_kind == AggregationKind.SUM
    assert record.valid is False
    assert CancelledRequestCountMetric().parse_record(record, MetricRecordDict()) == 1

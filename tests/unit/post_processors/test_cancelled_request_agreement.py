# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import pytest
from pytest import approx

from aiperf.common.models import ErrorDetails, ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.cancelled_request_count_metric import (
    CancelledRequestCountMetric,
)
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.good_request_fraction_metric import GoodRequestFractionMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_error_rate_metric import RequestErrorRateMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from tests.unit.metrics.conftest import create_record
from tests.unit.post_processors.conftest import create_metric_metadata


def _cancelled_record(start_ns: int) -> ParsedResponseRecord:
    record = create_record(
        start_ns=start_ns,
        error=ErrorDetails(
            type="RequestCancellationError",
            message="Request cancelled by external signal",
            code=499,
        ),
    )
    record.request.cancellation_perf_ns = start_ns
    return record


@pytest.mark.asyncio
async def test_cancelled_requests_do_not_inflate_error_or_goodput_denominators(
    mock_run,
) -> None:
    successes, errors, cancellations = 48, 6, 32
    records = [create_record(start_ns=1_000 * i) for i in range(successes)]
    records.extend(
        create_record(
            start_ns=500_000 + 1_000 * i,
            error=ErrorDetails(code=500, message="Internal server error"),
        )
        for i in range(errors)
    )
    records.extend(_cancelled_record(900_000 + 1_000 * i) for i in range(cancellations))

    processor = MetricRecordProcessor(mock_run)
    totals: dict[str, int] = defaultdict(int)
    counter_tags = (
        RequestCountMetric.tag,
        ErrorRequestCountMetric.tag,
        CancelledRequestCountMetric.tag,
    )
    for record in records:
        result = await processor.process_record(record, create_metric_metadata())
        for tag in counter_tags:
            totals[tag] += int(result.metrics.get(tag, 0))

    assert totals[RequestCountMetric.tag] == successes
    assert totals[ErrorRequestCountMetric.tag] == errors
    assert totals[CancelledRequestCountMetric.tag] == cancellations

    results = MetricResultsDict()
    results[RequestCountMetric.tag] = successes
    results[ErrorRequestCountMetric.tag] = errors
    results[CancelledRequestCountMetric.tag] = cancellations
    results["good_request_count"] = 40

    assert RequestErrorRateMetric().derive_value(results) == approx(
        100.0 * errors / (successes + errors)
    )
    assert GoodRequestFractionMetric().derive_value(results) == approx(
        40 / (successes + errors)
    )

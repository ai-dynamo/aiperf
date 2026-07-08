# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end agreement test for the cancellation/metrics fix.

Drives the REAL metric registry through ``MetricRecordProcessor`` (which owns the
valid/error/cancelled routing) for a synthetic run of successes + real errors +
client cancellations, then checks the aggregated counters and derived
rate/goodput metrics agree with the credit-side counts: cancellations are their
own bucket, never errors.
"""

from collections import defaultdict

import pytest
from pytest import approx

from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.cancelled_request_count_metric import (
    CancelledRequestCountMetric,
)
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.good_request_fraction_metric import GoodRequestFractionMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_error_rate_metric import RequestErrorRateMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from tests.unit.post_processors.conftest import _make_run, create_metric_metadata


def _request_info() -> RequestInfo:
    return RequestInfo(
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase="profiling",
        x_request_id="rid",
        x_correlation_id="cid",
        conversation_id="conv",
    )


def _valid_record(start_ns: int) -> ParsedResponseRecord:
    request = RequestRecord(
        request_info=_request_info(),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns + 50,
        error=None,
    )
    return ParsedResponseRecord(
        request=request,
        responses=[
            ParsedResponse(perf_ns=start_ns + 50, data=TextResponseData(text="hi"))
        ],
        token_counts=TokenCounts(input=8, output=1, reasoning=None),
    )


def _error_record(start_ns: int) -> ParsedResponseRecord:
    request = RequestRecord(
        request_info=_request_info(),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns,
        status=500,
        error=ErrorDetails(code=500, message="Internal server error"),
    )
    return ParsedResponseRecord(
        request=request,
        responses=[],
        token_counts=TokenCounts(input=None, output=None, reasoning=None),
    )


def _cancelled_record(start_ns: int) -> ParsedResponseRecord:
    request = RequestRecord(
        request_info=_request_info(),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=start_ns,
        error=ErrorDetails(
            type="RequestCancellationError",
            message="Request cancelled by external signal",
            code=499,
        ),
        cancellation_perf_ns=start_ns,
    )
    return ParsedResponseRecord(
        request=request,
        responses=[],
        token_counts=TokenCounts(input=None, output=None, reasoning=None),
    )


@pytest.mark.asyncio
async def test_cancelled_run_export_agrees_with_credit_counts(mock_user_config):
    """S successes + M errors + N cancellations: counters + rates agree with
    the credit-side (`completed=S, errors=M, cancelled=N`) view.

    Uses the real metric registry (no mock fixture) so the actual
    valid/error/cancelled routing runs.
    """
    successes, errors, cancellations = 48, 6, 32

    processor = MetricRecordProcessor(_make_run(mock_user_config))

    records: list[ParsedResponseRecord] = []
    records += [_valid_record(1_000 * i) for i in range(successes)]
    records += [_error_record(500_000 + 1_000 * i) for i in range(errors)]
    records += [_cancelled_record(900_000 + 1_000 * i) for i in range(cancellations)]

    totals: dict[str, int] = defaultdict(int)
    for record in records:
        metric_dict = await processor.process_record(record, create_metric_metadata())
        for tag in (
            RequestCountMetric.tag,
            ErrorRequestCountMetric.tag,
            CancelledRequestCountMetric.tag,
        ):
            if tag in metric_dict:
                totals[tag] += int(metric_dict[tag])

    # Counters agree with the credit-side buckets: cancellations are NOT errors.
    assert totals[RequestCountMetric.tag] == successes
    assert totals[ErrorRequestCountMetric.tag] == errors
    assert totals[CancelledRequestCountMetric.tag] == cancellations

    # Derived rate/goodput read the aggregated counters; cancellations are absent
    # from both, so the error rate reflects only real errors.
    results = MetricResultsDict()
    results[RequestCountMetric.tag] = totals[RequestCountMetric.tag]
    results[ErrorRequestCountMetric.tag] = totals[ErrorRequestCountMetric.tag]
    results[CancelledRequestCountMetric.tag] = totals[CancelledRequestCountMetric.tag]

    error_rate = RequestErrorRateMetric().derive_value(results)
    assert error_rate == approx(100.0 * errors / (successes + errors))

    # good_request_fraction denominator excludes the N cancellations: it is
    # successes + errors (54), not successes + errors + cancellations (86).
    results["good_request_count"] = 40
    good_fraction = GoodRequestFractionMetric().derive_value(results)
    assert good_fraction == approx(40 / (successes + errors))


@pytest.mark.asyncio
async def test_cancel_only_run_reports_zero_error_rate(mock_user_config):
    """A 40%-cancel/0-error run reports error_request_count==0 and 0% error rate."""
    successes, cancellations = 48, 32

    processor = MetricRecordProcessor(_make_run(mock_user_config))

    records = [_valid_record(1_000 * i) for i in range(successes)]
    records += [_cancelled_record(900_000 + 1_000 * i) for i in range(cancellations)]

    totals: dict[str, int] = defaultdict(int)
    for record in records:
        metric_dict = await processor.process_record(record, create_metric_metadata())
        for tag in (
            RequestCountMetric.tag,
            ErrorRequestCountMetric.tag,
            CancelledRequestCountMetric.tag,
        ):
            if tag in metric_dict:
                totals[tag] += int(metric_dict[tag])

    assert totals[ErrorRequestCountMetric.tag] == 0
    assert totals[CancelledRequestCountMetric.tag] == cancellations

    results = MetricResultsDict()
    results[RequestCountMetric.tag] = totals[RequestCountMetric.tag]
    results[CancelledRequestCountMetric.tag] = totals[CancelledRequestCountMetric.tag]
    assert RequestErrorRateMetric().derive_value(results) == approx(0.0)

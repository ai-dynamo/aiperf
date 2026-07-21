# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ContextOverflowCountMetric``.

Coverage:
- Metric is registered in ``MetricRegistry`` with the expected tag and flags.
- Counter increments by 1 per record where ``request.context_overflow=True``.
- Counter contributes 0 for records without the flag (mixed batch).
- All-zero result when no records carry the flag.
"""

from aiperf.common.enums import MetricFlags
from aiperf.common.models import ErrorDetails
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.context_overflow_count_metric import (
    ContextOverflowCountMetric,
)
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


def _make_overflow_record(flag: bool) -> object:
    record = create_record(
        error=ErrorDetails(code=400, type="Bad Request", message="ctx-overflow")
    )
    record.request.context_overflow = flag
    return record


def test_metric_is_registered_with_expected_tag_and_flags() -> None:
    cls = MetricRegistry.get_class(ContextOverflowCountMetric.tag)
    assert cls is ContextOverflowCountMetric
    assert cls.tag == "context_overflow_count"
    assert cls.flags.has_flags(MetricFlags.ERROR_ONLY)
    assert cls.flags.has_flags(MetricFlags.NO_INDIVIDUAL_RECORDS)


def test_metric_counts_overflow_records() -> None:
    """Three overflow records out of five = count of 3."""
    records = [
        _make_overflow_record(True),
        _make_overflow_record(False),
        _make_overflow_record(True),
        _make_overflow_record(False),
        _make_overflow_record(True),
    ]
    results = run_simple_metrics_pipeline(records, ContextOverflowCountMetric.tag)
    assert results[ContextOverflowCountMetric.tag] == 3


def test_metric_returns_zero_when_no_overflow_records() -> None:
    """All five records non-overflow -> count is missing or zero."""
    records = [_make_overflow_record(False) for _ in range(5)]
    results = run_simple_metrics_pipeline(records, ContextOverflowCountMetric.tag)
    # The aggregate counter only increments when a per-record value flows in.
    # When _parse_record returns 0, aggregate is still incremented by 0; if no
    # records contributed at all the tag may be absent. Both shapes mean "0".
    assert results.get(ContextOverflowCountMetric.tag, 0) == 0


def test_metric_returns_zero_when_no_records() -> None:
    results = run_simple_metrics_pipeline([], ContextOverflowCountMetric.tag)
    assert results.get(ContextOverflowCountMetric.tag, 0) == 0


def test_metric_increments_by_one_per_overflow_record() -> None:
    """Single overflow record -> count is 1."""
    records = [_make_overflow_record(True)]
    results = run_simple_metrics_pipeline(records, ContextOverflowCountMetric.tag)
    assert results[ContextOverflowCountMetric.tag] == 1

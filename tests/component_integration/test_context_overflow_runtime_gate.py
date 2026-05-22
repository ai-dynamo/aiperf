# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the runtime context-overflow gate.

Wires:
1. Classifier (``is_context_overflow_response``) -> per-request flag on
   ``RequestRecord.context_overflow``
2. ``ContextOverflowCountMetric`` aggregates the flag across records.
3. ``cli_runner._sum_runtime_response_counts`` sums per-run metric totals
   into the carrier keys consumed by ``AggregateConfidenceJsonExporter``.
4. The exporter feeds those into ``compute_submission_outcome`` to flip
   ``submission_valid=false`` when overflow rate exceeds 1%.

These tests bypass the network/orchestrator layer (full subprocess
benchmarking is covered elsewhere) and instead pin the contract between
the runtime counters and the exporter.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.cli_runner import _sum_runtime_response_counts
from aiperf.common.models import ErrorDetails
from aiperf.exporters.aggregate import (
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.exporters.aggregate.aggregate_base_exporter import (
    CONTEXT_OVERFLOW_REASON,
)
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.context_overflow_count_metric import (
    ContextOverflowCountMetric,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline

pytestmark = pytest.mark.component_integration


def _make_metric(avg: float, unit: str = "requests"):
    """Tiny stand-in for JsonMetricResult used by RunResult.summary_metrics."""
    from aiperf.common.models.export_models import JsonMetricResult

    return JsonMetricResult(unit=unit, avg=avg)


def _make_run(*, valid: int, errors: int, overflow: int):
    """Build a RunResult-shaped object with the metrics cli_runner reads."""
    return SimpleNamespace(
        success=True,
        summary_metrics={
            "request_count": _make_metric(valid),
            "error_request_count": _make_metric(errors),
            "context_overflow_count": _make_metric(overflow),
        },
    )


def _export_and_load_sync(aggregate: AggregateResult, tmp_path: Path) -> dict:
    """Run the async exporter end-to-end and return parsed JSON."""
    import asyncio

    config = AggregateExporterConfig(result=aggregate, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config)
    out_path = asyncio.get_event_loop().run_until_complete(exporter.export())
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1: classifier -> per-record flag -> aggregate metric.
# ---------------------------------------------------------------------------


def test_metric_aggregates_overflow_records_end_to_end():
    """Mix of overflow / non-overflow records produces correct aggregate count."""
    overflow_record_count = 7
    non_overflow_count = 93

    records = []
    for _ in range(overflow_record_count):
        record = create_record(
            error=ErrorDetails(
                code=400,
                type="Bad Request",
                message="context length exceeded for this prompt",
            )
        )
        # Simulate the inference_result_parser tagging step:
        record.request.context_overflow = True
        records.append(record)
    for _ in range(non_overflow_count):
        record = create_record()
        record.request.context_overflow = False
        records.append(record)

    results = run_simple_metrics_pipeline(records, ContextOverflowCountMetric.tag)
    assert results[ContextOverflowCountMetric.tag] == overflow_record_count


# ---------------------------------------------------------------------------
# Stage 2: cli_runner helper sums per-run summary metrics.
# ---------------------------------------------------------------------------


def test_sum_runtime_counts_single_run():
    runs = [_make_run(valid=485, errors=4, overflow=11)]
    total, overflow = _sum_runtime_response_counts(runs)
    assert total == 500
    assert overflow == 11


def test_sum_runtime_counts_includes_skipped_overflows_in_total_denominator():
    runs = [_make_run(valid=489, errors=0, overflow=11)]
    total, overflow = _sum_runtime_response_counts(runs)
    assert total == 500
    assert overflow == 11


def test_sum_runtime_counts_multi_run():
    """Confidence reporting: counts sum across all successful runs."""
    runs = [
        _make_run(valid=200, errors=0, overflow=0),
        _make_run(valid=190, errors=2, overflow=8),
        _make_run(valid=205, errors=1, overflow=4),
    ]
    total, overflow = _sum_runtime_response_counts(runs)
    assert total == 200 + 200 + 210
    assert overflow == 0 + 8 + 4


def test_sum_runtime_counts_empty_runs_returns_zero():
    total, overflow = _sum_runtime_response_counts([])
    assert total == 0
    assert overflow == 0


def test_sum_runtime_counts_handles_missing_metrics():
    """Run that didn't surface the new metric (older runs) shouldn't crash."""
    run = SimpleNamespace(
        success=True,
        summary_metrics={
            "request_count": _make_metric(100),
            # error_request_count and context_overflow_count omitted.
        },
    )
    total, overflow = _sum_runtime_response_counts([run])
    assert total == 100
    assert overflow == 0


# ---------------------------------------------------------------------------
# Stage 3: full carrier-key -> exporter -> submission_valid plumbing.
# ---------------------------------------------------------------------------


def test_runtime_overflow_rate_above_threshold_flips_submission_valid_false(tmp_path):
    """N/(N+M) > 0.01 -> submission_valid=false with overflow reason in JSON."""
    runs = [_make_run(valid=489, errors=11, overflow=11)]
    total, overflow = _sum_runtime_response_counts(runs)
    aggregate = AggregateResult(
        aggregation_type="confidence",
        num_runs=1,
        num_successful_runs=1,
        failed_runs=[],
        metrics={},
        metadata={
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": total,
            "_context_overflow_count": overflow,
        },
    )

    data = _export_and_load_sync(aggregate, tmp_path)
    md = data["metadata"]
    assert md["submission_valid"] is False
    assert CONTEXT_OVERFLOW_REASON in md["submission_invalid_reasons"]
    # Sanity: rate is 11/511 ≈ 2.2%, well over the 1% threshold.
    assert overflow / total > 0.01


def test_runtime_overflow_rate_at_one_percent_boundary_remains_valid(tmp_path):
    """N/(N+M) == 0.01 (strict greater-than rule) -> submission_valid=true."""
    # Precisely 5 overflow / 500 total = 1.0% boundary.
    runs = [_make_run(valid=495, errors=0, overflow=5)]
    total, overflow = _sum_runtime_response_counts(runs)
    assert total == 500 and overflow == 5
    aggregate = AggregateResult(
        aggregation_type="confidence",
        num_runs=1,
        num_successful_runs=1,
        failed_runs=[],
        metrics={},
        metadata={
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": total,
            "_context_overflow_count": overflow,
        },
    )

    data = _export_and_load_sync(aggregate, tmp_path)
    md = data["metadata"]
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md


def test_metric_class_is_discoverable_via_registry():
    """Registry-level smoke test that the new metric is auto-registered."""
    cls = MetricRegistry.get_class("context_overflow_count")
    assert cls is ContextOverflowCountMetric

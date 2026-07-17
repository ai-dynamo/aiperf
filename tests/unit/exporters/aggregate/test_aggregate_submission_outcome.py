# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate (multi-run) scenario-submission verdict on the confidence export.

Covers the cross-run fold ported from AgentX
(``aggregate_confidence_json_exporter.py`` + ``aggregate_base_exporter.py``):
the static scenario-lock outcome combined ACROSS RUNS with the cross-run
context-overflow rate and cancellation, via ``compute_submission_outcome``.

Uses real ``AggregateResult`` / ``ConfidenceMetric`` / ``ScenarioOutcome``
objects (no MagicMock) so the metric-key plumbing and the rate-equivalence
between summed totals and per-run means are exercised end-to-end.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.scenario.base import ScenarioOutcome
from aiperf.exporters.aggregate import (
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.orchestrator.aggregation.confidence import ConfidenceMetric


def _count_metric(mean: float) -> ConfidenceMetric:
    """A degenerate count ``ConfidenceMetric`` whose cross-run mean is ``mean``."""
    return ConfidenceMetric(
        mean=mean,
        std=0.0,
        min=mean,
        max=mean,
        cv=0.0,
        se=0.0,
        ci_low=mean,
        ci_high=mean,
        t_critical=float("nan"),
        unit="requests",
    )


def _make_result(*, metrics: dict, metadata: dict) -> AggregateResult:
    """A confidence ``AggregateResult`` over two successful runs."""
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=2,
        num_successful_runs=2,
        failed_runs=[],
        metrics=dict(metrics),
        metadata=dict(metadata),
    )


def _export_metadata(tmp_path, result: AggregateResult) -> dict:
    """Run the exporter and return the parsed top-level ``metadata`` dict."""
    config = AggregateExporterConfig(result=result, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config=config)
    payload = orjson.loads(exporter._generate_content())
    return payload["metadata"]


def test_cross_run_overflow_rate_flips_submission_invalid_from_means(tmp_path):
    """Cross-run overflow rate > limit flips submission_valid False from means.

    Neither single run exceeds 1% in isolation, but the cross-run mean rate
    (mean(overflow) / mean(total)) does: 3 / 100 = 3% > 1%. Derived purely
    from the confidence ``*_avg`` count metrics (no orchestrator carrier keys),
    proving the self-sufficient fallback path.
    """
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(97.0),
            "error_request_count_avg": _count_metric(0.0),
            "context_overflow_count_avg": _count_metric(3.0),
        },
        metadata={"scenario": "inferencex-agentx-mvp", "confidence_level": 0.95},
    )
    md = _export_metadata(tmp_path, result)
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False
    assert "context_overflow_rate_exceeded" in md["submission_invalid_reasons"]


def test_carrier_keys_sum_across_runs_and_are_stripped(tmp_path):
    """Orchestrator-stamped cross-run carrier totals win and never leak.

    The summed ``_total_responses`` / ``_context_overflow_count`` carriers
    (200 / 6 = 3% > 1%) drive the verdict; all underscore-prefixed carrier
    keys are popped from the public metadata.
    """
    outcome = ScenarioOutcome(scenario_name="inferencex-agentx-mvp")
    result = _make_result(
        metrics={},
        metadata={
            "_scenario_name": outcome.scenario_name,
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 200,
            "_context_overflow_count": 6,
            "_was_cancelled": False,
        },
    )
    md = _export_metadata(tmp_path, result)
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["context_overflow_rate_exceeded"]
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
        "_was_cancelled",
    ):
        assert key not in md


def test_lock_violation_propagates_into_aggregate(tmp_path):
    """A real --unsafe-override ScenarioOutcome lock violation propagates.

    No overflow, no cancellation: the aggregate verdict is exactly the static
    lock outcome (False + unsafe_override) carried through the validator
    carrier keys.
    """
    outcome = ScenarioOutcome(
        scenario_name="inferencex-agentx-mvp",
        submission_valid=False,
        submission_invalid_reasons=["unsafe_override"],
    )
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(100.0),
            "context_overflow_count_avg": _count_metric(0.0),
        },
        metadata={
            "_scenario_name": outcome.scenario_name,
            "_validator_submission_valid": outcome.submission_valid,
            "_validator_submission_invalid_reasons": outcome.submission_invalid_reasons,
            "_total_responses": 100,
            "_context_overflow_count": 0,
        },
    )
    md = _export_metadata(tmp_path, result)
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["unsafe_override"]


def test_cancellation_flips_aggregate_invalid(tmp_path):
    """A cancelled multi-run aggregate is never a valid submission."""
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(50.0),
            "context_overflow_count_avg": _count_metric(0.0),
        },
        metadata={
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_total_responses": 50,
            "_context_overflow_count": 0,
            "_was_cancelled": True,
        },
    )
    md = _export_metadata(tmp_path, result)
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["run_cancelled"]


def test_clean_scenario_run_is_valid(tmp_path):
    """A clean scenario run (under threshold, no lock, no cancel) is valid."""
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(1000.0),
            "context_overflow_count_avg": _count_metric(1.0),  # 0.1% < 1%
        },
        metadata={"scenario": "inferencex-agentx-mvp"},
    )
    md = _export_metadata(tmp_path, result)
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md


def test_no_scenario_run_omits_submission_fields(tmp_path):
    """A non-scenario aggregate carries no submission fields (null-safe omit)."""
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(10.0),
            "context_overflow_count_avg": _count_metric(9.0),  # huge rate, ignored
        },
        metadata={"confidence_level": 0.95},
    )
    md = _export_metadata(tmp_path, result)
    assert "scenario" not in md
    assert "submission_valid" not in md
    assert "submission_invalid_reasons" not in md
    # Untouched aggregate metadata still flows through.
    assert md["aggregation_type"] == "confidence"
    assert md["num_profile_runs"] == 2


@pytest.mark.parametrize(
    "overflow_mean,total_request_mean,expect_valid",
    [
        (1.0, 99.0, True),  # 1/100 == 1% boundary, accepted (strictly >)
        (2.0, 98.0, False),  # 2/100 == 2% > 1%
    ],
)  # fmt: skip
def test_boundary_rate_from_means(
    tmp_path, overflow_mean, total_request_mean, expect_valid
):
    """The strictly-greater rate operator holds when derived from means."""
    result = _make_result(
        metrics={
            "request_count_avg": _count_metric(total_request_mean),
            "context_overflow_count_avg": _count_metric(overflow_mean),
        },
        metadata={"scenario": "inferencex-agentx-mvp"},
    )
    md = _export_metadata(tmp_path, result)
    assert md["submission_valid"] is expect_valid

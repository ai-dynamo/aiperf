# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial component-integration tests for `submission_valid` stamping.

Each test exercises the full
`AggregateConfidenceJsonExporter._aggregate_to_export_data` -> file ->
JSON-decode path so the stamping behavior is verified end-to-end through
the JSON output that ships in `profile_export_aiperf_aggregate.json`.

Wiring contract:
- Scenario name and validator outcome are passed via underscore-prefixed
  keys on `AggregateResult.metadata`:
    * `_scenario_name`
    * `_validator_submission_valid`
    * `_validator_submission_invalid_reasons`
    * `_total_responses`
    * `_context_overflow_count`
    * `_was_cancelled`
- The exporter pops those keys (so they do not pollute the output) and
  feeds them through `compute_submission_outcome()` +
  `_build_run_metadata_dict()` to emit the final
  `submission_valid` / `submission_invalid_reasons` fields.

These tests pin the helper-and-exporter integration; the matching
loader -> trajectory -> strategy -> aggregate -> exporter chain is
exercised end-to-end by ``test_agentic_replay_e2e.py`` and the CLI-surface
test ``test_agentic_replay_cli_e2e.py``.
"""

import json
from pathlib import Path

import pytest

from aiperf.exporters.aggregate import (
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.exporters.aggregate.aggregate_base_exporter import (
    CONTEXT_OVERFLOW_RATE_LIMIT,
    CONTEXT_OVERFLOW_REASON,
    RUN_CANCELLED_REASON,
    compute_submission_outcome,
)
from aiperf.orchestrator.aggregation.base import AggregateResult

pytestmark = pytest.mark.component_integration


def _make_aggregate(metadata: dict) -> AggregateResult:
    """Build a minimal AggregateResult carrying the given metadata."""
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=1,
        num_successful_runs=1,
        failed_runs=[],
        metrics={},
        metadata=metadata,
    )


async def _export_and_load(aggregate: AggregateResult, tmp_path: Path) -> dict:
    """Write the aggregate via the JSON exporter and return the parsed JSON."""
    config = AggregateExporterConfig(result=aggregate, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config)
    out_path = await exporter.export()
    with open(out_path) as f:
        return json.load(f)


async def test_clean_scenario_run_emits_submission_valid_true(tmp_path):
    """Spec 8.4.6 #1: clean `--scenario inferencex-agentx-mvp` -> submission_valid: true."""
    aggregate = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 500,
            "_context_overflow_count": 0,
        }
    )

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md
    # Underscore-prefixed carrier keys are stripped from output.
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
    ):
        assert key not in md


async def test_unsafe_override_with_violation_flips_submission_valid_false(tmp_path):
    """Spec 8.4.6 #2: --unsafe-override + violation -> false with reasons."""
    aggregate = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": False,
            "_validator_submission_invalid_reasons": ["unsafe_override"],
            "_total_responses": 500,
            "_context_overflow_count": 0,
        }
    )

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False
    assert md["submission_invalid_reasons"] == ["unsafe_override"]


async def test_runtime_context_overflow_above_threshold_flips_false(tmp_path):
    """Spec 8.4.6 #3: clean validator but >1% overflow rate -> false with overflow reason."""
    # 11 / 500 = 2.2% >> 1% threshold.
    aggregate = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 500,
            "_context_overflow_count": 11,
        }
    )

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["submission_valid"] is False
    assert CONTEXT_OVERFLOW_REASON in md["submission_invalid_reasons"]


async def test_boundary_exactly_one_percent_overflow_remains_true(tmp_path):
    """Spec 8.4.6 #4: rate == 1.0% boundary -- strict greater-than only flips false.

    Pinned semantics: 5 overflows in 500 responses (rate == 0.01) does NOT
    flip submission_valid; 6 / 500 == 0.012 (> 0.01) does flip.
    """
    on_boundary = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 500,
            "_context_overflow_count": 5,
        }
    )
    data = await _export_and_load(on_boundary, tmp_path / "on")
    md = data["metadata"]
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md
    # Sanity: the boundary constant is the rate the test pins against.
    assert pytest.approx(0.01) == CONTEXT_OVERFLOW_RATE_LIMIT

    just_over = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 500,
            "_context_overflow_count": 6,
        }
    )
    data = await _export_and_load(just_over, tmp_path / "over")
    md = data["metadata"]
    assert md["submission_valid"] is False
    assert CONTEXT_OVERFLOW_REASON in md["submission_invalid_reasons"]


async def test_zero_responses_does_not_flip_on_overflow_rule(tmp_path):
    """Spec 8.4.6 #5: 0/0 overflow rate is treated as 0; submission_valid not flipped on overflow.

    Other failure-rate signals surface a 0-success run; the overflow rule
    specifically must not fire when total_responses == 0 (avoids divide-by-zero
    and avoids declaring "100% overflow" for a 0-response run).
    """
    aggregate = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 0,
            "_context_overflow_count": 0,
        }
    )

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    # Validator was clean and the overflow rule does not fire on a 0-response run.
    assert md["submission_valid"] is True
    assert CONTEXT_OVERFLOW_REASON not in md.get("submission_invalid_reasons", [])

    # Sanity: also pin the helper directly for this case.
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=True,
        validator_reasons=[],
        total_responses=0,
        context_overflow_count=0,
    )
    assert valid is True
    assert reasons == []


async def test_cancelled_run_flips_submission_valid_false(tmp_path):
    """A cancelled run (Ctrl+C) is never a valid submission, even when the
    validator was clean and no runtime threshold was crossed."""
    aggregate = _make_aggregate(
        {
            "_scenario_name": "inferencex-agentx-mvp",
            "_validator_submission_valid": True,
            "_validator_submission_invalid_reasons": [],
            "_total_responses": 500,
            "_context_overflow_count": 0,
            "_was_cancelled": True,
        }
    )

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False
    assert RUN_CANCELLED_REASON in md["submission_invalid_reasons"]
    # Carrier key is stripped from output.
    assert "_was_cancelled" not in md


async def test_bare_timing_mode_no_scenario_omits_submission_valid(tmp_path):
    """Spec 8.4.6 #6: bare agentic_replay timing mode (no --scenario) omits the field."""
    # No `_scenario_name` key, no validator outcome -- non-scenario run.
    aggregate = _make_aggregate({"confidence_level": 0.95})

    data = await _export_and_load(aggregate, tmp_path)

    md = data["metadata"]
    assert "submission_valid" not in md
    assert "submission_invalid_reasons" not in md
    assert "scenario" not in md
    # Existing non-scenario metadata still flows through.
    assert md["confidence_level"] == 0.95

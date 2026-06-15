# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the aggregate run-metadata helpers.

`_build_run_metadata_dict` is the integration point used by Task 9 to merge
scenario-submission tracking fields (`scenario`, `submission_valid`,
`submission_invalid_reasons`) into the top-level
`profile_export_aiperf_aggregate.json` output. The helper intentionally
returns an empty dict when `scenario_name is None` so non-scenario runs are
never polluted with submission-tracking fields.

`compute_submission_outcome` folds runtime-only signals into the validator
verdict; the cancellation tests here pin that a cancelled run is never a
valid scenario submission (reason `"run_cancelled"`).
"""

from aiperf.exporters.aggregate.aggregate_base_exporter import (
    CONTEXT_OVERFLOW_REASON,
    RUN_CANCELLED_REASON,
    _build_run_metadata_dict,
    compute_submission_outcome,
)


def test_submission_valid_omitted_when_scenario_unset() -> None:
    md = _build_run_metadata_dict(scenario_name=None, submission_valid=None)
    assert "submission_valid" not in md
    assert md == {}


def test_submission_valid_true_when_scenario_set_and_clean() -> None:
    md = _build_run_metadata_dict(
        scenario_name="inferencex-agentx-mvp", submission_valid=True
    )
    assert md["submission_valid"] is True
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert "submission_invalid_reasons" not in md


def test_submission_valid_false_with_reason() -> None:
    md = _build_run_metadata_dict(
        scenario_name="inferencex-agentx-mvp",
        submission_valid=False,
        submission_invalid_reasons=[
            "unsafe_override",
            "context_overflow_rate_exceeded",
        ],
    )
    assert md["submission_valid"] is False
    assert "unsafe_override" in md["submission_invalid_reasons"]
    assert "context_overflow_rate_exceeded" in md["submission_invalid_reasons"]


def test_cancelled_run_flips_submission_valid_false() -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=True,
        was_cancelled=True,
    )
    assert valid is False
    assert reasons == [RUN_CANCELLED_REASON]


def test_not_cancelled_run_keeps_submission_valid_true() -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=True,
        was_cancelled=False,
    )
    assert valid is True
    assert reasons == []


def test_cancelled_run_appends_reason_to_existing_reasons() -> None:
    # 11 / 500 = 2.2% overflow rate already flips submission_valid;
    # cancellation adds its own reason exactly once.
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=False,
        validator_reasons=["unsafe_override"],
        total_responses=500,
        context_overflow_count=11,
        was_cancelled=True,
    )
    assert valid is False
    assert reasons == ["unsafe_override", CONTEXT_OVERFLOW_REASON, RUN_CANCELLED_REASON]


def test_cancelled_run_without_scenario_omits_submission_valid() -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name=None,
        validator_submission_valid=None,
        was_cancelled=True,
    )
    assert valid is None
    assert reasons == []

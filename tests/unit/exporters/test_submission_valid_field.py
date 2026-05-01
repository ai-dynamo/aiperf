# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `_build_run_metadata_dict` aggregate run-metadata helper.

The helper is the integration point used by Task 9 to merge scenario-submission
tracking fields (`scenario`, `submission_valid`, `submission_invalid_reasons`)
into the top-level `profile_export_aiperf_aggregate.json` output. The helper
intentionally returns an empty dict when `scenario_name is None` so non-scenario
runs are never polluted with submission-tracking fields.
"""

from aiperf.exporters.aggregate.aggregate_base_exporter import (
    _build_run_metadata_dict,
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

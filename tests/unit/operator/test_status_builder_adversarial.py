# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator StatusBuilder terminal status handling.

Focuses on:
- Terminal Complete/Failed mutual exclusion when prior conditions are stale or contradictory.
- ResultsAvailable gating so Completed does not mean kubectl-wait success too early.
- Cancelled behavior as a distinct terminal phase, not a failed benchmark.
- observedGeneration stamping as an explicit successful-reconcile action.
- currentPhase/subPhase clearing at terminal transitions.

Out of scope:
- Monitor-handler JobSet and Pod state transitions; see
  tests/unit/operator/test_monitor_state_machine_edges.py.
- Timestamp parser/formatter unit coverage; see tests/unit/operator/test_status.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pytest import param

from aiperf.kubernetes.phase import Phase
from aiperf.operator.status import ConditionType, StatusBuilder

# =============================================================================
# Helpers
# =============================================================================


ConditionRecord = dict[str, object]
StatusRecord = dict[str, object]


@dataclass(slots=True)
class _Patch:
    """Minimal kopf.Patch stand-in exposing the mutable status dict."""

    status: dict[str, object] = field(default_factory=dict)


def _condition(
    condition_type: str,
    status: str,
    *,
    reason: str = "Existing",
    message: str = "existing status from prior reconcile",
    transition_time: str = "2026-05-18T10:30:00Z",
) -> ConditionRecord:
    return {
        "type": condition_type,
        "status": status,
        "reason": reason,
        "message": message,
        "lastTransitionTime": transition_time,
    }


def _status_with(*conditions: ConditionRecord) -> StatusRecord:
    return {"conditions": list(conditions)}


def _builder(
    existing_status: StatusRecord | None = None,
) -> tuple[StatusBuilder, _Patch]:
    patch = _Patch()
    return StatusBuilder(patch, existing_status), patch


def _conditions_by_type(patch: _Patch) -> dict[str, ConditionRecord]:
    raw_conditions = patch.status.get("conditions", [])
    assert isinstance(raw_conditions, list)
    by_type: dict[str, ConditionRecord] = {}
    for raw_condition in raw_conditions:
        assert isinstance(raw_condition, dict)
        condition_type = raw_condition["type"]
        assert isinstance(condition_type, str)
        by_type[condition_type] = raw_condition
    return by_type


# =============================================================================
# Terminal condition mutual exclusion
# =============================================================================


class TestStatusBuilderTerminalConditionRepair:
    """Terminal phases repair stale or contradictory Complete/Failed conditions."""

    def test_finalize_completed_with_prior_failed_true_repairs_to_complete_true(
        self,
    ) -> None:
        """Completed + ResultsAvailable=True must clear a stale Failed=True."""
        existing_status = _status_with(
            _condition("ResultsAvailable", "True", reason="ResultsStored"),
            _condition("Complete", "False", reason="PreviousFailure"),
            _condition("Failed", "True", reason="ControllerCrash"),
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.COMPLETED)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert by_type["Complete"]["status"] == "True"
        assert by_type["Complete"]["reason"] == "ResultsStored"
        assert by_type["Failed"]["status"] == "False"
        assert by_type["Failed"]["reason"] == "JobCompleted"

    def test_finalize_failed_with_prior_complete_true_repairs_to_failed_true(
        self,
    ) -> None:
        """Failed phase must not leave a stale Complete=True behind."""
        existing_status = _status_with(
            _condition("ResultsAvailable", "True", reason="ResultsStored"),
            _condition("Complete", "True", reason="ResultsStored"),
            _condition("Failed", "False", reason="JobCompleted"),
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.FAILED)
        builder.set_error("controller pod aiperf-bench-7f2a crashed before shutdown")
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert by_type["Failed"]["status"] == "True"
        assert by_type["Failed"]["message"] == (
            "controller pod aiperf-bench-7f2a crashed before shutdown"
        )
        assert by_type["Complete"]["status"] == "False"
        assert by_type["Complete"]["reason"] == "JobFailed"

    def test_finalize_cancelled_with_both_prior_terminals_true_clears_both(
        self,
    ) -> None:
        """Cancelled is its own terminal outcome even after contradictory status."""
        existing_status = _status_with(
            _condition("Complete", "True", reason="ResultsStored"),
            _condition("Failed", "True", reason="JobFailed"),
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.CANCELLED)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert by_type["Complete"]["status"] == "False"
        assert by_type["Complete"]["reason"] == "JobCancelled"
        assert by_type["Failed"]["status"] == "False"
        assert by_type["Failed"]["reason"] == "JobCancelled"

    @pytest.mark.parametrize(
        "phase,expected_complete,expected_failed",
        [
            param(Phase.COMPLETED, "True", "False", id="completed-repairs-failed"),
            param(Phase.FAILED, "False", "True", id="failed-repairs-complete"),
            param(Phase.CANCELLED, "False", "False", id="cancelled-clears-both"),
        ],
    )  # fmt: skip
    def test_finalize_terminal_phase_never_leaves_complete_and_failed_true(
        self,
        phase: Phase,
        expected_complete: str,
        expected_failed: str,
    ) -> None:
        """The batchv1.Job-style terminal booleans must stay mutually exclusive."""
        existing_status = _status_with(
            _condition("ResultsAvailable", "True", reason="ResultsStored"),
            _condition("Complete", "True", reason="MalformedPriorStatus"),
            _condition("Failed", "True", reason="MalformedPriorStatus"),
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(phase)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert by_type["Complete"]["status"] == expected_complete
        assert by_type["Failed"]["status"] == expected_failed
        assert not (
            by_type["Complete"]["status"] == "True"
            and by_type["Failed"]["status"] == "True"
        )


# =============================================================================
# ResultsAvailable gating and malformed prior conditions
# =============================================================================


class TestStatusBuilderResultsAvailableGating:
    """Completed only becomes Complete=True once ResultsAvailable is exactly True."""

    @pytest.mark.parametrize(
        "results_available_status",
        [
            param("False", id="false-string"),
            param("true", id="lowercase-true-is-not-k8s-true"),
            param("TRUE", id="uppercase-true-is-not-k8s-true"),
            param("", id="empty-status"),
        ],
    )  # fmt: skip
    def test_finalize_completed_with_non_true_results_available_does_not_latch(
        self, results_available_status: str
    ) -> None:
        """Only Kubernetes' canonical status string True unlocks Complete=True."""
        existing_status = _status_with(
            _condition(
                "ResultsAvailable",
                results_available_status,
                reason="MalformedResultsCondition",
            )
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.COMPLETED)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert "Complete" not in by_type
        assert "Failed" not in by_type
        assert patch.status["phase"] == "Completed"
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None

    def test_finalize_completed_ignores_unknown_results_typo_condition(
        self,
    ) -> None:
        """A typo like ResultAvailable must not satisfy the results gate."""
        existing_status = _status_with(
            _condition("ResultAvailable", "True", reason="TypoFromManualPatch")
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.COMPLETED)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert "ResultAvailable" not in by_type
        assert "Complete" not in by_type
        assert "Failed" not in by_type

    def test_finalize_completed_with_results_available_true_in_existing_status_latches(
        self,
    ) -> None:
        """A follow-up reconcile can derive Complete from server-persisted ResultsAvailable."""
        existing_status = _status_with(
            _condition("ResultsAvailable", "True", reason="ResultsStored")
        )
        builder, patch = _builder(existing_status)

        builder.set_phase(Phase.COMPLETED)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert by_type["Complete"]["status"] == "True"
        assert by_type["Failed"]["status"] == "False"


# =============================================================================
# Phase clearing and observedGeneration stamping
# =============================================================================


class TestStatusBuilderTerminalPatchShape:
    """Terminal patches clear in-flight labels and stamp generation only on request."""

    @pytest.mark.parametrize(
        "phase",
        [
            param(Phase.COMPLETED, id="completed"),
            param(Phase.FAILED, id="failed"),
            param(Phase.CANCELLED, id="cancelled"),
        ],
    )  # fmt: skip
    def test_set_phase_terminal_clears_current_and_sub_phase_together(
        self, phase: Phase
    ) -> None:
        """Terminal status must not leave either kubectl progress column stale."""
        builder, patch = _builder()
        patch.status["currentPhase"] = "profiling"
        patch.status["subPhase"] = "shutdown"

        builder.set_phase(phase)

        assert patch.status["phase"] == str(phase)
        assert patch.status["currentPhase"] is None
        assert patch.status["subPhase"] is None

    def test_finalize_without_explicit_generation_does_not_stamp_observed_generation(
        self,
    ) -> None:
        """finalize() must not acknowledge a spec generation by accident."""
        builder, patch = _builder()

        builder.set_phase(Phase.RUNNING)
        builder.conditions.set_true(
            ConditionType.WORKERS_READY,
            "WorkersReady",
            "all worker pods for aiperf-bench-7f2a are ready",
        )
        builder.finalize()

        assert "observedGeneration" not in patch.status

    def test_set_observed_generation_can_share_patch_with_terminal_conditions(
        self,
    ) -> None:
        """Successful terminal reconcile stamps generation and terminal booleans."""
        builder, patch = _builder()

        builder.set_phase(Phase.COMPLETED)
        builder.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsStored",
            "results for aiperf-bench-7f2a are stored on the operator PVC",
        )
        builder.set_observed_generation(42)
        builder.finalize()

        by_type = _conditions_by_type(patch)
        assert patch.status["observedGeneration"] == 42
        assert by_type["Complete"]["status"] == "True"
        assert by_type["Failed"]["status"] == "False"

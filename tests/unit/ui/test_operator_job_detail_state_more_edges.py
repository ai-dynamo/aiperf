# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for job-detail run-state derivation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

JOB_DETAIL_STATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "job-detail-state.js"
)


def derive_state(
    *, phase: str | None, epoch: object = None, run_epoch: object = None
) -> dict[str, object]:
    epoch_line = (
        "epoch: undefined" if epoch is _UNDEFINED else f"epoch: {json.dumps(epoch)}"
    )
    script = f"""
        import {{ deriveJobRunState }} from {JOB_DETAIL_STATE_PATH.as_uri()!r};
        const state = deriveJobRunState({{
          phase: {json.dumps(phase)},
          {epoch_line},
          runEpoch: {json.dumps(run_epoch)},
        }});
        console.log(JSON.stringify(state));
    """
    return json.loads(run_node(script))


_UNDEFINED = object()


@pytest.mark.parametrize(
    ("phase", "expected_flag", "expected_lower"),
    [
        param("Completed", "isCompleted", "completed", id="completed"),
        param("Succeeded", "isCompleted", "succeeded", id="succeeded-alias"),
        param("Cancelled", "isCancelled", "cancelled", id="cancelled"),
        param("Canceled", "isCancelled", "canceled", id="canceled-alias"),
        param("Failed", None, "failed", id="failed"),
        param("Error", None, "error", id="error-alias"),
        param("PartiallyFailed", "isPartiallyFailed", "partiallyfailed", id="partially-failed"),
        param("Archived", "isArchived", "archived", id="archived"),
    ],
)  # fmt: skip
def test_terminal_phases_stop_polling_and_hide_live_panels(
    phase: str,
    expected_flag: str | None,
    expected_lower: str,
) -> None:
    state = derive_state(phase=phase, epoch="100", run_epoch="100")

    assert state["phaseLower"] == expected_lower
    assert state["isTerminal"] is True
    assert state["pollingDone"] is True
    assert state["showLiveRunPanels"] is False
    if expected_flag is not None:
        assert state[expected_flag] is True


@pytest.mark.parametrize(
    ("phase", "expected_lower"),
    [
        param("Running", "running", id="running"),
        param("Pending", "pending", id="pending"),
        param(None, "unknown", id="missing-phase"),
        param("Mystery", "mystery", id="unknown-phase"),
    ],
)  # fmt: skip
def test_non_terminal_phases_keep_live_panels_available(
    phase: str | None, expected_lower: str
) -> None:
    state = derive_state(phase=phase, epoch="100", run_epoch="100")

    assert state["phaseLower"] == expected_lower
    assert state["isTerminal"] is False
    assert state["pollingDone"] is False
    assert state["showLiveRunPanels"] is True


@pytest.mark.parametrize(
    ("epoch", "run_epoch", "expected"),
    [
        param(_UNDEFINED, "100", True, id="unpinned-route-is-current"),
        param("100", "100", True, id="same-string-epoch"),
        param("099", "100", False, id="different-string-epoch"),
        param(100, 100, True, id="same-numeric-epoch"),
        param(100, "100", True, id="numeric-route-string-status"),
        param("100", 100, True, id="string-route-numeric-status"),
        param(99, "100", False, id="different-numeric-string-epoch"),
        param("100", None, False, id="pinned-route-without-live-epoch"),
    ],
)  # fmt: skip
def test_viewing_current_run_compares_epoch_values_across_wire_types(
    epoch: object,
    run_epoch: object,
    expected: bool,
) -> None:
    state = derive_state(phase="Running", epoch=epoch, run_epoch=run_epoch)

    assert state["viewingCurrentRun"] is expected


def test_viewing_archived_epoch_of_rerunning_job_is_not_current() -> None:
    state = derive_state(phase="Running", epoch="1779050863", run_epoch="1779050999")

    assert state["isRunning"] is True
    assert state["viewingCurrentRun"] is False
    assert state["showLiveRunPanels"] is True


def test_unknown_phase_is_not_accidentally_terminal_or_running() -> None:
    state = derive_state(phase="Paused", epoch="200", run_epoch="200")

    assert state["phaseLower"] == "paused"
    assert state["isRunning"] is False
    assert state["isCompleted"] is False
    assert state["isCancelled"] is False
    assert state["isPartiallyFailed"] is False
    assert state["isArchived"] is False
    assert state["isTerminal"] is False
    assert state["viewingCurrentRun"] is True
    assert state["pollingDone"] is False
    assert state["showLiveRunPanels"] is True

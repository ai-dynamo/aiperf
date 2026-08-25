# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial phase-normalization tests for the operator UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_THEME_JS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "lib" / "theme.js"
_JOB_DETAIL_STATE_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "job-detail-state.js"
)
_SWEEP_DETAIL_HELPERS_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)


def _source(path: Path) -> str:
    return path.read_text()


def _node_json(script: str) -> object:
    return json.loads(run_node(script))


def test_phase_color_falls_back_to_unknown_for_new_or_empty_phases() -> None:
    script = f"""
        import {{ colors, phaseColor }} from {_THEME_JS.as_uri()!r};
        const phases = [null, undefined, '', 'Paused', 'BackoffRetrying', 'partiallyFailed'];
        console.log(JSON.stringify(phases.map((phase) => phaseColor(phase) === colors.phaseUnknown)));
    """

    assert _node_json(script) == [True, True, True, True, True, True]


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        param("Pending", "phasePending", id="pending-camelcase"),
        param("INITIALIZING", "phasePending", id="initializing-uppercase"),
        param("running", "phaseRunning", id="running-lowercase"),
        param("Completed", "phaseCompleted", id="completed-camelcase"),
        param("succeeded", "phaseCompleted", id="succeeded-lowercase"),
        param("Archived", "phaseCompleted", id="archived-camelcase"),
        param("failed", "phaseFailed", id="failed-lowercase"),
        param("ERROR", "phaseFailed", id="error-uppercase"),
    ],
)  # fmt: skip
def test_phase_color_normalizes_known_lowercase_and_camelcase_variants(
    phase: str,
    expected: str,
) -> None:
    script = f"""
        import {{ colors, phaseColor }} from {_THEME_JS.as_uri()!r};
        console.log(JSON.stringify(phaseColor({phase!r}) === colors[{expected!r}]));
    """

    assert _node_json(script) is True


@pytest.mark.parametrize(
    ("phase", "terminal", "running", "completed", "cancelled", "partial", "archived"),
    [
        param("pending", False, False, False, False, False, False, id="pending-open"),
        param("running", False, True, False, False, False, False, id="running-live"),
        param("Completed", True, False, True, False, False, False, id="completed-terminal"),
        param("succeeded", True, False, True, False, False, False, id="succeeded-terminal"),
        param("failed", True, False, False, False, False, False, id="failed-terminal"),
        param("ERROR", True, False, False, False, False, False, id="error-terminal"),
        param("cancelled", True, False, False, True, False, False, id="cancelled-terminal"),
        param("canceled", True, False, False, True, False, False, id="canceled-terminal"),
        param("partiallyFailed", True, False, False, False, True, False, id="partial-camelcase"),
        param("Archived", True, False, False, False, False, True, id="archived-terminal"),
    ],
)  # fmt: skip
def test_job_run_state_terminal_and_running_phase_sets_are_normalized(
    phase: str,
    terminal: bool,
    running: bool,
    completed: bool,
    cancelled: bool,
    partial: bool,
    archived: bool,
) -> None:
    script = f"""
        import {{ deriveJobRunState }} from {_JOB_DETAIL_STATE_JS.as_uri()!r};
        const state = deriveJobRunState({{ phase: {phase!r}, epoch: undefined, runEpoch: '7' }});
        console.log(JSON.stringify({{
          phaseLower: state.phaseLower,
          isTerminal: state.isTerminal,
          isRunning: state.isRunning,
          isCompleted: state.isCompleted,
          isCancelled: state.isCancelled,
          isPartiallyFailed: state.isPartiallyFailed,
          isArchived: state.isArchived,
          pollingDone: state.pollingDone,
          showLiveRunPanels: state.showLiveRunPanels,
        }}));
    """

    state = _node_json(script)

    assert state["phaseLower"] == phase.lower()
    assert state["isTerminal"] is terminal
    assert state["isRunning"] is running
    assert state["isCompleted"] is completed
    assert state["isCancelled"] is cancelled
    assert state["isPartiallyFailed"] is partial
    assert state["isArchived"] is archived
    assert state["pollingDone"] is terminal
    assert state["showLiveRunPanels"] is not terminal


def test_new_subphase_values_do_not_accidentally_become_terminal_run_phases() -> None:
    subphases = [
        "initializing",
        "configuring",
        "ready",
        "profiling",
        "processing",
        "stopping",
        "shutdown",
    ]
    script = f"""
        import {{ deriveJobRunState }} from {_JOB_DETAIL_STATE_JS.as_uri()!r};
        import {{ shouldShowSweepDiagnostics }} from {_SWEEP_DETAIL_HELPERS_JS.as_uri()!r};
        const phases = {json.dumps(subphases)};
        console.log(JSON.stringify(phases.map((phase) => {{
          const state = deriveJobRunState({{ phase, epoch: undefined, runEpoch: '9' }});
          return {{
            phase,
            isTerminal: state.isTerminal,
            isRunning: state.isRunning,
            pollingDone: state.pollingDone,
            showLiveRunPanels: state.showLiveRunPanels,
            showSweepDiagnostics: shouldShowSweepDiagnostics(phase),
          }};
        }})));
    """

    states = _node_json(script)

    assert states == [
        {
            "phase": phase,
            "isTerminal": False,
            "isRunning": False,
            "pollingDone": False,
            "showLiveRunPanels": True,
            "showSweepDiagnostics": False,
        }
        for phase in subphases
    ]


def test_sweep_diagnostics_running_phase_set_is_narrow_and_case_insensitive() -> None:
    script = f"""
        import {{ shouldShowSweepDiagnostics }} from {_SWEEP_DETAIL_HELPERS_JS.as_uri()!r};
        const phases = ['Pending', 'running', 'Aggregating', 'Completed', 'partiallyFailed', 'archived'];
        console.log(JSON.stringify(phases.map((phase) => shouldShowSweepDiagnostics(phase))));
    """

    assert _node_json(script) == [True, True, True, False, False, False]


def test_static_phase_sets_include_expected_adversarial_aliases() -> None:
    job_state_src = _source(_JOB_DETAIL_STATE_JS)
    sweep_helper_src = _source(_SWEEP_DETAIL_HELPERS_JS)
    theme_src = _source(_THEME_JS)

    assert (
        "const CANCELLED_PHASES = new Set(['cancelled', 'canceled']);" in job_state_src
    )
    assert "'partiallyfailed'" in job_state_src
    assert "'archived'" in job_state_src
    assert (
        "const RUNNING_PHASES = new Set(['pending', 'running', 'aggregating']);"
        in sweep_helper_src
    )
    assert "p === 'completed' || p === 'succeeded' || p === 'archived'" in theme_src
    assert "return colors.phaseUnknown;" in theme_src

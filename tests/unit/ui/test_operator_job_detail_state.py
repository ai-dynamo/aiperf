# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for job-detail state helpers."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_archived_phase_is_terminal_and_not_live() -> None:
    script = f"""
        import {{ deriveJobRunState }} from {JOB_DETAIL_STATE_PATH.as_uri()!r};
        const state = deriveJobRunState({{
          phase: 'Archived',
          epoch: '1779050863',
          runEpoch: null,
        }});
        console.log(JSON.stringify(state));
    """

    state = json.loads(run_node(script))

    assert state == {
        "phaseLower": "archived",
        "isRunning": False,
        "isCompleted": False,
        "isCancelled": False,
        "isPartiallyFailed": False,
        "isArchived": True,
        "isTerminal": True,
        "viewingCurrentRun": False,
        "pollingDone": True,
        "showLiveRunPanels": False,
    }

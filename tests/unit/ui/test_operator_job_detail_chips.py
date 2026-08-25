# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for job-detail chip cleanup helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
RUN_SELECTOR_PATH = UI_ROOT / "lib" / "run-selector.js"
CONDITION_HELPERS_PATH = UI_ROOT / "components" / "conditions-helpers.js"


def test_final_run_selector_uses_single_epoch_row_without_latest_pill() -> None:
    script = f"""
        import {{ buildRunSelectorRows }} from {RUN_SELECTOR_PATH.as_uri()!r};
        const rows = buildRunSelectorRows({{
          namespace: 'bench',
          name: 'job',
          hasLive: true,
          isRunning: false,
          current: '1779024475',
          epochs: [{{
            epoch: '1779024475',
            isLatest: true,
            mtimeEpoch: 1779024475,
            fileCount: 12,
          }}],
        }});
        console.log(JSON.stringify(rows));
    """

    rows = json.loads(run_node(script))

    assert rows == [
        {
            "kind": "epoch",
            "epoch": "1779024475",
            "label": "1779024475",
            "selected": True,
            "href": "#/jobs/bench/job/runs/1779024475",
            "fileCount": 12,
            "mtimeEpoch": 1779024475,
            "isLatest": False,
        }
    ]


def test_condition_chips_hide_success_noise_and_prettify_warnings() -> None:
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'ConfigValid', status: 'True' }},
          {{ type: 'EndpointReachable', status: 'True' }},
          {{ type: 'PreflightPassed', status: 'True' }},
          {{ type: 'PreflightHasWarnings', status: 'False', reason: 'WarningsPresent' }},
          {{ type: 'ResourcesCreated', status: 'True' }},
          {{ type: 'ResultsAvailable', status: 'True' }},
          {{ type: 'Failed', status: 'False' }},
          {{ type: 'Complete', status: 'True' }},
        ]);
        console.log(JSON.stringify(badges.map(b => {{
          return {{ type: b.type, label: b.label, className: b.className }};
        }})));
    """

    badges = json.loads(run_node(script))

    assert badges == [
        {
            "type": "PreflightHasWarnings",
            "label": "Preflight warnings",
            "className": "condition-badge--progress",
        },
        {
            "type": "Failed",
            "label": "Failed",
            "className": "condition-badge--false",
        },
    ]

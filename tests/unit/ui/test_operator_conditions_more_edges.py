# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for operator condition badge helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

CONDITION_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "conditions-helpers.js"
)


def test_visible_condition_badges_hide_noisy_success_conditions_case_insensitively() -> (
    None
):
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'ConfigValid', status: 'True' }},
          {{ type: 'EndpointReachable', status: 'TRUE' }},
          {{ type: 'ResourcesCreated', status: 'true' }},
          {{ type: 'ResultsAvailable', status: 'False', reason: 'WaitingForArtifacts' }},
        ]);
        console.log(JSON.stringify(badges.map(b => {{
          return {{ type: b.type, label: b.label, className: b.className }};
        }})));
    """

    assert json.loads(run_node(script)) == [
        {
            "type": "ResultsAvailable",
            "label": "Results",
            "className": "condition-badge--progress",
        }
    ]


def test_visible_condition_badges_hide_false_terminal_counterpart_conditions() -> None:
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'Complete', status: 'False', reason: 'FailedConditionFalse' }},
          {{ type: 'Failed', status: 'False', reason: 'CompleteConditionFalse' }},
          {{ type: 'ResultsAvailable', status: 'False', reason: 'WaitingForArtifacts' }},
        ]);
        console.log(JSON.stringify(badges.map(b => b.type)));
    """

    assert json.loads(run_node(script)) == ["ResultsAvailable"]


def test_warning_conditions_are_prettified_and_progress_classed() -> None:
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'PreflightHasWarnings', status: 'False', reason: 'WarningsPresent' }},
          {{ type: 'EndpointHasWarnings', status: 'Unknown', message: 'TLS warning ignored' }},
        ]);
        console.log(JSON.stringify(badges.map(b => {{
          return {{ type: b.type, label: b.label, className: b.className }};
        }})));
    """

    assert json.loads(run_node(script)) == [
        {
            "type": "PreflightHasWarnings",
            "label": "Preflight warnings",
            "className": "condition-badge--progress",
        },
        {
            "type": "EndpointHasWarnings",
            "label": "Endpoint Warnings",
            "className": "condition-badge--progress",
        },
    ]


def test_unknown_condition_type_falls_back_to_readable_label_and_unknown_class() -> (
    None
):
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'CacheHasWarmupDrift', status: 'Maybe', reason: 'ControllerDidNotSay' }},
        ]);
        console.log(JSON.stringify(badges.map(b => {{
          return {{ type: b.type, label: b.label, className: b.className }};
        }})));
    """

    assert json.loads(run_node(script)) == [
        {
            "type": "CacheHasWarmupDrift",
            "label": "Cache Warmup Drift",
            "className": "condition-badge--unknown",
        }
    ]


def test_visible_condition_badge_summary_keeps_filtered_order_and_overflow_count() -> (
    None
):
    script = f"""
        import {{ visibleConditionBadgeSummary }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const summary = visibleConditionBadgeSummary([
          {{ type: 'ConfigValid', status: 'True' }},
          {{ type: 'FirstProblem', status: 'False' }},
          {{ type: 'EndpointReachable', status: 'True' }},
          {{ type: 'SecondProblem', status: 'Unknown' }},
          {{ type: 'ThirdProblem', status: 'False' }},
        ], 2);
        console.log(JSON.stringify({{
          types: summary.badges.map(b => b.type),
          overflow: summary.overflow,
        }}));
    """

    assert json.loads(run_node(script)) == {
        "types": ["FirstProblem", "SecondProblem"],
        "overflow": 1,
    }


def test_missing_status_reason_and_message_fields_do_not_break_badge_mapping() -> None:
    script = f"""
        import {{ visibleConditionBadges }} from {CONDITION_HELPERS_PATH.as_uri()!r};
        const badges = visibleConditionBadges([
          {{ type: 'ControllerHasNoOpinion' }},
          {{ type: 'WorkerNeedsRecords', status: 'False' }},
        ]);
        console.log(JSON.stringify(badges.map(b => {{
          return {{
            type: b.type,
            label: b.label,
            className: b.className,
            hasMessage: Object.prototype.hasOwnProperty.call(b, 'message'),
          }};
        }})));
    """

    assert json.loads(run_node(script)) == [
        {
            "type": "ControllerHasNoOpinion",
            "label": "Controller No Opinion",
            "className": "condition-badge--unknown",
            "hasMessage": True,
        },
        {
            "type": "WorkerNeedsRecords",
            "label": "Worker Needs Records",
            "className": "condition-badge--false",
            "hasMessage": True,
        },
    ]

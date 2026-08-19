# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-detail pure helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

SWEEP_DETAIL_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)


def test_archived_sweep_variations_use_cell_metrics_when_child_jobs_are_gone() -> None:
    script = f"""
        import {{ buildSweepVariations }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = buildSweepVariations({{
          manifest: [{{ name: 'sweep-v00', variationIndex: 0, variationLabel: 'concurrency=8' }}],
          childSummaries: {{ 'sweep-v00': {{ summary: null, phase: null }} }},
          cells: {{ cells: [{{
            variation_index: 0,
            variation_label: 'concurrency=8',
            metrics: {{ request_throughput: {{ avg: 11 }}, request_latency: {{ p99: 22 }} }},
          }}] }},
        }});
        console.log(JSON.stringify(variations));
    """

    variations = json.loads(run_node(script))

    # `std` is null, not 0. An archived cell contributes one observation, and
    # one observation does not estimate spread -- `std: 0` would assert "every
    # trial landed on the same number", which is a reproducibility result this
    # archive never measured. `cv` was already null here for the same reason;
    # only `std` was inconsistent.
    assert variations[0]["n_trials"] == 1
    assert variations[0]["perMetric"]["request_throughput.avg"] == {
        "mean": 11,
        "std": None,
        "cv": None,
        "n": 1,
    }
    assert variations[0]["perMetric"]["request_latency.p99"] == {
        "mean": 22,
        "std": None,
        "cv": None,
        "n": 1,
    }


def test_manifest_falls_back_to_archived_detail_children() -> None:
    script = f"""
        import {{ resolveSweepManifest }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const manifest = resolveSweepManifest({{
          detail: {{
            children: [{{
              name: 'sweep-v00',
              namespace: 'bench',
              phase: 'Archived',
              variationIndex: 0,
              variationLabel: 'latin_hypercube_0000',
            }}],
          }},
          archivedChildren: [],
        }});
        console.log(JSON.stringify(manifest));
    """

    assert json.loads(run_node(script)) == [
        {
            "name": "sweep-v00",
            "namespace": "bench",
            "phase": "Archived",
            "variationIndex": 0,
            "variationLabel": "latin_hypercube_0000",
        }
    ]


def test_archived_sweep_detail_hides_diagnostics_panel() -> None:
    script = f"""
        import {{ shouldShowSweepDiagnostics }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          archived: shouldShowSweepDiagnostics('Archived'),
          succeeded: shouldShowSweepDiagnostics('Succeeded'),
          running: shouldShowSweepDiagnostics('Running'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "archived": False,
        "succeeded": False,
        "running": True,
    }


def test_build_trial_board_rows_groups_trials_and_statuses() -> None:
    script = f"""
        import {{ buildTrialBoardRows }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const rows = buildTrialBoardRows({{
          manifest: [
            {{ name: 'sweep-v00-t0', namespace: 'bench', variationIndex: 0, variationLabel: 'concurrency=8', trialIndex: 0 }},
            {{ name: 'sweep-v00-t1', namespace: 'bench', variationIndex: 0, variationLabel: 'concurrency=8', trialIndex: 1 }},
            {{ name: 'sweep-v01-t0', namespace: 'bench', variationIndex: 1, variationLabel: 'concurrency=16', trialIndex: 0 }},
          ],
          childSummaries: {{
            'sweep-v00-t0': {{ phase: 'Succeeded', progressPercent: 100, summary: {{ output_token_throughput: {{ avg: 1000 }} }} }},
            'sweep-v00-t1': {{ phase: 'Running', progressPercent: 25, summary: null }},
            'sweep-v01-t0': {{ phase: 'Failed', progressPercent: 10, summary: null }},
          }},
        }});
        console.log(JSON.stringify(rows));
    """

    rows = json.loads(run_node(script))

    assert [row["variation_index"] for row in rows] == [0, 1]
    assert rows[0]["label"] == "concurrency=8"
    assert [trial["state"] for trial in rows[0]["trials"]] == ["succeeded", "running"]
    assert rows[1]["trials"][0]["state"] == "failed"


def test_child_sweep_state_matches_rollup_phase_buckets() -> None:
    script = f"""
        import {{ buildTrialBoardRows, childSweepState }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const rows = buildTrialBoardRows({{
          manifest: [
            {{ name: 'missing-phase', namespace: 'bench', variationIndex: 0, trialIndex: 0 }},
            {{ name: 'status-only', namespace: 'bench', variationIndex: 0, trialIndex: 1, status: 'Succeeded' }},
          ],
          childSummaries: {{}},
        }});
        console.log(JSON.stringify({{
          aggregating: childSweepState('Aggregating'),
          pending: childSweepState('Pending'),
          emptyString: childSweepState(''),
          undefinedPhase: childSweepState(undefined),
          completed: childSweepState('Completed'),
          archived: childSweepState('Archived'),
          partiallyFailed: childSweepState('PartiallyFailed'),
          cancelled: childSweepState('Cancelled'),
          profiling: childSweepState('Profiling'),
          processing: childSweepState('Processing'),
          queued: childSweepState('Queued'),
          initializing: childSweepState('Initializing'),
          nullPhase: childSweepState(null),
          missingPhase: rows[0].trials[0].state,
          statusFallback: rows[0].trials[1].state,
          weird: childSweepState('WaitingForMoonlight'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "aggregating": "running",
        "pending": "pending",
        "emptyString": "pending",
        "undefinedPhase": "pending",
        "completed": "succeeded",
        "archived": "succeeded",
        "partiallyFailed": "failed",
        "cancelled": "cancelled",
        "profiling": "running",
        "processing": "running",
        "queued": "pending",
        "initializing": "pending",
        "nullPhase": "pending",
        "missingPhase": "pending",
        "statusFallback": "succeeded",
        "weird": "unknown",
    }


def test_pick_sweep_winner_uses_output_throughput_by_default() -> None:
    script = f"""
        import {{ pickSweepWinner }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = [
          {{ variation_index: 0, label: 'concurrency=8', n_trials: 2, perMetric: {{ 'output_token_throughput.avg': {{ mean: 900, cv: 0.03, n: 2 }} }} }},
          {{ variation_index: 1, label: 'concurrency=16', n_trials: 2, perMetric: {{ 'output_token_throughput.avg': {{ mean: 1200, cv: 0.04, n: 2 }} }} }},
        ];
        console.log(JSON.stringify(pickSweepWinner({{ variations }})));
    """

    winner = json.loads(run_node(script))

    assert winner["variation_index"] == 1
    assert winner["metricKey"] == "output_token_throughput.avg"
    assert winner["higherIsBetter"] is True


def test_pick_sweep_winner_inverts_latency_metric() -> None:
    script = f"""
        import {{ pickSweepWinner }} from {SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = [
          {{ variation_index: 0, label: 'concurrency=8', n_trials: 2, perMetric: {{ 'request_latency.p99': {{ mean: 80, cv: 0.03, n: 2 }} }} }},
          {{ variation_index: 1, label: 'concurrency=16', n_trials: 2, perMetric: {{ 'request_latency.p99': {{ mean: 120, cv: 0.04, n: 2 }} }} }},
        ];
        console.log(JSON.stringify(pickSweepWinner({{ variations, metricKey: 'request_latency.p99' }})));
    """

    winner = json.loads(run_node(script))

    assert winner["variation_index"] == 0
    assert winner["higherIsBetter"] is False

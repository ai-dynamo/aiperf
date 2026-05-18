# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-detail pure helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SWEEP_DETAIL_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


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

    variations = json.loads(_run_node(script))

    assert variations[0]["n_trials"] == 1
    assert variations[0]["perMetric"]["request_throughput.avg"] == {
        "mean": 11,
        "std": 0,
        "cv": None,
        "n": 1,
    }
    assert variations[0]["perMetric"]["request_latency.p99"] == {
        "mean": 22,
        "std": 0,
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

    assert json.loads(_run_node(script)) == [
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

    assert json.loads(_run_node(script)) == {
        "archived": False,
        "succeeded": False,
        "running": True,
    }

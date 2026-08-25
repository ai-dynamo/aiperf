# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for compare, leaderboard, and history UI helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_PAGES_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui" / "pages"
)
COMPARE_PATH = UI_PAGES_DIR / "compare.js"
LEADERBOARD_PATH = UI_PAGES_DIR / "leaderboard.js"
HISTORY_PATH = UI_PAGES_DIR / "history.js"


def _compare_helper_script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        const palette = {{
          mauve: '#cba6f7', blue: '#89b4fa', green: '#a6e3a1', peach: '#fab387',
          pink: '#f5c2e7', teal: '#94e2d5', sapphire: '#74c7ec', yellow: '#f9e2af',
          lavender: '#b4befe', maroon: '#eba0ac', red: '#f38ba8', overlay0: '#6c7086',
          mantle: '#181825', text: '#cdd6f4', surface0: '#313244', overlay1: '#7f849c',
        }};
        const modelColor = (model) => model === 'model-a' ? '#111111' : '#222222';
        const fmtNumber = (value) => String(value);
        const source = fs.readFileSync({json.dumps(str(COMPARE_PATH))}, 'utf8');
        const helpers = source
          .slice(0, source.indexOf('export function Compare()'))
          .replace(/^import .*$/gm, '')
          .replace(/^export \{{[^\\r\\n]+\}};$/gm, '');
        eval(helpers + '\\n' + {json.dumps(body)});
    """


def test_compare_lab_groups_frontiers_by_namespace_and_model_without_aggregating() -> (
    None
):
    script = _compare_helper_script(
        """
        const entries = [
          { metric: 'request_throughput', stat: 'avg', values: {
            'ns-a/run-1': 100, 'ns-b/run-2': 90, 'ns-a/run-3': 110, 'ns-b/run-4': 80,
          } },
          { metric: 'request_latency', stat: 'p99', values: {
            'ns-a/run-1': 20, 'ns-b/run-2': 12, 'ns-a/run-3': 30, 'ns-b/run-4': 10,
          } },
        ];
        const meta = {
          'ns-a/run-1': { model: 'model-a' },
          'ns-a/run-3': { model: 'model-a' },
          'ns-b/run-2': { model: 'model-b' },
          'ns-b/run-4': { model: 'model-b' },
        };
        const splitKey = (key) => {
          const idx = key.indexOf('/');
          return { ns: key.slice(0, idx), jobId: key.slice(idx + 1) };
        };
        const axis = LAB_AXES[0];
        const points = buildLabPoints(
          entries, axis, ['ns-a/run-1', 'ns-b/run-2', 'ns-a/run-3', 'ns-b/run-4'], splitKey, meta,
        );
        const datasets = buildLabDatasets(points, axis, null);
        console.log(JSON.stringify({
          clusterKeys: Array.from(new Set(points.map((p) => p.clusterKey))).sort(),
          labels: datasets.map((d) => d.label).sort(),
          frontierSizes: datasets.filter((d) => d.showLine).map((d) => d.data.length).sort(),
        }));
        """
    )

    out = json.loads(run_node(script))

    assert out == {
        "clusterKeys": ["ns-a · model-a", "ns-b · model-b"],
        "frontierSizes": [2, 2],
        "labels": [
            "ns-a · model-a",
            "ns-a · model-a · frontier",
            "ns-b · model-b",
            "ns-b · model-b · frontier",
        ],
    }


def test_compare_scatter_points_skip_missing_metric_values_and_gpu_telemetry() -> None:
    script = _compare_helper_script(
        """
        const entries = [
          { metric: 'output_token_throughput_per_user', stat: 'avg', values: {
            'ns-a/run-1': 10, 'ns-a/run-2': 20, 'ns-a/run-3': 30,
          } },
          { metric: 'output_token_throughput', stat: 'avg', values: {
            'ns-a/run-1': 80, 'ns-a/run-2': null,
          } },
        ];
        const splitKey = (key) => {
          const idx = key.indexOf('/');
          return { ns: key.slice(0, idx), jobId: key.slice(idx + 1) };
        };
        const points = buildScatterPoints(
          entries,
          { metric: 'output_token_throughput_per_user', stat: 'avg' },
          { metric: 'output_token_throughput', stat: 'avg' },
          ['ns-a/run-1', 'ns-a/run-2', 'ns-a/run-3'],
          splitKey,
          { 'ns-a/run-1': { gpu_count: 4, gpu_name: 'NVIDIA H100' }, 'ns-a/run-2': { gpu_count: 4 } },
          true,
        );
        const missingMetricPoints = buildScatterPoints(
          entries,
          { metric: 'does_not_exist', stat: 'avg' },
          { metric: 'output_token_throughput', stat: 'avg' },
          ['ns-a/run-1'], splitKey, {}, false,
        );
        console.log(JSON.stringify({ points, missingMetricPoints }));
        """
    )

    out = json.loads(run_node(script))

    assert out["missingMetricPoints"] == []
    assert [point["key"] for point in out["points"]] == ["ns-a/run-1"]
    assert out["points"][0]["y"] == 20
    assert out["points"][0]["gpuFamily"] == "H100"


def test_history_sorting_is_stable_by_start_time_then_job_id() -> None:
    source = HISTORY_PATH.read_text()

    assert "Date.parse(a.start_time)" in source
    assert "Date.parse(b.start_time)" in source
    assert "return ta - tb" in source
    assert "String(a.job_id ?? '').localeCompare(String(b.job_id ?? ''))" in source


def test_history_query_params_drive_filters_and_clear_chips() -> None:
    source = HISTORY_PATH.read_text()

    assert "const ns = q.ns ?? ''" in source
    assert "const urlModel = q.model ?? ''" in source
    assert "const urlEndpoint = q.endpoint ?? ''" in source
    assert "setQuery({ model })" in source
    assert "setQuery({ endpoint })" in source
    assert "setQuery({ ns: undefined })" in source


def test_leaderboard_requests_full_filterable_limit_and_ranks_by_direction() -> None:
    """Full fetch limit, and rank order imposed client-side.

    The "keeps API sorting" half of this assertion was wrong: the API always
    sorts descending because the UI never sends `order`, which put the slowest
    run at rank 1 for latency metrics.
    """
    source = LEADERBOARD_PATH.read_text()

    assert "const LEADERBOARD_FETCH_LIMIT = 1000;" in source
    assert (
        "getLeaderboard(selected.metric, selected.stat, LEADERBOARD_FETCH_LIMIT)"
        in source
    )
    assert "const top10 = filtered.slice(0, 10)" in source
    assert "function rankEntries(" in source


def test_compare_api_preserves_selected_namespace_qualified_run_identity() -> None:
    source = COMPARE_PATH.read_text()

    assert "const bareJobIds = keys.map((k) => splitKey(k).jobId);" not in source
    assert (
        "const bareJobIds = selectedKeys.map((k) => splitKey(k).jobId);" not in source
    )
    assert re.search(r"api\.compareJobs\((keys|selectedKeys)\)", source)

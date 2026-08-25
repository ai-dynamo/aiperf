# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial identity/filtering tests for compare, leaderboard, and history UI."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
API_PATH = UI_ROOT / "lib" / "api.js"
COMPARE_PATH = UI_ROOT / "pages" / "compare.js"
COMPARE_FILTERS_PATH = UI_ROOT / "pages" / "compare-filters.js"
HISTORY_PATH = UI_ROOT / "pages" / "history.js"
LEADERBOARD_PATH = UI_ROOT / "pages" / "leaderboard.js"


def _compare_helper_script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        const palette = {{
          mauve: '#cba6f7', blue: '#89b4fa', green: '#a6e3a1', peach: '#fab387',
          pink: '#f5c2e7', teal: '#94e2d5', sapphire: '#74c7ec', yellow: '#f9e2af',
          lavender: '#b4befe', maroon: '#eba0ac', red: '#f38ba8', overlay0: '#6c7086',
          mantle: '#181825', text: '#cdd6f4', surface0: '#313244', overlay1: '#7f849c',
        }};
        const modelColor = (model) => 'color:' + model;
        const fmtNumber = (value) => String(value);
        const source = fs.readFileSync({json.dumps(str(COMPARE_PATH))}, 'utf8');
        const helpers = source
          .slice(0, source.indexOf('export function Compare()'))
          .replace(/^import .*$/gm, '')
          .replace('export {{ applyJobFilters, extractFacets, extractCrossFacets, FILTER_NONE }};', '');
        eval(helpers + '\\n' + {json.dumps(body)});
    """


def _leaderboard_script(entries: list[dict[str, object]], body: str) -> str:
    return f"""
        import fs from 'node:fs';
        import {{ applyJobFilters, extractCrossFacets, FILTER_NONE }} from {COMPARE_FILTERS_PATH.as_uri()!r};

        const source = fs.readFileSync({json.dumps(str(LEADERBOARD_PATH))}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function Leaderboard', 'function Leaderboard');
        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
        const stateValues = [
          {{ metric: 'request_throughput', stat: 'avg' }},
          new Set(),
          new Set(),
          new Set(),
          {{ entries: {json.dumps(entries)} }},
          false,
          null,
        ];
        function useState(initial) {{
          const value = stateValues.length ? stateValues.shift() : (typeof initial === 'function' ? initial() : initial);
          return [value, () => {{}}];
        }}
        function useEffect() {{}}
        function useMemo(fn) {{ return fn(); }}
        const api = {{ getLeaderboard: async () => ({{ entries: [] }}) }};
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        const palette = {{
          mauve: '#cba6f7', blue: '#89b4fa', green: '#a6e3a1', peach: '#fab387',
          pink: '#f5c2e7', teal: '#94e2d5', sapphire: '#74c7ec', yellow: '#f9e2af',
          flamingo: '#f2cdcd', lavender: '#b4befe', subtext1: '#bac2de', overlay0: '#6c7086',
          surface0: '#313244', surface1: '#45475a', error: '#f38ba8',
        }};
        function buildJobPath(entry) {{ return '/job/' + JSON.stringify(entry); }}
        function navigate() {{}}
        function MetricSelector() {{}}
        function ChartWrapper() {{}}
        function LoadingPanel() {{}}
        function fmtNumber(value) {{ return Number(value).toFixed(2); }}
        eval(source + '\\nglobalThis.Leaderboard = Leaderboard;');

        function collectValues(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{ for (const item of node) collectValues(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{ for (const value of node.values) collectValues(value, out); return out; }}
          out.push(node);
          return out;
        }}
        function collectText(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{ for (const item of node) collectText(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{
            for (const string of node.strings) out.push(String(string));
            for (const value of node.values) collectText(value, out);
            return out;
          }}
          if (typeof node === 'string' || typeof node === 'number') out.push(String(node));
          return out;
        }}
        function chartData(rendered) {{ return collectValues(rendered).find((value) => value?.datasets?.[0]?.data); }}
        {body}
    """


def _history_script(
    entries: list[dict[str, object]], query_value: dict[str, str], body: str
) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({json.dumps(str(HISTORY_PATH))}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function History', 'function History');
        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
        const query = {{ value: {json.dumps(query_value)} }};
        function setQuery() {{}}
        const stateValues = [
          {{ metric: 'request_throughput', stat: 'avg' }},
          query.value.model ?? '',
          query.value.endpoint ?? '',
          {{ entries: {json.dumps(entries)} }},
          false,
          null,
        ];
        function useState(initial) {{
          const value = stateValues.length ? stateValues.shift() : (typeof initial === 'function' ? initial() : initial);
          return [value, () => {{}}];
        }}
        function useEffect() {{}}
        const api = {{ getHistory: async () => ({{ entries: [] }}) }};
        const palette = {{ blue: '#89b4fa', teal: '#94e2d5', peach: '#fab387', overlay0: '#6c7086', surface0: '#313244' }};
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        function buildJobPath(entry) {{ return '/job/' + JSON.stringify(entry); }}
        function navigate() {{}}
        function MetricSelector() {{}}
        function ChartWrapper() {{}}
        function LoadingPanel() {{}}
        function NsPill() {{}}
        function ModelPill() {{}}
        function fmtNumber(value) {{ return Number(value).toFixed(3); }}
        eval(source + '\\nglobalThis.History = History;');

        function collectValues(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{ for (const item of node) collectValues(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{ for (const value of node.values) collectValues(value, out); return out; }}
          out.push(node);
          return out;
        }}
        function collectText(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{ for (const item of node) collectText(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{
            for (const string of node.strings) out.push(String(string));
            for (const value of node.values) collectText(value, out);
            return out;
          }}
          if (typeof node === 'string' || typeof node === 'number') out.push(String(node));
          return out;
        }}
        function chartData(rendered) {{ return collectValues(rendered).find((value) => value?.datasets?.[0]?.data); }}
        {body}
    """


def _api_script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({json.dumps(str(API_PATH))}, 'utf8')
          .replace(/import \{{[\s\S]*?\}} from '\.\/state\.js';/, "const setError = () => {{}};");
        const module = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
        const api = module.api;
        {body}
    """


def test_compare_keeps_same_job_id_distinct_by_namespace_and_does_not_aggregate() -> (
    None
):
    script = _compare_helper_script(
        """
        const entries = [
          { metric: 'request_throughput', stat: 'avg', values: { 'alpha/same': 10, 'beta/same': 40 } },
          { metric: 'request_latency', stat: 'p99', values: { 'alpha/same': 80, 'beta/same': 20 } },
        ];
        const splitKey = (key) => {
          const idx = key.indexOf('/');
          return { ns: key.slice(0, idx), jobId: key.slice(idx + 1) };
        };
        const points = buildLabPoints(
          entries,
          LAB_AXES[0],
          ['alpha/same', 'beta/same'],
          splitKey,
          { 'alpha/same': { model: 'org/Model A' }, 'beta/same': { model: 'org/Model A' } },
        );
        const datasets = buildLabDatasets(points, LAB_AXES[0], null);
        console.log(JSON.stringify({
          keys: points.map((p) => p.key),
          values: points.map((p) => [p.x, p.y]),
          clusters: datasets.filter((d) => !d.showLine).map((d) => d.label).sort(),
        }));
        """
    )

    assert json.loads(run_node(script)) == {
        "keys": ["alpha/same", "beta/same"],
        "values": [[10, 80], [40, 20]],
        "clusters": ["alpha · org/Model A", "beta · org/Model A"],
    }


def test_compare_filters_model_names_with_slashes_and_spaces_exactly() -> None:
    script = f"""
        import {{ applyJobFilters, extractCrossFacets }} from {COMPARE_FILTERS_PATH.as_uri()!r};
        const jobs = [
          {{ job_id: 'a', namespace: 'ns', model: 'meta/llama 3.1 70b', endpoint: 'vllm' }},
          {{ job_id: 'b', namespace: 'ns', model: 'meta/llama', endpoint: 'vllm' }},
          {{ job_id: 'c', namespace: 'ns', model: 'llama 3.1 70b', endpoint: 'vllm' }},
        ];
        const filters = {{
          nsFilter: new Set(),
          modelFilter: new Set(['meta/llama 3.1 70b']),
          endpointFilter: new Set(),
          search: '',
        }};
        const filtered = applyJobFilters(jobs, filters).map((j) => j.job_id);
        const modelFacets = Array.from(extractCrossFacets(jobs, filters).model.keys()).sort();
        console.log(JSON.stringify({{ filtered, modelFacets }}));
    """

    assert json.loads(run_node(script)) == {
        "filtered": ["a"],
        "modelFacets": ["llama 3.1 70b", "meta/llama", "meta/llama 3.1 70b"],
    }


def test_compare_scatter_skips_missing_metrics_without_zero_filling() -> None:
    script = _compare_helper_script(
        """
        const entries = [
          { metric: 'output_token_throughput_per_user', stat: 'avg', values: { 'ns/ok': 8, 'ns/missing-y': 9, 'ns/zero': 0 } },
          { metric: 'output_token_throughput', stat: 'avg', values: { 'ns/ok': 32, 'ns/missing-y': null, 'ns/zero': 0 } },
        ];
        const splitKey = (key) => ({ ns: key.split('/')[0], jobId: key.split('/').slice(1).join('/') });
        const points = buildScatterPoints(
          entries,
          { metric: 'output_token_throughput_per_user', stat: 'avg' },
          { metric: 'output_token_throughput', stat: 'avg' },
          ['ns/ok', 'ns/missing-y', 'ns/zero'],
          splitKey,
          { 'ns/ok': { gpu_count: 4 }, 'ns/missing-y': { gpu_count: 4 }, 'ns/zero': { gpu_count: 4 } },
          true,
        );
        console.log(JSON.stringify(points.map((p) => ({ key: p.key, x: p.x, y: p.y }))));
        """
    )

    assert json.loads(run_node(script)) == [
        {"key": "ns/ok", "x": 8, "y": 8},
        {"key": "ns/zero", "x": 0, "y": 0},
    ]


def test_compare_scatter_and_lab_skip_nan_infinity_values() -> None:
    script = _compare_helper_script(
        """
        const entries = [
          { metric: 'request_throughput', stat: 'avg', values: {
            'ns/ok': 10, 'ns/nan-x': 'NaN', 'ns/inf-y': 30, 'ns/zero': 0,
          } },
          { metric: 'request_latency', stat: 'p99', values: {
            'ns/ok': 50, 'ns/nan-x': 40, 'ns/inf-y': 'Infinity', 'ns/zero': 0,
          } },
        ];
        const keys = ['ns/ok', 'ns/nan-x', 'ns/inf-y', 'ns/zero'];
        const splitKey = (key) => ({ ns: key.split('/')[0], jobId: key.split('/').slice(1).join('/') });
        const meta = {
          'ns/ok': { gpu_count: 1, model: 'm' },
          'ns/nan-x': { gpu_count: 1, model: 'm' },
          'ns/inf-y': { gpu_count: 1, model: 'm' },
          'ns/zero': { gpu_count: 1, model: 'm' },
        };
        const scatter = buildScatterPoints(
          entries,
          { metric: 'request_throughput', stat: 'avg' },
          { metric: 'request_latency', stat: 'p99' },
          keys,
          splitKey,
          meta,
          false,
        );
        const lab = buildLabPoints(entries, LAB_AXES[0], keys, splitKey, meta);
        console.log(JSON.stringify({
          scatter: scatter.map((p) => ({ key: p.key, x: p.x, y: p.y })),
          lab: lab.map((p) => ({ key: p.key, x: p.x, y: p.y })),
        }));
        """
    )

    assert json.loads(run_node(script)) == {
        "scatter": [
            {"key": "ns/ok", "x": 10, "y": 50},
            {"key": "ns/zero", "x": 0, "y": 0},
        ],
        "lab": [
            {"key": "ns/ok", "x": 10, "y": 50},
            {"key": "ns/zero", "x": 0, "y": 0},
        ],
    }


def test_leaderboard_missing_values_are_not_ranked_or_zero_filled() -> None:
    entries = [
        {
            "job_id": "missing",
            "namespace": "prod",
            "model": "meta/llama 3",
            "value": None,
            "rank": None,
        },
        {
            "job_id": "valid",
            "namespace": "prod",
            "model": "meta/llama 3",
            "value": 42,
            "rank": None,
        },
    ]
    script = _leaderboard_script(
        entries,
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify({ labels: chartData(rendered).labels, text: collectText(rendered) }));
        """,
    )

    out = json.loads(run_node(script))

    assert out["labels"] == ["valid"]
    assert "missing" not in out["text"]
    assert "null" not in out["text"]


def test_leaderboard_null_api_ranks_do_not_collapse_distinct_same_model_runs() -> None:
    entries = [
        {
            "job_id": "run-a",
            "namespace": "prod",
            "model": "meta/llama 3",
            "value": 12,
            "rank": None,
        },
        {
            "job_id": "run-b",
            "namespace": "prod",
            "model": "meta/llama 3",
            "value": 11,
            "rank": None,
        },
        {
            "job_id": "run-c",
            "namespace": "prod",
            "model": "meta/llama 3",
            "value": 10,
            "rank": None,
        },
    ]
    script = _leaderboard_script(
        entries,
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify({ labels: chartData(rendered).labels, text: collectText(rendered) }));
        """,
    )

    out = json.loads(run_node(script))

    assert out["labels"] == ["run-a", "run-b", "run-c"]
    assert all(rank in out["text"] for rank in ["1", "2", "3"])
    assert "null" not in out["text"]


def test_history_filters_do_not_treat_malicious_query_strings_as_selectors() -> None:
    entries = [
        {
            "job_id": "visible",
            "namespace": "prod",
            "model": "safe model",
            "endpoint": "https://api.example",
            "start_time": "2026-05-01T00:00:00Z",
            "value": 1,
        },
        {
            "job_id": "hidden",
            "namespace": "prod&ns=dev",
            "model": "safe model",
            "endpoint": "https://api.example",
            "start_time": "2026-05-02T00:00:00Z",
            "value": 2,
        },
    ]
    script = _history_script(
        entries,
        {"ns": "prod&ns=dev", "model": "safe", "endpoint": "api.example"},
        """
        const rendered = History();
        const text = collectText(rendered);
        console.log(JSON.stringify({
          ids: text.filter((value) => ['visible', 'hidden'].includes(value)),
          data: chartData(rendered).datasets[0].data,
        }));
        """,
    )

    assert json.loads(run_node(script)) == {"ids": ["hidden"], "data": [2]}


def test_history_keeps_repeated_runs_as_separate_points_without_aggregation() -> None:
    entries = [
        {
            "job_id": "repeat-a",
            "namespace": "prod",
            "model": "m",
            "endpoint": "e",
            "start_time": "2026-05-01T10:00:00Z",
            "value": 10,
        },
        {
            "job_id": "repeat-b",
            "namespace": "prod",
            "model": "m",
            "endpoint": "e",
            "start_time": "2026-05-01T10:00:00Z",
            "value": 20,
        },
    ]
    script = _history_script(
        entries,
        {},
        """
        const chart = chartData(History());
        console.log(JSON.stringify({ labels: chart.labels, data: chart.datasets[0].data }));
        """,
    )

    out = json.loads(run_node(script))

    assert len(out["data"]) == 2
    assert out["data"] == [10, 20]
    assert out["labels"][0] == out["labels"][1]


def test_compare_api_uses_repeated_selected_params_and_encodes_injection_strings() -> (
    None
):
    script = _api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({ url, method: opts.method ?? null, body: opts.body ?? null });
          return { ok: true, status: 200, json: async () => ({ ok: true }), text: async () => 'ok' };
        };
        const selected = [
          'prod/same',
          'dev/same',
          'team space/model/name',
          'safe&jobs=evil&metric=request_latency',
          'x?jobs=evil#frag',
        ];
        await api.compareJobs(selected);
        const url = calls[0].url;
        const params = new URLSearchParams(url.slice(url.indexOf('?') + 1));
        console.log(JSON.stringify({
          url,
          method: calls[0].method,
          body: calls[0].body,
          jobs: params.getAll('jobs'),
          metrics: params.getAll('metric'),
        }));
        """
    )

    out = json.loads(run_node(script))

    assert out["method"] is None
    assert out["body"] is None
    assert out["jobs"] == [
        "prod/same",
        "dev/same",
        "team space/model/name",
        "safe&jobs=evil&metric=request_latency",
        "x?jobs=evil#frag",
    ]
    assert out["metrics"] == []
    assert "safe%26jobs%3Devil%26metric%3Drequest_latency" in out["url"]
    assert "x%3Fjobs%3Devil%23frag" in out["url"]

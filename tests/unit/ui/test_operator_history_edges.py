# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for the operator history page."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
HISTORY_PATH = UI_ROOT / "pages" / "history.js"


def _history_page_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(HISTORY_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function History', 'function History');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        function useState(initial) {{
          const value = globalThis.__historyStates.shift();
          return [value === undefined ? (typeof initial === 'function' ? initial() : initial) : value, () => {{}}];
        }}
        function useEffect() {{}}
        const api = {{ getHistory: async () => ({{ entries: [] }}) }};
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        const palette = {{
          blue: '#89b4fa', teal: '#94e2d5', peach: '#fab387', overlay0: '#6c7086',
          overlay1: '#7f849c', surface0: '#313244',
        }};
        const query = {{ value: {{}} }};
        const setQueryCalls = [];
        function setQuery(update) {{ setQueryCalls.push(update); }}
        const navigations = [];
        function buildJobPath(entry) {{ return '/job/' + JSON.stringify(entry); }}
        function navigate(path) {{ navigations.push(path); }}
        function MetricSelector() {{}}
        function ChartWrapper() {{}}
        function LoadingPanel() {{}}
        function NsPill(props) {{ return {{ component: 'NsPill', props }}; }}
        function ModelPill(props) {{ return {{ component: 'ModelPill', props }}; }}
        function fmtNumber(value) {{ return String(value); }}

        eval(source + '\\nglobalThis.History = History;');

        function renderHistory({{
          entries = [], queryValue = {{}}, modelState, endpointState,
          loading = false, error = null,
          selected = {{ metric: 'request_throughput', stat: 'avg' }},
        }} = {{}}) {{
          query.value = queryValue;
          globalThis.__historyStates = [
            selected,
            modelState ?? queryValue.model ?? '',
            endpointState ?? queryValue.endpoint ?? '',
            {{ entries }},
            loading,
            error,
          ];
          setQueryCalls.length = 0;
          navigations.length = 0;
          return History();
        }}

        function collectValues(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{
            for (const item of node) collectValues(item, out);
            return out;
          }}
          if (typeof node === 'object' && node.__html) {{
            for (const value of node.values) collectValues(value, out);
            return out;
          }}
          out.push(node);
          return out;
        }}

        function collectText(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{
            for (const item of node) collectText(item, out);
            return out;
          }}
          if (typeof node === 'object' && node.__html) {{
            for (const string of node.strings) out.push(String(string));
            for (const value of node.values) collectText(value, out);
            return out;
          }}
          if (typeof node === 'string' || typeof node === 'number') out.push(String(node));
          return out;
        }}

        function visibleJobIds(node, ids) {{
          const text = collectText(node);
          return text.filter((value) => ids.includes(value));
        }}

        function findChartData(node) {{
          return collectValues(node).find((value) => value?.datasets?.[0]?.data);
        }}

        function runRowNavigation(node) {{
          for (const value of collectValues(node)) {{
            if (typeof value !== 'function') continue;
            try {{
              value();
            }} catch {{
              continue;
            }}
            if (navigations.length) return navigations.at(-1);
          }}
          return null;
        }}

        {expression}
    """


def test_history_orders_archived_and_current_runs_by_start_time_then_job_id() -> None:
    entries = [
        {
            "job_id": "current-newer",
            "namespace": "ns",
            "phase": "Running",
            "start_time": "2026-05-03T00:00:00Z",
            "value": 3,
        },
        {
            "job_id": "archived-newest",
            "namespace": "ns",
            "phase": "Archived",
            "start_time": "2026-05-04T00:00:00Z",
            "value": 4,
        },
        {
            "job_id": "archived-oldest",
            "namespace": "ns",
            "phase": "Archived",
            "start_time": "2026-05-01T00:00:00Z",
            "value": 1,
        },
        {
            "job_id": "current-middle-a",
            "namespace": "ns",
            "phase": "Completed",
            "start_time": "2026-05-02T00:00:00Z",
            "value": 2,
        },
        {
            "job_id": "current-middle-b",
            "namespace": "ns",
            "phase": "Completed",
            "start_time": "2026-05-02T00:00:00Z",
            "value": 2,
        },
    ]
    script = _history_page_script(
        f"""
        const ids = {json.dumps([entry["job_id"] for entry in entries])};
        const rendered = renderHistory({{ entries: {json.dumps(entries)} }});
        console.log(JSON.stringify(visibleJobIds(rendered, ids)));
        """
    )

    assert json.loads(run_node(script)) == [
        "archived-oldest",
        "current-middle-a",
        "current-middle-b",
        "current-newer",
        "archived-newest",
    ]


def test_history_chart_keeps_each_run_as_its_own_dated_point_without_bucketing() -> (
    None
):
    entries = [
        {
            "job_id": "morning",
            "namespace": "ns",
            "start_time": "2026-05-02T09:00:00Z",
            "value": 10,
        },
        {
            "job_id": "evening",
            "namespace": "ns",
            "start_time": "2026-05-02T18:00:00Z",
            "value": 20,
        },
    ]
    script = _history_page_script(
        f"""
        const chart = findChartData(renderHistory({{ entries: {json.dumps(entries)} }}));
        console.log(JSON.stringify({{
          pointCount: chart.datasets[0].data.length,
          values: chart.datasets[0].data,
          labels: chart.labels,
        }}));
        """
    )

    out = json.loads(run_node(script))

    assert out["pointCount"] == 2
    assert out["values"] == [10, 20]
    assert out["labels"][0] == out["labels"][1]


def test_history_filters_model_endpoint_substrings_case_insensitively_and_namespace_exactly() -> (
    None
):
    entries = [
        {
            "job_id": "match",
            "namespace": "prod",
            "model": "Llama-3",
            "endpoint": "https://API.example/v1",
            "start_time": "2026-05-01T00:00:00Z",
            "value": 1,
        },
        {
            "job_id": "wrong-model",
            "namespace": "prod",
            "model": "Mixtral",
            "endpoint": "https://API.example/v1",
            "start_time": "2026-05-02T00:00:00Z",
            "value": 2,
        },
        {
            "job_id": "wrong-endpoint",
            "namespace": "prod",
            "model": "llama-3-small",
            "endpoint": "https://other.example/v1",
            "start_time": "2026-05-03T00:00:00Z",
            "value": 3,
        },
        {
            "job_id": "wrong-namespace",
            "namespace": "production",
            "model": "llama-3",
            "endpoint": "https://api.example/v2",
            "start_time": "2026-05-04T00:00:00Z",
            "value": 4,
        },
    ]
    script = _history_page_script(
        f"""
        const ids = {json.dumps([entry["job_id"] for entry in entries])};
        const rendered = renderHistory({{
          entries: {json.dumps(entries)},
          queryValue: {{ ns: 'prod', model: 'llama', endpoint: 'api.example' }},
        }});
        console.log(JSON.stringify({{
          ids: visibleJobIds(rendered, ids),
          countText: collectText(rendered).filter((value) => value.includes(' of ') || value.includes('filtered')),
        }}));
        """
    )

    out = json.loads(run_node(script))

    assert out["ids"] == ["match"]
    assert "1 of 4 runs" in out["countText"]


def test_history_missing_timestamps_sort_first_and_render_missing_date_marker() -> None:
    entries = [
        {
            "job_id": "dated",
            "namespace": "ns",
            "start_time": "2026-05-01T00:00:00Z",
            "value": 1,
        },
        {"job_id": "missing-b", "namespace": "ns", "value": 2},
        {"job_id": "missing-a", "namespace": "ns", "start_time": None, "value": 3},
    ]
    script = _history_page_script(
        f"""
        const rendered = renderHistory({{ entries: {json.dumps(entries)} }});
        console.log(JSON.stringify({{
          ids: visibleJobIds(rendered, ['dated', 'missing-a', 'missing-b']),
          missingMarkers: collectText(rendered).filter((value) => value === '—').length,
        }}));
        """
    )

    out = json.loads(run_node(script))

    assert out["ids"] == ["missing-a", "missing-b", "dated"]
    assert out["missingMarkers"] >= 2


def test_history_row_links_delegate_run_epoch_and_child_run_epoch_entries_to_router() -> (
    None
):
    entry = {
        "job_id": "sweep-child",
        "namespace": "bench",
        "start_time": "2026-05-01T00:00:00Z",
        "value": 1,
        "runEpoch": "parent-epoch",
        "childRunEpoch": 7,
    }
    script = _history_page_script(
        f"""
        const path = runRowNavigation(renderHistory({{ entries: [{json.dumps(entry)}] }}));
        console.log(JSON.stringify(JSON.parse(path.slice('/job/'.length))));
        """
    )

    out = json.loads(run_node(script))

    assert out["job_id"] == "sweep-child"
    assert out["runEpoch"] == "parent-epoch"
    assert out["childRunEpoch"] == 7


def test_history_empty_states_distinguish_no_runs_from_filters_hiding_runs() -> None:
    hidden_entry = {
        "job_id": "hidden",
        "namespace": "prod",
        "model": "llama",
        "endpoint": "https://api.example/v1",
        "start_time": "2026-05-01T00:00:00Z",
        "value": 1,
    }
    script = _history_page_script(
        f"""
        const emptyText = collectText(renderHistory({{ entries: [] }}));
        const filteredText = collectText(renderHistory({{
          entries: [{json.dumps(hidden_entry)}],
          queryValue: {{ ns: 'dev', model: 'mixtral', endpoint: 'other' }},
        }}));
        const emptyCopy = emptyText.join('');
        const filteredCopy = filteredText.join('');
        console.log(JSON.stringify({{
          emptyHasNoCompletedCopy: emptyCopy.includes('No completed benchmarks yet'),
          filteredHasHiddenCopy: filteredCopy.includes('No data points match the current filters'),
          filteredMentionsNamespace: filteredText.includes('/namespace'),
        }}));
        """
    )

    assert json.loads(run_node(script)) == {
        "emptyHasNoCompletedCopy": True,
        "filteredHasHiddenCopy": True,
        "filteredMentionsNamespace": True,
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator UI JobTable behavior.

``job-table.js`` imports browser-only Preact modules through the HTML import map,
so these tests evaluate the component with tiny hook/template stubs and inspect
its rendered template values. The goal is to exercise the table's pure data
choices without a browser renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import FORMAT_JS, run_node

JOB_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "job-table.js"
)


def _node_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(JOB_TABLE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function JobTable', 'function JobTable');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        function useState(initial) {{
          return [typeof initial === 'function' ? initial() : initial, () => {{}}];
        }}
        function useMemo(fn) {{ return fn(); }}
        function useEffect() {{}}
        function useRef() {{ return {{ current: null }}; }}
        function phaseColor() {{ return '#89b4fa'; }}
        const palette = {{ surface0: '#313244', blue: '#89b4fa' }};
{FORMAT_JS}
        const navigations = [];
        function navigate(path) {{ navigations.push(path); }}
        function NsPill(props) {{ return {{ component: 'NsPill', props }}; }}
        function RelativeTime(props) {{ return {{ component: 'RelativeTime', props }}; }}

        eval(source + '\\nglobalThis.JobTable = JobTable;');

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

        function flattenText(node) {{
          if (node == null || node === false) return '';
          if (Array.isArray(node)) return node.map(flattenText).join('');
          if (typeof node === 'object' && node.__html) {{
            let text = '';
            for (let i = 0; i < node.strings.length; i++) {{
              text += node.strings[i];
              if (i < node.values.length) text += flattenText(node.values[i]);
            }}
            return text;
          }}
          if (typeof node === 'function') return `[Function ${{node.name}}]`;
          if (typeof node === 'object') return JSON.stringify(node);
          return String(node);
        }}

        function rowIds(rendered) {{
          return collectValues(rendered)
            .filter((value) => typeof value === 'string'
              && value.startsWith('job-row-')
              && !value.startsWith('job-row-ns-'));
        }}

        {expression}
    """


def test_archived_rows_survive_case_insensitive_phase_filter() -> None:
    jobs = [
        {"namespace": "bench", "name": "archived", "phase": "Archived"},
        {"namespace": "bench", "name": "running", "phase": "Running"},
    ]
    script = _node_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          filter: ['archived'],
          sort: {{ key: 'name', dir: 1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify({{ rows: rowIds(rendered), text: flattenText(rendered) }}));
        """
    )
    out = json.loads(run_node(script))
    assert out["rows"] == ["job-row-bench-archived"]
    assert "Archived" in out["text"]


def test_sweep_child_rows_show_parent_variation_and_trial_metadata() -> None:
    jobs = [
        {
            "namespace": "sweep ns",
            "name": "child-0",
            "phase": "Completed",
            "sweepName": "sweep/one",
            "variationLabel": "batch=16",
            "variationIndex": 3,
            "trialIndex": 2,
        }
    ]
    script = _node_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'name', dir: 1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify(flattenText(rendered)));
        """
    )
    text = json.loads(run_node(script))
    assert "sweep/one" in text
    assert "batch=16" in text
    assert "trial 2" in text


def test_missing_workers_progress_and_metrics_render_as_placeholders() -> None:
    jobs = [
        {
            "namespace": "bench",
            "name": "sparse",
            "phase": "Completed",
            "workersReady": None,
            "workersTotal": None,
            "progressPercent": None,
            "throughputRps": None,
            "latencyP99Ms": None,
        }
    ]
    script = _node_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'name', dir: 1 }},
          onSortChange: () => {{}},
        }});
        const text = flattenText(rendered);
        console.log(JSON.stringify({{ text, placeholderCount: (text.match(/---/g) ?? []).length }}));
        """
    )
    out = json.loads(run_node(script))
    assert out["placeholderCount"] == 4
    assert "NaN" not in out["text"]
    assert "undefined" not in out["text"]


def test_sorting_by_age_throughput_and_latency_uses_expected_direction_and_missing_sink() -> (
    None
):
    jobs = [
        {
            "namespace": "bench",
            "name": "old-fast-slow",
            "phase": "Completed",
            "created": "2026-01-01T00:00:00Z",
            "throughputRps": 200,
            "latencyP99Ms": 900,
        },
        {
            "namespace": "bench",
            "name": "new-slow-fast",
            "phase": "Completed",
            "created": "2026-01-03T00:00:00Z",
            "throughputRps": 100,
            "latencyP99Ms": 100,
        },
        {
            "namespace": "bench",
            "name": "missing-metrics",
            "phase": "Archived",
            "created": "2026-01-02T00:00:00Z",
            "throughputRps": None,
            "latencyP99Ms": None,
        },
    ]
    script = _node_script(
        f"""
        const jobs = {json.dumps(jobs)};
        const cases = {{
          ageDesc: {{ key: 'age', dir: -1 }},
          throughputDesc: {{ key: 'throughput', dir: -1 }},
          latencyAsc: {{ key: 'latency', dir: 1 }},
        }};
        const out = Object.fromEntries(Object.entries(cases).map(([name, sort]) => [
          name,
          rowIds(JobTable({{ jobs, sort, onSortChange: () => {{}} }})),
        ]));
        console.log(JSON.stringify(out));
        """
    )
    out = json.loads(run_node(script))
    assert out["ageDesc"] == [
        "job-row-bench-new-slow-fast",
        "job-row-bench-missing-metrics",
        "job-row-bench-old-fast-slow",
    ]
    assert out["throughputDesc"] == [
        "job-row-bench-old-fast-slow",
        "job-row-bench-new-slow-fast",
        "job-row-bench-missing-metrics",
    ]
    assert out["latencyAsc"] == [
        "job-row-bench-new-slow-fast",
        "job-row-bench-old-fast-slow",
        "job-row-bench-missing-metrics",
    ]


def test_job_table_keeps_row_navigation_encoded_by_delegating_to_caller() -> None:
    source = JOB_TABLE_PATH.read_text()
    assert "onRowClick && onRowClick(job)" in source
    assert "#/jobs/${" not in source
    assert "`/jobs/${" not in source
    assert (
        "#/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}"
        in source
    )
    assert (
        "navigate(`/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`)"
        in source
    )

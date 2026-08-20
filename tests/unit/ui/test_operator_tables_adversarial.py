# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial pure-data tests for operator UI job, sweep, and cells tables."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import FORMAT_JS, run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_TABLE_PATH = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components" / "job-table.js"
)
_SWEEPS_PAGE_PATH = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "sweeps.js"
)
_CELLS_TABLE_PATH = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components" / "cells-table.js"
)


def _job_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_JOB_TABLE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function JobTable', 'function JobTable');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        function useState(initial) {{ return [typeof initial === 'function' ? initial() : initial, () => {{}}]; }}
        function useMemo(fn) {{ return fn(); }}
        function useEffect() {{}}
        function useRef() {{ return {{ current: null }}; }}
        function phaseColor() {{ return '#89b4fa'; }}
        const palette = {{ surface0: '#313244', blue: '#89b4fa' }};
{FORMAT_JS}
        function navigate() {{}}
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

        function templateStrings(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{
            for (const item of node) templateStrings(item, out);
            return out;
          }}
          if (typeof node === 'object' && node.__html) {{
            out.push(...node.strings);
            for (const value of node.values) templateStrings(value, out);
          }}
          return out;
        }}

        function rowIds(rendered) {{
          return collectValues(rendered)
            .filter((value) => typeof value === 'string'
              && value.startsWith('job-row-')
              && !value.startsWith('job-row-ns-'));
        }}

        {expression}
    """


def _sweeps_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_SWEEPS_PAGE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function Sweeps', 'function Sweeps');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        function useMemo(fn) {{ return fn(); }}
        function useEffect() {{}}
        const stateSetters = [];
        let stateIndex = 0;
        function useState(initial) {{
          stateIndex += 1;
          let value = typeof initial === 'function' ? initial() : initial;
          if (globalThis.__forceSweepsLoaded && stateIndex === 2) value = false;
          stateSetters.push((next) => {{ value = next; }});
          return [value, stateSetters[stateSetters.length - 1]];
        }}

        const api = {{ listSweeps: async () => {{ throw new Error('not called'); }} }};
        function poll() {{}}
        const sweeps = {{ value: [] }};
        const freshness = {{ value: {{}} }};
        function FreshnessPill(props) {{ return {{ component: 'FreshnessPill', props }}; }}
        function StaleBanner(props) {{ return {{ component: 'StaleBanner', props }}; }}
        function dedupeByNsName(rows) {{ return rows; }}
        function navigate() {{}}
        const query = {{ value: {{}} }};
        function setQuery(update) {{
          query.value = Object.fromEntries(
            Object.entries({{ ...query.value, ...update }}).filter(([, value]) => value !== undefined)
          );
        }}
        const palette = {{
          blue: '#89b4fa', green: '#a6e3a1', mantle: '#181825', overlay0: '#6c7086',
          red: '#f38ba8', surface0: '#313244', surface1: '#45475a', teal: '#94e2d5',
          text: '#cdd6f4', yellow: '#f9e2af',
        }};
        function phaseColor() {{ return '#89b4fa'; }}
        function NsPill(props) {{ return {{ component: 'NsPill', props }}; }}
        function ModelPill(props) {{ return {{ component: 'ModelPill', props }}; }}
        function RelativeTime(props) {{ return {{ component: 'RelativeTime', props }}; }}
        function LoadingPanel(props) {{ return {{ component: 'LoadingPanel', props }}; }}

        eval(source + '\\nglobalThis.Sweeps = Sweeps;');

        function renderSweeps({{ rows = [], q = {{}}, forceLoaded = true }} = {{}}) {{
          sweeps.value = rows;
          query.value = q;
          stateSetters.length = 0;
          stateIndex = 0;
          globalThis.__forceSweepsLoaded = forceLoaded;
          return Sweeps();
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
            .filter((value) => typeof value === 'string' && value.startsWith('sweep-row-'))
            .filter((value) => !value.startsWith('sweep-row-ns-') && !value.startsWith('sweep-row-model-'));
        }}

        {expression}
    """


def _cells_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_CELLS_TABLE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function CellsTable', 'function CellsTable');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        const palette = {{ red: '#f38ba8' }};

        eval(source + '\\nglobalThis.CellsTable = CellsTable;');

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
            .filter((value) => typeof value === 'string' && value.startsWith('sweep-cell-row-'));
        }}

        {expression}
    """


def test_job_table_duplicate_names_across_namespaces_stay_distinct_and_stable() -> None:
    jobs = [
        {"namespace": "team-a", "name": "same", "phase": "Running", "throughputRps": 7},
        {"namespace": "team-b", "name": "same", "phase": "Running", "throughputRps": 7},
        {
            "namespace": "team-c",
            "name": "later",
            "phase": "Running",
            "throughputRps": 7,
        },
    ]
    script = _job_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'throughput', dir: -1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify(rowIds(rendered)));
        """
    )

    assert json.loads(run_node(script)) == [
        "job-row-team-a-same",
        "job-row-team-b-same",
        "job-row-team-c-later",
    ]


def test_job_table_mixed_numeric_strings_sort_numerically_with_missing_values_last() -> (
    None
):
    jobs = [
        {
            "namespace": "bench",
            "name": "two",
            "phase": "Completed",
            "throughputRps": "2",
        },
        {
            "namespace": "bench",
            "name": "missing",
            "phase": "Completed",
            "throughputRps": None,
        },
        {
            "namespace": "bench",
            "name": "ten",
            "phase": "Completed",
            "throughputRps": "10",
        },
    ]
    script = _job_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'throughput', dir: -1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify(rowIds(rendered)));
        """
    )

    assert json.loads(run_node(script)) == [
        "job-row-bench-ten",
        "job-row-bench-two",
        "job-row-bench-missing",
    ]


def test_job_table_html_like_and_extremely_long_labels_remain_interpolated_values() -> (
    None
):
    long_label = "dim=" + ("x" * 512)
    html_like_name = "<img src=x onerror=alert(1)>"
    jobs = [
        {
            "namespace": "bench",
            "name": html_like_name,
            "phase": "Completed",
            "sweepName": "<script>alert(1)</script>",
            "variationLabel": long_label,
        }
    ]
    script = _job_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'name', dir: 1 }},
          onSortChange: () => {{}},
        }});
        const text = flattenText(rendered);
        const templates = templateStrings(rendered).join('');
        console.log(JSON.stringify({{ text, templatesContainInputs: templates.includes({json.dumps(html_like_name)}) || templates.includes({json.dumps(long_label)}) }}));
        """
    )

    out = json.loads(run_node(script))
    assert html_like_name in out["text"]
    assert long_label in out["text"]
    assert out["templatesContainInputs"] is False


def test_sweeps_table_mixed_numeric_strings_sort_numerically() -> None:
    rows = [
        {"namespace": "bench", "name": "two", "phase": "Running", "failed_runs": "2"},
        {"namespace": "bench", "name": "ten", "phase": "Running", "failed_runs": "10"},
        {"namespace": "bench", "name": "zero", "phase": "Running", "failed_runs": 0},
    ]
    script = _sweeps_script(
        f"""
        const rendered = renderSweeps({{
          rows: {json.dumps(rows)},
          q: {{ sort: 'failed:desc' }},
        }});
        console.log(JSON.stringify(rowIds(rendered)));
        """
    )

    assert json.loads(run_node(script)) == [
        "sweep-row-bench-ten",
        "sweep-row-bench-two",
        "sweep-row-bench-zero",
    ]


def test_sweeps_table_accepts_mixed_snake_and_camel_case_rollup_rows() -> None:
    rows = [
        {
            "namespace": "bench",
            "name": "snake",
            "phase": "Running",
            "completed_runs": 2,
            "failed_runs": 1,
            "total_variations": 3,
        },
        {
            "namespace": "bench",
            "name": "camel",
            "phase": "Running",
            "completedRuns": 2,
            "failedRuns": 1,
            "totalVariations": 3,
        },
    ]
    script = _sweeps_script(
        f"""
        const rendered = renderSweeps({{
          rows: {json.dumps(rows)},
          q: {{ sort: 'name:asc' }},
        }});
        const text = flattenText(rendered);
        console.log(JSON.stringify({{ rows: rowIds(rendered), text }}));
        """
    )

    out = json.loads(run_node(script))
    assert out["rows"] == ["sweep-row-bench-camel", "sweep-row-bench-snake"]
    assert out["text"].count("2 / 3") == 2
    assert out["text"].count("1") >= 2


def test_sweeps_table_duplicate_names_long_labels_and_html_like_text_are_preserved() -> (
    None
):
    long_name = "sweep-" + ("n" * 512)
    html_like_model = "<b>llama</b>"
    rows = [
        {
            "namespace": "team-a",
            "name": "dupe",
            "phase": "Running",
            "model": html_like_model,
        },
        {
            "namespace": "team-b",
            "name": "dupe",
            "phase": "Running",
            "model": html_like_model,
        },
        {
            "namespace": "team-c",
            "name": long_name,
            "phase": "Succeeded",
            "model": "mixtral",
        },
    ]
    script = _sweeps_script(
        f"""
        const rendered = renderSweeps({{
          rows: {json.dumps(rows)},
          q: {{ sort: 'name:asc' }},
        }});
        console.log(JSON.stringify({{ rows: rowIds(rendered), text: flattenText(rendered) }}));
        """
    )

    out = json.loads(run_node(script))
    assert out["rows"] == [
        "sweep-row-team-a-dupe",
        "sweep-row-team-b-dupe",
        "sweep-row-team-c-" + long_name,
    ]
    assert html_like_model in out["text"]
    assert long_name in out["text"]


def test_cells_table_mixed_snake_camel_rows_keep_input_order_and_extreme_labels() -> (
    None
):
    long_label = "lr=" + ("0" * 512)
    cells = [
        {
            "variation_index": 7,
            "variation_label": "<b>snake</b>",
            "trials_completed": 1,
            "trials_failed": 0,
            "values": {"lr": "0.1"},
            "metrics": {"throughput": {"mean": 12.3456}},
        },
        {
            "variationIndex": 8,
            "variationLabel": long_label,
            "trialsCompleted": 2,
            "trialsFailed": 1,
            "values": {"lr": None},
            "metrics": {"throughput": {"mean": 123.456}},
        },
    ]
    script = _cells_script(
        f"""
        const rendered = CellsTable({{
          dimensions: [{{ name: 'lr' }}],
          cells: {json.dumps(cells)},
          metric: 'throughput',
          stat: 'mean',
        }});
        console.log(JSON.stringify({{ rows: rowIds(rendered), text: flattenText(rendered) }}));
        """
    )

    out = json.loads(run_node(script))
    assert out["rows"] == ["sweep-cell-row-7", "sweep-cell-row-8"]
    assert "<b>snake</b>" in out["text"]
    assert long_label in out["text"]
    # 12.3456 is the mean of one trial's throughput; the old assertion pinned
    # "12.346", i.e. five significant figures on an average the underlying
    # measurement never resolved that finely. Four significant figures matches
    # the band already used by the pareto tooltip and trial board.
    assert "12.35" in out["text"]
    assert "12.346" not in out["text"]
    assert "123.5" in out["text"]


def test_cells_table_numeric_string_metrics_format_without_throwing() -> None:
    cells = [
        {
            "variation_index": 1,
            "variation_label": "string-metric",
            "trials_completed": "10",
            "trials_failed": "2",
            "values": {"batch": "16"},
            "metrics": {"throughput": {"mean": "12.3456"}},
        }
    ]
    script = _cells_script(
        f"""
        const rendered = CellsTable({{
          dimensions: [{{ name: 'batch' }}],
          cells: {json.dumps(cells)},
          metric: 'throughput',
          stat: 'mean',
        }});
        console.log(JSON.stringify({{ rows: rowIds(rendered), text: flattenText(rendered) }}));
        """
    )

    out = json.loads(run_node(script))
    assert out["rows"] == ["sweep-cell-row-1"]
    # Numeric strings take the same four-significant-figure path as numbers;
    # the old assertion pinned the over-precise "12.346".
    assert "12.35" in out["text"]

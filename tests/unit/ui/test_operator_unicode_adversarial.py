# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unicode, RTL, and control-character adversarial tests for operator UI helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import FORMAT_JS, run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_ROUTER_PATH = _UI_ROOT / "lib" / "router.js"
_ROUTER_HELPERS_PATH = _UI_ROOT / "lib" / "router-helpers.js"
_JOB_TABLE_PATH = _UI_ROOT / "components" / "job-table.js"
_PILLS_PATH = _UI_ROOT / "components" / "pills.js"
_RELAUNCH_PATH = _UI_ROOT / "components" / "relaunch-button.js"
_API_PATH = _UI_ROOT / "lib" / "api.js"
_EVENTS_TAB_PATH = _UI_ROOT / "components" / "diagnostics-events-tab.js"
_LOGS_TAB_PATH = _UI_ROOT / "components" / "diagnostics-logs-tab.js"


ADV_NS = "研究‮-ns\x1f"
ADV_JOB = "job-שלום-‮abc\x00"
ADV_MODEL = "mistral/東京‮-7b\x1b"
ADV_MESSAGE = "warning שלום ‮abc\x00\nnext line"


def _router_import_script() -> str:
    return f"""
        import {{ readFileSync }} from 'node:fs';
        let source = readFileSync({str(_ROUTER_PATH)!r}, 'utf8');
        source = source.replace(
          "import {{ signal }} from '@preact/signals';",
          "const signal = (value) => ({{ value }});",
        );
        source = source.replace(
          "import {{ normalizePath, replaceHash }} from './router-helpers.js';",
          "import {{ normalizePath, replaceHash }} from {_ROUTER_HELPERS_PATH.as_uri()!r};",
        );
        const routerModuleUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const router = await import(routerModuleUrl);
    """


def _job_table_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_JOB_TABLE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function JobTable', 'function JobTable');

        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
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
          if (Array.isArray(node)) {{ for (const item of node) collectValues(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{ for (const value of node.values) collectValues(value, out); return out; }}
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


def _pills_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_PILLS_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace(/export function /g, 'function ');

        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
        const palette = {{ teal: '#94e2d5', indigo: '#89b4fa' }};
        function modelColor() {{ return '#cba6f7'; }}
        eval(source + '\\nglobalThis.NsPill = NsPill; globalThis.ModelPill = ModelPill;');

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

        function collectValues(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{ for (const item of node) collectValues(item, out); return out; }}
          if (typeof node === 'object' && node.__html) {{ for (const value of node.values) collectValues(value, out); return out; }}
          out.push(node);
          return out;
        }}

        {expression}
    """


def _relaunch_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_RELAUNCH_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace(/export function /g, 'function ');
        function html() {{}}
        const palette = {{ green: '#a6e3a1' }};
        function navigate() {{}}
        eval(source + '\\nglobalThis.serializeYaml = serializeYaml;');
        {expression}
    """


def _api_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(_API_PATH)!r}, 'utf8');
        source = source.replace(/import \{{[\s\S]*?\}} from '\.\/state\.js';/, 'function setError() {{}}');
        source = source.replace(/^export /gm, '');
        eval(source + '\\nglobalThis.api = api;');
        {expression}
    """


def _events_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_EVENTS_TAB_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function EventsTab', 'function EventsTab');
        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
        function useEffect() {{}}
        function useRef() {{ return {{ current: null }}; }}
        let useStateCalls = 0;
        function useState(initial) {{
          useStateCalls += 1;
          if (useStateCalls === 1) return [globalThis.__eventsState, () => {{}}];
          if (useStateCalls === 2) return ['all', () => {{}}];
          if (useStateCalls === 3) return [1700000000000, () => {{}}];
          return [typeof initial === 'function' ? initial() : initial, () => {{}}];
        }}
        const api = {{}};
        function poll() {{}}
        eval(source + '\\nglobalThis.EventsTab = EventsTab;');

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
        {expression}
    """


def _logs_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_LOGS_TAB_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function LogsTab', 'function LogsTab');
        function html(strings, ...values) {{ return {{ __html: true, strings: Array.from(strings), values }}; }}
        function useEffect() {{}}
        function useRef() {{ return {{ current: null }}; }}
        let useStateCalls = 0;
        function useState(initial) {{
          useStateCalls += 1;
          if (useStateCalls === 1) return [globalThis.__selectedPod, () => {{}}];
          if (useStateCalls === 2) return [globalThis.__selectedContainer, () => {{}}];
          if (useStateCalls === 3) return [200, () => {{}}];
          if (useStateCalls === 4) return [false, () => {{}}];
          if (useStateCalls === 5) return [globalThis.__tail, () => {{}}];
          if (useStateCalls === 6) return [globalThis.__err, () => {{}}];
          if (useStateCalls === 7) return [true, () => {{}}];
          return [typeof initial === 'function' ? initial() : initial, () => {{}}];
        }}
        const api = {{}};
        eval(source + '\\nglobalThis.LogsTab = LogsTab;');

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
        {expression}
    """


def test_router_builds_routes_and_queries_with_unicode_rtl_and_controls_encoded() -> (
    None
):
    script = f"""
        global.window = {{ location: {{ hash: '#/' }}, addEventListener() {{}} }};
        {_router_import_script()}
        const {{ buildRoute, hashUrl, matchRoute }} = router;
        const path = buildRoute('/jobs/:namespace/:name', {{
          namespace: {json.dumps(ADV_NS)},
          name: {json.dumps(ADV_JOB)},
        }});
        const href = hashUrl('/jobs', {{ ns: {json.dumps(ADV_NS)}, model: {json.dumps(ADV_MODEL)}, empty: '' }});
        console.log(JSON.stringify({{ path, href, params: matchRoute('/jobs/:namespace/:name', path) }}));
    """

    out = json.loads(run_node(script))
    assert out["params"] == {"namespace": ADV_NS, "name": ADV_JOB}
    assert (
        out["path"]
        == "/jobs/%E7%A0%94%E7%A9%B6%E2%80%AE-ns%1F/job-%D7%A9%D7%9C%D7%95%D7%9D-%E2%80%AEabc%00"
    )
    assert (
        out["href"]
        == "#/jobs?ns=%E7%A0%94%E7%A9%B6%E2%80%AE-ns%1F&model=mistral%2F%E6%9D%B1%E4%BA%AC%E2%80%AE-7b%1B"
    )


def test_model_namespace_and_job_labels_remain_values_not_template_html() -> None:
    script = _pills_script(
        f"""
        const ns = NsPill({{ ns: {json.dumps(ADV_NS)}, onClick: () => {{}}, testId: 'ns-adv' }});
        const model = ModelPill({{ model: {json.dumps(ADV_MODEL)}, onClick: () => {{}}, testId: 'model-adv' }});
        console.log(JSON.stringify({{
          nsText: flattenText(ns),
          modelText: flattenText(model),
          nsTitle: collectValues(ns).find(v => typeof v === 'string' && v.startsWith('Filter by namespace')),
          modelTitle: collectValues(model).find(v => typeof v === 'string' && v.startsWith('Filter by model')),
        }}));
        """
    )

    out = json.loads(run_node(script))
    assert ADV_NS in out["nsText"]
    assert ADV_MODEL in out["modelText"]
    assert out["nsTitle"] == "Filter by namespace: " + ADV_NS
    assert out["modelTitle"] == "Filter by model: " + ADV_MODEL


def test_job_table_sorts_unicode_labels_deterministically_and_preserves_text() -> None:
    jobs = [
        {
            "namespace": ADV_NS,
            "name": "ב-job",
            "phase": "Completed",
            "throughputRps": "2",
        },
        {
            "namespace": ADV_NS,
            "name": ADV_JOB,
            "phase": "Completed",
            "throughputRps": "10",
        },
        {
            "namespace": ADV_NS,
            "name": "東京-job",
            "phase": "Completed",
            "throughputRps": None,
        },
    ]
    script = _job_table_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'throughput', dir: -1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify({{ rows: rowIds(rendered), text: flattenText(rendered) }}));
        """
    )

    out = json.loads(run_node(script))
    assert out["rows"] == [
        "job-row-" + ADV_NS + "-" + ADV_JOB,
        "job-row-" + ADV_NS + "-ב-job",
        "job-row-" + ADV_NS + "-東京-job",
    ]
    assert ADV_JOB in out["text"]
    assert "東京-job" in out["text"]


def test_yaml_serializer_quotes_unicode_rtl_control_and_multiline_strings() -> None:
    script = _relaunch_script(
        f"""
        const yaml = serializeYaml({{
          metadata: {{ name: {json.dumps(ADV_JOB)}, namespace: {json.dumps(ADV_NS)} }},
          spec: {{ model: {json.dumps(ADV_MODEL)}, prompt: {json.dumps(ADV_MESSAGE)} }},
        }});
        console.log(JSON.stringify(yaml));
        """
    )

    yaml = json.loads(run_node(script))
    assert "name: 'job-שלום-‮abc\x00'" in yaml
    assert "namespace: '研究‮-ns\x1f'" in yaml
    assert "model: 'mistral/東京‮-7b\x1b'" in yaml
    assert "prompt: |\n      warning שלום ‮abc\x00\n      next line" in yaml


def test_artifact_file_urls_encode_unicode_rtl_control_filename_segments() -> None:
    filename = "plots/latency שלום‮\x1f.csv"
    script = _api_script(
        f"""
        console.log(JSON.stringify({{
          jobBundle: api.resultBundleUrl({json.dumps(ADV_NS)}, {json.dumps(ADV_JOB)}, '1700000000'),
          sweepFile: api.sweepArtifactFileUrl({json.dumps(ADV_NS)}, {json.dumps(ADV_JOB)}, '1700000000', {json.dumps(filename)}),
        }}));
        """
    )

    assert json.loads(run_node(script)) == {
        "jobBundle": "/api/v1/results/%E7%A0%94%E7%A9%B6%E2%80%AE-ns%1F/job-%D7%A9%D7%9C%D7%95%D7%9D-%E2%80%AEabc%00/runs/1700000000.zip",
        "sweepFile": "/api/v1/sweeps/%E7%A0%94%E7%A9%B6%E2%80%AE-ns%1F/job-%D7%A9%D7%9C%D7%95%D7%9D-%E2%80%AEabc%00/epochs/1700000000/artifacts/plots/latency%20%D7%A9%D7%9C%D7%95%D7%9D%E2%80%AE%1F.csv",
    }


def test_event_messages_with_unicode_rtl_and_controls_render_as_text_values() -> None:
    event = {
        "type": "Warning",
        "reason": "FailedScheduling",
        "message": ADV_MESSAGE,
        "last_timestamp": "2026-05-18T12:34:56Z",
        "involved_object": {"kind": "Pod", "name": ADV_JOB},
        "count": 2,
    }
    script = _events_script(
        f"""
        globalThis.__eventsState = {{ kind: 'ok', events: [{json.dumps(event)}] }};
        const rendered = EventsTab({{ ns: {json.dumps(ADV_NS)}, name: {json.dumps(ADV_JOB)}, active: true }});
        console.log(JSON.stringify(flattenText(rendered)));
        """
    )

    text = json.loads(run_node(script))
    assert ADV_MESSAGE in text
    assert "Pod/" + ADV_JOB in text
    assert "×2" in text


def test_log_messages_with_unicode_rtl_controls_and_newlines_remain_single_text_node() -> (
    None
):
    line_one = "controller שלום‮\x00 started"
    line_two = "worker 東京\x1b[31m warning"
    script = _logs_script(
        f"""
        globalThis.__selectedPod = {json.dumps(ADV_JOB)};
        globalThis.__selectedContainer = 'control-plane';
        globalThis.__tail = [{json.dumps(line_one)}, {json.dumps(line_two)}];
        globalThis.__err = {json.dumps(ADV_MESSAGE)};
        const rendered = LogsTab({{
          ns: {json.dumps(ADV_NS)},
          name: {json.dumps(ADV_JOB)},
          pods: [{{ name: {json.dumps(ADV_JOB)}, phase: 'Running', containers: ['control-plane'] }}],
          active: true,
        }});
        console.log(JSON.stringify(flattenText(rendered)));
        """
    )

    text = json.loads(run_node(script))
    assert line_one + "\n" + line_two in text
    assert ADV_MESSAGE in text
    assert "1 line" not in text
    assert "2 lines" in text

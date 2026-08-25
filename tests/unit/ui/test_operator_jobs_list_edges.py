# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for the operator jobs list page and job-detail links."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
JOBS_PAGE_PATH = UI_ROOT / "pages" / "jobs.js"
JOB_TABLE_PATH = UI_ROOT / "components" / "job-table.js"
ROUTER_PATH = UI_ROOT / "lib" / "router.js"
ROUTER_HELPERS_PATH = UI_ROOT / "lib" / "router-helpers.js"


def _job_table_script(expression: str) -> str:
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
        function fmtNumber(value, digits) {{ return Number(value).toFixed(digits); }}
        function fmtThroughput(value) {{ return String(value); }}
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

        function rowIds(rendered) {{
          return collectValues(rendered)
            .filter((value) => typeof value === 'string'
              && value.startsWith('job-row-')
              && !value.startsWith('job-row-ns-'));
        }}

        {expression}
    """


def _jobs_page_script(
    local_jobs: list[dict[str, object]], query_value: dict[str, str], expression: str
) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(JOBS_PAGE_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function Jobs', 'function Jobs');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        function useState(initial) {{
          return [typeof initial === 'function' ? initial() : initial, () => {{}}];
        }}
        function useEffect() {{}}
        function useMemo(fn) {{ return fn(); }}
        const api = {{ listJobs: async () => ({{ jobs: [] }}) }};
        function poll() {{}}
        const jobs = {{ value: {json.dumps(local_jobs)} }};
        const freshness = {{ value: {{ jobs: null }} }};
        function dedupeByNsName(items) {{ return items; }}
        function buildJobPath(job) {{ return '/detail/' + (job.name ?? job.job_id ?? ''); }}
        function navigate() {{}}
        const query = {{ value: {json.dumps(query_value)} }};
        function setQuery() {{}}
        const palette = {{
          mantle: '#181825', surface0: '#313244', surface1: '#45475a', text: '#cdd6f4',
          overlay0: '#6c7086', teal: '#94e2d5', mauve: '#cba6f7', peach: '#fab387'
        }};
        function JobTable() {{}}
        function FreshnessPill() {{}}
        function StaleBanner() {{}}
        function LoadingPanel() {{}}

        eval(source + '\\nglobalThis.Jobs = Jobs;');

        function findJobTableJobs(node) {{
          if (node == null || node === false) return null;
          if (Array.isArray(node)) {{
            for (const item of node) {{
              const found = findJobTableJobs(item);
              if (found) return found;
            }}
            return null;
          }}
          if (typeof node === 'object' && node.__html) {{
            for (let i = 0; i < node.values.length; i++) {{
              if (node.values[i] === JobTable) return node.values[i + 1];
            }}
            for (const value of node.values) {{
              const found = findJobTableJobs(value);
              if (found) return found;
            }}
          }}
          return null;
        }}

        const rendered = Jobs();
        const renderedJobs = findJobTableJobs(rendered) ?? [];
        {expression}
    """


def _router_import_script() -> str:
    return f"""
        import {{ readFileSync }} from 'node:fs';
        let source = readFileSync({str(ROUTER_PATH)!r}, 'utf8');
        source = source.replace(
          "import {{ signal }} from '@preact/signals';",
          "const signal = (value) => ({{ value }});",
        );
        source = source.replace(
          "import {{ normalizePath, replaceHash }} from './router-helpers.js';",
          "import {{ normalizePath, replaceHash }} from {ROUTER_HELPERS_PATH.as_uri()!r};",
        );
        const routerModuleUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const router = await import(routerModuleUrl);
    """


def test_job_table_sorts_current_and_archived_jobs_by_age_without_phase_bucketing() -> (
    None
):
    jobs = [
        {
            "namespace": "bench",
            "name": "new-archived",
            "phase": "Archived",
            "created": "2026-05-03T00:00:00Z",
        },
        {
            "namespace": "bench",
            "name": "old-running",
            "phase": "Running",
            "created": "2026-05-01T00:00:00Z",
        },
        {
            "namespace": "bench",
            "name": "mid-complete",
            "phase": "Completed",
            "created": "2026-05-02T00:00:00Z",
        },
    ]
    script = _job_table_script(
        f"""
        const rendered = JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: 'age', dir: -1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify(rowIds(rendered)));
        """
    )

    assert json.loads(run_node(script)) == [
        "job-row-bench-new-archived",
        "job-row-bench-mid-complete",
        "job-row-bench-old-running",
    ]


def test_jobs_page_phase_filters_match_expected_status_buckets_case_insensitively() -> (
    None
):
    local_jobs = [
        {"namespace": "ns", "name": "starting", "phase": "Initializing"},
        {"namespace": "ns", "name": "live", "phase": "RUNNING"},
        {"namespace": "ns", "name": "done", "phase": "Succeeded"},
        {"namespace": "ns", "name": "bad", "phase": "Error"},
        {"namespace": "ns", "name": "archived", "phase": "Archived"},
    ]
    script = _jobs_page_script(
        local_jobs,
        {},
        """
        const buckets = {};
        for (const phase of ['running', 'completed', 'failed']) {
          query.value = { phase };
          buckets[phase] = findJobTableJobs(Jobs()).map((job) => job.name);
        }
        query.value = {};
        buckets.all = findJobTableJobs(Jobs()).map((job) => job.name);
        console.log(JSON.stringify(buckets));
        """,
    )

    assert json.loads(run_node(script)) == {
        "running": ["starting", "live"],
        "completed": ["done"],
        "failed": ["bad"],
        "all": ["starting", "live", "done", "bad", "archived"],
    }


def test_jobs_page_search_matches_name_or_namespace_case_insensitively() -> None:
    local_jobs = [
        {"namespace": "Team-A", "name": "load-test", "phase": "Running"},
        {"namespace": "default", "name": "MODEL-SMOKE", "phase": "Completed"},
        {"namespace": "default", "name": "unrelated", "phase": "Failed"},
    ]
    script = _jobs_page_script(
        local_jobs,
        {"q": "team"},
        """
        const byNamespace = renderedJobs.map((job) => job.name);
        query.value = { q: 'model' };
        const byName = findJobTableJobs(Jobs()).map((job) => job.name);
        console.log(JSON.stringify({ byNamespace, byName }));
        """,
    )

    assert json.loads(run_node(script)) == {
        "byNamespace": ["load-test"],
        "byName": ["MODEL-SMOKE"],
    }


def test_jobs_page_namespace_and_model_filters_stack_exactly() -> None:
    local_jobs = [
        {
            "namespace": "prod",
            "name": "llama-prod",
            "phase": "Running",
            "model": "llama",
        },
        {
            "namespace": "prod",
            "name": "mixtral-prod",
            "phase": "Running",
            "model": "mixtral",
        },
        {"namespace": "dev", "name": "llama-dev", "phase": "Running", "model": "llama"},
        {
            "namespace": "production",
            "name": "llama-production",
            "phase": "Running",
            "model": "llama",
        },
    ]
    script = _jobs_page_script(
        local_jobs,
        {"ns": "prod", "model": "llama"},
        """
        console.log(JSON.stringify(renderedJobs.map((job) => job.name)));
        """,
    )

    assert json.loads(run_node(script)) == ["llama-prod"]


def test_build_job_path_encodes_names_and_pins_run_epoch_variants() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        {_router_import_script()}
        const {{ buildJobPath }} = router;
        const paths = {{
          nameOnly: buildJobPath({{ namespace: 'team/a', name: 'bench/job 1' }}),
          jobId: buildJobPath({{ namespace: 'analytics', job_id: 'archive/job' }}),
          nameBeatsJobId: buildJobPath({{ namespace: 'ns', name: 'job-name', job_id: 'job-id' }}),
          runEpoch: buildJobPath({{ namespace: 'ns', name: 'job', runEpoch: '2026-05-18T03:04:05Z' }}),
          snakeRunEpoch: buildJobPath({{ namespace: 'ns', job_id: 'job/id', run_epoch: 'epoch/with slash' }}),
          childRunEpoch: buildJobPath({{ namespace: 'ns', name: 'child', runEpoch: 'parent', childRunEpoch: 42 }}),
        }};
        console.log(JSON.stringify(paths));
    """

    assert json.loads(run_node(script)) == {
        "nameOnly": "/jobs/team%2Fa/bench%2Fjob%201",
        "jobId": "/jobs/analytics/archive%2Fjob",
        "nameBeatsJobId": "/jobs/ns/job-name",
        "runEpoch": "/jobs/ns/job/runs/2026-05-18T03%3A04%3A05Z",
        "snakeRunEpoch": "/jobs/ns/job%2Fid/runs/epoch%2Fwith%20slash",
        "childRunEpoch": "/jobs/ns/child/runs/42",
    }


def test_jobs_page_row_click_delegates_to_build_job_path_for_links() -> None:
    source = JOBS_PAGE_PATH.read_text()
    assert "navigate(buildJobPath(job))" in source
    assert "function handleRowClick(job)" in source
    assert "onRowClick=${handleRowClick}" in source


def test_jobs_page_uses_shared_freshness_source_and_stale_banner() -> None:
    source = (UI_ROOT / "pages" / "jobs.js").read_text(encoding="utf-8")

    assert (
        "import { FreshnessPill, StaleBanner } from '../components/freshness.js';"
        in source
    )
    assert "freshness.value.jobs" in source
    assert "source: 'jobs'" in source
    assert '<${StaleBanner} source=${jobsFreshness} label="Jobs list" />' in source
    assert "<${FreshnessPill} source=${jobsFreshness}" in source

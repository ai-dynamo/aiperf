# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator leaderboard behavior."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
LEADERBOARD_PATH = UI_ROOT / "pages" / "leaderboard.js"
COMPARE_FILTERS_PATH = UI_ROOT / "pages" / "compare-filters.js"
ROUTER_PATH = UI_ROOT / "lib" / "router.js"
ROUTER_HELPERS_PATH = UI_ROOT / "lib" / "router-helpers.js"


def _leaderboard_script(
    state_values_js: str, body: str, *, run_effects: bool = False
) -> str:
    effect_impl = "fn();" if run_effects else ""
    effect_runner = ""
    return f"""
        import fs from 'node:fs';
        import {{ applyJobFilters, extractCrossFacets, FILTER_NONE }} from {COMPARE_FILTERS_PATH.as_uri()!r};

        const source = fs.readFileSync({str(LEADERBOARD_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function Leaderboard', 'function Leaderboard');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}
        const effects = [];
        const stateValues = {state_values_js};
        function useState(initial) {{
          const value = stateValues.length ? stateValues.shift() : (typeof initial === 'function' ? initial() : initial);
          return [value, () => {{}}];
        }}
        function useEffect(fn) {{ {effect_impl} }}
        function useMemo(fn) {{ return fn(); }}
        const apiCalls = [];
        const api = {{
          getLeaderboard(metric, stat, limit) {{
            apiCalls.push({{ metric, stat, limit }});
            return Promise.resolve({{ entries: [] }});
          }},
        }};
        {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
        const palette = {{
          mauve: '#cba6f7', blue: '#89b4fa', green: '#a6e3a1', peach: '#fab387',
          pink: '#f5c2e7', teal: '#94e2d5', sapphire: '#74c7ec', yellow: '#f9e2af',
          flamingo: '#f2cdcd', lavender: '#b4befe', subtext1: '#bac2de', overlay0: '#6c7086',
          overlay1: '#7f849c', surface0: '#313244', surface1: '#45475a', error: '#f38ba8',
        }};
        function buildJobPath(entry) {{
          return '/jobs/' + encodeURIComponent(entry.namespace ?? 'default') + '/' + encodeURIComponent(entry.name ?? entry.job_id ?? '');
        }}
        function navigate() {{}}
        function MetricSelector() {{}}
        function ChartWrapper() {{}}
        function LoadingPanel() {{}}
        function fmtNumber(value, digits) {{ return Number(value).toFixed(digits); }}

        eval(source + '\\nglobalThis.Leaderboard = Leaderboard;');

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
        function chartData(rendered) {{
          return collectValues(rendered).find((value) => value && value.datasets && value.labels);
        }}
        function allStrings(rendered) {{
          return collectValues(rendered).filter((value) => typeof value === 'string');
        }}

        {effect_runner}
        {body}
    """


def _leaderboard_state(
    entries: list[dict[str, object]],
    *,
    selected: dict[str, str] | None = None,
    ns_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
) -> str:
    selected_js = json.dumps(
        selected or {"metric": "request_throughput", "stat": "avg"}
    )
    ns_js = json.dumps(ns_filter or [])
    model_js = json.dumps(model_filter or [])
    entries_js = json.dumps(entries)
    return f"""
      [
        {selected_js},
        new Set({ns_js}),
        new Set({model_js}),
        new Set(),
        {{ entries: {entries_js} }},
        false,
        null,
      ]
    """


def test_metric_and_stat_selection_drive_api_request_with_full_limit() -> None:
    script = _leaderboard_script(
        "[{ metric: 'request_latency_p99', stat: 'p99' }, new Set(), new Set(), new Set(), null, false, null]",
        """
        Leaderboard();
        await Promise.resolve();
        console.log(JSON.stringify(apiCalls));
        """,
        run_effects=True,
    )

    assert json.loads(run_node(script)) == [
        {"metric": "request_latency_p99", "stat": "p99", "limit": 1000}
    ]


def test_leaderboard_ranks_client_side_because_api_order_is_always_desc() -> None:
    """The page must impose rank order itself.

    This test previously asserted the opposite -- that leaderboard.js leaves
    ordering to the API. It cannot: ``api.getLeaderboard`` sends only
    metric/stat/limit, so the endpoint keeps its ``order="desc"`` default and
    latency metrics come back worst-first. See
    tests/unit/ui/test_operator_ranking_presentation.py for the behavioural
    regressions.
    """
    source = LEADERBOARD_PATH.read_text()

    assert (
        "getLeaderboard(selected.metric, selected.stat, LEADERBOARD_FETCH_LIMIT)"
        in source
    )
    assert "function rankEntries(" in source
    assert re.search(r"rankEntries\(\s*entries\.filter", source)


def test_leaderboard_skips_missing_metric_values_before_ranking() -> None:
    entries = [
        {
            "job_id": "missing",
            "namespace": "ns",
            "model": "llama",
            "value": None,
            "unit": "tok/s",
        },
        {
            "job_id": "with-value",
            "namespace": "ns",
            "model": "llama",
            "value": 12.5,
            "unit": "tok/s",
        },
    ]
    script = _leaderboard_script(
        _leaderboard_state(entries),
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify(chartData(rendered).labels));
        """,
    )

    assert json.loads(run_node(script)) == ["with-value"]


def test_namespace_and_model_filters_stack_without_substring_matches() -> None:
    entries = [
        {"job_id": "prod-llama", "namespace": "prod", "model": "llama", "value": 30},
        {
            "job_id": "prod-mixtral",
            "namespace": "prod",
            "model": "mixtral",
            "value": 25,
        },
        {
            "job_id": "production-llama",
            "namespace": "production",
            "model": "llama",
            "value": 20,
        },
        {"job_id": "dev-llama", "namespace": "dev", "model": "llama", "value": 15},
    ]
    script = _leaderboard_script(
        _leaderboard_state(entries, ns_filter=["prod"], model_filter=["llama"]),
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify(chartData(rendered).labels));
        """,
    )

    assert json.loads(run_node(script)) == ["prod-llama"]


def test_chart_limit_is_top_ten_but_table_keeps_filterable_fetch_limit() -> None:
    entries = [
        {
            "job_id": f"job-{idx:02d}",
            "namespace": "ns",
            "model": "llama",
            "value": 100 - idx,
        }
        for idx in range(12)
    ]
    script = _leaderboard_script(
        _leaderboard_state(entries),
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify({
          labels: chartData(rendered).labels,
          strings: allStrings(rendered),
        }));
        """,
    )

    out = json.loads(run_node(script))

    assert out["labels"] == [f"job-{idx:02d}" for idx in range(10)]
    assert "12 runs" in out["strings"]


def test_related_runs_are_not_aggregated_by_namespace_or_model() -> None:
    entries = [
        {"job_id": "run-a", "namespace": "same", "model": "llama", "value": 10},
        {"job_id": "run-b", "namespace": "same", "model": "llama", "value": 20},
        {"job_id": "run-c", "namespace": "same", "model": "llama", "value": 30},
    ]
    script = _leaderboard_script(
        _leaderboard_state(entries),
        """
        const rendered = Leaderboard();
        console.log(JSON.stringify(chartData(rendered).labels));
        """,
    )

    # One bar per run, never a merged "same/llama" bar. Order is best-first
    # for the selected higher-is-better metric.
    assert json.loads(run_node(script)) == ["run-c", "run-b", "run-a"]


def test_job_detail_link_builder_encodes_namespace_name_and_epoch() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
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
        const router = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
        const paths = {{
          plain: router.buildJobPath({{ namespace: 'team/a', job_id: 'bench/job 1' }}),
          named: router.buildJobPath({{ namespace: 'ns', name: 'job name', job_id: 'ignored' }}),
          epoch: router.buildJobPath({{ namespace: 'ns', job_id: 'job/id', run_epoch: 'epoch/1' }}),
        }};
        console.log(JSON.stringify(paths));
    """

    assert json.loads(run_node(script)) == {
        "plain": "/jobs/team%2Fa/bench%2Fjob%201",
        "named": "/jobs/ns/job%20name",
        "epoch": "/jobs/ns/job%2Fid/runs/epoch%2F1",
    }

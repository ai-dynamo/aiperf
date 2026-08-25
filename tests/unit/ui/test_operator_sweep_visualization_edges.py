# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep visualization edge tests for variation table/chart/Pareto behavior."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS_DIR = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_SWEEP_DETAIL_HELPERS_PATH = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)
_SWEEP_DETAIL_PAGE_PATH = _SWEEP_DETAIL_HELPERS_PATH.with_name("sweep-detail.js")


def _run_component_script(
    component_name: str, export_names: list[str], body: str
) -> dict[str, object]:
    component_path = _COMPONENTS_DIR / component_name
    chart_wrapper_stub = (
        ""
        if component_name == "chart-wrapper.js"
        else """
          function ChartWrapper(props) { return { component: 'ChartWrapper', props }; }
          globalThis.__ChartWrapper = ChartWrapper;
        """
    )
    script = f"""
        import fs from 'node:fs';

        const path = {str(component_path)!r};
        let source = fs.readFileSync(path, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        const prelude = `
          {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
          const palette = {{
            blue: '#89b4fa', mantle: '#181825', text: '#cdd6f4',
            surface0: '#313244', overlay0: '#6c7086', overlay1: '#7f849c'
          }};
          const fmtNumber = (value, decimals = 1, fallback = '---') =>
            value == null || typeof value !== 'number' || !isFinite(value)
              ? fallback
              : value.toFixed(decimals);
          const fmtInt = (value, fallback = '---') =>
            value == null || typeof value !== 'number' || !isFinite(value)
              ? fallback
              : String(Math.round(value));
          const columnDecimals = (values, decimals = 1) => {{
            let widest = decimals;
            for (const v of values ?? []) {{
              if (typeof v !== 'number' || !isFinite(v)) continue;
              const abs = Math.abs(v);
              const needed = abs === 0 ? decimals
                : abs < 0.01 ? Math.max(decimals, 5)
                : abs < 1 ? Math.max(decimals, 4)
                : decimals;
              if (needed > widest) widest = needed;
            }}
            return widest;
          }};
          const fmtFixed = (value, decimals = 1, fallback = '---') =>
            value == null || typeof value !== 'number' || !isFinite(value)
              ? fallback
              : value.toFixed(Math.max(0, decimals));
          const useMemo = (fn) => fn();
          const html = (strings, ...values) => ({{ strings: Array.from(strings), values }});
          {chart_wrapper_stub}
        `;
        source = `${{prelude}}\n${{source}}\nexport {{ {", ".join(export_names)} }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {body}
    """
    return json.loads(run_node(script))


def test_build_sweep_variations_filters_missing_and_non_finite_stats() -> None:
    script = f"""
        import {{ buildSweepVariations }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = buildSweepVariations({{
          headlineMetrics: [
            {{key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'}},
            {{key: 'request_latency', stat: 'p99', label: 'P99', unit: 'ms'}},
          ],
          manifest: [
            {{name: 'sweep-v00-t0', variationIndex: 0, variationLabel: 'concurrency=8'}},
            {{name: 'sweep-v00-t1', variationIndex: 0, variationLabel: 'concurrency=8'}},
            {{name: 'sweep-v00-t2', variationIndex: 0, variationLabel: 'concurrency=8'}},
          ],
          childSummaries: {{
            'sweep-v00-t0': {{summary: {{request_throughput: {{avg: 10}}, request_latency: {{p99: Infinity}}}}}},
            'sweep-v00-t1': {{summary: {{request_throughput: {{avg: NaN}}, request_latency: {{p99: 50}}}}}},
            'sweep-v00-t2': {{summary: {{request_throughput: {{}}, request_latency: {{p99: 70}}}}}},
          }},
          cells: {{cells: []}},
        }});
        console.log(JSON.stringify(variations[0]));
    """

    variation = json.loads(run_node(script))

    assert variation["n_trials"] == 3
    # Three trials ran, but NaN and a missing key left one usable throughput
    # observation, so spread is unmeasured: `std` is null like `cv`, not 0.
    # Reporting 0 here would be the worst version of the defect -- two trials
    # were discarded and the surviving number would claim they agreed.
    assert variation["perMetric"]["request_throughput.avg"] == {
        "mean": 10,
        "std": None,
        "cv": None,
        "n": 1,
    }
    assert variation["perMetric"]["request_latency.p99"] == {
        "mean": 60,
        "std": 10,
        "cv": 1 / 6,
        "n": 2,
    }


def test_build_sweep_variations_keeps_duplicate_labels_as_distinct_indices() -> None:
    script = f"""
        import {{ buildSweepVariations }} from {_SWEEP_DETAIL_HELPERS_PATH.as_uri()!r};
        const variations = buildSweepVariations({{
          headlineMetrics: [{{key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'}}],
          manifest: [
            {{name: 'sweep-v00-t0', variationIndex: 0, variationLabel: 'shared-label'}},
            {{name: 'sweep-v01-t0', variationIndex: 1, variationLabel: 'shared-label'}},
          ],
          childSummaries: {{
            'sweep-v00-t0': {{summary: {{request_throughput: {{avg: 10}}}}}},
            'sweep-v01-t0': {{summary: {{request_throughput: {{avg: 20}}}}}},
          }},
          cells: {{cells: []}},
        }});
        console.log(JSON.stringify(variations.map(v => ({{
          index: v.variation_index,
          label: v.label,
          mean: v.perMetric['request_throughput.avg'].mean,
          trials: `${{v.n_trials}}/${{v.n_total}}`,
        }}))));
    """

    assert json.loads(run_node(script)) == [
        {"index": 0, "label": "shared-label", "mean": 10, "trials": "1/1"},
        {"index": 1, "label": "shared-label", "mean": 20, "trials": "1/1"},
    ]


def test_variations_table_renders_rows_by_index_and_missing_cell_dashes() -> None:
    result = _run_component_script(
        "variations-table.js",
        ["VariationsTable"],
        """
        function flatten(node, out = []) {
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {
            for (const child of node) flatten(child, out);
          } else if (typeof node === 'object') {
            if (node.strings) out.push(...node.strings);
            if (node.values) flatten(node.values, out);
          } else {
            out.push(String(node));
          }
          return out;
        }
        const rendered = helpers.VariationsTable({
          headlineMetrics: [
            {key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'},
            {key: 'request_latency', stat: 'p99', label: 'P99', unit: 'ms'},
            {key: 'time_to_first_token', stat: 'p50', label: 'TTFT', unit: 'ms'},
          ],
          variations: [
            {variation_index: 7, label: 'benchmark.concurrency=8', n_trials: 2, n_total: 3, perMetric: {
              'request_throughput.avg': {mean: 123.4, cv: 0.125},
              'request_latency.p99': {mean: 45.678, cv: null},
            }},
            {variation_index: 8, label: 'benchmark.concurrency=16', n_trials: 1, n_total: 3, perMetric: {
              'request_throughput.avg': {mean: 234.5, cv: 0},
            }},
          ],
        });
        const flat = flatten(rendered);
        console.log(JSON.stringify({
          hasIndexRows: flat.includes('variation-row-7') && flat.includes('variation-row-8'),
          trialNumbers: flat.filter(v => ['1', '2', '3'].includes(v)),
          headers: flat.filter(v => ['Throughput', 'P99', 'TTFT'].includes(v)),
          dashCells: flat.filter(v => v === '—').length,
          cvValues: flat.filter(v => ['12.50%', '0.00%'].includes(v)),
        }));
        """,
    )

    assert result == {
        "hasIndexRows": True,
        "trialNumbers": ["2", "3", "1", "3"],
        "headers": ["Throughput", "P99"],
        "dashCells": 1,
        "cvValues": ["12.50%", "0.00%"],
    }


def test_variations_chart_preserves_duplicate_labels_and_index_aligned_series() -> None:
    result = _run_component_script(
        "variations-chart.js",
        ["VariationsChart"],
        """
        const rendered = helpers.VariationsChart({
          metricLabel: 'Output tok/s',
          unit: 'tok/s',
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=8', mean: 100, std: 5, cv: 0.05, n: 2},
            {variation_index: 1, label: 'other.concurrency=8', mean: null, std: 20, cv: null, n: 0},
            {variation_index: 2, label: '', mean: 200, std: null, cv: 0, n: 1},
          ],
        });
        const data = rendered.values.find(v => v && Array.isArray(v.datasets));
        console.log(JSON.stringify({
          labels: data.labels,
          plus: data.datasets[0].data,
          minus: data.datasets[1].data,
          mean: data.datasets[2].data,
        }));
        """,
    )

    # Variation 2 is `{mean: 200, std: null, n: 1}`. The band used to coerce
    # that null to 0 and emit plus == minus == 200 -- a zero-width band drawn
    # through the point, which reads as a measurement of perfect
    # reproducibility from a variation that ran once and estimated no spread at
    # all. The old expectation encoded exactly that. The band is now absent
    # there while the mean is still plotted, matching the tooltip, which has
    # always said "1 trial, spread unknown" for this same point.
    assert result == {
        "labels": ["concurrency=8", "concurrency=8", "v2"],
        "plus": [105, None, None],
        "minus": [95, None, None],
        "mean": [100, None, 200],
    }


def test_pareto_same_xy_ties_render_all_points_but_only_one_frontier_step() -> None:
    result = _run_component_script(
        "variations-pareto.js",
        ["VariationsPareto"],
        """
        const xMetric = {key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'};
        const yMetric = {key: 'request_latency', stat: 'p99', label: 'P99 latency', unit: 'ms'};
        const rendered = helpers.VariationsPareto({
          xMetric,
          yMetric,
          yIsSmallerBetter: true,
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=8', perMetric: {
              'request_throughput.avg': {mean: 100}, 'request_latency.p99': {mean: 50},
            }},
            {variation_index: 1, label: 'benchmark.concurrency=8-copy', perMetric: {
              'request_throughput.avg': {mean: 100}, 'request_latency.p99': {mean: 50},
            }},
            {variation_index: 2, label: 'benchmark.concurrency=16', perMetric: {
              'request_throughput.avg': {mean: 120}, 'request_latency.p99': {mean: 45},
            }},
          ],
        });
        const datasets = rendered.values.find(v => v && Array.isArray(v.datasets))?.datasets ?? [];
        console.log(JSON.stringify({
          scatterNames: datasets[0]?.data.map(p => p.jobName),
          frontier: datasets[1]?.data,
        }));
        """,
    )

    assert result["scatterNames"] == [
        "concurrency=8",
        "concurrency=8-copy",
        "concurrency=16",
    ]
    assert result["frontier"] == [
        {"x": 100, "y": 50, "jobName": "concurrency=8"},
        {"x": 120, "y": 45, "jobName": "concurrency=16"},
    ]


def test_metric_selector_query_defaults_are_elided_from_url() -> None:
    source = _SWEEP_DETAIL_PAGE_PATH.read_text()

    assert "const DEFAULT_CHART_METRIC_KEY = 'output_token_throughput.avg';" in source
    assert "const DEFAULT_PARETO_AXIS_KEY = 'tps_p99';" in source
    assert "const urlMetric = query.value.metric ?? DEFAULT_CHART_METRIC_KEY;" in source
    assert "const urlAxis = query.value.axis ?? DEFAULT_PARETO_AXIS_KEY;" in source
    assert (
        "onchange=${e => setQuery({ metric: e.target.value === DEFAULT_CHART_METRIC_KEY ? undefined : e.target.value })}"
        in source
    )
    assert (
        "onclick=${() => setQuery({ axis: a.key === DEFAULT_PARETO_AXIS_KEY ? undefined : a.key })}"
        in source
    )
    assert "key: 'tps_ttft'" in source
    assert "y: { key: 'time_to_first_token',     stat: 'p50', label: 'TTFT'" in source

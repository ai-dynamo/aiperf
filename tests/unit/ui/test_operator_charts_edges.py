# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for operator UI chart data shaping."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import CHART_TYPOGRAPHY_JS_IN_TEMPLATE, run_node

COMPONENTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
)


def _run_component_script(
    component_name: str, export_names: list[str], body: str
) -> dict[str, object]:
    component_path = COMPONENTS_DIR / component_name
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


def test_chart_wrapper_fingerprint_handles_sparse_scatter_points() -> None:
    result = _run_component_script(
        "chart-wrapper.js",
        ["dataFingerprint"],
        """
        console.log(JSON.stringify({
          missing: helpers.dataFingerprint(null),
          empty: helpers.dataFingerprint({datasets: []}),
          sparse: helpers.dataFingerprint({datasets: [
            {label: 'scatter', data: [null, {x: 1, y: 2}, undefined, {x: 3, y: 4}]},
            {label: 'line', data: [5, null, 7]},
          ]}),
        }));
        """,
    )

    assert result == {
        "missing": "",
        "empty": "",
        "sparse": "scatter:;1,2;;3,4|line:5;;7",
    }


def test_latency_timeline_filters_missing_and_non_finite_values() -> None:
    result = _run_component_script(
        "latency-timeline-chart.js",
        ["recordLatencyMs", "strideSample"],
        """
        const records = [
          null,
          {error: 'request failed', metrics: {request_latency: {value: 1, unit: 'ms'}}},
          {metrics: {}},
          {metrics: {request_latency: {value: NaN, unit: 'ms'}}},
          {metrics: {request_latency: {value: Infinity, unit: 'ms'}}},
          {metrics: {request_latency: {value: 2500000, unit: 'ns'}}},
          {metrics: {request_latency: {value: 2500, unit: 'us'}}},
          {metrics: {request_latency: {value: 2.5, unit: 'ms'}}},
          {metrics: {request_latency: {value: 0.0025, unit: 's'}}},
          {metrics: {request_latency: {value: 9, unit: 'custom'}}},
        ];
        console.log(JSON.stringify({
          latencies: records.map(helpers.recordLatencyMs),
          sampled: helpers.strideSample([0, 1, 2, 3, 4, 5, 6], 3),
          unchangedIdentity: helpers.strideSample(records, 100) === records,
        }));
        """,
    )

    assert result == {
        "latencies": [None, None, None, None, None, 2.5, 2.5, 2.5, 2.5, 9],
        "sampled": [0, 3, 6],
        "unchangedIdentity": True,
    }


def test_variations_chart_treats_nan_and_infinity_as_missing_series_data() -> None:
    result = _run_component_script(
        "variations-chart.js",
        ["VariationsChart"],
        """
        const rendered = helpers.VariationsChart({
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=8', mean: NaN, std: 1, cv: 0.1, n: 1},
            {variation_index: 1, label: 'benchmark.concurrency=16', mean: Infinity, std: 1, cv: 0.2, n: 1},
            {variation_index: 2, label: 'benchmark.concurrency=32', mean: null, std: 1, cv: 0.3, n: 1},
          ],
          metricLabel: 'Request throughput',
          unit: 'req/s',
        });
        console.log(JSON.stringify({
          renderedChartWrapper: rendered.values.some(v => v === globalThis.__ChartWrapper),
          text: rendered.strings.join(''),
          chartData: rendered.values.find(v => v && v.datasets) ?? null,
        }));
        """,
    )

    assert result["renderedChartWrapper"] is False
    assert "No " in result["text"]
    assert result["chartData"] is None


def test_variations_chart_preserves_empty_series_empty_state() -> None:
    result = _run_component_script(
        "variations-chart.js",
        ["VariationsChart"],
        """
        const noVariations = helpers.VariationsChart({variations: [], metricLabel: 'Latency', unit: 'ms'});
        const allMissing = helpers.VariationsChart({
          variations: [{variation_index: 0, label: 'x=1', mean: null}],
          metricLabel: 'Latency',
          unit: 'ms',
        });
        console.log(JSON.stringify({
          noVariationsChart: noVariations.values.some(v => v === globalThis.__ChartWrapper),
          allMissingChart: allMissing.values.some(v => v === globalThis.__ChartWrapper),
        }));
        """,
    )

    assert result == {"noVariationsChart": False, "allMissingChart": False}


def test_variations_pareto_orders_frontier_and_deduplicates_equal_frontier_points() -> (
    None
):
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
              'request_throughput.avg': {mean: 100}, 'request_latency.p99': {mean: 80},
            }},
            {variation_index: 1, label: 'benchmark.concurrency=16', perMetric: {
              'request_throughput.avg': {mean: 150}, 'request_latency.p99': {mean: 90},
            }},
            {variation_index: 2, label: 'benchmark.concurrency=32', perMetric: {
              'request_throughput.avg': {mean: 200}, 'request_latency.p99': {mean: 70},
            }},
            {variation_index: 3, label: 'benchmark.concurrency=32-duplicate', perMetric: {
              'request_throughput.avg': {mean: 200}, 'request_latency.p99': {mean: 70},
            }},
            {variation_index: 4, label: 'benchmark.concurrency=bad', perMetric: {
              'request_throughput.avg': {mean: NaN}, 'request_latency.p99': {mean: 10},
            }},
          ],
        });
        const datasets = rendered.values.find(v => v && Array.isArray(v.datasets))?.datasets ?? [];
        console.log(JSON.stringify({
          scatter: datasets[0]?.data,
          frontier: datasets[1]?.data,
          xTitle: rendered.values.find(v => v?.scales)?.scales?.x?.title?.text,
          yTitle: rendered.values.find(v => v?.scales)?.scales?.y?.title?.text,
        }));
        """,
    )

    assert result["scatter"] == [
        {"x": 100, "y": 80, "jobName": "concurrency=8", "cluster": "sweep"},
        {"x": 150, "y": 90, "jobName": "concurrency=16", "cluster": "sweep"},
        {"x": 200, "y": 70, "jobName": "concurrency=32", "cluster": "sweep"},
        {"x": 200, "y": 70, "jobName": "concurrency=32-duplicate", "cluster": "sweep"},
    ]
    # On an (x, y) tie the frontier keeps the first-seen point, so the duplicate
    # collapses to one frontier step while both points still render in scatter.
    assert result["frontier"] == [
        {"x": 100, "y": 80, "jobName": "concurrency=8"},
        {"x": 200, "y": 70, "jobName": "concurrency=32"},
    ]
    assert result["xTitle"] == "Throughput (req/s)"
    assert result["yTitle"] == "P99 latency (ms)"


def test_variations_table_drops_columns_with_only_non_finite_or_missing_values() -> (
    None
):
    result = _run_component_script(
        "variations-table.js",
        ["VariationsTable"],
        """
        const rendered = helpers.VariationsTable({
          headlineMetrics: [
            {key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'},
            {key: 'request_latency', stat: 'p99', label: 'P99', unit: 'ms'},
          ],
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=8', n_trials: 1, n_total: 1, perMetric: {
              'request_throughput.avg': {mean: NaN, cv: NaN},
              'request_latency.p99': {mean: null},
            }},
            {variation_index: 1, label: 'benchmark.concurrency=16', n_trials: 1, n_total: 1, perMetric: {
              'request_throughput.avg': {mean: Infinity, cv: 0.1},
            }},
          ],
        });
        console.log(JSON.stringify({
          labels: rendered.values.filter(v => v === 'Throughput' || v === 'P99'),
          text: rendered.strings.join(''),
        }));
        """,
    )

    assert result["labels"] == []


def test_chart_wrapper_options_fingerprint_includes_callbacks_for_stable_updates() -> (
    None
):
    source = (COMPONENTS_DIR / "chart-wrapper.js").read_text()

    assert "JSON.stringify(options)" not in source
    assert "callbacks" in source

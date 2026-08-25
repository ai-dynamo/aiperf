# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial correctness tests for operator UI chart data shaping."""

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
          const useMemo = (fn) => fn();
          const html = (strings, ...values) => ({{ strings: Array.from(strings), values }});
          {CHART_TYPOGRAPHY_JS_IN_TEMPLATE}
          {chart_wrapper_stub}
        `;
        source = `${{prelude}}\n${{source}}\nexport {{ {", ".join(export_names)} }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {body}
    """
    return json.loads(run_node(script))


def test_latency_timeline_rejects_non_finite_and_negative_latencies() -> None:
    result = _run_component_script(
        "latency-timeline-chart.js",
        ["recordLatencyMs"],
        """
        const records = [
          {metrics: {request_latency: {value: NaN, unit: 'ms'}}},
          {metrics: {request_latency: {value: Infinity, unit: 'ms'}}},
          {metrics: {request_latency: {value: -1, unit: 'ms'}}},
          {metrics: {request_latency: {value: -1000, unit: 'us'}}},
          {metrics: {request_latency: {value: -1000000, unit: 'ns'}}},
          {metrics: {request_latency: {value: -0.25, unit: 's'}}},
          {metrics: {request_latency: {value: 12.5, unit: 'ms'}}},
        ];
        console.log(JSON.stringify({latencies: records.map(helpers.recordLatencyMs)}));
        """,
    )

    assert result == {"latencies": [None, None, None, None, None, None, 12.5]}


def test_latency_timeline_huge_series_sampling_stays_bounded_and_ordered() -> None:
    result = _run_component_script(
        "latency-timeline-chart.js",
        ["strideSample"],
        """
        const values = Array.from({length: 25003}, (_, i) => i);
        const sampled = helpers.strideSample(values, 10000);
        console.log(JSON.stringify({
          count: sampled.length,
          first: sampled[0],
          last: sampled.at(-1),
          strictlyIncreasing: sampled.every((value, index) => index === 0 || value > sampled[index - 1]),
          unchangedIdentity: helpers.strideSample(values, 30000) === values,
        }));
        """,
    )

    assert result["count"] <= 10000
    assert result["first"] == 0
    assert result["last"] <= 25002
    assert result["strictlyIncreasing"] is True
    assert result["unchangedIdentity"] is True


def test_variations_chart_drops_non_finite_series_and_preserves_empty_state() -> None:
    result = _run_component_script(
        "variations-chart.js",
        ["VariationsChart"],
        """
        const rendered = helpers.VariationsChart({
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=8', mean: NaN, std: 1, cv: 0.1, n: 1},
            {variation_index: 1, label: 'benchmark.concurrency=16', mean: Infinity, std: 1, cv: 0.2, n: 1},
            {variation_index: 2, label: 'benchmark.concurrency=32', mean: -Infinity, std: 1, cv: 0.3, n: 1},
            {variation_index: 3, label: 'benchmark.concurrency=64', mean: null, std: 1, cv: 0.4, n: 1},
          ],
          metricLabel: 'Request throughput',
          unit: 'req/s',
        });
        console.log(JSON.stringify({
          hasChartData: rendered.values.some(v => v && v.datasets),
          text: rendered.strings.join(''),
        }));
        """,
    )

    assert result["hasChartData"] is False
    assert "No " in result["text"]


def test_variations_chart_preserves_duplicate_and_prototype_like_labels() -> None:
    result = _run_component_script(
        "variations-chart.js",
        ["VariationsChart"],
        """
        const rendered = helpers.VariationsChart({
          variations: [
            {variation_index: 0, label: 'benchmark.__proto__=alpha', mean: 10, std: 1, cv: 0.1, n: 1},
            {variation_index: 1, label: 'benchmark.__proto__=alpha', mean: 11, std: 1, cv: 0.1, n: 1},
            {variation_index: 2, label: 'benchmark.constructor=beta', mean: 12, std: 1, cv: 0.1, n: 1},
          ],
          metricLabel: 'Latency',
          unit: 'ms',
        });
        const chartData = rendered.values.find(v => v && v.datasets);
        console.log(JSON.stringify({
          labels: chartData?.labels,
          uniqueCount: new Set(chartData?.labels ?? []).size,
          prototypePolluted: Object.prototype.alpha === true || Object.prototype.beta === true,
        }));
        """,
    )

    # Labels stay index-aligned with the data series, so duplicate short-labels
    # are preserved verbatim rather than deduped. The adversarial value here is
    # that prototype-like labels never pollute Object.prototype.
    assert result["labels"] == [
        "__proto__=alpha",
        "__proto__=alpha",
        "constructor=beta",
    ]
    assert result["uniqueCount"] == 2
    assert result["prototypePolluted"] is False


def test_chart_wrapper_option_fingerprint_accounts_for_callback_changes() -> None:
    source = (COMPONENTS_DIR / "chart-wrapper.js").read_text()

    assert "JSON.stringify(options)" not in source
    assert "callbacks" in source
    assert "function" in source or "toString" in source


def test_pareto_filters_non_finite_points_and_keeps_zero_axis_padding_positive() -> (
    None
):
    result = _run_component_script(
        "variations-pareto.js",
        ["VariationsPareto"],
        """
        const xMetric = {key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'};
        const yMetric = {key: 'request_latency', stat: 'p99', label: 'P99 latency', unit: 'ms'};
        const perMetric = (x, y) => ({
          'request_throughput.avg': {mean: x},
          'request_latency.p99': {mean: y},
        });
        const rendered = helpers.VariationsPareto({
          xMetric,
          yMetric,
          yIsSmallerBetter: true,
          variations: [
            {variation_index: 0, label: 'benchmark.concurrency=0a', perMetric: perMetric(0, 0)},
            {variation_index: 1, label: 'benchmark.concurrency=0b', perMetric: perMetric(0, 0)},
            {variation_index: 2, label: 'benchmark.concurrency=nan', perMetric: perMetric(NaN, 1)},
            {variation_index: 3, label: 'benchmark.concurrency=inf', perMetric: perMetric(1, Infinity)},
          ],
        });
        const datasets = rendered.values.find(v => v && Array.isArray(v.datasets))?.datasets ?? [];
        const options = rendered.values.find(v => v?.scales);
        console.log(JSON.stringify({
          scatter: datasets[0]?.data,
          frontier: datasets[1]?.data ?? [],
          xMin: options?.scales?.x?.min,
          xMax: options?.scales?.x?.max,
          yMin: options?.scales?.y?.min,
          yMax: options?.scales?.y?.max,
        }));
        """,
    )

    assert result["scatter"] == [
        {"x": 0, "y": 0, "jobName": "concurrency=0a", "cluster": "sweep"},
        {"x": 0, "y": 0, "jobName": "concurrency=0b", "cluster": "sweep"},
    ]
    assert result["frontier"] == []
    assert result["xMin"] < 0 < result["xMax"]
    assert result["yMin"] < 0 < result["yMax"]


def test_pareto_ties_keep_all_scatter_points_but_frontier_only_strict_improvements() -> (
    None
):
    result = _run_component_script(
        "variations-pareto.js",
        ["VariationsPareto"],
        """
        const xMetric = {key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'};
        const yMetric = {key: 'request_latency', stat: 'p99', label: 'P99 latency', unit: 'ms'};
        const metricBag = Object.create(null);
        metricBag['request_throughput.avg'] = {mean: 100};
        metricBag['request_latency.p99'] = {mean: 10};
        const perMetric = (x, y) => ({
          'request_throughput.avg': {mean: x},
          'request_latency.p99': {mean: y},
        });
        const rendered = helpers.VariationsPareto({
          xMetric,
          yMetric,
          yIsSmallerBetter: true,
          variations: [
            {variation_index: 0, label: 'benchmark.__proto__=a', perMetric: metricBag},
            {variation_index: 1, label: 'benchmark.constructor=a', perMetric: perMetric(110, 10)},
            {variation_index: 2, label: 'benchmark.toString=b', perMetric: perMetric(120, 9)},
          ],
        });
        const datasets = rendered.values.find(v => v && Array.isArray(v.datasets))?.datasets ?? [];
        console.log(JSON.stringify({
          scatterNames: datasets[0]?.data.map(point => point.jobName),
          frontier: datasets[1]?.data,
          prototypePolluted: Object.prototype.a === true || Object.prototype.b === true,
        }));
        """,
    )

    assert result["scatterNames"] == ["__proto__=a", "constructor=a", "toString=b"]
    assert result["frontier"] == [
        {"x": 100, "y": 10, "jobName": "__proto__=a"},
        {"x": 120, "y": 9, "jobName": "toString=b"},
    ]
    assert result["prototypePolluted"] is False

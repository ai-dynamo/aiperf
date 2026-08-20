# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation regressions for sweep-variation components.

These cover defects where the rendered output misrepresented the underlying
measurement: opaque cell identities standing in for the swept parameter
values, and rounding that hid real differences between operating points.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import (
    CHART_TYPOGRAPHY_JS_IN_TEMPLATE,
    FORMAT_JS_IN_TEMPLATE,
    run_node,
)

COMPONENTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
)

# The real `lib/format.js` is spliced in: the components under test import it
# for real at runtime and the harness strips imports, so a hand-copy here would
# quietly assert its own decimal rules rather than the console's.
_PRELUDE = (
    """
  const palette = new Proxy({}, { get: (_t, key) => '#' + String(key) });
"""
    + FORMAT_JS_IN_TEMPLATE
    + CHART_TYPOGRAPHY_JS_IN_TEMPLATE
    + """
  const useMemo = (fn) => fn();
  const useState = (initial) => [initial, () => {}];
  const html = (strings, ...values) => ({ strings: Array.from(strings), values });
  function ChartWrapper(props) { return { component: 'ChartWrapper', props }; }
  globalThis.__ChartWrapper = ChartWrapper;
"""
)


def run_component(
    component_name: str, export_names: list[str], body: str
) -> dict[str, object]:
    """Load one UI component under Node with stubbed imports and run `body`."""
    component_path = COMPONENTS_DIR / component_name
    script = f"""
        import fs from 'node:fs';
        const path = {str(component_path)!r};
        let source = fs.readFileSync(path, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `{_PRELUDE}\n${{source}}\nexport {{ {", ".join(export_names)} }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {body}
    """
    return json.loads(run_node(script))


def _pareto_render(extra_variation_fields: str) -> dict[str, object]:
    return run_component(
        "variations-pareto.js",
        ["VariationsPareto"],
        f"""
        const rendered = helpers.VariationsPareto({{
          xMetric: {{key: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s'}},
          yMetric: {{key: 'request_latency', stat: 'p99', label: 'Latency P99', unit: 'ms'}},
          yIsSmallerBetter: true,
          variations: [
            {{variation_index: 0, label: 'search_iter_0004', {extra_variation_fields}
              perMetric: {{
                'request_throughput.avg': {{mean: 3.47}},
                'request_latency.p99': {{mean: 812.4}},
              }}}},
            {{variation_index: 1, label: 'search_iter_0008', {extra_variation_fields}
              perMetric: {{
                'request_throughput.avg': {{mean: 3.52}},
                'request_latency.p99': {{mean: 640.9}},
              }}}},
          ],
        }});
        const datasets = rendered.values.find(v => v && Array.isArray(v.datasets))?.datasets ?? [];
        const options = rendered.values.find(v => v?.scales);
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        console.log(JSON.stringify({{
          scatterNames: datasets[0]?.data.map(p => p.jobName),
          legendDisplay: options.plugins.legend.display,
          datasetLabels: datasets.map(d => d.label),
          caption: flatten(rendered).replace(/\\s+/g, ' ').trim(),
          tooltip: options.plugins.tooltip.callbacks.label(
            {{raw: {{x: 3.47, y: 812.4}}}},
          ),
        }}));
        """,
    )


def test_pareto_identifies_points_by_swept_values_not_search_iter_cell_id() -> None:
    """Adaptive cell ids say nothing; the swept values are the identity."""
    with_values = _pareto_render("valuesLabel: 'concurrency=17',")
    assert with_values["scatterNames"] == ["concurrency=17", "concurrency=17"]

    # Without `valuesLabel` (older archives) the raw label is still the fallback.
    without_values = _pareto_render("")
    assert without_values["scatterNames"] == ["search_iter_0004", "search_iter_0008"]


def test_pareto_tooltip_keeps_enough_resolution_to_order_adjacent_points() -> None:
    """3.47 and 3.52 req/s must not both collapse to a whole number."""
    rendered = _pareto_render("valuesLabel: 'concurrency=17',")
    assert rendered["tooltip"] == [
        "Throughput: 3.47 req/s",
        "Latency P99: 812.40 ms",
    ]


def test_pareto_explains_its_marks_in_words_instead_of_a_one_entry_legend() -> None:
    rendered = _pareto_render("valuesLabel: 'concurrency=17',")
    assert rendered["legendDisplay"] is False
    assert rendered["datasetLabels"] == [
        "Variation (mean across its trials)",
        "Pareto frontier",
    ]
    caption = rendered["caption"]
    assert (
        "Each dot is one variation, plotted at the mean across its trials." in caption
    )
    # The frontier direction must be spelled out: "best" flips per axis pair.
    assert "Dashed line is the Pareto frontier" in caption
    assert "lowest Latency P99" in caption
    assert "at each Throughput" in caption
    assert "concurrency=17 → concurrency=17" in caption


def test_pareto_tooltip_drops_decimals_only_once_they_are_noise() -> None:
    result = run_component(
        "variations-pareto.js",
        ["VariationsPareto"],
        """
        const rendered = helpers.VariationsPareto({
          xMetric: {key: 'output_token_throughput', stat: 'avg', label: 'Token Throughput', unit: 'tok/s'},
          yMetric: {key: 'request_latency', stat: 'p99', label: 'Latency P99', unit: 'ms'},
          yIsSmallerBetter: true,
          variations: [
            {variation_index: 0, valuesLabel: 'concurrency=1', perMetric: {
              'output_token_throughput.avg': {mean: 14235.6},
              'request_latency.p99': {mean: 42.25},
            }},
          ],
        });
        const options = rendered.values.find(v => v?.scales);
        console.log(JSON.stringify({
          tooltip: options.plugins.tooltip.callbacks.label({raw: {x: 14235.6, y: 42.25}}),
        }));
        """,
    )
    # Thousands keep the separator and shed decimals; tens keep two.
    assert result["tooltip"] == [
        "Token Throughput: 14,236 tok/s",
        "Latency P99: 42.25 ms",
    ]


def _variations_chart_render(extra_variation_fields: str) -> dict[str, object]:
    return run_component(
        "variations-chart.js",
        ["VariationsChart"],
        f"""
        const rendered = helpers.VariationsChart({{
          metricLabel: 'Output tok/s',
          unit: 'tok/s',
          variations: [
            {{variation_index: 0, label: 'search_iter_0004', {extra_variation_fields}
              mean: 1420.5, std: 33.25, cv: 0.0234, n: 3}},
            {{variation_index: 1, label: 'search_iter_0008', {extra_variation_fields}
              mean: 1655.25, std: 12.5, cv: 0.0076, n: 3}},
          ],
        }});
        const data = rendered.values.find(v => v && v.datasets);
        const options = rendered.values.find(v => v?.scales);
        console.log(JSON.stringify({{
          labels: data.labels,
          tensions: data.datasets.map(d => d.tension),
          tooltipMulti: options.plugins.tooltip.callbacks.label({{dataIndex: 0, parsed: {{y: 1420.5}}}}),
          tooltipSingle: helpers.VariationsChart({{
            metricLabel: 'Output tok/s',
            unit: 'tok/s',
            variations: [{{variation_index: 0, label: 'x=1', mean: 42.25, std: 0, cv: null, n: 1}}],
          }}).values.find(v => v?.scales).plugins.tooltip.callbacks.label(
            {{dataIndex: 0, parsed: {{y: 42.25}}}},
          ),
        }}));
        """,
    )


def test_variation_curve_ticks_show_swept_values_not_search_iter_cell_id() -> None:
    with_values = _variations_chart_render("valuesLabel: 'concurrency=17',")
    assert with_values["labels"] == ["concurrency=17", "concurrency=17"]

    without_values = _variations_chart_render("")
    assert without_values["labels"] == ["search_iter_0004", "search_iter_0008"]


def test_variation_curve_does_not_spline_between_discrete_operating_points() -> None:
    """A smoothed curve would imply measurements between two variations."""
    rendered = _variations_chart_render("valuesLabel: 'concurrency=17',")
    assert rendered["tensions"] == [0, 0, 0]


def test_variation_curve_tooltip_quantifies_the_band_it_draws() -> None:
    rendered = _variations_chart_render("valuesLabel: 'concurrency=17',")
    # Magnitude-aware decimals + separators, the drawn +/-1 std stated
    # numerically, and the sample size the mean rests on.
    assert rendered["tooltipMulti"] == "  1,421 ±33 tok/s (cv 2.3%) — mean of 3 trials"


def test_variation_curve_tooltip_refuses_to_render_a_single_trial_as_zero_spread() -> (
    None
):
    """meanStd reports std 0 for n<2; that is unmeasured, not reproducible."""
    rendered = _variations_chart_render("valuesLabel: 'concurrency=17',")
    assert rendered["tooltipSingle"] == "  42.25 tok/s — 1 trial, spread unknown"
    assert "±" not in str(rendered["tooltipSingle"])


def test_live_variations_columns_name_the_metric_and_stat_they_show() -> None:
    """ "TPS" read as requests/sec; the column is output-token throughput."""
    helpers_uri = (COMPONENTS_DIR / "live-variations-helpers.js").as_uri()
    component_path = COMPONENTS_DIR / "live-variations-card.js"
    script = f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(component_path)!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `import {{ parseVariationValues, titleCase, trialContributesMetrics }} from {helpers_uri!r};\n{_PRELUDE}\n${{source}}\nexport {{ LiveVariationsCard }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        const rendered = helpers.LiveVariationsCard({{
          manifest: [
            {{name: 'sweep-v00-t0', variation_index: 0, variation_label: 'concurrency=8', trial_index: 0}},
          ],
          childData: {{
            'sweep-v00-t0': {{
              phase: 'Succeeded',
              summary: {{outputTokenThroughputTps: 1420.5, requestLatencyP99Ms: 812.4, ttftMs: 42.2}},
            }},
          }},
        }});
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        console.log(JSON.stringify({{text: flatten(rendered).replace(/\\s+/g, ' ')}}));
    """
    text = json.loads(run_node(script))["text"]

    assert "output tok/s" in text
    assert "request latency p99 · ms" in text
    assert "TTFT avg · ms" in text
    # The ambiguous heading is gone.
    assert "TPS" not in text
    # And the columns declare that they average completed trials only.
    assert "Mean across this variation" in text


def test_cells_table_heading_names_the_metric_and_its_unit() -> None:
    """A raw snake_case tag with no unit is not a user-facing column heading."""
    result = run_component(
        "cells-table.js",
        ["CellsTable"],
        """
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        const known = helpers.CellsTable({
          dimensions: [{name: 'concurrency'}],
          cells: [{variation_index: 0, variation_label: 'c=8', metrics: {request_throughput: {avg: 12.5}}}],
          metric: 'request_throughput',
          stat: 'avg',
        });
        const unknown = helpers.CellsTable({
          dimensions: [{name: 'concurrency'}],
          cells: [{variation_index: 0, variation_label: 'c=8', metrics: {throughput: {mean: 12.5}}}],
          metric: 'throughput',
          stat: 'mean',
        });
        console.log(JSON.stringify({
          known: flatten(known).replace(/\\s+/g, ' '),
          unknown: flatten(unknown).replace(/\\s+/g, ' '),
        }));
        """,
    )

    # The visible heading names the metric and unit. The raw tag survives only
    # inside the hover title, which is deliberate: it tells an operator which
    # key the column came from without putting it in the reader's way.
    assert ">Request throughput avg · req/s</th>" in result["known"]
    assert ">request_throughput (avg)</th>" not in result["known"]
    # An unrecognised key gets a humanized label and no invented unit.
    assert "throughput mean" in result["unknown"]
    assert "req/s" not in result["unknown"]


def test_cells_chart_axis_and_tooltip_carry_the_unit() -> None:
    result = run_component(
        "cells-chart.js",
        ["CellsChart"],
        """
        const rendered = helpers.CellsChart({
          dimensions: [{name: 'concurrency', values: [8, 16]}],
          cells: [
            {variation_index: 0, values: {concurrency: 8}, metrics: {request_throughput: {avg: 12.5}}},
            {variation_index: 1, values: {concurrency: 16}, metrics: {request_throughput: {avg: 18.25}}},
          ],
          metric: 'request_throughput',
          stat: 'avg',
        });
        const options = rendered.values.find(v => v?.scales);
        const data = rendered.values.find(v => v && v.datasets);
        console.log(JSON.stringify({
          yTitle: options.scales.y.title.text,
          seriesLabel: data.datasets[0].label,
          tooltip: options.plugins.tooltip.callbacks.label(
            {parsed: {y: 12.5}, dataset: {label: data.datasets[0].label}},
          ),
        }));
        """,
    )

    assert result["yTitle"] == "Request throughput avg · req/s"
    assert result["seriesLabel"] == "Request throughput avg · req/s"
    assert result["tooltip"].endswith(" req/s")


def _phase_bar_render(phases_js: str) -> str:
    return run_component(
        "phase-bar.js",
        ["PhaseBar"],
        f"""
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        console.log(JSON.stringify({{
          text: flatten(helpers.PhaseBar({{phases: {phases_js}}})).replace(/\\s+/g, ' '),
        }}));
        """,
    )["text"]


def test_phase_bar_never_reports_100_percent_before_the_phase_is_done() -> None:
    """Math.round(99.9) is 100; an unfinished phase must not claim completion."""
    text = _phase_bar_render("[{name: 'profiling', completed: 999, total: 1000}]")
    assert "in progress" in text
    assert "(99%)" in text
    assert "(100%)" not in text
    assert "aria-valuenow=100" not in text.replace('"', "")


def test_phase_bar_reports_100_percent_once_the_phase_is_actually_done() -> None:
    text = _phase_bar_render("[{name: 'profiling', completed: 1000, total: 1000}]")
    assert "done" in text
    assert "(100%)" in text

    # Over-counted phases still clamp to 100 rather than showing 101%.
    over = _phase_bar_render("[{name: 'profiling', completed: 1010, total: 1000}]")
    assert "(100%)" in over


def test_trial_detail_panel_keeps_resolution_and_one_locale() -> None:
    """A single-digit req/s must not print as a whole number, and every number
    on the page must use the same separators as fmtNumber elsewhere."""
    helpers_uri = (COMPONENTS_DIR.parent / "pages" / "sweep-detail-helpers.js").as_uri()
    component_path = COMPONENTS_DIR / "sweep-live-trial-board.js"
    script = f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(component_path)!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `import {{ buildTrialBoardRows }} from {helpers_uri!r};\n{_PRELUDE}\n${{source}}\nexport {{ DetailPanel }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        const rendered = helpers.DetailPanel({{
          row: {{variation_index: 0, label: 'concurrency=8'}},
          trial: {{
            trial_index: 0,
            name: 'sweep-v00-t0',
            namespace: 'ns',
            state: 'succeeded',
            summary: {{
              outputTokenThroughputTps: 14235.6,
              requestThroughputRps: 3.47,
              requestLatencyP99Ms: 812.44,
              ttftMs: 42.25,
            }},
          }},
        }});
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        console.log(JSON.stringify({{text: flatten(rendered).replace(/\\s+/g, ' ')}}));
    """
    text = json.loads(run_node(script))["text"]

    assert "14,236 tok/s" in text
    # Was "3 req/s" under maximumFractionDigits: 0.
    assert "3.47 req/s" in text
    assert "812.44 ms" in text
    assert "42.25 ms" in text

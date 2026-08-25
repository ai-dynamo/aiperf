# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numeric-presentation contracts for operator UI table, chart and KPI cells.

Each test here pins a property of how a *number* reaches the reader, not how
the surrounding markup is shaped:

* displayed precision never exceeds what the measurement resolves, and never
  rounds two distinguishable values onto the same string;
* a value that is present is never rendered as missing;
* a percentage never rounds up to 100% while the underlying counts say
  something is still outstanding;
* every surface of one page formats numbers through the same locale, so a
  chart axis and the table beside it never disagree on separators;
* a unit is either correct or absent, never guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import (
    CHART_TYPOGRAPHY_JS_IN_TEMPLATE,
    FORMAT_JS_IN_TEMPLATE,
    run_node,
)

_UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
_COMPONENTS_DIR = _UI_DIR / "components"

# Stubs for every symbol the components under test import, plus the real
# `lib/format.js` -- the formatters own the decimal counts these tests are
# about, so a hand-copy of them would assert against itself.
_PRELUDE = (
    """
  const palette = new Proxy({}, { get: (_t, key) => '#' + String(key) });
"""
    + FORMAT_JS_IN_TEMPLATE
    + CHART_TYPOGRAPHY_JS_IN_TEMPLATE
    + """
  const useMemo = (fn) => fn();
  const useState = (initial) => [typeof initial === 'function' ? initial() : initial, () => {}];
  const useEffect = () => {};
  const useRef = () => ({ current: null });
  const html = (strings, ...values) => ({ strings: Array.from(strings), values });
  const phaseColor = () => '#89b4fa';
  const navigate = () => {};
  function NsPill(props) { return { component: 'NsPill', props }; }
  function RelativeTime(props) { return { component: 'RelativeTime', props }; }
  function Sparkline(props) { return { component: 'Sparkline', props }; }
  function ChartWrapper(props) { return { component: 'ChartWrapper', props }; }
"""
)

_FLATTEN = """
  const flatten = (node) => {
    if (node == null || node === false) return '';
    if (Array.isArray(node)) return node.map(flatten).join('');
    if (typeof node === 'object' && node.strings) {
      return node.strings
        .map((s, i) => s + flatten(node.values[i] ?? '')).join('');
    }
    if (typeof node === 'object') return '';
    return String(node);
  };
"""


def _run_component(component: str, exports: list[str], body: str) -> dict:
    """Load one UI component under Node with stubbed imports and run `body`."""
    path = _COMPONENTS_DIR / component
    script = f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(path)!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replace(/^export \\{{[^}}]*\\}};\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `{_PRELUDE}\n${{source}}\nexport {{ {", ".join(exports)} }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {_FLATTEN}
        {body}
    """
    return json.loads(run_node(script))


# ---------------------------------------------------------------------------
# cells-chart.js
# ---------------------------------------------------------------------------


def test_cells_chart_sets_the_same_locale_the_dom_formatters_use() -> None:
    """Chart.js reads `options.locale` straight into `Intl.NumberFormat` and
    never assigns a default. lib/format.js formats with the viewer's locale
    (`toLocaleString` with no explicit locale), so the chart must state that
    same locale explicitly -- leaving it unset is not equivalent, it is what
    put "1.234,5" on the y-axis beside "1,234.5" in the table below it.

    The harness pins LC_ALL, so `navigator.language` resolves to en-US here."""
    out = _run_component(
        "cells-chart.js",
        ["CellsChart"],
        """
        const rendered = helpers.CellsChart({
          dimensions: [{name: 'concurrency', values: [8, 16]}],
          cells: [
            {variation_index: 0, values: {concurrency: 8},
             metrics: {request_throughput: {avg: 1234.5}}},
            {variation_index: 1, values: {concurrency: 16},
             metrics: {request_throughput: {avg: 2345.6}}},
          ],
          metric: 'request_throughput',
          stat: 'avg',
        });
        const options = rendered.values.find(v => v?.scales);
        console.log(JSON.stringify({ locale: options.locale ?? null }));
        """,
    )

    assert out["locale"] == "en-US"  # == navigator.language under the pinned LC_ALL


def test_cells_chart_tooltip_orders_faithfully_and_carries_the_unit() -> None:
    """The tooltip is the only place a reader gets a mark's numeric value, so
    two distinguishable cells must not round onto the same string, and the
    number must arrive with the unit it is measured in.

    `req/s` and `ms` readouts are pinned to two decimals across the console so
    a value keeps one shape wherever it appears; the resolution assertion here
    is that distinguishable operating points stay distinguishable, not a
    particular digit count."""
    out = _run_component(
        "cells-chart.js",
        ["CellsChart"],
        """
        const rendered = helpers.CellsChart({
          dimensions: [{name: 'concurrency', values: [8, 16]}],
          cells: [
            {variation_index: 0, values: {concurrency: 8},
             metrics: {request_throughput: {avg: 12.3456}}},
            {variation_index: 1, values: {concurrency: 16},
             metrics: {request_throughput: {avg: 12.4111}}},
          ],
          metric: 'request_throughput',
          stat: 'avg',
        });
        const options = rendered.values.find(v => v?.scales);
        const label = options.plugins.tooltip.callbacks.label;
        console.log(JSON.stringify({
          a: label({parsed: {y: 12.3456}, dataset: {label: 'x'}}),
          b: label({parsed: {y: 12.4111}, dataset: {label: 'x'}}),
          big: label({parsed: {y: 12345.678}, dataset: {label: 'x'}}),
        }));
        """,
    )

    assert out["a"] == "x: 12.35 req/s"
    # Distinguishable operating points keep distinguishable strings.
    assert out["b"] != out["a"]
    # A large value stays grouped and keeps its unit rather than becoming a
    # bare run of digits.
    assert out["big"] == "x: 12,345.68 req/s"


# ---------------------------------------------------------------------------
# cells-table.js
# ---------------------------------------------------------------------------


def test_cells_table_renders_every_magnitude_distinctly_and_grouped() -> None:
    """Five cells spanning four orders of magnitude must produce five distinct
    strings: a rounding rule that merges two rows shows a tie the underlying
    measurements do not have."""
    out = _run_component(
        "cells-table.js",
        ["CellsTable"],
        """
        const cell = (idx, value) => ({
          variation_index: idx, variation_label: 'v' + idx,
          trials_completed: 3, trials_failed: 0, values: {},
          metrics: {request_throughput: {avg: value}},
        });
        const rendered = helpers.CellsTable({
          dimensions: [],
          cells: [cell(0, 1.23456), cell(1, 12.3456), cell(2, 123.456),
                  cell(3, 1234.56), cell(4, 12345.678)],
          metric: 'request_throughput',
          stat: 'avg',
        });
        console.log(JSON.stringify({ text: flatten(rendered) }));
        """,
    )

    text = out["text"]
    for expected in ("1.23", "12.35", "123.46", "1,234.56", "12,345.68"):
        assert expected in text, expected


# ---------------------------------------------------------------------------
# job-table.js
# ---------------------------------------------------------------------------


def _job_table_text(jobs: list[dict], sort_key: str = "name") -> str:
    out = _run_component(
        "job-table.js",
        ["JobTable"],
        f"""
        const rendered = helpers.JobTable({{
          jobs: {json.dumps(jobs)},
          sort: {{ key: {json.dumps(sort_key)}, dir: 1 }},
          onSortChange: () => {{}},
        }});
        console.log(JSON.stringify({{ text: flatten(rendered) }}));
        """,
    )
    return str(out["text"])


def test_job_table_string_typed_throughput_renders_its_value_not_a_placeholder() -> (
    None
):
    """`finiteNumber` accepts numeric strings for sorting and for the relative
    bar, so the same value must not print as missing in the cell beside it."""
    text = _job_table_text(
        [
            {
                "namespace": "bench",
                "name": "stringy",
                "phase": "Completed",
                "throughputRps": "12.5",
            }
        ]
    )

    assert "12.50 req/s" in text
    assert "--- req/s" not in text


def test_job_table_progress_never_reports_100_percent_before_the_job_reports_it() -> (
    None
):
    """Math.round(99.6) is 100; an unfinished job must not claim completion.
    Mirrors the same rule already enforced on components/phase-bar.js."""
    text = _job_table_text(
        [
            {
                "namespace": "bench",
                "name": "nearly",
                "phase": "Running",
                "progressPercent": 99.6,
            }
        ]
    )

    assert "99%" in text
    assert "100%" not in text
    assert "width: 99%" in text


def test_job_table_progress_reports_100_percent_once_the_job_reports_it() -> None:
    text = _job_table_text(
        [
            {
                "namespace": "bench",
                "name": "done",
                "phase": "Completed",
                "progressPercent": 100,
            }
        ]
    )

    assert "100%" in text


def test_job_table_progress_bar_width_stays_inside_its_track() -> None:
    """An over-reported percent must not draw past the track, and a negative
    one must not draw backwards."""
    over = _job_table_text(
        [{"namespace": "b", "name": "over", "phase": "Running", "progressPercent": 140}]
    )
    under = _job_table_text(
        [
            {
                "namespace": "b",
                "name": "under",
                "phase": "Running",
                "progressPercent": -20,
            }
        ]
    )

    assert "width: 100%" in over
    assert "140%" not in over
    assert "width: 0%" in under
    assert "-20%" not in under


def test_job_table_sortable_numeric_columns_keep_distinguishable_rows_distinct() -> (
    None
):
    """A sortable column whose rounding merges two rows shows a tie that the
    sort itself does not believe in. Both columns scale decimals to magnitude
    so relative resolution stays constant."""
    text = _job_table_text(
        [
            {
                "namespace": "b",
                "name": "a",
                "phase": "Completed",
                "throughputRps": 3.47,
                "latencyP99Ms": 41.6,
            },
            {
                "namespace": "b",
                "name": "b",
                "phase": "Completed",
                "throughputRps": 3.52,
                "latencyP99Ms": 42.4,
            },
        ]
    )

    # Under the older one-decimal / zero-decimal counts these printed
    # "3.5"/"3.5" and "42"/"42" -- two ties that are not ties.
    assert "3.47 req/s" in text
    assert "3.52 req/s" in text
    assert "41.60 ms" in text
    assert "42.40 ms" in text


def test_job_table_latency_column_states_one_unit_for_every_row() -> None:
    """A column that silently switched to seconds past 1000 ms made two rows
    incomparable at a glance -- "812 ms" above "1.2 s" reads as the smaller
    number being the slower one. Every row now states the same unit."""
    text = _job_table_text(
        [
            {
                "namespace": "b",
                "name": "ms",
                "phase": "Completed",
                "latencyP99Ms": 812.44,
            },
            {
                "namespace": "b",
                "name": "s",
                "phase": "Completed",
                "latencyP99Ms": 1234.5,
            },
        ]
    )

    assert "812.44 ms" in text
    assert "1,234.50 ms" in text
    # The old rule printed "812 ms" beside "1.2 s": rounded away the tie-break
    # digits and changed unit mid-column.
    assert "812 ms" not in text
    assert "1.23 s" not in text


def test_job_table_throughput_bar_is_not_drawn_across_incomparable_runs() -> None:
    """Each AIPerfJob is its own experiment; a bar normalised against the
    fastest run of a *different* model asserts a ranking the data cannot
    support. Two different models must not scale against each other."""
    text = _job_table_text(
        [
            {
                "namespace": "bench",
                "name": "small",
                "phase": "Completed",
                "model": "tiny",
                "throughputRps": 3000,
            },
            {
                "namespace": "bench",
                "name": "big",
                "phase": "Completed",
                "model": "huge",
                "throughputRps": 30,
            },
        ]
    )

    # Neither group has a second member, so no relative bar is drawn at all --
    # in particular the 30 req/s run does not get a 1%-full bar because some
    # unrelated model reached 3000.
    assert "transition: width 0.3s" not in text
    assert "3,000.00 req/s" in text
    assert "30.00 req/s" in text


def test_job_table_throughput_bar_scales_within_one_comparable_group() -> None:
    """Sweep children share namespace and model, which is exactly the case the
    bar exists for: ranking one workload's operating points."""
    text = _job_table_text(
        [
            {
                "namespace": "bench",
                "name": "c8",
                "phase": "Completed",
                "model": "same",
                "throughputRps": 50,
            },
            {
                "namespace": "bench",
                "name": "c16",
                "phase": "Completed",
                "model": "same",
                "throughputRps": 100,
            },
            {
                "namespace": "bench",
                "name": "other-model",
                "phase": "Completed",
                "model": "different",
                "throughputRps": 100000,
            },
        ]
    )

    # 50 of the group max 100, not of the unrelated 100000.
    assert "width: 50.0%" in text
    assert "width: 100.0%" in text
    assert "width: 0.1%" not in text


# ---------------------------------------------------------------------------
# realtime-kpi-grid.js
# ---------------------------------------------------------------------------


def _kpi_grid_text(summary: dict, slos: dict | None = None) -> str:
    """Render the goodput / success-rate tile. Rendered directly rather than
    through RealtimeKpiGrid because the grid returns child components as
    unevaluated functions under this preact-free harness."""
    out = _run_component(
        "realtime-kpi-grid.js",
        ["ReliabilityTile"],
        f"""
        const rendered = helpers.ReliabilityTile({{
          summary: {json.dumps(summary)},
          slos: {json.dumps(slos)},
          timeseries: {{}},
        }});
        console.log(JSON.stringify({{ text: flatten(rendered) }}));
        """,
    )
    return str(out["text"])


def test_goodput_tile_never_shows_100_percent_while_requests_failed() -> None:
    """The chip is derived from exact counts and the sub-line from the rounded
    percentage; rounding 99.96 up made the two halves of one tile disagree."""
    text = _kpi_grid_text(
        {
            "good_request_count": {"avg": 9996},
            "request_count": {"avg": 10000},
            "goodput": {"avg": 120.5},
        },
        {"request_latency": 500},
    )

    assert "4 failed" in text
    assert "99.9%" in text
    assert "100.0%" not in text


def test_goodput_tile_shows_100_percent_when_nothing_failed() -> None:
    text = _kpi_grid_text(
        {
            "good_request_count": {"avg": 10000},
            "request_count": {"avg": 10000},
            "goodput": {"avg": 120.5},
        },
        {"request_latency": 500},
    )

    assert "0 failed" in text
    assert "100.0%" in text


def test_success_rate_tile_never_rounds_a_failed_run_up_to_100_percent() -> None:
    """One error in a million is still not a clean run."""
    text = _kpi_grid_text(
        {
            "request_count": {"avg": 1000000},
            "error_request_count": {"avg": 1},
        }
    )

    assert "1 errors" in text
    assert "99.99%" in text
    assert "100.00%" not in text


def test_success_rate_tile_shows_100_percent_on_a_clean_run() -> None:
    text = _kpi_grid_text(
        {
            "request_count": {"avg": 1000},
            "error_request_count": {"avg": 0},
        }
    )

    assert "0 errors" in text
    assert "100.00%" in text


def test_success_rate_denominator_is_successes_plus_errors() -> None:
    """`request_count` counts successful requests only, so the population a
    rate is taken over is request_count + error_request_count. Dividing by
    `request_count` alone gave 3 errors against 1 success a 300% error rate,
    floored to a 0% success rate."""
    text = _kpi_grid_text(
        {
            "request_count": {"avg": 1},
            "error_request_count": {"avg": 3},
        }
    )

    # 3 of 4 failed -> 25% success, not 0%.
    assert "25.00%" in text
    assert "0.00%" not in text


def test_success_rate_prefers_the_authoritative_total_requests_scalar() -> None:
    text = _kpi_grid_text(
        {
            "request_count": {"avg": 1},
            "error_request_count": {"avg": 3},
            "total_requests": 4,
        }
    )

    assert "25.00%" in text


def test_success_rate_states_the_population_it_was_measured_over() -> None:
    """An error count without its denominator is not interpretable: 5 errors
    of 10 and 5 errors of 5,000,000 are different runs."""
    text = _kpi_grid_text(
        {
            "request_count": {"avg": 4999995},
            "error_request_count": {"avg": 5},
        }
    )

    assert "5 of 5,000,000" in text


def _kpi_tile_text(tag: str, metric: dict, slos: dict) -> str:
    out = _run_component(
        "realtime-kpi-grid.js",
        ["KpiTile", "TILES"],
        f"""
        const spec = helpers.TILES.find(t => t.tag === {json.dumps(tag)});
        const rendered = helpers.KpiTile({{
          spec,
          metric: {json.dumps(metric)},
          slos: {json.dumps(slos)},
          series: [],
        }});
        console.log(JSON.stringify({{ text: flatten(rendered) }}));
        """,
    )
    return str(out["text"])


def test_slo_chip_states_the_unit_of_the_threshold_it_compares_against() -> None:
    """The headline carries a unit; a bare "<= 500" beside it makes the reader
    assume the threshold shares it, which is load-bearing for a pass/fail
    verdict and was never stated."""
    text = _kpi_tile_text(
        "request_latency", {"p99": 412.5, "avg": 300}, {"request_latency": 500}
    )

    assert "≤ 500.00 ms" in text


def test_slo_chip_threshold_avoids_float_artefacts_and_uses_one_locale() -> None:
    """A threshold is an author-supplied constant, so it is shown as authored:
    grouped like every other number, with no float noise and no fabricated
    trailing zeros."""
    noisy = _kpi_tile_text(
        "request_throughput",
        {"current": 0.5, "avg": 0.5},
        {"request_throughput": 0.30000000000000004},
    )
    big = _kpi_tile_text(
        "request_throughput",
        {"current": 5000, "avg": 5000},
        {"request_throughput": 12000},
    )

    assert "≥ 0.30 req/s" in noisy
    assert "0.30000000000000004" not in noisy
    assert "≥ 12,000.00 req/s" in big

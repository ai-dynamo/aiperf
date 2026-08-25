# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single trial must not be reported as zero spread.

``meanStd`` (pages/sweep-detail-helpers.js) returned ``std: 0`` for one trial.
Zero is a measurement result -- "every trial landed on the same number" -- and
one observation supports no such claim. The variation chart consumed that value
for its error band and drew a zero-width band through the point, which reads as
a confident measurement of perfect reproducibility rather than the absence of
one.

The discipline already existed on either side of the defect and only ``std`` was
left out: ``cv`` has always been null for n<2, and ``sweep-live-trial-board.js``
says "1 trial, spread unknown". These tests pin ``std: null`` and pin that every
consumer renders it as unmeasured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import (
    CHART_TYPOGRAPHY_JS_IN_TEMPLATE,
    FORMAT_JS_IN_TEMPLATE,
    run_node,
)

_UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
_HELPERS = _UI_DIR / "pages" / "sweep-detail-helpers.js"
_COMPONENTS = _UI_DIR / "components"

# Mirrors the stub prelude used by the other chart harnesses in this directory,
# with the real lib/format.js spliced in so the rendered numbers are the ones
# a reader would see.
_PRELUDE = (
    """
  const palette = new Proxy({}, { get: (_t, key) => '#' + String(key) });
"""
    + FORMAT_JS_IN_TEMPLATE
    + CHART_TYPOGRAPHY_JS_IN_TEMPLATE
    + """
  const useMemo = (fn) => fn();
  const html = (strings, ...values) => ({ strings: Array.from(strings), values });
  function ChartWrapper(props) { return { component: 'ChartWrapper', props }; }
  globalThis.__ChartWrapper = ChartWrapper;
"""
)


def _render_chart(variations_js: str) -> dict[str, object]:
    """Render VariationsChart over `variations_js` and return its datasets."""
    script = f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(_COMPONENTS / "variations-chart.js")!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `{_PRELUDE}\n${{source}}\nexport {{ VariationsChart }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const mod = await import(moduleUri);
        const rendered = mod.VariationsChart({{
          metricLabel: 'Output tok/s', unit: 'tok/s', variations: {variations_js},
        }});
        const data = rendered.values.find(v => v && v.datasets);
        const options = rendered.values.find(v => v?.scales);
        const byLabel = Object.fromEntries(
          (data?.datasets ?? []).map(d => [d.label, d.data]),
        );
        console.log(JSON.stringify({{
          plus: byLabel['mean + std'] ?? null,
          minus: byLabel['mean - std'] ?? null,
          means: byLabel['Output tok/s'] ?? null,
          tooltips: (data?.labels ?? []).map((_l, i) =>
            options.plugins.tooltip.callbacks.label({{
              dataIndex: i, parsed: {{y: {variations_js}[i].mean}},
            }})),
        }}));
    """
    return json.loads(run_node(script))


def _mean_std(values_js: str) -> dict[str, object]:
    """meanStd is module-private; reach it through its only public caller.

    One manifest child per value, each with its own summary, so the metric has
    exactly ``len(values)`` observations to aggregate. An empty list still needs
    one child (an empty manifest short-circuits to ``[]``) but gives it no
    summary, which is the genuine zero-observation case.
    """
    script = f"""
        import {{ buildSweepVariations }} from {_HELPERS.as_uri()!r};
        const values = {values_js};
        const manifest = values.length > 0
          ? values.map((_v, i) => ({{ name: `v0-t${{i}}`, variation_index: 0, trial_index: i }}))
          : [{{ name: 'v0-t0', variation_index: 0, trial_index: 0 }}];
        const childSummaries = Object.fromEntries(
          values.map((v, i) => [`v0-t${{i}}`, {{ summary: {{ m: {{ avg: v }} }} }}]),
        );
        const out = buildSweepVariations({{
          manifest, childSummaries, cells: null, statusRuns: null,
          headlineMetrics: [{{ key: 'm', stat: 'avg', label: 'M', unit: 'x' }}],
        }});
        console.log(JSON.stringify(out[0].perMetric['m.avg']));
    """
    return json.loads(run_node(script))


# ---------------------------------------------------------------------------
# meanStd
# ---------------------------------------------------------------------------


def test_single_trial_reports_unmeasured_spread_not_zero_spread() -> None:
    """`std: 0` asserts a reproducibility result one observation cannot support."""
    stats = _mean_std("[42.25]")

    assert stats["n"] == 1
    assert stats["mean"] == 42.25
    assert stats["std"] is None, (
        "One trial does not estimate spread. `std: 0` claims every trial landed "
        "on the same number, which is a measurement nobody made."
    )
    assert stats["cv"] is None


def test_two_trials_do_estimate_spread() -> None:
    """The n<2 guard must not swallow a real, small spread."""
    stats = _mean_std("[10, 20]")

    assert stats["n"] == 2
    assert stats["mean"] == 15
    assert stats["std"] == 5
    assert stats["cv"] == pytest.approx(1 / 3)


def test_two_identical_trials_report_a_genuine_zero_spread() -> None:
    """Zero stays available for the case that actually measured it."""
    stats = _mean_std("[7, 7]")

    assert stats["n"] == 2
    assert stats["std"] == 0, (
        "Two trials that agreed IS a measured zero. Only the unmeasured case "
        "became null; conflating the two would lose a real result."
    )


def test_no_trials_leaves_every_statistic_absent() -> None:
    stats = _mean_std("[]")

    assert stats == {"mean": None, "std": None, "cv": None, "n": 0}


# ---------------------------------------------------------------------------
# VariationsChart error band
# ---------------------------------------------------------------------------

_ONE_TRIAL = "[{variation_index: 0, label: 'concurrency=17', mean: 1648, std: null, cv: null, n: 1}]"
_THREE_TRIALS = "[{variation_index: 0, label: 'concurrency=17', mean: 1648, std: 33.25, cv: 0.02, n: 3}]"
_MIXED = (
    "[{variation_index: 0, label: 'concurrency=8', mean: 1000, std: 50, cv: 0.05, n: 3},"
    " {variation_index: 1, label: 'concurrency=17', mean: 1648, std: null, cv: null, n: 1}]"
)


def test_single_trial_variation_gets_no_error_band() -> None:
    """A zero-width band is a drawn claim of perfect reproducibility."""
    chart = _render_chart(_ONE_TRIAL)

    assert chart["means"] == [1648], "the mean is still plotted"
    assert chart["plus"] == [None]
    assert chart["minus"] == [None]


def test_multi_trial_variation_keeps_its_error_band() -> None:
    chart = _render_chart(_THREE_TRIALS)

    assert chart["plus"] == [1648 + 33.25]
    assert chart["minus"] == [1648 - 33.25]


def test_band_is_omitted_per_point_not_per_chart() -> None:
    """A sweep with mixed trial counts must not lose the bands it did measure,
    nor gain one where it measured nothing."""
    chart = _render_chart(_MIXED)

    assert chart["plus"] == [1050, None]
    assert chart["minus"] == [950, None]


@pytest.mark.parametrize(
    "variations_js",
    [
        param(
            "[{variation_index: 0, label: 'x', mean: 100, std: 0, cv: null, n: 1}]",
            id="upstream-coerced-std-to-zero",
        ),
        param(
            "[{variation_index: 0, label: 'x', mean: 100, std: 5, cv: 0.05, n: 1}]",
            id="upstream-supplied-std-without-trials",
        ),
        param(
            "[{variation_index: 0, label: 'x', mean: 100, std: 5, cv: 0.05}]",
            id="upstream-omitted-n",
        ),
    ],
)  # fmt: skip
def test_band_stays_absent_when_a_caller_supplies_spread_it_did_not_measure(
    variations_js: str,
) -> None:
    """The chart gates on `n`, not only on `std` being finite.

    `pages/sweep-detail.js` currently maps `std: r?.std ?? 0` on the way into
    this component, which would turn the honest null straight back into a
    zero-width band. Gating on the trial count means the shading cannot
    contradict the tooltip regardless of what a caller sends.
    """
    chart = _render_chart(variations_js)

    assert chart["plus"] == [None]
    assert chart["minus"] == [None]


def test_tooltip_and_band_never_disagree_about_whether_spread_is_known() -> None:
    """Both use the same `n >= 2` test, so one cannot claim what the other denies."""
    chart = _render_chart(_MIXED)
    tooltips = chart["tooltips"]
    assert isinstance(tooltips, list)

    assert "±" in tooltips[0]
    assert "mean of 3 trials" in tooltips[0]
    assert chart["plus"][0] is not None

    assert "±" not in tooltips[1]
    assert "1 trial, spread unknown" in tooltips[1]
    assert chart["plus"][1] is None

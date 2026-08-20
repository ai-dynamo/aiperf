# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sweep winner must follow the sweep's objective, not the chart selector.

Regression for two defects seen together on the `gemma-bo4` adaptive sweep:

1. ``pickSweepWinner`` was called with the chart's currently-selected metric, so
   clicking "ITL avg" in the chart re-crowned the winner as the lowest-ITL
   variation (``search_iter_0000``) instead of the objective's best.
2. Even on the correct metric it ignored ``slaFilters``, so the winner was the
   highest raw throughput regardless of feasibility -- concurrency 309 at
   TTFT p99 18.5s, against a declared constraint of p99 < 500ms.

The real answer for that sweep was concurrency 17.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

HELPERS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)

# Trimmed from the real gemma-bo4 run: (label, tok/s, TTFT p99 ms).
_GEMMA_BO4 = [
    ("search_iter_0000", 139.1, 137.3),
    ("search_iter_0005", 3661.5, 18522.0),  # best throughput, wildly infeasible
    ("search_iter_0008", 1648.0, 465.0),  # the true winner
    ("search_iter_0007", 1740.6, 593.2),  # better tok/s, breaches SLA
]


def _variations_js() -> str:
    return json.dumps(
        [
            {
                "variation_index": i,
                "label": label,
                "perMetric": {
                    "output_token_throughput.avg": {"mean": tps, "cv": None, "n": 1},
                    "time_to_first_token.p99": {"mean": ttft, "cv": None, "n": 1},
                    "inter_token_latency.avg": {"mean": 6.8 + i, "cv": None, "n": 1},
                },
            }
            for i, (label, tps, ttft) in enumerate(_GEMMA_BO4)
        ]
    )


def _run(expr: str) -> dict | None:
    script = f"""
        import {{ pickObjectiveWinner, pickSweepWinner, isVariationFeasible }}
          from {HELPERS.as_uri()!r};
        const variations = {_variations_js()};
        console.log(JSON.stringify({expr}));
    """
    return json.loads(run_node(script))


def test_objective_winner_respects_sla_constraint() -> None:
    """conc-17 wins, not the higher-throughput but infeasible points."""
    winner = _run(
        "pickObjectiveWinner({ variations, "
        "objectives: [{metric:'output_token_throughput', stat:'avg', direction:'maximize'}], "
        "slaFilters: [{metricTag:'time_to_first_token', stat:'p99', op:'lt', threshold:500}] })"
    )
    assert winner is not None
    assert winner["label"] == "search_iter_0008"
    assert winner["mean"] == 1648.0
    assert winner["higherIsBetter"] is True
    assert winner["constrained"] is True
    assert winner["feasibleCount"] == 2


def test_without_sla_filters_the_infeasible_peak_wins() -> None:
    """Pins that the SLA filter -- not luck -- is what excludes the peak."""
    winner = _run(
        "pickObjectiveWinner({ variations, "
        "objectives: [{metric:'output_token_throughput', stat:'avg', direction:'maximize'}], "
        "slaFilters: [] })"
    )
    assert winner["label"] == "search_iter_0005"


def test_chart_metric_ranking_reproduces_the_reported_bug() -> None:
    """The old path really does crown search_iter_0000 when the chart is on ITL."""
    winner = _run(
        "pickSweepWinner({ variations, metricKey: 'inter_token_latency.avg' })"
    )
    assert winner["label"] == "search_iter_0000"


def test_no_objectives_returns_null_so_callers_fall_back() -> None:
    """Grid/zip/scenario sweeps declare no objective; caller keeps old behaviour."""
    assert (
        _run("pickObjectiveWinner({ variations, objectives: null, slaFilters: null })")
        is None
    )


def test_missing_constrained_metric_is_infeasible_not_feasible() -> None:
    """An unmeasured constraint must not read as a satisfied one."""
    feasible = _run(
        "isVariationFeasible({ perMetric: { 'output_token_throughput.avg': {mean: 10} } }, "
        "[{metricTag:'time_to_first_token', stat:'p99', op:'lt', threshold:500}])"
    )
    assert feasible is False


def _fmt(values_js: str):
    script = f"""
        import {{ formatVariationValues }} from {HELPERS.as_uri()!r};
        console.log(JSON.stringify(formatVariationValues({values_js})));
    """
    return json.loads(run_node(script))


class TestVariationValueLabels:
    """Adaptive variations must read as what they tried, not `search_iter_NNNN`.

    The planner's label IS the artifact-path cell identity, so it cannot be
    renamed -- but on its own it tells a reader nothing. `status.runs[].values`
    carries the swept parameters and was simply never surfaced.
    """

    def test_json_string_form_from_the_cr_is_parsed(self) -> None:
        assert _fmt(r"'{\"phases.profiling.concurrency\":17}'") == "concurrency=17"

    def test_dotted_path_is_shortened_to_its_leaf(self) -> None:
        """The `phases.profiling.` prefix is identical on every row; drop it."""
        assert _fmt(r"{'phases.profiling.rate': 12.5}") == "rate=12.5"

    def test_multi_dimensional_values_are_joined(self) -> None:
        got = _fmt(r"{'phases.profiling.concurrency': 8, 'phases.profiling.rate': 100}")
        assert got == "concurrency=8, rate=100"

    def test_unparseable_or_empty_returns_null_so_caller_falls_back(self) -> None:
        assert _fmt("null") is None
        assert _fmt("{}") is None
        assert _fmt("'[1,2]'") is None

    def test_variations_get_values_label_from_status_runs(self) -> None:
        script = f"""
            import {{ buildSweepVariations }} from {HELPERS.as_uri()!r};
            const variations = buildSweepVariations({{
              manifest: [{{ name: 'bo-v08', variationIndex: 8, variationLabel: 'search_iter_0008' }}],
              childSummaries: {{ 'bo-v08': {{ summary: {{ output_token_throughput: {{ avg: 1648 }} }} }} }},
              cells: null,
              statusRuns: [{{ index: 8, label: 'search_iter_0008',
                             values: '{{"phases.profiling.concurrency":17}}' }}],
            }});
            console.log(JSON.stringify(variations));
        """
        variations = json.loads(run_node(script))
        assert variations[0]["valuesLabel"] == "concurrency=17"
        # The planner id is retained -- it is still the artifact-path identity.
        assert variations[0]["label"] == "search_iter_0008"

    def test_missing_status_runs_leaves_values_label_null(self) -> None:
        script = f"""
            import {{ buildSweepVariations }} from {HELPERS.as_uri()!r};
            console.log(JSON.stringify(buildSweepVariations({{
              manifest: [{{ name: 'bo-v00', variationIndex: 0, variationLabel: 'search_iter_0000' }}],
              childSummaries: {{}}, cells: null, statusRuns: null,
            }})));
        """
        variations = json.loads(run_node(script))
        assert variations[0]["valuesLabel"] is None

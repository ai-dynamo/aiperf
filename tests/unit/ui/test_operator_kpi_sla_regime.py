# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The headline KPI tiles must declare which SLA regime their number is from.

The defect these cover: the sweep-detail KPI row headlined the unconstrained
peak (3,842 tok/s at concurrency=9, TTFT p99 18,380 ms against a declared limit
of 500) in a larger, gold, award-toned font directly beside a
feasibility-filtered winner card reading 1,648 tok/s. Both numbers were correct;
nothing on the page said they answered different questions.

The fix labels rather than filters, so these tests pin two things that a
source grep cannot distinguish: that a breach is announced when there IS a
checkable constraint, and that NOTHING is announced when there is not -- an
unconstrained sweep must not gain a feasibility claim it never earned.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_SWEEP_DETAIL_JS = _UI / "pages" / "sweep-detail.js"
_KPI_CARD_JS = _UI / "components" / "kpi-card.js"


def _slice(path: Path, marker: str) -> str:
    """Return the fenced dependency-free block from a UI module."""
    source = path.read_text(encoding="utf-8")
    begin, end = f"// aiperf:{marker}:begin", f"// aiperf:{marker}:end"
    assert begin in source and end in source, (
        f"{path.name} must keep the {marker} sentinel comments; they are what "
        "makes this logic testable without a bundler."
    )
    block = source.split(begin, 1)[1].split(end, 1)[0]
    assert "import " not in block, (
        f"the {marker} block must stay dependency-free so node can evaluate it"
    )
    return block


def _eval_sweep(body: str) -> object:
    return json.loads(
        run_node(_slice(_SWEEP_DETAIL_JS, "sweep-detail-pure") + "\n" + body)
    )


def _eval_kpi_card(body: str) -> object:
    return json.loads(run_node(_slice(_KPI_CARD_JS, "kpi-card-pure") + "\n" + body))


# A variation shaped the way ``buildSweepVariations`` emits them: ``perMetric``
# keyed ``"<metric>.<stat>"`` with a ``mean``.
def _variation(
    index: int, label: str, throughput: float, ttft_p99: float | None
) -> dict:
    per_metric: dict[str, dict[str, float]] = {
        "output_token_throughput.avg": {"mean": throughput, "cv": None, "n": 1},
    }
    if ttft_p99 is not None:
        per_metric["time_to_first_token.p99"] = {"mean": ttft_p99, "cv": None, "n": 1}
    return {
        "variation_index": index,
        "label": label,
        "valuesLabel": label,
        "perMetric": per_metric,
    }


_TTFT_UNDER_500 = {
    "metricTag": "time_to_first_token",
    "stat": "p99",
    "op": "lt",
    "threshold": 500.0,
}

# The gemma-bo4 shape: the throughput peak is also the worst TTFT offender.
_GEMMA_VARIATIONS = [
    _variation(0, "concurrency=4", 136.0, 141.5),
    _variation(5, "concurrency=9", 3842.0, 18380.0),
    _variation(8, "concurrency=12", 1632.0, 465.0),
]


def _claim(**kwargs: object) -> dict:
    return _eval_sweep(
        f"console.log(JSON.stringify(slaClaimState({json.dumps(kwargs)})));"
    )


# ---------------------------------------------------------------------------
# slaClaimState -- what the page is entitled to say.
# ---------------------------------------------------------------------------


def test_a_sweep_with_no_constraints_earns_no_feasibility_claim() -> None:
    # sla_filter_count == 0 makes every `feasible` flag in the API response
    # vacuously true (sweeps_models.py:401-412). Rendering "meets SLA" off a
    # vacuous flag advertises an SLA the run never had.
    claim = _claim(slaFilters=None, slaFilterCount=0, variations=_GEMMA_VARIATIONS)

    assert claim["state"] == "off"
    assert claim["filters"] == []


def test_an_explicit_zero_filter_count_overrides_a_spec_that_declares_filters() -> None:
    # The count is what the SEARCH applied at trial-scoring time. A spec that
    # declares constraints the search never enforced did not constrain anything.
    claim = _claim(
        slaFilters=[_TTFT_UNDER_500],
        slaFilterCount=0,
        variations=_GEMMA_VARIATIONS,
    )

    assert claim["state"] == "off"


def test_a_grid_sweep_with_neither_filters_nor_a_search_summary_stays_silent() -> None:
    # gemma-conc2 / cp-sweep: sweep_type grid, sla_filters null, no
    # search_summary at all, so slaFilterCount arrives as null rather than 0.
    claim = _claim(slaFilters=None, slaFilterCount=None, variations=_GEMMA_VARIATIONS)

    assert claim["state"] == "off"


def test_a_constrained_run_whose_filter_definitions_were_lost_says_so_instead_of_guessing() -> (
    None
):
    # `spec_summary.sla_filters` is legitimately null on archives written before
    # the field existed. The run WAS constrained; this page just cannot check it.
    claim = _claim(slaFilters=None, slaFilterCount=2, variations=_GEMMA_VARIATIONS)

    assert claim["state"] == "unevaluable"
    assert claim["reason"] == "definitions"
    assert claim["declaredCount"] == 2


def test_a_constraint_on_an_uncollected_metric_is_unevaluable_not_universally_infeasible() -> (
    None
):
    # `isVariationFeasible` treats a missing metric as infeasible, which is right
    # for picking a winner and catastrophic for a tile: a constraint on a metric
    # outside HEADLINE_METRICS would stamp "breaches SLA" on every tile of every
    # sweep, which is a fabrication, not a caution.
    claim = _claim(
        slaFilters=[{**_TTFT_UNDER_500, "metricTag": "energy_per_token"}],
        slaFilterCount=1,
        variations=_GEMMA_VARIATIONS,
    )

    assert claim["state"] == "unevaluable"
    assert claim["reason"] == "unmeasured"


def test_a_measurable_constraint_is_active() -> None:
    claim = _claim(
        slaFilters=[_TTFT_UNDER_500],
        slaFilterCount=1,
        variations=_GEMMA_VARIATIONS,
    )

    assert claim["state"] == "active"
    assert claim["filters"] == [_TTFT_UNDER_500]


def test_a_malformed_filter_is_dropped_rather_than_evaluated() -> None:
    # An unknown operator or a non-numeric threshold cannot be checked and must
    # not be silently treated as satisfied.
    claim = _claim(
        slaFilters=[{**_TTFT_UNDER_500, "op": "approximately"}],
        slaFilterCount=1,
        variations=_GEMMA_VARIATIONS,
    )

    assert claim["state"] == "unevaluable"
    assert claim["reason"] == "definitions"


@pytest.mark.parametrize(
    ("op", "symbol"),
    [
        param("lt", "<", id="lt"),
        param("le", "≤", id="le"),
        param("gt", ">", id="gt"),
        param("ge", "≥", id="ge"),
    ],
)  # fmt: skip
def test_every_declared_operator_renders_a_symbol(op: str, symbol: str) -> None:
    rendered = _eval_sweep(
        "console.log(JSON.stringify(formatSlaFilter("
        + json.dumps({**_TTFT_UNDER_500, "op": op})
        + ")));"
    )

    assert rendered == f"time_to_first_token p99 {symbol} 500"


def test_the_filter_metric_key_matches_the_default_isVariationFeasible_uses() -> None:
    # sweep-detail-helpers.js:367 defaults a missing stat to p95. Reading a
    # different stat here would quote an observed number that had nothing to do
    # with the verdict printed beside it.
    key = _eval_sweep(
        "console.log(JSON.stringify(slaFilterMetricKey({metricTag: 'time_to_first_token'})));"
    )

    assert key == "time_to_first_token.p95"


# ---------------------------------------------------------------------------
# kpiSlaAnnotation -- what the tile actually says.
# ---------------------------------------------------------------------------


def _annotate(**kwargs: object) -> dict:
    return _eval_sweep(
        f"console.log(JSON.stringify(kpiSlaAnnotation({json.dumps(kwargs)})));"
    )


def test_an_unconstrained_tile_gains_no_note_and_keeps_its_tooltip() -> None:
    note = _annotate(
        claim={"state": "off", "filters": [], "declaredCount": 0},
        feasible=None,
        observedText=[],
        alternative=None,
        attributionTitle="concurrency=9 · search_iter_0005",
    )

    assert note == {
        "note": None,
        "noteTone": None,
        "tone": None,
        "title": "concurrency=9 · search_iter_0005",
    }


def test_an_infeasible_peak_is_announced_demoted_and_explained() -> None:
    note = _annotate(
        claim={"state": "active", "filters": [_TTFT_UNDER_500], "declaredCount": 1},
        feasible=False,
        observedText=["time_to_first_token p99 = 18,380.0"],
        alternative={"valueText": "1,632 tok/s", "attribution": "concurrency=12"},
        attributionTitle="concurrency=9 · search_iter_0005",
    )

    assert note["note"] == "breaches SLA"
    assert note["noteTone"] == "bad"
    # `warn` overrides the gold "award" tone the lead throughput tile otherwise
    # gets. A rejected operating point rendered as a trophy is most of what made
    # this tile read as the sweep's answer.
    assert note["tone"] == "warn"
    assert "time_to_first_token p99 < 500" in note["title"]
    assert "observed time_to_first_token p99 = 18,380.0" in note["title"]
    assert "1,632 tok/s at concurrency=12" in note["title"]
    # The alternative is scoped: it walks this page's per-variation means, while
    # the winner card ranks the planner's recorded objective values.
    assert "Of the variations charted here" in note["title"]


def test_an_infeasible_peak_with_no_feasible_alternative_says_none_exists() -> None:
    note = _annotate(
        claim={"state": "active", "filters": [_TTFT_UNDER_500], "declaredCount": 1},
        feasible=False,
        observedText=["time_to_first_token p99 = 18,380.0"],
        alternative=None,
        attributionTitle="concurrency=9",
    )

    assert "No variation charted here satisfied the SLA" in note["title"]


def test_a_feasible_extremum_is_marked_positively_without_changing_its_tone() -> None:
    note = _annotate(
        claim={"state": "active", "filters": [_TTFT_UNDER_500], "declaredCount": 1},
        feasible=True,
        observedText=["time_to_first_token p99 = 427.1"],
        alternative=None,
        attributionTitle="concurrency=13 · search_iter_0009",
    )

    assert note["note"] == "meets SLA"
    assert note["noteTone"] == "ok"
    assert note["tone"] is None


def test_an_unevaluable_tile_states_that_it_is_unfiltered_rather_than_implying_it_is() -> (
    None
):
    note = _annotate(
        claim={
            "state": "unevaluable",
            "filters": [],
            "declaredCount": 2,
            "reason": "definitions",
        },
        feasible=None,
        observedText=[],
        alternative=None,
        attributionTitle="concurrency=9",
    )

    assert note["note"] is None
    assert note["tone"] is None
    assert "2 SLA constraints" in note["title"]
    assert "NOT filtered for feasibility" in note["title"]


def test_an_unevaluable_tile_singularises_a_lone_constraint() -> None:
    note = _annotate(
        claim={
            "state": "unevaluable",
            "filters": [_TTFT_UNDER_500],
            "declaredCount": 1,
            "reason": "unmeasured",
        },
        feasible=None,
        observedText=[],
        alternative=None,
        attributionTitle=None,
    )

    assert "1 SLA constraint," in note["title"]
    assert "not among the metrics this page collects" in note["title"]


# ---------------------------------------------------------------------------
# Attribution width -- a caption that overflows its tile attributes one tile's
# variation to its neighbour.
# ---------------------------------------------------------------------------


def test_a_grid_variation_label_is_shortened_to_its_leaf_dimension() -> None:
    # Grid sweeps carry no `valuesLabel`, so the tile falls back to
    # `variation_label`, which is the full dotted path. 31 characters do not fit
    # a ~134px caption; the prefix is identical on every variation anyway.
    result = _eval_sweep(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 3, label: 'phases.profiling.concurrency=64', valuesLabel: null,
        })));
        """
    )

    assert result["text"] == "concurrency=64"
    # The unshortened label stays recoverable: it is what the artifact paths and
    # the children manifest are keyed by.
    assert result["title"] == "phases.profiling.concurrency=64"


def test_shortening_leaves_a_dotted_value_alone() -> None:
    # Only the key side of an `=` is a dimension path. A float or a model path on
    # the value side must survive intact.
    result = _eval_sweep(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 1, label: 'phases.profiling.request_rate=2.5', valuesLabel: null,
        })));
        """
    )

    assert result["text"] == "request_rate=2.5"


def test_shortening_handles_a_multi_dimension_label() -> None:
    result = _eval_sweep(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 1,
          label: 'phases.profiling.concurrency=8, endpoint.streaming=true',
          valuesLabel: null,
        })));
        """
    )

    assert result["text"] == "concurrency=8, streaming=true"


def test_shortening_leaves_a_planner_cell_id_untouched() -> None:
    result = _eval_sweep(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 5, label: 'search_iter_0005', valuesLabel: null,
        })));
        """
    )

    assert result["text"] == "search_iter_0005"


# ---------------------------------------------------------------------------
# kpiValueFontClass -- the tile must shrink a long value, never clip it.
# ---------------------------------------------------------------------------

# The named steps, largest first. The helper returns a class; style.css turns it
# into a size, so "steps down" is only true if the two agree.
_FONT_LADDER = (
    "metric-val--hero",
    "metric-val--legacy",
    "metric-val--secondary",
    "metric-val--compact",
)


def _font_class_sizes() -> dict[str, float]:
    """Resolve each ladder class to its px size through style.css."""
    css = (_UI / "style.css").read_text(encoding="utf-8")
    tokens = {
        name: float(rem) * 16
        for name, rem in re.findall(r"(--font-size-[a-z0-9]+): ([\d.]+)rem;", css)
    }
    sizes = {}
    for cls in _FONT_LADDER:
        match = re.search(
            rf"\.{re.escape(cls)} {{ font-size: var\((--[a-z0-9-]+)\)", css
        )
        assert match, f"{cls} must have a font-size rule in style.css"
        sizes[cls] = tokens[match.group(1)]
    return sizes


def test_the_font_ladder_actually_descends() -> None:
    """The helper's contract is "step DOWN one named level"; that is a claim
    about style.css, not about the helper, and nothing else checks it."""
    sizes = [_font_class_sizes()[cls] for cls in _FONT_LADDER]

    assert sizes == sorted(sizes, reverse=True), sizes
    assert len(set(sizes)) == len(sizes), sizes


@pytest.mark.parametrize(
    ("value", "variant", "expected"),
    [
        # A 160px `.kpi-row` track leaves the hero value ~86px, about six
        # characters at 30px. Six or fewer keeps the base size, so every
        # short-valued tile on every page is untouched.
        param("14", "hero", "metric-val--hero", id="short-hero-unchanged"),
        param("2158", "hero", "metric-val--hero", id="four-chars-unchanged"),
        param("3,842", "hero", "metric-val--hero", id="five-chars-unchanged"),
        param("82,215", "hero", "metric-val--hero", id="six-chars-unchanged"),
        param("2,158.9", "hero", "metric-val--legacy", id="seven-chars-steps-down"),
        param("123456789", "hero", "metric-val--legacy", id="nine-chars"),
        param("1,234,567.8", "hero", "metric-val--legacy", id="eleven-chars"),
        param("2,158.9", "legacy", "metric-val--secondary", id="legacy-base"),
        # The secondary variant already starts two steps down, so it lands on
        # the smallest named size rather than continuing past it.
        param("2,158.9", "secondary", "metric-val--compact", id="secondary-hits-floor"),
    ],
)  # fmt: skip
def test_value_font_steps_down_with_length(
    value: str, variant: str, expected: str
) -> None:
    got = _eval_kpi_card(
        f"console.log(JSON.stringify(kpiValueFontClass({json.dumps(value)}, {json.dumps(variant)})));"
    )

    assert got == expected
    if expected != _FONT_LADDER[0]:
        sizes = _font_class_sizes()
        base = {"hero": "metric-val--hero", "legacy": "metric-val--legacy"}.get(
            variant, "metric-val--secondary"
        )
        assert sizes[got] < sizes[base]


def test_value_font_never_collapses_below_the_sub_line() -> None:
    # `.metric-sub` is the tile's smallest text. A value that shrank to or below
    # it would stop reading as the tile's headline, which is a different failure
    # from clipping but just as misleading. The ladder bottoms out instead.
    got = _eval_kpi_card(
        "console.log(JSON.stringify(kpiValueFontClass('9'.repeat(120), 'secondary')));"
    )

    assert got == _FONT_LADDER[-1]
    assert _font_class_sizes()[got] >= 15


def test_value_font_tolerates_a_numeric_or_absent_value() -> None:
    got = _eval_kpi_card(
        "console.log(JSON.stringify([kpiValueFontClass(42, 'hero'), "
        "kpiValueFontClass(null, 'hero'), kpiValueFontClass('1,234,567.8', 'nonsense')]));"
    )

    # Unknown variants fall back to the hero base rather than throwing.
    assert got == ["metric-val--hero", "metric-val--hero", "metric-val--legacy"]


# ---------------------------------------------------------------------------
# Wiring -- the model above only matters if the page and the card use it.
# ---------------------------------------------------------------------------


def test_the_page_computes_feasibility_from_the_shared_helper() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    # Reimplementing the comparators here would let the tiles and the winner
    # card disagree about who passes; the op->symbol map is presentation only.
    assert "isVariationFeasible" in source
    assert "from './sweep-detail-helpers.js'" in source
    assert "const claim = slaClaimState({" in source


def test_the_page_treats_a_missing_search_summary_as_no_opinion_not_as_zero() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert (
        "detail?.search_summary\n        ? detail.search_summary.sla_filter_count"
        in source
    )
    assert ": null," in source


def test_the_tiles_render_the_sla_note_and_its_tone() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert "k.slaNote &&" in source
    assert "kpi-sla-note-" in source
    # A breach note has to read as a breach: the note carries its own tone
    # colour rather than inheriting the tile's.
    assert "k.slaNoteTone === 'bad' ? palette.amber" in source

    # The other half of the same defect: an infeasible peak must not keep an
    # award colouring. The headline tiles pass no `tone` at all now, so there
    # is no award tone left to contradict the note.
    tiles = source.split("${headlineKpis.map((k, i) => {", 1)[1].split("})}", 1)[0]
    assert "tone=" not in tiles


def test_the_card_clamps_its_sub_line_and_scales_its_value() -> None:
    source = _KPI_CARD_JS.read_text(encoding="utf-8")

    assert "text-overflow:ellipsis" in source
    assert "kpiValueFontClass(shown, variant)" in source
    # The rich sub-line must be a sibling of the icon+value row, not a child of
    # the body column, or it inherits the ~86px width that ellipsed away even
    # "concurrency=9".
    body_close = source.index(
        '${sub && html`<div class="metric-sub" style=${richSubStyle}'
    )
    assert source.index('class="metric-card__body"') < body_close
    assert source.rindex("</div>\n      </div>", 0, body_close) > 0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The swept parameter values must survive from the manifest to every surface.

Adaptive planners name variations ``search_iter_NNNN``. That string is the
artifact-path cell identity and cannot be renamed, but it describes nothing, so
each sweep surface has to lead with the values the variation actually tried and
demote the identity. These tests pin that contract for the two surfaces fed by
the children manifest -- the Live Variations card and the live trial board --
plus the shared resolver both go through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

_UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
_HELPERS = _UI_DIR / "pages" / "sweep-detail-helpers.js"
_COMPONENTS = _UI_DIR / "components"
_LIVE_HELPERS = _COMPONENTS / "live-variations-helpers.js"

# Enough of the real `lib/theme.js` and `lib/format.js` to render; the harness
# strips the components' imports, mirroring the approach in
# test_operator_sweep_variation_presentation.py.
#
# `live-variations-helpers.js` is deliberately NOT stubbed here -- it is
# re-imported for real below. It owns `parseVariationValues`, whose guards are
# the subject of several tests in this file; a stub of the code under test
# would assert nothing.
_PRELUDE = """
  const palette = new Proxy({}, { get: (_t, key) => '#' + String(key) });
  const fmtNumber = (value, decimals = 1, fallback = '---') =>
    value == null || typeof value !== 'number' || !isFinite(value)
      ? fallback
      : value.toFixed(decimals);
  const useState = (initial) => [initial, () => {}];
  const navigate = () => {};
  const buildJobPath = () => '';
  const html = (strings, ...values) => ({ strings: Array.from(strings), values });
"""

# `html` above returns a plain {strings, values} tree rather than VDOM; this
# walks it back into the text a reader would see, which is what the ordering
# assertions are about.
_FLATTEN = """
  const flatten = (node) => Array.isArray(node)
    ? node.map(flatten).join('')
    : (node && node.strings)
      ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
      : String(node ?? '');
"""

_ADAPTIVE_MANIFEST = json.dumps(
    [
        {
            "name": "sweep-v08-t0",
            "namespace": "bench",
            "variation_index": 8,
            "variation_label": "search_iter_0008",
            "variation_values": '{"phases.profiling.concurrency":17}',
            "trial_index": 0,
        }
    ]
)

# The disk/archived read path returns camelCase (ChildrenManifestEntry uses
# to_camel), the live CR path snake_case. Both must resolve.
_ADAPTIVE_MANIFEST_CAMEL = json.dumps(
    [
        {
            "name": "sweep-v08-t0",
            "namespace": "bench",
            "variationIndex": 8,
            "variationLabel": "search_iter_0008",
            "variationValues": '{"phases.profiling.concurrency":17}',
            "trialIndex": 0,
        }
    ]
)

_GRID_MANIFEST = json.dumps(
    [
        {
            "name": "sweep-v00-t0",
            "namespace": "bench",
            "variation_index": 0,
            "variation_label": "benchmark.phases.profiling.concurrency-32",
            "trial_index": 0,
        }
    ]
)


def _eval_helpers(body: str) -> object:
    script = f"""
        import {{
          buildSweepVariations, buildTrialBoardRows, formatVariationValues,
          indexVariationValues,
        }} from {_HELPERS.as_uri()!r};
        {body}
    """
    return json.loads(run_node(script))


def _render_component(
    component: str, exports: list[str], body: str, *, import_helpers: bool = False
) -> object:
    """Load one component with imports stripped and evaluate `body` against it."""
    helper_import = (
        f"import {{ buildTrialBoardRows }} from {_HELPERS.as_uri()!r};"
        if import_helpers
        else ""
    )
    # Restored for real rather than stubbed: these are the card's own pure
    # helpers, and `parseVariationValues` is exercised directly below.
    live_helper_import = (
        "import { parseVariationValues, titleCase, trialContributesMetrics } "
        f"from {_LIVE_HELPERS.as_uri()!r};"
    )
    script = f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(_COMPONENTS / component)!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `{helper_import}\n{live_helper_import}\n{_PRELUDE}\n${{source}}\nexport {{ {", ".join(exports)} }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {_FLATTEN}
        {body}
    """
    return json.loads(run_node(script))


# ---------------------------------------------------------------------------
# Shared resolver
# ---------------------------------------------------------------------------


def test_manifest_values_resolve_without_status_runs() -> None:
    """status.runs[] is only appended at child terminal phase, so a live sweep
    must resolve values from the manifest alone."""
    resolved = _eval_helpers(
        f"""
        const index = indexVariationValues({{
          manifest: {_ADAPTIVE_MANIFEST}, statusRuns: null,
        }});
        console.log(JSON.stringify([...index].map(([k, v]) => [k, v.valuesLabel])));
        """
    )

    assert resolved == [[8, "concurrency=17"]]


def test_manifest_wins_over_status_runs_because_it_is_less_truncated() -> None:
    """Both decode the same annotation, but the manifest budget is 2048 bytes
    against status.runs' 256, so the manifest copy is preferred."""
    resolved = _eval_helpers(
        f"""
        const index = indexVariationValues({{
          manifest: {_ADAPTIVE_MANIFEST},
          statusRuns: [{{ index: 8, values: '{{"__aiperf_truncated__":true}}' }}],
        }});
        console.log(JSON.stringify(index.get(8).valuesLabel));
        """
    )

    assert resolved == "concurrency=17"


def test_status_runs_still_fills_gaps_for_archives_without_manifest_values() -> None:
    resolved = _eval_helpers(
        f"""
        const index = indexVariationValues({{
          manifest: {_GRID_MANIFEST},
          statusRuns: [{{ index: 0, values: '{{"phases.profiling.concurrency":32}}' }}],
        }});
        console.log(JSON.stringify(index.get(0).valuesLabel));
        """
    )

    assert resolved == "concurrency=32"


@pytest.mark.parametrize(
    "raw",
    [
        param('\'{"__aiperf_truncated__":true,"limitBytes":256}\'', id="truncated"),
        param("'{\"concurrency\":null}'", id="null-value"),
        param("'{}'", id="empty-object"),
        param("''", id="empty-string"),
        param("null", id="missing"),
        param("'[1,2]'", id="array"),
        param('\'{"variables.tuning":{"a":1}}\'', id="only-nested-values"),
    ],
)  # fmt: skip
def test_format_variation_values_never_emits_a_half_formed_label(raw: str) -> None:
    """A partial descriptor is worse than the planner id: it looks authoritative
    while naming a parameter that was never applied."""
    assert (
        _eval_helpers(f"console.log(JSON.stringify(formatVariationValues({raw})))")
        is None
    )


def test_format_variation_values_keeps_scalars_alongside_a_nested_value() -> None:
    """A nested object has no honest one-line rendering, so it is skipped -- but
    it must not suppress the scalars that do describe the operating point."""
    assert (
        _eval_helpers(
            "console.log(JSON.stringify(formatVariationValues("
            '\'{"variables.tuning":{"a":1},"phases.profiling.concurrency":17}\')))'
        )
        == "concurrency=17"
    )


# ---------------------------------------------------------------------------
# buildTrialBoardRows -- same contract as buildSweepVariations
# ---------------------------------------------------------------------------


def test_trial_board_rows_carry_values_label_like_sweep_variations() -> None:
    rows, variations = _eval_helpers(
        f"""
        const rows = buildTrialBoardRows({{
          manifest: {_ADAPTIVE_MANIFEST}, childSummaries: {{}},
        }});
        const variations = buildSweepVariations({{
          manifest: {_ADAPTIVE_MANIFEST}, childSummaries: {{}}, cells: null,
          statusRuns: null,
        }});
        console.log(JSON.stringify([
          rows.map(r => [r.label, r.valuesLabel]),
          variations.map(v => [v.label, v.valuesLabel]),
        ]));
        """
    )

    assert rows == [["search_iter_0008", "concurrency=17"]]
    assert rows == variations


def test_trial_board_rows_accept_the_camel_case_archived_shape() -> None:
    rows = _eval_helpers(
        f"""
        console.log(JSON.stringify(buildTrialBoardRows({{
          manifest: {_ADAPTIVE_MANIFEST_CAMEL}, childSummaries: {{}},
        }}).map(r => r.valuesLabel)));
        """
    )

    assert rows == ["concurrency=17"]


def test_trial_board_rows_leave_values_label_null_for_older_archives() -> None:
    rows = _eval_helpers(
        f"""
        console.log(JSON.stringify(buildTrialBoardRows({{
          manifest: {_GRID_MANIFEST}, childSummaries: {{}},
        }}).map(r => [r.label, r.valuesLabel])));
        """
    )

    assert rows == [["benchmark.phases.profiling.concurrency-32", None]]


# ---------------------------------------------------------------------------
# SweepLiveTrialBoard
# ---------------------------------------------------------------------------


def _board_text(manifest: str) -> str:
    return _render_component(
        "sweep-live-trial-board.js",
        ["SweepLiveTrialBoard"],
        f"""
        const rendered = helpers.SweepLiveTrialBoard({{
          manifest: {manifest}, childSummaries: {{}},
        }});
        console.log(JSON.stringify(flatten(rendered).replace(/\\s+/g, ' ')));
        """,
        import_helpers=True,
    )  # type: ignore[return-value]


def test_trial_board_leads_with_values_and_keeps_the_cell_id_reachable() -> None:
    text = _board_text(_ADAPTIVE_MANIFEST)

    assert "concurrency=17" in text
    # The id is the artifact directory name, so it must stay on screen --
    # demoted, not deleted.
    assert "search_iter_0008" in text
    assert "Cell identity assigned by the sweep planner" in text
    assert text.index("concurrency=17") < text.rindex("search_iter_0008")


def test_trial_board_does_not_print_the_label_twice_for_grid_sweeps() -> None:
    """Grid labels are already descriptive; showing the same string as both
    headline and demoted id is noise."""
    text = _board_text(_GRID_MANIFEST)

    assert "benchmark.phases.profiling.concurrency-32" in text
    assert "Cell identity assigned by the sweep planner" not in text


# ---------------------------------------------------------------------------
# LiveVariationsCard
# ---------------------------------------------------------------------------


def _card(manifest: str, exports: list[str], body: str) -> object:
    return _render_component("live-variations-card.js", exports, body)


def test_card_chips_come_from_values_when_the_label_is_a_planner_counter() -> None:
    groups = _card(
        _ADAPTIVE_MANIFEST,
        ["groupVariations"],
        f"""
        console.log(JSON.stringify(helpers.groupVariations(
          {_ADAPTIVE_MANIFEST}, {{}}, null,
        ).map(g => ({{ chips: g.chips, fromValues: g.fromValues }}))));
        """,
    )

    assert groups == [
        {"chips": [{"name": "Concurrency", "value": "17"}], "fromValues": True}
    ]


def test_card_falls_back_to_label_parsing_for_grid_sweeps() -> None:
    groups = _card(
        _GRID_MANIFEST,
        ["groupVariations"],
        f"""
        console.log(JSON.stringify(helpers.groupVariations(
          {_GRID_MANIFEST}, {{}}, null,
        ).map(g => ({{ chips: g.chips, fromValues: g.fromValues }}))));
        """,
    )

    assert groups == [
        {"chips": [{"name": "Concurrency", "value": "32"}], "fromValues": False}
    ]


@pytest.mark.parametrize(
    "raw",
    [
        param('\'{"__aiperf_truncated__":true}\'', id="truncated"),
        param("'garbage'", id="unparseable"),
        param("'{\"concurrency\":null}'", id="null-value"),
        param('\'{"variables.tuning":{"a":1}}\'', id="only-nested-values"),
        param("'[1,2]'", id="array"),
        param("''", id="empty-string"),
        param("null", id="missing"),
    ],
)  # fmt: skip
def test_card_value_parsing_yields_no_partial_chip(raw: str) -> None:
    """Read from the module that OWNS the chips, not through the card.

    `parseVariationValues` moved to `live-variations-helpers.js` when the three
    parallel implementations of "describe a variation by its swept values" were
    collapsed onto `sweptValueEntries`. Asserting against the owner keeps this
    test honest about where the guards live; that the card uses them is now
    structural (it imports the function) rather than something to re-verify.
    """
    chips = json.loads(
        run_node(
            f"""
            import {{ parseVariationValues }} from {_LIVE_HELPERS.as_uri()!r};
            console.log(JSON.stringify(parseVariationValues({raw})));
            """
        )
    )

    assert chips == []


@pytest.mark.parametrize(
    "raw",
    [
        param('\'{"__aiperf_truncated__":true}\'', id="truncated"),
        param("'garbage'", id="unparseable"),
        param("'{\"concurrency\":null}'", id="null-value"),
        param('\'{"variables.tuning":{"a":1}}\'', id="only-nested-values"),
        param('\'{"phases.profiling.concurrency":17}\'', id="scalar"),
        param('\'{"phases.profiling.":17}\'', id="trailing-dot-path"),
        param('\'{"a.b":1,"c.d":2}\'', id="multiple-scalars"),
    ],
)  # fmt: skip
def test_chips_and_display_string_never_disagree_on_which_parameters_show(
    raw: str,
) -> None:
    """One concept, one implementation.

    The chips and the display string may differ in presentation -- a title-cased
    name in its own element vs. `leaf=value` inline -- but never in WHICH
    parameters they describe. Three implementations of this used to exist and
    disagreed on exactly the inputs above.
    """
    out = json.loads(
        run_node(
            f"""
            import {{ parseVariationValues }} from {_LIVE_HELPERS.as_uri()!r};
            import {{ formatVariationValues }} from {_HELPERS.as_uri()!r};
            const chips = parseVariationValues({raw});
            console.log(JSON.stringify({{
              chipCount: chips.length,
              chipValues: chips.map(c => c.value),
              label: formatVariationValues({raw}),
            }}));
            """
        )
    )

    if out["chipCount"] == 0:
        assert out["label"] is None
    else:
        assert out["label"] is not None
        assert out["label"].split(", ") == [
            f"{part.split('=')[0]}={value}"
            for part, value in zip(
                out["label"].split(", "), out["chipValues"], strict=True
            )
        ]


def test_card_renders_values_first_and_the_planner_id_underneath() -> None:
    text = _card(
        _ADAPTIVE_MANIFEST,
        ["LiveVariationsCard"],
        f"""
        const rendered = helpers.LiveVariationsCard({{
          manifest: {_ADAPTIVE_MANIFEST}, childData: {{}},
        }});
        console.log(JSON.stringify(flatten(rendered).replace(/\\s+/g, ' ')));
        """,
    )

    assert isinstance(text, str)
    assert "Concurrency" in text
    assert "search_iter_0008" in text
    assert text.index("Concurrency") < text.rindex("search_iter_0008")

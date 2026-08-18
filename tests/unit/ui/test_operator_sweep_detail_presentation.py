# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests for ``sweep-detail.js`` presentation helpers.

The page module imports browser import-map specifiers (``htm/preact``) that bare
node cannot resolve and the repo ships no bundler, so the page's pure helpers are
fenced off inside sentinel comments and sliced out here. That buys real
behavioural coverage instead of the source-substring assertions the rest of the
page's tests are limited to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SWEEP_DETAIL_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "sweep-detail.js"
)

_BEGIN = "// aiperf:sweep-detail-pure:begin"
_END = "// aiperf:sweep-detail-pure:end"


def _pure_block() -> str:
    """Return the dependency-free helper block from the page source."""
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")
    assert _BEGIN in source and _END in source, (
        "sweep-detail.js must keep the pure-helper sentinel comments; they are "
        "what makes page-level logic testable without a bundler."
    )
    block = source.split(_BEGIN, 1)[1].split(_END, 1)[0]
    assert "import " not in block, (
        "the pure-helper block must stay dependency-free so it can be evaluated "
        "standalone in node"
    )
    return block


def _eval(body: str) -> object:
    return json.loads(run_node(_pure_block() + "\n" + body))


# ---------------------------------------------------------------------------
# currentCellCaption -- status.currentCell is never cleared, so a terminal
# sweep kept rendering "running variation 13/22".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase_mode",
    ["terminal", "unknown"],
)  # fmt: skip
def test_current_cell_caption_is_suppressed_outside_live_phases(
    phase_mode: str,
) -> None:
    caption = _eval(
        f"""
        console.log(JSON.stringify(currentCellCaption({{
          currentCell: {{ variationIndex: 13, label: 'search_iter_0013', trial: 0 }},
          phaseMode: {phase_mode!r},
          valuesLabel: 'concurrency=309',
        }})));
        """
    )

    assert caption is None


def test_current_cell_caption_is_suppressed_when_no_cell_is_recorded() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: null, phaseMode: 'live', valuesLabel: null,
        })));
        """
    )

    assert caption is None


def test_current_cell_caption_leads_with_swept_values_not_the_planner_id() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: { variationIndex: 13, label: 'search_iter_0013', trial: 0 },
          phaseMode: 'live',
          valuesLabel: 'concurrency=309',
        })));
        """
    )

    assert caption == "running variation 13 · concurrency=309 · trial 0"


def test_current_cell_caption_drops_the_misleading_progress_denominator() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: { variationIndex: 13, label: 'search_iter_0013', trial: 0 },
          phaseMode: 'live',
          valuesLabel: 'concurrency=309',
        })));
        """
    )

    # `variationIndex` names the child (`<sweep>-v13`); it is not a count of
    # finished work, so it must never be rendered as `13/22`.
    assert "/" not in caption
    assert "22" not in caption


def test_current_cell_caption_falls_back_to_the_planner_label() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: { variationIndex: 2, label: 'search_iter_0002', trial: 1 },
          phaseMode: 'live',
          valuesLabel: null,
        })));
        """
    )

    assert caption == "running variation 2 · search_iter_0002 · trial 1"


def test_current_cell_caption_tolerates_a_cell_missing_index_and_trial() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: { label: '' },
          phaseMode: 'live',
          valuesLabel: null,
        })));
        """
    )

    assert caption == "running a variation"


def test_current_cell_caption_keeps_variation_zero_visible() -> None:
    caption = _eval(
        """
        console.log(JSON.stringify(currentCellCaption({
          currentCell: { variationIndex: 0, label: 'search_iter_0000', trial: 0 },
          phaseMode: 'live',
          valuesLabel: 'concurrency=8',
        })));
        """
    )

    assert caption == "running variation 0 · concurrency=8 · trial 0"


def test_page_gates_the_current_cell_line_on_the_helper() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert "running variation ${currentCell.variationIndex" not in source
    assert "${liveCaption && html`" in source


# ---------------------------------------------------------------------------
# sweepPresentationModel -- adaptive sweeps have two intentionally different
# surfaces: a live optimization study and a terminal planner result. A browser
# ranking during the study is a current leader, never a final winner.
# ---------------------------------------------------------------------------


def _presentation_model(**kwargs: object) -> dict:
    args = json.dumps(kwargs)
    return _eval(f"console.log(JSON.stringify(sweepPresentationModel({args})));")


def test_live_adaptive_sweep_is_an_optimization_study_not_a_verdict() -> None:
    model = _presentation_model(
        phaseMode="live",
        sweepType="adaptive_search",
        hasPlannerVerdict=True,
        isFailed=False,
        isCancelled=False,
    )

    assert model == {
        "kind": "study",
        "showsPlannerVerdict": False,
        "leaderLabel": "Current leader",
    }


def test_terminal_adaptive_sweep_shows_only_the_planners_final_verdict() -> None:
    model = _presentation_model(
        phaseMode="terminal",
        sweepType="adaptive_search",
        hasPlannerVerdict=True,
        isFailed=False,
        isCancelled=False,
    )

    assert model == {
        "kind": "result",
        "showsPlannerVerdict": True,
        "leaderLabel": None,
    }


@pytest.mark.parametrize(
    "phase_mode,is_failed,is_cancelled",
    [
        param("terminal", True, False, id="failed"),
        param("terminal", False, True, id="cancelled"),
        param("unknown", False, False, id="unknown"),
    ],
)  # fmt: skip
def test_non_result_states_never_expose_a_final_recommendation(
    phase_mode: str, is_failed: bool, is_cancelled: bool
) -> None:
    model = _presentation_model(
        phaseMode=phase_mode,
        sweepType="adaptive_search",
        hasPlannerVerdict=True,
        isFailed=is_failed,
        isCancelled=is_cancelled,
    )

    assert model["showsPlannerVerdict"] is False


def test_grid_sweep_uses_variation_analysis_not_a_winner_summary() -> None:
    model = _presentation_model(
        phaseMode="terminal",
        sweepType="grid",
        hasPlannerVerdict=True,
        isFailed=False,
        isCancelled=False,
    )

    assert model == {
        "kind": "variations",
        "showsPlannerVerdict": False,
        "leaderLabel": None,
    }


# ---------------------------------------------------------------------------
# variationsCardModel -- `status.totalVariations` is `max_iterations` for
# adaptive_search (operator/handlers/sweep/create.py:287), i.e. a ceiling.
# ---------------------------------------------------------------------------


def _variations_card(**kwargs: object) -> dict:
    args = json.dumps(kwargs)
    return _eval(f"console.log(JSON.stringify(variationsCardModel({args})));")


@pytest.mark.parametrize(
    "sweep_type,bounded",
    [
        param("adaptive_search", True, id="adaptive_search"),
        param("grid", False, id="grid"),
        param("zip", False, id="zip"),
        param("sobol", False, id="sobol"),
        param("scenarios", False, id="scenarios"),
        param(None, False, id="missing"),
    ],
)  # fmt: skip
def test_only_adaptive_search_declares_a_bounded_variation_count(
    sweep_type: str | None, bounded: bool
) -> None:
    result = _eval(
        f"console.log(JSON.stringify(isBoundedVariationCount({json.dumps(sweep_type)})));"
    )

    assert result is bounded


def test_grid_sweep_still_reports_its_exact_total_and_progress() -> None:
    card = _variations_card(
        sweepType="grid",
        totalVariations=6,
        observedVariations=3,
        phaseMode="live",
        finishedRuns=3,
        plannedRuns=6,
    )

    assert card["value"] == 6
    assert card["progress"] == 50
    assert card["sub"] is None


def test_completed_grid_sweep_keeps_a_full_progress_bar() -> None:
    card = _variations_card(
        sweepType="grid",
        totalVariations=6,
        observedVariations=6,
        phaseMode="terminal",
        finishedRuns=6,
        plannedRuns=6,
    )

    assert card["value"] == 6
    assert card["progress"] == 100


def test_converged_adaptive_sweep_reports_variations_actually_run() -> None:
    # gemma-bo4: max_iterations 22, search stopped at 14. Showing "22" beside a
    # part-filled bar reads as truncation.
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=22,
        observedVariations=14,
        phaseMode="terminal",
        finishedRuns=14,
        plannedRuns=22,
    )

    assert card["value"] == 14
    assert card["progress"] is None
    assert card["sub"] == "stopped early · limit 22"
    assert "not a target" in card["title"]


def test_terminal_adaptive_sweep_that_used_its_whole_budget_says_so() -> None:
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=22,
        observedVariations=22,
        phaseMode="terminal",
        finishedRuns=22,
        plannedRuns=22,
    )

    assert card["value"] == 22
    assert card["sub"] == "hit limit 22"
    assert card["progress"] is None


def test_live_adaptive_sweep_keeps_a_bar_but_marks_the_ceiling_as_a_ceiling() -> None:
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=22,
        observedVariations=5,
        phaseMode="live",
        finishedRuns=4,
        plannedRuns=22,
    )

    assert card["value"] == 5
    assert card["sub"] == "of up to 22"
    assert card["progress"] == 18


def test_adaptive_sweep_without_a_manifest_falls_back_to_the_ceiling() -> None:
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=22,
        observedVariations=0,
        phaseMode="live",
        finishedRuns=0,
        plannedRuns=22,
    )

    assert card["value"] == 22
    assert card["sub"] == "up to 22"


def test_adaptive_subtitle_never_claims_a_convergence_reason() -> None:
    # `convergence_reason` lives in the search_history.json artifact, not on the
    # CR status, so a cancelled adaptive sweep would be mislabelled "converged".
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=22,
        observedVariations=3,
        phaseMode="terminal",
        finishedRuns=3,
        plannedRuns=22,
    )

    assert "converg" not in card["sub"].lower()


def test_progress_bar_measures_runs_against_planned_runs_not_variations() -> None:
    # 6 variations x 3 trials = 18 planned runs. After 6 runs the bar must read
    # 33%, not 100%.
    card = _variations_card(
        sweepType="grid",
        totalVariations=6,
        observedVariations=2,
        phaseMode="live",
        finishedRuns=6,
        plannedRuns=18,
    )

    assert card["progress"] == 33


def test_progress_bar_counts_cancelled_runs_as_finished() -> None:
    card = _variations_card(
        sweepType="grid",
        totalVariations=4,
        observedVariations=4,
        phaseMode="terminal",
        finishedRuns=4,  # 2 completed + 1 failed + 1 cancelled
        plannedRuns=4,
    )

    assert card["progress"] == 100


def test_page_feeds_the_variations_card_run_scoped_counters() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert (
        "finishedRuns: (s.completed_runs ?? 0) + (s.failed_runs ?? 0) "
        "+ (s.cancelled_runs ?? 0)," in source
    )
    assert "plannedRuns: status.maxTotalRuns || s.total_variations," in source
    assert "title=${completedTitle}" in source


@pytest.mark.parametrize(
    "source",
    [
        param("live", id="live"),
        param("archived", id="archived"),
        param("both", id="both"),
    ],
)  # fmt: skip
def test_every_provenance_chip_value_is_explained(source: str) -> None:
    title = _eval(f"console.log(JSON.stringify(sweepSourceTitle({source!r})));")

    assert isinstance(title, str) and len(title) > 20


def test_unknown_provenance_values_get_no_invented_explanation() -> None:
    result = _eval(
        """
        console.log(JSON.stringify({
          unknown: sweepSourceTitle('mystery'),
          missing: sweepSourceTitle(null),
        }));
        """
    )

    assert result == {"unknown": None, "missing": None}


def test_page_attaches_the_provenance_explanation_to_the_chip() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert "title=${sweepSourceTitle(s.source) ?? undefined}" in source


# ---------------------------------------------------------------------------
# nonSuccessCardModel -- `cancelled_runs` is a distinct terminal bucket from
# `failed_runs` (routers/sweeps_models.py:80-88).
# ---------------------------------------------------------------------------


def _non_success_card(**kwargs: object) -> dict:
    args = json.dumps(kwargs)
    return _eval(f"console.log(JSON.stringify(nonSuccessCardModel({args})));")


def test_a_cancelled_only_sweep_is_not_reported_as_failed() -> None:
    card = _non_success_card(failedRuns=0, cancelledRuns=2)

    assert card["label"] == "Cancelled"
    assert card["value"] == 2
    assert card["tone"] == "neutral"


def test_failures_stay_the_headline_when_a_sweep_has_both() -> None:
    card = _non_success_card(failedRuns=3, cancelledRuns=2)

    assert card["label"] == "Failed"
    assert card["value"] == 3
    assert card["tone"] == "bad"
    assert card["sub"] == "+2 cancelled"


def test_a_clean_sweep_shows_zero_failures_without_alarm_colouring() -> None:
    card = _non_success_card(failedRuns=0, cancelledRuns=0)

    assert card == {
        "label": "Failed",
        "value": 0,
        "tone": "neutral",
        "sub": None,
        "title": None,
    }


def test_non_success_card_singularises_its_tooltip() -> None:
    card = _non_success_card(failedRuns=0, cancelledRuns=1)

    assert card["title"].startswith("1 run was cancelled.")


def test_page_renders_the_non_success_tile_from_the_helper() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert 'label="Failed"' not in source
    assert "label=${nonSuccessCard.label}" in source


def test_kpi_attribution_leads_with_swept_values_and_keeps_the_id_on_hover() -> None:
    result = _eval(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 5, label: 'search_iter_0005', valuesLabel: 'concurrency=309',
        })));
        """
    )

    assert result == {
        "text": "concurrency=309",
        "title": "concurrency=309 · search_iter_0005",
    }


def test_kpi_attribution_falls_back_to_the_planner_label() -> None:
    result = _eval(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 5, label: 'search_iter_0005', valuesLabel: null,
        })));
        """
    )

    assert result == {"text": "search_iter_0005", "title": "search_iter_0005"}


def test_kpi_attribution_falls_back_to_the_variation_index() -> None:
    result = _eval(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 0, label: '', valuesLabel: null,
        })));
        """
    )

    assert result == {"text": "v0", "title": "v0"}


def test_kpi_attribution_tolerates_a_missing_variation() -> None:
    result = _eval("console.log(JSON.stringify(kpiVariationAttribution(null)));")

    assert result == {"text": "---", "title": None}


def test_kpi_attribution_does_not_duplicate_an_already_meaningful_label() -> None:
    # Grid sweeps label variations with the values themselves, so the tooltip
    # would otherwise read "concurrency=8 · concurrency=8".
    result = _eval(
        """
        console.log(JSON.stringify(kpiVariationAttribution({
          variation_index: 1, label: 'concurrency=8', valuesLabel: 'concurrency=8',
        })));
        """
    )

    assert result == {"text": "concurrency=8", "title": "concurrency=8"}


def test_headline_kpi_tiles_use_the_attribution_helper() -> None:
    source = _SWEEP_DETAIL_JS.read_text(encoding="utf-8")

    assert "top.v.label || `v${top.v.variation_index}`" not in source
    assert "const attribution = kpiVariationAttribution(top.v);" in source
    assert "title=${k.variationTitle ?? undefined}" in source


def test_variations_card_tolerates_a_sweep_with_no_declared_total() -> None:
    card = _variations_card(
        sweepType="adaptive_search",
        totalVariations=0,
        observedVariations=0,
        phaseMode="unknown",
        finishedRuns=0,
        plannedRuns=0,
    )

    assert card["value"] == 0
    assert card["sub"] is None
    assert card["title"] is None

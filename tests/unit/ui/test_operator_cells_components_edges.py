# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for sweep cells table/chart component behavior.

The components import browser import-map modules (``htm/preact`` and Chart.js
wrappers), so these tests pin the pure data-shaping expectations statically until
those rules are extracted into importable helpers.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CELLS_TABLE_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components" / "cells-table.js"
)
_CELLS_CHART_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components" / "cells-chart.js"
)
_SWEEP_DETAIL_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "sweep-detail.js"
)


def _source(path: Path) -> str:
    return path.read_text()


def test_cells_table_accepts_snake_case_and_camel_case_cell_fields() -> None:
    src = _source(_CELLS_TABLE_JS)

    expected_fallbacks = [
        "variation_index",
        "variationIndex",
        "variation_label",
        "variationLabel",
        "trials_completed",
        "trialsCompleted",
        "trials_failed",
        "trialsFailed",
    ]
    for token in expected_fallbacks:
        assert token in src, f"CellsTable must support {token} cell fields"


def test_cells_chart_keeps_missing_metric_values_as_null_points() -> None:
    src = _source(_CELLS_CHART_JS)

    assert "cell?.metrics?.[metric]?.[stat] ?? null" in src
    assert "spanGaps: true" in src
    assert "if (v == null) return `${ctx.dataset.label}: (no data)`;" in src


def test_cells_chart_has_dimensionless_single_cell_fallback() -> None:
    src = _source(_CELLS_CHART_JS)

    assert "No swept dimensions in this sweep." not in src
    assert "variation_index" in src or "variationIndex" in src


def test_cells_table_click_target_uses_first_child_job_shape() -> None:
    table_src = _source(_CELLS_TABLE_JS)
    detail_src = _source(_SWEEP_DETAIL_JS)

    assert "onclick=${() => onCellClick && onCellClick(c)}" in table_src
    assert "c.children?.[0] && navigate(buildJobPath(c.children[0]))" in detail_src

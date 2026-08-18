# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static edge tests for operator phase and record progress components."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PHASE_BAR_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components" / "phase-bar.js"
)
_RECORD_PROCESSING_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "record-processing.js"
)
_JOB_DETAIL_JS = (
    _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "job-detail.js"
)


def _source(path: Path) -> str:
    return path.read_text()


def test_phase_bar_zero_totals_stay_pending_with_zero_percent() -> None:
    src = _source(_PHASE_BAR_JS)

    assert "phase.total > 0 ? Math.round" in src
    assert ": 0" in src
    assert "phase.completed >= phase.total && phase.total > 0" in src


def test_phase_bar_caps_over_complete_progress_at_100_percent() -> None:
    src = _source(_PHASE_BAR_JS)

    assert "Math.min(100" in src
    assert "aria-valuenow=${Math.min(100" in src or "aria-valuenow=${pct}" not in src


def test_record_processing_tolerates_missing_phase_records() -> None:
    src = _source(_RECORD_PROCESSING_JS)

    assert "if (p == null || typeof p !== 'object') continue;" in src
    assert "if (p == null || typeof p !== 'object') return null;" in src
    assert "p.recordsSuccess ?? 0" in src
    assert "p.recordsError ?? 0" in src


def test_record_processing_uses_request_and_record_completion_booleans() -> None:
    src = _source(_RECORD_PROCESSING_JS)

    assert "isRequestsComplete" in src
    assert "isRecordsComplete" in src
    assert "const sendingComplete = p.sendingComplete ?? false;" not in src


def test_phase_ordering_preserves_operator_status_order() -> None:
    detail_src = _source(_JOB_DETAIL_JS)
    phase_src = _source(_PHASE_BAR_JS)
    phase_mapping = detail_src[
        detail_src.index("const phasesArray =") : detail_src.index("const pods =")
    ]
    phase_bar_loop = phase_src[
        phase_src.index("${phases.map") : phase_src.index("</div>\n  `;")
    ]

    assert "Object.entries(rawPhases).map(([phaseName, p]) => ({" in phase_mapping
    assert "${phases.map((phase) => {" in phase_bar_loop
    assert ".sort(" not in phase_mapping
    assert ".sort(" not in phase_bar_loop


def test_phase_progress_components_keep_narrow_view_overflow_bounded() -> None:
    detail_src = _source(_JOB_DETAIL_JS)
    phase_src = _source(_PHASE_BAR_JS)

    assert '<div style="overflow-x: auto; max-width: 100%">' in detail_src
    assert '<div class="phase-bar" style="min-width:0;max-width:100%">' in phase_src
    assert 'style="min-width:0;flex:1 1 0"' in phase_src
    assert "overflow:hidden;text-overflow:ellipsis;white-space:nowrap" in phase_src

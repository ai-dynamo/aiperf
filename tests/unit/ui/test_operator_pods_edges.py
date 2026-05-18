# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator pod display components."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_PAGES = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages"

_PODS_BAR_JS = _COMPONENTS / "pods-bar.js"
_PODS_STRIP_JS = _COMPONENTS / "pods-strip.js"
_DIAGNOSTICS_PANEL_JS = _COMPONENTS / "diagnostics-panel.js"
_DIAGNOSTICS_PODS_TAB_JS = _COMPONENTS / "diagnostics-pods-tab.js"
_JOB_DETAIL_JS = _PAGES / "job-detail.js"
_SWEEP_DETAIL_JS = _PAGES / "sweep-detail.js"


def _source(path: Path) -> str:
    return path.read_text()


def test_empty_pods_are_not_reported_as_healthy() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)
    pods_strip = _source(_PODS_STRIP_JS)

    assert "No pods</div>" in pods_bar
    assert "No pods</div>" in pods_tab
    assert "list.length === 0" in pods_strip
    assert "no pods" in pods_strip.lower()
    assert "metaParts.push('all healthy');" not in pods_strip


def test_crashloop_badges_count_nested_kubernetes_reasons() -> None:
    pods_strip = _source(_PODS_STRIP_JS)
    job_detail = _source(_JOB_DETAIL_JS)
    sweep_detail = _source(_SWEEP_DETAIL_JS)

    for src in (pods_strip, job_detail, sweep_detail):
        assert "crashloop" in src.lower()
        assert "containerStatuses" in src
        assert "waiting?.reason" in src or "state?.waiting?.reason" in src


def test_readiness_counts_use_ready_state_not_running_phase() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_strip = _source(_PODS_STRIP_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    assert "pods.filter((p) => p.ready).length" in pods_bar
    assert "pods.filter((p) => p.ready).length" in pods_tab
    assert "list.filter((p) => p.ready).length" in pods_strip
    assert "return ph === 'running';" not in pods_strip


def test_restart_totals_include_missing_top_level_restarts_fallbacks() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    for src in (pods_bar, pods_tab):
        assert "p.restarts ?? 0" in src
        assert "containerStatuses" in src
        assert "restartCount" in src


def test_missing_pod_fields_get_stable_display_fallbacks() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    for src in (pods_bar, pods_tab):
        assert "pod.name ??" in src
        assert "unknown pod" in src.lower()
        assert "key=${pod.name}" not in src
        assert "title=${pod.name}" not in src


def test_pods_strip_generates_diagnostics_pods_navigation() -> None:
    pods_strip = _source(_PODS_STRIP_JS)
    diagnostics_panel = _source(_DIAGNOSTICS_PANEL_JS)
    job_detail = _source(_JOB_DETAIL_JS)

    assert "can navigate to ?diag=pods" in pods_strip
    assert "onBarClick=${onExpand}" in pods_strip
    assert "onPodClick=${onExpand}" in pods_strip
    assert "url.searchParams.set('diag', tab);" in diagnostics_panel
    assert "import { PodsStrip }" in job_detail
    assert "diag=pods" in job_detail
    assert "onExpand" in job_detail


def test_archived_runs_do_not_mount_pods_or_logs_tabs() -> None:
    diagnostics_panel = _source(_DIAGNOSTICS_PANEL_JS)
    job_detail = _source(_JOB_DETAIL_JS)

    assert "archived ? ['events', 'conditions'] : ALL_TABS" in diagnostics_panel
    assert "mode=${viewingCurrentRun ? (isRunning ? 'live' : 'completed') : 'archived'}" in job_detail
    assert "archived=${!viewingCurrentRun}" in job_detail
    assert "${showLiveRunPanels && html`" not in job_detail

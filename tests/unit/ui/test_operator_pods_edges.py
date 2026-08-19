# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator pod display components."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_PAGES = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages"

_PODS_BAR_JS = _COMPONENTS / "pods-bar.js"
_DIAGNOSTICS_PANEL_JS = _COMPONENTS / "diagnostics-panel.js"
_DIAGNOSTICS_PODS_TAB_JS = _COMPONENTS / "diagnostics-pods-tab.js"
_JOB_DETAIL_JS = _PAGES / "job-detail.js"
_SWEEP_DETAIL_JS = _PAGES / "sweep-detail.js"


def _source(path: Path) -> str:
    return path.read_text()


def test_empty_pods_are_not_reported_as_healthy() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    assert "No pods</div>" in pods_bar
    assert "No pods</div>" in pods_tab


def test_crashloop_badges_count_nested_kubernetes_reasons() -> None:
    job_detail = _source(_JOB_DETAIL_JS)
    sweep_detail = _source(_SWEEP_DETAIL_JS)

    # The detail pages consume the already-flattened ``pod.reason`` and only need
    # to count crashloops for the diagnostics badge.
    for src in (job_detail, sweep_detail):
        assert "crashloop" in src.lower()
        assert "/crashloop/i.test(p.reason" in src


def test_readiness_counts_use_ready_state_not_running_phase() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    assert "pods.filter((p) => p.ready).length" in pods_bar
    assert "pods.filter((p) => p.ready).length" in pods_tab


def test_restart_totals_include_missing_top_level_restarts_fallbacks() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    # Pods arrive with a flattened ``restarts`` count; a missing value falls back
    # to 0 so the summed total never becomes NaN.
    for src in (pods_bar, pods_tab):
        assert "p.restarts ?? 0" in src


def test_missing_pod_fields_get_stable_display_fallbacks() -> None:
    pods_bar = _source(_PODS_BAR_JS)
    pods_tab = _source(_DIAGNOSTICS_PODS_TAB_JS)

    # A missing phase renders as ``unknown`` rather than a blank cell; the pod
    # name is the stable React key and tooltip text.
    for src in (pods_bar, pods_tab):
        assert "(pod.phase ?? 'unknown')" in src
        assert "key=${pod.name}" in src
        assert '<td class="pods-table-name" title=${pod.name}>${pod.name}</td>' in src


def test_diagnostics_panel_owns_pods_navigation() -> None:
    diagnostics_panel = _source(_DIAGNOSTICS_PANEL_JS)
    job_detail = _source(_JOB_DETAIL_JS)

    # The diagnostics panel owns the URL-backed ?diag=<tab> navigation and
    # mounts the dedicated pods tab; job-detail mounts the panel.
    assert "url.searchParams.set('diag', tab);" in diagnostics_panel
    assert "'pods'" in diagnostics_panel
    assert "import { DiagnosticsPanel }" in job_detail
    assert "<${DiagnosticsPanel}" in job_detail


def test_archived_runs_do_not_mount_pods_or_logs_tabs() -> None:
    diagnostics_panel = _source(_DIAGNOSTICS_PANEL_JS)
    job_detail = _source(_JOB_DETAIL_JS)

    # Archived runs restrict the diagnostics panel to events + conditions; the
    # pods/logs tabs are dropped because the pod CRs are gone.
    assert "archived ? ['events', 'conditions'] : ALL_TABS" in diagnostics_panel
    assert (
        "mode=${viewingCurrentRun ? (isRunning ? 'live' : 'completed') : 'archived'}"
        in job_detail
    )
    assert "archived=${!viewingCurrentRun}" in job_detail
    # Live phase/chart panels stay gated behind showLiveRunPanels; archived
    # gating is enforced by the panel's tab list above, not by removing it.
    assert "${showLiveRunPanels && html`" in job_detail

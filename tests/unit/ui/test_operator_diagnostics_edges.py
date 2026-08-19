# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for the operator diagnostics panel and tabs."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_API_JS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "lib" / "api.js"
_DIAGNOSTICS_PANEL_JS = _COMPONENTS / "diagnostics-panel.js"
_DIAGNOSTICS_EVENTS_TAB_JS = _COMPONENTS / "diagnostics-events-tab.js"
_DIAGNOSTICS_LOGS_TAB_JS = _COMPONENTS / "diagnostics-logs-tab.js"
_DIAGNOSTICS_PODS_TAB_JS = _COMPONENTS / "diagnostics-pods-tab.js"


def _source(path: Path) -> str:
    return path.read_text()


def test_diagnostics_query_param_parsing_is_diag_only_and_allowlisted() -> None:
    src = _source(_DIAGNOSTICS_PANEL_JS)

    assert "const ALL_TABS = ['events', 'logs', 'conditions', 'pods'];" in src
    assert "const url = new URL(window.location.href);" in src
    assert "const t = url.searchParams.get('diag');" in src
    assert "return ALL_TABS.includes(t) ? t : null;" in src
    assert "url.searchParams.set('diag', tab);" in src
    assert "window.history.replaceState(null, '', url.toString());" in src


def test_archived_diagnostics_fallback_uses_conditions_not_events_or_removed_tabs() -> (
    None
):
    src = _source(_DIAGNOSTICS_PANEL_JS)

    assert (
        "const availableTabs = useMemo(() => archived ? ['events', 'conditions'] : ALL_TABS, [archived]);"
        in src
    )
    assert (
        "const defaultTab = (mode === 'live' && !archived) ? 'events' : 'conditions';"
        in src
    )
    assert "setActive(defaultTab);" in src
    assert "setActive(availableTabs[0]);" not in src


def test_diagnostics_tab_rendering_keeps_removed_archived_tabs_unmounted() -> None:
    src = _source(_DIAGNOSTICS_PANEL_JS)

    assert "${availableTabs.map((tab) => {" in src
    assert "${renderedActive === 'events' && html`<${EventsTab}" in src
    assert "${renderedActive === 'logs' && html`<${LogsTab}" in src
    assert "${renderedActive === 'conditions' && html`<${ConditionsTab}" in src
    assert "${renderedActive === 'pods' && html`<${PodsTab}" in src
    assert "if (!availableTabs.includes(active))" in src


def test_empty_pods_events_and_logs_have_explicit_empty_states() -> None:
    pods_src = _source(_DIAGNOSTICS_PODS_TAB_JS)
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "if (!pods || pods.length === 0)" in pods_src
    assert "No pods</div>" in pods_src
    # "No events recorded for this run." now belongs to kind='ok' with an empty
    # list. It used to be the 404 branch, which claimed a data fact the UI had
    # no way to know.
    assert "if (state.kind === 'none')" not in events_src
    assert "No events recorded for this run." in events_src
    assert "'No warning events.'" in events_src
    assert 'data-testid="run-events-loading"' in events_src
    assert "if (podList.length === 0)" in logs_src
    assert "no pods yet" in logs_src
    assert "No pods yet — logs will appear here once workers are scheduled." in logs_src


def test_hidden_events_and_logs_tabs_do_not_fetch_or_stream() -> None:
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "if (!active) return;" in events_src
    assert "}, [ns, name, kind, active]);" in events_src
    assert "if (!active) return;" in logs_src
    assert (
        "}, [ns, name, selectedPod, selectedContainer, follow, tailLines, kind, active]);"
        in logs_src
    )


def test_job_and_sweep_diagnostics_use_different_api_roots() -> None:
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)
    api_src = _source(_API_JS)

    assert "kind === 'sweep'" in events_src
    assert "? await api.getSweepEvents(ns, name)" in events_src
    assert ": await api.getJobEvents(ns, name);" in events_src
    assert (
        "const fetchLogs = kind === 'sweep' ? api.getSweepLogs : api.getJobLogs;"
        in logs_src
    )
    assert (
        "`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/events`"
        in api_src
    )
    assert (
        "`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/events`"
        in api_src
    )
    assert (
        "`${BASE}/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/logs?${params}`"
        in api_src
    )
    assert (
        "`${BASE}/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/logs?${params}`"
        in api_src
    )

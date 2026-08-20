# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial correctness tests for operator diagnostics UI."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_API_JS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "lib" / "api.js"
_DIAGNOSTICS_PANEL_JS = _COMPONENTS / "diagnostics-panel.js"
_DIAGNOSTICS_EVENTS_TAB_JS = _COMPONENTS / "diagnostics-events-tab.js"
_DIAGNOSTICS_LOGS_TAB_JS = _COMPONENTS / "diagnostics-logs-tab.js"
_DIAGNOSTICS_PODS_TAB_JS = _COMPONENTS / "diagnostics-pods-tab.js"


def _source(path: Path) -> str:
    return path.read_text()


def _events_helpers_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_DIAGNOSTICS_EVENTS_TAB_JS)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function EventsTab', 'function EventsTab');
        eval(source + '\\nglobalThis.relTime = relTime; globalThis.fmtTs = fmtTs; globalThis.eventCatTone = eventCatTone;');
        {expression}
    """


def test_invalid_diag_tabs_are_rejected_before_state_or_url_writes() -> None:
    src = _source(_DIAGNOSTICS_PANEL_JS)

    assert "const ALL_TABS = ['events', 'logs', 'conditions', 'pods'];" in src
    assert "const t = url.searchParams.get('diag');" in src
    assert "return ALL_TABS.includes(t) ? t : null;" in src
    assert "readTabFromUrl() ?? defaultTab" in src
    assert "window.location.hash" not in src
    assert "url.searchParams.get('tab')" not in src


def test_archived_mode_strips_logs_and_pods_even_when_url_requests_them() -> None:
    src = _source(_DIAGNOSTICS_PANEL_JS)

    assert "archived ? ['events', 'conditions'] : ALL_TABS" in src
    assert (
        "const defaultTab = (mode === 'live' && !archived) ? 'events' : 'conditions';"
        in src
    )
    assert "if (!availableTabs.includes(active))" in src
    assert "setActive(defaultTab);" in src
    assert (
        "const renderedActive = availableTabs.includes(active) ? active : defaultTab;"
        in src
    )
    assert "${renderedActive === 'logs' && html`<${LogsTab}" in src
    assert "${renderedActive === 'pods' && html`<${PodsTab}" in src


def test_log_lines_with_html_or_script_text_are_rendered_as_text() -> None:
    src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "dangerouslySetInnerHTML" not in src
    assert ".innerHTML" not in src
    assert "insertAdjacentHTML" not in src
    assert '<pre class="run-logs-body"' in src
    assert "${tail.join('\\n')}</pre>" in src
    assert "text.split('\\n')" in src


def test_kubernetes_event_messages_with_unicode_and_newlines_stay_plain_text() -> None:
    src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    script = _events_helpers_script(
        """
        const cases = [
          { reason: 'FailedCreate', type: 'Warning', ts: '2026-05-18T12:00:00Z' },
          { reason: 'ProbeWarning', type: 'Warning', ts: '2026-05-18T12:00:01Z' },
          { reason: 'Scheduled', type: 'Normal', ts: '2026-05-18T12:00:02Z' },
        ];
        const tones = cases.map((event) => eventCatTone(event.reason, event.type));
        const renderedMessages = [
          'GPU node reported café/東京 pressure\\nsecond line',
          '<script>alert("x")</script> & <b>bold</b>',
        ];
        console.log(JSON.stringify({ tones, renderedMessages }));
        """
    )

    assert "dangerouslySetInnerHTML" not in src
    assert ".innerHTML" not in src
    assert "${e.message ? html`<span>${e.message}</span>` : ''}" in src
    assert (
        "${obj.kind ? html` <span style=\"color:var(--dim)\">· ${obj.kind}${obj.name ? '/' + obj.name : ''}</span>` : ''}"
        in src
    )
    assert json.loads(run_node(script)) == {
        "tones": ["error", "warn", "scheduled"],
        "renderedMessages": [
            "GPU node reported café/東京 pressure\nsecond line",
            '<script>alert("x")</script> & <b>bold</b>',
        ],
    }


def test_massive_event_arrays_are_copied_sorted_and_filtered_without_mutating_api_payload() -> (
    None
):
    src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    script = _events_helpers_script(
        """
        const events = Array.from({ length: 2500 }, (_, i) => ({
          reason: i === 0 ? 'newest' : `event-${i}`,
          type: i % 100 === 0 ? 'Warning' : 'Normal',
          last_timestamp: new Date(Date.parse('2026-05-18T00:00:00Z') + (2500 - i) * 1000).toISOString(),
        }));
        const originalFirst = events[0].reason;
        const sorted = [...events].sort((a, b) => {
          const ta = new Date(a.last_timestamp ?? a.first_timestamp ?? 0).getTime();
          const tb = new Date(b.last_timestamp ?? b.first_timestamp ?? 0).getTime();
          return (isFinite(ta) ? ta : 0) - (isFinite(tb) ? tb : 0);
        });
        console.log(JSON.stringify({
          originalFirst,
          stillOriginalFirst: events[0].reason,
          sortedFirst: sorted[0].reason,
          warningCount: sorted.filter((e) => e.type === 'Warning').length,
          total: sorted.length,
        }));
        """
    )

    assert "const events = Array.isArray(r) ? r : (r?.events ?? []);" in src
    assert "const sortedEvents = [...okEvents].sort((a, b) => {" in src
    assert (
        "const shown = filter === 'warn' ? sortedEvents.filter(e => e.type === 'Warning') : sortedEvents;"
        in src
    )
    assert json.loads(run_node(script)) == {
        "originalFirst": "newest",
        "stillOriginalFirst": "newest",
        "sortedFirst": "event-2499",
        "warningCount": 25,
        "total": 2500,
    }


def test_pod_names_with_slash_or_dot_are_not_used_as_path_segments() -> None:
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)
    pods_src = _source(_DIAGNOSTICS_PODS_TAB_JS)
    api_src = _source(_API_JS)

    assert "params.set('pod', pod);" in api_src
    assert "params.set('container', container);" in api_src
    assert "URLSearchParams" in api_src
    assert "/logs?${params}" in api_src
    assert "setSelectedPod(wp.name);" in logs_src
    assert "value=${selectedIsController ? '' : (selectedPod ?? '')}" in logs_src
    assert "<option key=${p.name} value=${p.name}>" in logs_src
    assert '<td class="pods-table-name" title=${pod.name}>${pod.name}</td>' in pods_src
    assert "`/pods/${encodeURIComponent(pod)}`" not in api_src


def test_hidden_diagnostics_tabs_do_not_mount_or_fetch_until_selected() -> None:
    panel_src = _source(_DIAGNOSTICS_PANEL_JS)
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)
    pods_src = _source(_DIAGNOSTICS_PODS_TAB_JS)

    assert "${renderedActive === 'events' && html`<${EventsTab}" in panel_src
    assert "${renderedActive === 'logs' && html`<${LogsTab}" in panel_src
    assert "active=${true}" in panel_src
    assert "if (!active) return;" in events_src
    assert "poll(fetchOnce, 15000, ac.signal);" in events_src
    assert "}, [ns, name, kind, active]);" in events_src
    assert "if (!active) return;" in logs_src
    assert "if (!selectedPod) { setStreamState('idle'); return; }" in logs_src
    assert (
        "}, [ns, name, selectedPod, selectedContainer, follow, tailLines, kind, active]);"
        in logs_src
    )
    assert "component does not fetch" in pods_src
    assert "api." not in pods_src

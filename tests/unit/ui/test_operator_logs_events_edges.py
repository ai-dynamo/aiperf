# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator logs and events UI components."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "components"
_LOG_STRIP_JS = _COMPONENTS / "log-strip.js"
_DIAGNOSTICS_EVENTS_TAB_JS = _COMPONENTS / "diagnostics-events-tab.js"
_DIAGNOSTICS_LOGS_TAB_JS = _COMPONENTS / "diagnostics-logs-tab.js"


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


def _log_strip_helpers_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_LOG_STRIP_JS)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function LogStrip', 'function LogStrip');
        eval(source + '\\nglobalThis.phaseSeverity = phaseSeverity; globalThis.fmtTs = fmtTs;');
        {expression}
    """


def test_log_strip_phase_severity_counts_are_derived_from_event_severity() -> None:
    script = _log_strip_helpers_script(
        """
        const phases = ['Pending', 'Running', 'Completed', 'Succeeded', 'Failed', 'Error', null, 'Unknown'];
        const severities = phases.map((phase) => phaseSeverity(phase));
        const events = severities.map((severity) => ({ severity }));
        const counts = {
          all: events.length,
          warn: events.filter((e) => e.severity === 'warn').length,
          error: events.filter((e) => e.severity === 'error').length,
        };
        console.log(JSON.stringify({ severities, counts }));
        """
    )

    assert json.loads(run_node(script)) == {
        "severities": [
            "info",
            "info",
            "info",
            "info",
            "error",
            "error",
            "info",
            "info",
        ],
        "counts": {"all": 8, "warn": 0, "error": 2},
    }


def test_log_strip_filters_by_exact_severity_and_keeps_collapsed_tail_window() -> None:
    src = _source(_LOG_STRIP_JS)

    assert "warn: events.filter(e => e.severity === 'warn').length" in src
    assert "error: events.filter(e => e.severity === 'error').length" in src
    assert "if (filter === 'all') return true;" in src
    assert "return e.severity === filter;" in src
    assert (
        "const visible = collapsed ? filtered.slice(-COLLAPSED_ROWS) : filtered;" in src
    )
    assert "const hiddenCount = filtered.length - visible.length;" in src


def test_events_tab_formats_malformed_timestamps_as_dashes() -> None:
    script = _events_helpers_script(
        """
        Date.now = () => Date.parse('2026-05-18T12:00:00Z');
        console.log(JSON.stringify({
          missingTs: fmtTs(null),
          malformedTs: fmtTs('not-a-date'),
          validTs: fmtTs('2026-05-18T01:02:03Z'),
          missingRelative: relTime(null),
          malformedRelative: relTime('not-a-date'),
          validRelative: relTime('2026-05-18T11:59:30Z'),
        }));
        """
    )

    assert json.loads(run_node(script)) == {
        "missingTs": "—",
        "malformedTs": "—",
        "validTs": "01:02:03",
        "missingRelative": "—",
        "malformedRelative": "—",
        "validRelative": "30s ago",
    }


def test_events_tab_warning_filter_and_reason_tones_cover_kubernetes_edges() -> None:
    script = _events_helpers_script(
        """
        const events = [
          { type: 'Normal', reason: 'Scheduled' },
          { type: 'Warning', reason: 'Unhealthy' },
          { type: 'Warning', reason: 'BackOff' },
          { type: 'Normal', reason: 'FailedCreate' },
          { type: 'Warning', reason: null },
        ];
        const warnReasons = events.filter((e) => e.type === 'Warning').map((e) => e.reason ?? 'missing');
        const tones = events.map((e) => eventCatTone(e.reason, e.type));
        console.log(JSON.stringify({ warnReasons, tones }));
        """
    )

    assert json.loads(run_node(script)) == {
        "warnReasons": ["Unhealthy", "BackOff", "missing"],
        "tones": ["scheduled", "warn", "error", "error", "warning"],
    }


def test_events_tab_sorting_treats_malformed_timestamps_as_oldest() -> None:
    script = _events_helpers_script(
        """
        const events = [
          { reason: 'late', last_timestamp: '2026-05-18T03:00:00Z' },
          { reason: 'bad', last_timestamp: 'not-a-date' },
          { reason: 'early', first_timestamp: '2026-05-18T01:00:00Z' },
        ];
        const sorted = [...events].sort((a, b) => {
          const ta = new Date(a.last_timestamp ?? a.first_timestamp ?? 0).getTime();
          const tb = new Date(b.last_timestamp ?? b.first_timestamp ?? 0).getTime();
          return (isFinite(ta) ? ta : 0) - (isFinite(tb) ? tb : 0);
        });
        console.log(JSON.stringify(sorted.map((event) => event.reason)));
        """
    )

    assert json.loads(run_node(script)) == ["bad", "early", "late"]


def test_logs_tab_follow_and_non_follow_paths_use_different_response_contracts() -> (
    None
):
    src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "if (follow) {" in src
    assert "follow: true, tailLines: clampedTail, signal: ac.signal" in src
    assert "const reader = res.body?.getReader();" in src
    assert "const text = await res.text();" in src
    assert "while (true) {" in src
    assert "if (leftover) appendText(leftover + '\\n');" in src
    assert "} else {" in src
    assert "follow: false, tailLines: clampedTail, signal: ac.signal" in src
    assert "appendText(text);" in src


def test_hidden_logs_and_events_tabs_gate_all_network_work_on_active_prop() -> None:
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "if (!active) return;" in events_src
    assert "poll(fetchOnce, 15000, ac.signal);" in events_src
    assert "}, [ns, name, kind, active]);" in events_src
    assert "if (!active) return;" in logs_src
    assert "if (!selectedPod) { setStreamState('idle'); return; }" in logs_src
    assert (
        "const fetchLogs = kind === 'sweep' ? api.getSweepLogs : api.getJobLogs;"
        in logs_src
    )
    assert (
        "}, [ns, name, selectedPod, selectedContainer, follow, tailLines, kind, active]);"
        in logs_src
    )


def test_logs_and_events_empty_states_are_explicit_and_specific() -> None:
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    assert "if (state.kind === 'loading')" in events_src
    assert "loading…" in events_src
    assert 'data-testid="run-events-loading"' in events_src
    assert "Loading events…" in events_src
    # The genuinely-empty case is kind='ok' with an empty array. There is no
    # 'none' state any more: it was only ever reachable from an HTTP 404, and a
    # 404 is not evidence that a run recorded no events.
    assert "if (state.kind === 'none')" not in events_src
    assert "No events recorded for this run." in events_src
    assert "'No warning events.'" in events_src
    assert "if (podList.length === 0)" in logs_src
    assert "no pods yet" in logs_src
    assert "No pods yet — logs will appear here once workers are scheduled." in logs_src


def test_logs_and_events_error_states_preserve_actionable_messages() -> None:
    events_src = _source(_DIAGNOSTICS_EVENTS_TAB_JS)
    logs_src = _source(_DIAGNOSTICS_LOGS_TAB_JS)

    # Status comes off the error object, not a regex over its message: a 500
    # whose body contains "404" used to be routed to the empty-state branch.
    assert "const status = httpStatusOf(err);" in events_src
    assert "if (status === 404)" in events_src
    assert "Events endpoint returned 404" in events_src
    assert "not an empty " in events_src
    assert "msg: describeEventsError(err)," in events_src
    # A failed refresh keeps the last good snapshot instead of erasing it.
    assert "events: prev.events ?? null," in events_src
    assert 'data-testid="run-events-stale"' in events_src
    assert "Refresh failed" in events_src
    assert "run-events--err" in events_src
    assert (
        '<div class="run-event run-event--error" data-testid="run-events-error">'
        "${state.msg}</div>" in events_src
    )
    assert (
        "if (httpStatusOf(e) === 404) setErr('Pod not found (it may have been evicted).');"
        in logs_src
    )
    assert "else setErr(e.message);" in logs_src
    assert '${err && html`<div class="run-logs-error">${err}</div>`}' in logs_src

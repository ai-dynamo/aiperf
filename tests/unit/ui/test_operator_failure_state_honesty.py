# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-, empty-, and loading-state honesty checks for the operator UI.

Every case here is a claim the UI used to make that it could not support:
a 404 reported as "no data", an all-healthy status block reported as a blank
pane, an optional counter summed into ``NaN``. The theme is that "we could
not load this", "there is nothing here", and "everything is fine" are three
different sentences and must not render as the same one.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
COMPONENTS = UI_ROOT / "components"
LIB = UI_ROOT / "lib"


_HELPERS_IMPORT = (
    "import { visibleConditionBadgeSummary } from "
    f"'{(COMPONENTS / 'conditions-helpers.js').as_uri()}';"
)

# api.js pulls in state.js, which imports @preact/signals — unavailable to a
# bare node run. Swap the import for inert stand-ins; httpStatusOf touches none
# of them.
_STATE_STUBS = (
    "function clearFreshnessSource() {} "
    "function markFreshnessAttempt() {} "
    "function markFreshnessFailure() {} "
    "function markFreshnessStopped() {} "
    "function markFreshnessSuccess() {} "
    "function setError() {}"
)


def _conditions_import_script() -> str:
    """Load conditions.js with a string-rendering stand-in for htm/preact."""
    return f"""
        import {{ readFileSync }} from 'node:fs';
        const renderValue = (value) => Array.isArray(value)
          ? value.join('')
          : value == null || value === false
            ? ''
            : String(value);
        globalThis.__html = (strings, ...values) => strings.reduce(
          (acc, part, index) => acc + part + renderValue(values[index]),
          '',
        );
        let source = readFileSync({str(COMPONENTS / "conditions.js")!r}, 'utf8');
        source = source.replace(
          "import {{ html }} from 'htm/preact';",
          'const html = globalThis.__html;',
        ).replace(
          "import {{ visibleConditionBadgeSummary }} from './conditions-helpers.js';",
          {_HELPERS_IMPORT!r},
        );
        const url = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const mod = await import(url);
    """


def test_conditions_all_healthy_says_so_instead_of_rendering_nothing() -> None:
    """An archived job carries exactly one condition, Archived=True.

    Every all-True condition is filtered out of the badge row by design, and
    the component then returned null -- so the diagnostics tab header read
    "conditions 1" above a completely blank pane. Reproduced in the browser
    against the local results-server before the fix.
    """
    script = f"""
        {_conditions_import_script()}
        const output = mod.Conditions({{ conditions: [
          {{ type: 'Archived', status: 'True' }},
        ] }});
        console.log(JSON.stringify({{
          renderedNothing: output == null || String(output).trim() === '',
          saysHealthy: String(output).includes('All 1 condition healthy'),
          testid: String(output).includes('data-testid="conditions-all-ok"'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "renderedNothing": False,
        "saysHealthy": True,
        "testid": True,
    }


def test_conditions_all_healthy_pluralises_the_count() -> None:
    script = f"""
        {_conditions_import_script()}
        const output = mod.Conditions({{ conditions: [
          {{ type: 'ConfigValid', status: 'True' }},
          {{ type: 'WorkersReady', status: 'True' }},
          {{ type: 'ResultsAvailable', status: 'True' }},
        ] }});
        console.log(JSON.stringify(String(output).includes('All 3 conditions healthy')));
    """

    assert json.loads(run_node(script)) is True


def test_conditions_distinguishes_not_reported_from_none_reported() -> None:
    """``null`` means we have no status block; ``[]`` means the CR has none."""
    script = f"""
        {_conditions_import_script()}
        const unknown = String(mod.Conditions({{ conditions: null }}));
        const empty = String(mod.Conditions({{ conditions: [] }}));
        console.log(JSON.stringify({{
          unknownSaysNotReported: unknown.includes('Conditions not reported'),
          unknownAvoidsNoConditions: !unknown.includes('>No conditions<'),
          emptySaysNoConditions: empty.includes('No conditions'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "unknownSaysNotReported": True,
        "unknownAvoidsNoConditions": True,
        "emptySaysNoConditions": True,
    }


def test_diagnostics_badge_coerces_optional_counters_before_summing() -> None:
    """``undefined + 3`` rendered the literal badge text "NaN"."""
    source = (COMPONENTS / "diagnostics-panel.js").read_text(encoding="utf-8")

    assert "const warnCount = Number(conditionWarnCount) || 0;" in source
    assert "const crashCount = Number(podCrashCount) || 0;" in source
    assert (
        "(warnCount > 0 || crashCount > 0) ? (warnCount + crashCount) : null" in source
    )
    assert "conditionWarnCount + podCrashCount" not in source


def test_http_status_of_separates_a_server_answer_from_no_answer() -> None:
    """``null`` status means the request never reached an HTTP response.

    "the server said 404" and "the request did not complete" are different
    claims; a UI that collapses them tells the reader to go look at the wrong
    layer.
    """
    script = f"""
        import {{ readFileSync }} from 'node:fs';
        let source = readFileSync({str(LIB / "api.js")!r}, 'utf8');
        source = source.replace(
          /^import \\{{[\\s\\S]*?\\}} from '\\.\\/state\\.js';$/m,
          {_STATE_STUBS!r},
        );
        const url = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const {{ httpStatusOf }} = await import(url);
        const withStatus = new Error('API 404: nope');
        withStatus.status = 404;
        console.log(JSON.stringify({{
          fromStatusField: httpStatusOf(withStatus),
          networkFailure: httpStatusOf(new TypeError('Failed to fetch')),
          messageContaining404IsNotAStatus: httpStatusOf(
            new Error('API 500: upstream returned 404'),
          ),
          nullish: httpStatusOf(null),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "fromStatusField": 404,
        "networkFailure": None,
        "messageContaining404IsNotAStatus": None,
        "nullish": None,
    }


def _log_strip_script(expression: str) -> str:
    return f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(COMPONENTS / "log-strip.js")!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace('export function LogStrip', 'function LogStrip');
        eval(source + '\\nglobalThis.emptyStripMessage = emptyStripMessage;');
        {expression}
    """


def test_log_strip_empty_body_says_which_kind_of_empty_it_is() -> None:
    """A blank strip reads identically to a dead jobs feed.

    The Warn filter can never match -- phaseSeverity only ever returns 'info'
    or 'error' -- so clicking it produced a silent blank panel with no hint
    that the filter, not the data, was responsible.
    """
    script = _log_strip_script(
        """
        console.log(JSON.stringify({
          cold: emptyStripMessage(0, 'all'),
          filteredOut: emptyStripMessage(12, 'warn'),
          allFilterWithEntries: emptyStripMessage(12, 'all'),
        }));
        """
    )

    result = json.loads(run_node(script))
    assert result["cold"].startswith("No lifecycle events yet")
    assert result["filteredOut"] == "No warn events among the 12 recorded."
    assert result["allFilterWithEntries"] == "No events to show (12 recorded)."


def test_log_strip_renders_the_empty_message_element() -> None:
    source = (COMPONENTS / "log-strip.js").read_text(encoding="utf-8")
    style = (UI_ROOT / "style.css").read_text(encoding="utf-8")

    assert "visible.length === 0 && html" in source
    assert 'data-testid="log-strip-empty"' in source
    assert ".log-strip-empty" in style


def test_logs_tab_live_label_tracks_the_connection_not_the_follow_toggle() -> None:
    """A closed follow stream kept advertising "· live" forever.

    The read loop breaks on done and the async body returns; nothing reset the
    header, so a pod whose stream the apiserver closed looked identical to one
    still streaming.
    """
    source = (COMPONENTS / "diagnostics-logs-tab.js").read_text(encoding="utf-8")

    assert "const [streamState, setStreamState] = useState('idle');" in source
    assert "if (!ac.signal.aborted) setStreamState('streaming');" in source
    assert "if (!ac.signal.aborted) setStreamState('ended');" in source
    assert "streamState === 'streaming' ? ' · live'" in source
    assert "streamState === 'ended' ? ' · stream ended'" in source
    # The old unconditional claim must be gone.
    assert "${follow ? ' · live' : ''}" not in source


def test_logs_tab_separates_still_loading_from_no_output() -> None:
    source = (COMPONENTS / "diagnostics-logs-tab.js").read_text(encoding="utf-8")

    assert "'Loading logs…'" in source
    assert "'No log output from this container.'" in source
    assert 'data-testid="run-logs-empty"' in source


def test_logs_tab_reads_404_off_the_status_not_the_message() -> None:
    source = (COMPONENTS / "diagnostics-logs-tab.js").read_text(encoding="utf-8")

    assert "httpStatusOf(e) === 404" in source
    assert "/\\b404\\b/.test(e.message)" not in source

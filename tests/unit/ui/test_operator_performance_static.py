# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static performance footgun checks for the operator UI."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"

_EFFECT_START_RE = re.compile(r"useEffect\s*\(")
_ADD_EVENT_RE = re.compile(
    r"(?P<target>[\w.]+)\.addEventListener\(\s*['\"](?P<event>[^'\"]+)['\"]"
)
_JSON_STRINGIFY_RE = re.compile(r"JSON\.stringify\(")

# These are bounded/persistence/transport cases, not render-loop or deep-diff cases.
_ALLOWED_JSON_STRINGIFY_CALLS = {
    ("components/artifacts-card.js", 184),
    ("components/artifacts-card.js", 228),
    # saveHiddenCols -> localStorage. Serializing a short Set of column names for
    # persistence is not a render or effect diff, which is what this guard exists
    # to catch. Was line 52; moved when readoutDecimals was added above it.
    ("components/job-table.js", 72),
    ("components/relaunch-button.js", 147),
    ("components/server-metrics/helpers.js", 173),
    ("lib/api.js", 112),
    ("lib/api.js", 239),  # createJob POST body — bounded transport, not render/diff
    ("lib/api.js", 249),  # createSweep POST body — bounded transport, not render/diff
    ("lib/job-ws.js", 127),
}

# Existing debt: dependency is a short list of child names, not full child objects.
# Keep this explicit so future large-object stringify dependencies fail closed.
_ALLOWED_JSON_STRINGIFY_EFFECT_DEPS = {
    # childSummaries effect; dependency is a short list of child NAMES, not the
    # child objects. Pinned by line, so it moves whenever code is inserted above
    # it -- was 305 before the epoch-ownership guard was added, then 329 before
    # the children-fetch skip gained its rationale comment, then 332 after the
    # state-aware sweep presentation helpers were moved, then 328 after the
    # sweepConfig state + effect were inserted above this block.
    ("pages/sweep-detail.js", 328),
}

# Module-singleton listeners live for the app lifetime; component listeners must clean up.
_ALLOWED_APP_LIFETIME_LISTENERS = {
    ("lib/router.js", 69),
    ("lib/router.js", 70),
    ("lib/theme-switch.js", 78),
}


def _ui_js_files() -> list[Path]:
    return sorted(
        path
        for path in _UI_ROOT.rglob("*.js")
        if path.is_file() and "vendor" not in path.parts
    )


def _source(path: Path) -> str:
    return path.read_text()


def _relative(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _line_number(src: str, offset: int) -> int:
    return src[:offset].count("\n") + 1


def _line_has_effect_dependency_array(src: str, offset: int) -> bool:
    tail = src[offset : offset + 30000]
    return re.search(r"\},\s*\[[^\]]*\]\s*\)", tail) is not None


def _json_stringify_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        rel = _relative(path)
        src = _source(path)
        for match in _JSON_STRINGIFY_RE.finditer(src):
            line = _line_number(src, match.start())
            if (rel, line) in _ALLOWED_JSON_STRINGIFY_CALLS:
                continue
            if (rel, line) in _ALLOWED_JSON_STRINGIFY_EFFECT_DEPS:
                continue
            violations.append(
                f"{rel}:{line} JSON.stringify can hide large-object render/effect churn"
            )
    return violations


def _unclean_event_listener_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        rel = _relative(path)
        src = _source(path)
        for match in _ADD_EVENT_RE.finditer(src):
            line = _line_number(src, match.start())
            if (rel, line) in _ALLOWED_APP_LIFETIME_LISTENERS:
                continue
            event = match.group("event")
            target = match.group("target")
            cleanup = (
                f"{target}.removeEventListener('{event}'" in src
                or f'{target}.removeEventListener("{event}"' in src
            )
            one_shot = "once: true" in src[match.start() : match.start() + 500]
            if not cleanup and not one_shot:
                violations.append(
                    f"{rel}:{line} {target}.{event} listener has no cleanup or once:true"
                )
    return violations


def _effect_timer_poll_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        rel = _relative(path)
        src = _source(path)
        if (
            "poll(" in src
            and "useEffect" in src
            and (
                "new AbortController" not in src
                or ".signal" not in src
                or ".abort()" not in src
            )
        ):
            violations.append(f"{rel} poll effect is missing AbortController cleanup")
        if "setInterval(" in src and "clearInterval(" not in src:
            violations.append(f"{rel} interval effect is missing clearInterval cleanup")
    return violations


def _unstable_fetch_effect_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        rel = _relative(path)
        src = _source(path)
        for match in _JSON_STRINGIFY_RE.finditer(src):
            line = _line_number(src, match.start())
            if (rel, line) in _ALLOWED_JSON_STRINGIFY_EFFECT_DEPS:
                continue
            preceding_effect = src.rfind("useEffect", 0, match.start())
            following_deps = src.find("],", match.end(), match.end() + 300)
            if preceding_effect != -1 and following_deps != -1:
                violations.append(
                    f"{rel}:{line} fetch effect depends on JSON.stringify"
                )
        for match in _EFFECT_START_RE.finditer(src):
            effect_tail = src[match.start() : match.start() + 5000]
            if (
                "fetch(" not in effect_tail
                and "api." not in effect_tail
                and "poll(" not in effect_tail
            ):
                continue
            if not _line_has_effect_dependency_array(src, match.start()):
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} fetch/poll effect has no explicit dependency array"
                )
    return violations


def test_json_stringify_is_not_used_for_large_render_or_effect_diffs() -> None:
    assert _json_stringify_violations() == []


def test_polling_timers_and_render_tick_effects_have_cleanup() -> None:
    assert _effect_timer_poll_violations() == []


def test_component_event_listeners_are_removed_or_one_shot() -> None:
    assert _unclean_event_listener_violations() == []


def test_fetching_effects_have_stable_explicit_dependencies() -> None:
    assert _unstable_fetch_effect_violations() == []


def test_dashboard_recent_jobs_table_is_bounded_before_render_mapping() -> None:
    dashboard = _source(_UI_ROOT / "pages" / "dashboard.js")
    helpers = _source(_UI_ROOT / "pages" / "dashboard-helpers.js")
    assert "const recent = recentJobs(allJobs);" in dashboard
    assert "recent.map((job, i)" in dashboard
    assert "function recentJobs(jobList, limit = 5)" in helpers
    assert "if (top.length > limit) top.pop();" in helpers


def test_job_table_large_array_rendering_is_scroll_bounded_and_memoized() -> None:
    job_table = _source(_UI_ROOT / "components" / "job-table.js")
    style = _source(_UI_ROOT / "style.css")
    assert "const filtered = filter && filter.length > 0" in job_table
    assert "const sorted = [...filtered].sort" in job_table
    assert '<div class="job-table-wrapper">' in job_table
    assert "max-height: calc(100vh - 240px)" in style
    assert "overflow-y: auto" in style


def test_chart_wrapper_avoids_option_recreation_and_cleans_chart_instances() -> None:
    chart_wrapper = _source(_UI_ROOT / "components" / "chart-wrapper.js")
    assert "JSON.stringify" not in chart_wrapper
    assert "function dataFingerprint(data)" in chart_wrapper
    assert "function optionsFingerprint(value, seen = new WeakSet())" in chart_wrapper
    assert "chartRef.current.destroy();" in chart_wrapper
    assert "}, [type, hasData]);" in chart_wrapper
    assert "}, [data]);" in chart_wrapper
    assert "}, [options]);" in chart_wrapper

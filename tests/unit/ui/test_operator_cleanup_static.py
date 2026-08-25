# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static cleanup/leak checks for operator UI modules."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"

_LISTENER_RE = re.compile(
    r"(?P<target>[A-Za-z_$][\w$.]*)\.addEventListener\(\s*(?P<quote>['\"])(?P<event>[^'\"]+)(?P=quote)"
)
_ASSIGNED_TIMEOUT_RE = re.compile(
    r"(?:(?:const|let|var)\s+)?(?P<name>[A-Za-z_$][\w$]*)\s*=\s*setTimeout\("
)
_INTERVAL_RE = re.compile(r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*setInterval\(")
_POLL_RE = re.compile(r"\bpoll\s*\(")

_GLOBAL_LISTENER_ALLOWLIST = {
    # Router listeners are app-lifetime signal wiring installed once by module import.
    ("lib/router.js", "window", "hashchange"),
    ("lib/router.js", "window", "load"),
    # Theme follows OS color-scheme changes for the whole app and is guarded by initialized.
    ("lib/theme-switch.js", "mediaQuery", "change"),
}

# Keyed by the call text rather than by line number: an allowlist that drifts
# every time an unrelated edit shifts a line stops protecting anything, and
# re-numbering it is exactly the reflex that would silently re-admit a real leak.
_ONE_SHOT_TIMEOUT_ALLOWLIST = {
    # User-initiated feedback/download timers do not keep polling or external
    # resources alive.
    ("components/artifacts-card.js", "setTimeout(() => setCopyLabel('Copy'), 2000)"),
    ("components/artifacts-card.js", "setTimeout(() => downloadFile(f.name), i * 300)"),
    ("pages/job-detail.js", "setTimeout(() => setCopyLabel('Copy'), 2000)"),
    # Zero-delay parsing yield is bounded by the aborted signal checks around the loop.
    ("pages/job-detail.js", "setTimeout(r, 0)"),
}


def _ui_js_files() -> list[Path]:
    return sorted(path for path in _UI_ROOT.rglob("*.js") if "vendor" not in path.parts)


def _source(path: Path) -> str:
    return path.read_text()


def _relative(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _line_number(src: str, offset: int) -> int:
    return src[:offset].count("\n") + 1


def _has_remove_listener(src: str, target: str, event: str) -> bool:
    return (
        f"{target}.removeEventListener('{event}'" in src
        or f'{target}.removeEventListener("{event}"' in src
    )


def _call_text(src: str, open_paren: int) -> str:
    depth = 0
    for idx in range(open_paren, len(src)):
        char = src[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return src[open_paren : idx + 1]
    return src[open_paren:]


def _is_once_abort_listener(src: str, match: re.Match[str]) -> bool:
    open_paren = src.find("(", match.start())
    return match.group("event") == "abort" and "once: true" in _call_text(
        src, open_paren
    )


def _listener_cleanup_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        src = _source(path)
        rel = _relative(path)
        for match in _LISTENER_RE.finditer(src):
            target = match.group("target")
            event = match.group("event")
            if (rel, target, event) in _GLOBAL_LISTENER_ALLOWLIST:
                continue
            if _is_once_abort_listener(src, match):
                continue
            if not _has_remove_listener(src, target, event):
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} {target}.addEventListener('{event}') has no cleanup"
                )
    return violations


def _timer_cleanup_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        src = _source(path)
        rel = _relative(path)
        for match in _INTERVAL_RE.finditer(src):
            name = match.group("name")
            if f"clearInterval({name})" not in src:
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} setInterval handle {name!r} is never cleared"
                )
        for match in _ASSIGNED_TIMEOUT_RE.finditer(src):
            name = match.group("name")
            if f"clearTimeout({name})" not in src:
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} setTimeout handle {name!r} is never cleared"
                )
        for match in re.finditer(r"\bsetTimeout\(", src):
            line = _line_number(src, match.start())
            call = _call_text(src, src.index("(", match.start()))
            if (rel, f"setTimeout{call}") in _ONE_SHOT_TIMEOUT_ALLOWLIST:
                continue
            prefix = src[max(0, match.start() - 80) : match.start()]
            if re.search(r"(?:=|return\s+new\s+Promise\([^)]*=>\s*)\s*$", prefix):
                continue
            violations.append(
                f"{rel}:{line} untracked setTimeout requires explicit allowlist"
            )
    return violations


def _poll_cleanup_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_js_files():
        src = _source(path)
        rel = _relative(path)
        if "export function poll" in src:
            continue
        for match in _POLL_RE.finditer(src):
            call = _call_text(src, src.find("(", match.start()))
            if ".signal" not in call:
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} poll call does not receive an AbortSignal"
                )
                continue
            effect_start = src.rfind("useEffect", 0, match.start())
            effect_end = src.find("\n  },", match.end())
            effect_body = (
                src[effect_start:effect_end]
                if effect_start != -1 and effect_end != -1
                else src
            )
            if ".abort()" not in effect_body:
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel}:{line} poll AbortSignal is not aborted by effect cleanup"
                )
    return violations


def test_event_listeners_are_removed_or_declared_app_lifetime() -> None:
    assert _listener_cleanup_violations() == []


def test_intervals_and_tracked_timeouts_are_cleared() -> None:
    assert _timer_cleanup_violations() == []


def test_poll_loops_are_abortable_from_effect_cleanup() -> None:
    assert _poll_cleanup_violations() == []


def test_job_websocket_close_stops_reconnect_and_closes_socket() -> None:
    src = _source(_UI_ROOT / "lib" / "job-ws.js")

    assert "ws.onclose =" in src
    assert "reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);" in src
    assert "closed = true;" in src
    assert "clearTimeout(reconnectTimer);" in src
    assert "ws.close(1000, 'page leaving');" in src


def test_job_detail_websocket_effect_closes_subscription_handle() -> None:
    src = _source(_UI_ROOT / "pages" / "job-detail.js")
    ws_effect_start = src.index("const handle = openJobWs(namespace, name")
    ws_effect = src[
        ws_effect_start : src.index("}, [namespace, name, wsActive]);", ws_effect_start)
    ]
    cleanup = ws_effect.split("return () => {", 1)[1]

    # The cleanup does more than close now (it also drops the connected ref),
    # so assert the close happens rather than the exact one-liner it used to be.
    assert "handle.close();" in cleanup


def test_streaming_log_effects_abort_fetches_on_unmount() -> None:
    for rel in ("components/diagnostics-logs-tab.js",):
        src = _source(_UI_ROOT / rel)
        assert "const ac = new AbortController();" in src
        assert "signal: ac.signal" in src
        assert "if (ac.signal.aborted) return;" in src
        assert "return () => ac.abort();" in src


def test_global_listener_modules_are_singleton_lifetimes() -> None:
    router_src = _source(_UI_ROOT / "lib" / "router.js")
    theme_src = _source(_UI_ROOT / "lib" / "theme-switch.js")

    assert "window.addEventListener('hashchange', syncFromHash);" in router_src
    assert "window.addEventListener('load', syncFromHash);" in router_src
    assert "let initialized = false;" in theme_src
    assert "if (!isBrowser() || initialized) return;" in theme_src
    assert "mediaQuery.addEventListener('change', mediaListener);" in theme_src

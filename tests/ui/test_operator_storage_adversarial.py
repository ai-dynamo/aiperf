# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial static tests for operator UI Web Storage interactions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_CONFIG_REDACTION_JS = _UI_ROOT / "lib" / "config-redaction.js"
_JOB_TABLE_JS = _UI_ROOT / "components" / "job-table.js"
_THEME_SWITCH_JS = _UI_ROOT / "lib" / "theme-switch.js"
_INDEX_HTML = _UI_ROOT / "index.html"

_HIDDEN_COLS_KEY = "aiperf-ui-v1.job-table.hidden-cols"
_THEME_KEY = "aiperfTheme"


def _source(path: Path) -> str:
    return path.read_text()


def _without_comments(source: str) -> str:
    """Strip /* */, // and <!-- --> comments so assertions test code, not prose."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"<!--.*?-->", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


def _function_body(source: str, function_name: str) -> str:
    signature = re.search(
        rf"(?:export\s+)?function {re.escape(function_name)}\([^)]*\) \{{", source
    )
    assert signature is not None, f"{function_name} must remain statically testable"
    start = signature.end() - 1
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"{function_name} body was not balanced")


def _storage_literal_keys(source: str) -> list[str]:
    return re.findall(
        r"(?:sessionStorage|localStorage|window\.localStorage)\.(?:getItem|setItem|removeItem)\('([^']+)'",
        source,
    )


def test_storage_keys_do_not_collide_between_session_and_local_preferences() -> None:
    """Web Storage keys should be purpose-specific so writes cannot overwrite unrelated state."""
    sources = {
        "job-table.js": _source(_JOB_TABLE_JS),
        "theme-switch.js": _source(_THEME_SWITCH_JS),
        "index.html": _source(_INDEX_HTML),
    }
    literal_keys = {name: _storage_literal_keys(src) for name, src in sources.items()}
    all_literal_keys = [key for keys in literal_keys.values() for key in keys]

    assert _HIDDEN_COLS_KEY in _source(_JOB_TABLE_JS)
    # The theme preference key is intentionally gone (see the dark-only guard
    # below), so it can no longer collide with anything.
    assert _THEME_KEY not in all_literal_keys


def test_relaunch_storage_payload_redacts_sensitive_data_before_serializing() -> None:
    """The pure redaction helper should scrub nested secret-like keys before sessionStorage."""
    script = f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({json.dumps(str(_CONFIG_REDACTION_JS))}, 'utf8');
        const helpers = source.replace(/export /g, '');
        eval(helpers + `
          const redacted = redactConfigForYaml({{
            endpoint: {{
              api_key: 'sk-live',
              headers: {{ Authorization: 'Bearer token', safe: 'keep' }},
            }},
            nested: [{{ client_secret: 'secret-value', model: 'llama' }}],
            passwordFile: '/tmp/password.txt',
          }});
          console.log(JSON.stringify(redacted));
        `);
    """

    result = json.loads(run_node(script))

    assert result["endpoint"]["api_key"] == "[REDACTED]"
    assert result["endpoint"]["headers"]["Authorization"] == "[REDACTED]"
    assert result["endpoint"]["headers"]["safe"] == "keep"
    assert result["nested"][0]["client_secret"] == "[REDACTED]"
    assert result["nested"][0]["model"] == "llama"
    assert result["passwordFile"] == "[REDACTED]"


def test_local_storage_unavailable_or_malformed_preferences_fall_back_safely() -> None:
    """localStorage access should be guarded and default to visible columns."""
    job_table = _source(_JOB_TABLE_JS)
    load_body = _function_body(job_table, "loadHiddenCols")
    save_body = _function_body(job_table, "saveHiddenCols")

    assert "typeof localStorage === 'undefined'" in load_body
    assert "JSON.parse(raw)" in load_body
    assert "catch {\n    return new Set();\n  }" in load_body
    assert "typeof localStorage === 'undefined'" in save_body
    assert "localStorage.setItem(HIDDEN_COLS_STORAGE_KEY" in save_body
    assert "catch { /* quota / private mode — silent */ }" in save_body


def test_theme_is_a_constant_and_reads_no_persisted_preference() -> None:
    """The dashboard is dark-only; nothing may resolve a theme at runtime.

    This replaces a pair of tests that asserted ``getTheme``/``setTheme``
    degraded gracefully when localStorage threw. Those guarded the read path of
    a preference that must no longer exist: resolving ``'auto'`` against
    ``prefers-color-scheme`` put ``data-theme="light"`` on ``<html>`` for every
    light-OS visitor, and style.css only ever partially neutralized the light
    palette, so 13 of its 73 custom properties leaked into the dark UI. There is
    also no coherent light rendering available -- ``lib/theme.js`` is a
    hardcoded dark palette imported by every Chart.js consumer. Keeping the
    theme a constant is the invariant; storage robustness was only ever a proxy
    for it.
    """
    theme_switch = _source(_THEME_SWITCH_JS)
    index_html = _source(_INDEX_HTML)

    # Comments are stripped: both files keep a note explaining what the removed
    # preference resolution used to do, and that prose must stay greppable.
    for name, source in (("theme-switch.js", theme_switch), ("index.html", index_html)):
        code = _without_comments(source)
        assert "localStorage" not in code, name
        assert "matchMedia" not in code, name
        assert "prefers-color-scheme" not in code, name
        assert _THEME_KEY not in code, name
        assert "'light'" not in code, name

    assert "const THEME = 'dark';" in theme_switch
    assert "document.documentElement.dataset.theme = THEME;" in theme_switch
    assert "document.documentElement.dataset.theme = 'dark';" in index_html
    # The removed switching API must not come back without the CSS to support it.
    for dead in (
        "export function setTheme",
        "export function cycleTheme",
        "export function getResolvedTheme",
        "themechange",
    ):
        assert dead not in _without_comments(theme_switch), dead

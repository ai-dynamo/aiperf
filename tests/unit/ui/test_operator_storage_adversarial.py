# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial static tests for operator UI Web Storage interactions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_LAUNCH_JS = _UI_ROOT / "pages" / "launch.js"
_RELAUNCH_BUTTON_JS = _UI_ROOT / "components" / "relaunch-button.js"
_JOB_TABLE_JS = _UI_ROOT / "components" / "job-table.js"
_THEME_SWITCH_JS = _UI_ROOT / "lib" / "theme-switch.js"
_INDEX_HTML = _UI_ROOT / "index.html"

_PREFILL_KEY = "aiperf.launch.prefill"
_HIDDEN_COLS_KEY = "aiperf-ui-v1.job-table.hidden-cols"
_THEME_KEY = "aiperfTheme"


def _source(path: Path) -> str:
    return path.read_text()


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


def test_relaunch_prefill_quota_or_security_errors_block_navigation() -> None:
    """QuotaExceededError/SecurityError must not navigate to an empty Launch page."""
    body = _function_body(_source(_RELAUNCH_BUTTON_JS), "RelaunchButton")

    set_pos = body.index(f"sessionStorage.setItem('{_PREFILL_KEY}'")
    catch_pos = body.index("catch (err)", set_pos)
    warn_pos = body.index(
        "console.warn('Unable to prepare launch prefill', err);", catch_pos
    )
    return_pos = body.index("return;", warn_pos)
    navigate_pos = body.index("navigate('/launch')")

    assert set_pos < catch_pos < warn_pos < return_pos < navigate_pos


def test_launch_prefill_malformed_json_and_stale_ttl_are_ignored() -> None:
    """Bad or stale sessionStorage payloads should be consumed without mutating editor state."""
    body = _function_body(_source(_LAUNCH_JS), "Launch")

    parse_pos = body.index("JSON.parse(raw)")
    parse_catch_pos = body.index("catch (_e) { return; }", parse_pos)
    yaml_type_pos = body.index("typeof payload.yaml !== 'string'", parse_catch_pos)
    stale_pos = body.index("Date.now() - payload.at > 60000", yaml_type_pos)
    set_yaml_pos = body.index("setYaml(payload.yaml)", stale_pos)

    assert parse_pos < parse_catch_pos < yaml_type_pos < stale_pos < set_yaml_pos


def test_launch_prefill_is_one_shot_even_when_payload_is_malformed_or_stale() -> None:
    """The handoff key must be removed before JSON/TTL validation to prevent replay loops."""
    body = _function_body(_source(_LAUNCH_JS), "Launch")

    get_pos = body.index(f"sessionStorage.getItem('{_PREFILL_KEY}')")
    remove_pos = body.index(f"sessionStorage.removeItem('{_PREFILL_KEY}')")
    parse_pos = body.index("JSON.parse(raw)")
    stale_pos = body.index("Date.now() - payload.at > 60000")

    assert get_pos < remove_pos < parse_pos < stale_pos
    assert body.count(f"sessionStorage.removeItem('{_PREFILL_KEY}')") == 1


def test_storage_keys_do_not_collide_between_session_and_local_preferences() -> None:
    """Web Storage keys should be purpose-specific so writes cannot overwrite unrelated state."""
    sources = {
        "launch.js": _source(_LAUNCH_JS),
        "relaunch-button.js": _source(_RELAUNCH_BUTTON_JS),
        "job-table.js": _source(_JOB_TABLE_JS),
        "theme-switch.js": _source(_THEME_SWITCH_JS),
        "index.html": _source(_INDEX_HTML),
    }
    literal_keys = {name: _storage_literal_keys(src) for name, src in sources.items()}
    all_literal_keys = [key for keys in literal_keys.values() for key in keys]

    assert literal_keys["launch.js"] == [_PREFILL_KEY, _PREFILL_KEY]
    assert literal_keys["relaunch-button.js"] == [_PREFILL_KEY]
    assert _HIDDEN_COLS_KEY in _source(_JOB_TABLE_JS)
    assert _THEME_KEY in all_literal_keys
    assert len({_PREFILL_KEY, _HIDDEN_COLS_KEY, _THEME_KEY}) == 3
    assert _PREFILL_KEY not in {_HIDDEN_COLS_KEY, _THEME_KEY}


def test_relaunch_storage_payload_redacts_sensitive_data_before_serializing() -> None:
    """The pure redaction helper should scrub nested secret-like keys before sessionStorage."""
    script = f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({json.dumps(str(_RELAUNCH_BUTTON_JS))}, 'utf8');
        const helpers = source
          .slice(0, source.indexOf('/**\\n * Minimal YAML serializer'))
          .replace(/^import .*$/gm, '')
          .replace(/export /g, '');
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
    """localStorage access should be guarded and default to visible columns/theme auto."""
    job_table = _source(_JOB_TABLE_JS)
    theme_switch = _source(_THEME_SWITCH_JS)
    load_body = _function_body(job_table, "loadHiddenCols")
    save_body = _function_body(job_table, "saveHiddenCols")
    get_theme_body = _function_body(theme_switch, "getTheme")
    set_theme_body = _function_body(theme_switch, "setTheme")

    assert "typeof localStorage === 'undefined'" in load_body
    assert "JSON.parse(raw)" in load_body
    assert "catch {\n    return new Set();\n  }" in load_body
    assert "typeof localStorage === 'undefined'" in save_body
    assert "localStorage.setItem(HIDDEN_COLS_STORAGE_KEY" in save_body
    assert "catch { /* quota / private mode — silent */ }" in save_body
    assert "window.localStorage.getItem(STORAGE_KEY)" in get_theme_body
    assert "catch (_) {\n    return 'auto';\n  }" in get_theme_body
    assert "window.localStorage.setItem(STORAGE_KEY, pref)" in set_theme_body
    assert "localStorage unavailable (private mode, quota)" in set_theme_body


def test_index_theme_bootstrap_falls_back_when_local_storage_is_blocked() -> None:
    """The synchronous pre-React bootstrap must survive SecurityError from localStorage."""
    source = _source(_INDEX_HTML)

    assert "try {" in source
    assert "localStorage.getItem('aiperfTheme') || 'auto'" in source
    assert "catch (e)" in source
    assert "document.documentElement.dataset.theme = 'dark';" in source
